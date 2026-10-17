#include "pulsarlm/ops.hpp"
#include "pulsarlm/rope.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAFunctions.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

// Kernel correctness against an ATen reference. Also checks that the op is
// reachable through the dispatcher, i.e. registration took effect.

TEST_CASE("rmsnorm matches ATen reference", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const double eps = 1e-6;
    auto x = at::randn({4, 512}, at::device(at::kCUDA).dtype(at::kFloat));
    auto w = at::randn({512}, at::device(at::kCUDA).dtype(at::kFloat));

    auto got = pulsarlm::rmsnorm_cuda(x, w, eps);

    auto ms = x.pow(2).mean(-1, /*keepdim=*/true);
    auto ref = x * at::rsqrt(ms + eps) * w;

    REQUIRE(at::allclose(got, ref, /*rtol=*/1e-5, /*atol=*/1e-5));

    // eps is what keeps an all-zero row finite; on unit-variance input it shifts the
    // result far below any tolerance, so nothing above would notice it being dropped.
    auto zeros = at::zeros({4, 512}, at::device(at::kCUDA).dtype(at::kFloat));
    REQUIRE(at::allclose(pulsarlm::rmsnorm_cuda(zeros, w, eps), zeros));
}

// The mass side output sets each view page's W exp(S / W) per query head, with S the sum over the queries attending the
// page of their mass weight times the log of their max share of its keys, and W the sum of those weights. fp32 runs the
// scalar kernels, bf16 the tensor-core ones, so each case covers both.
namespace {

constexpr int64_t kPageSize = 16;
constexpr int64_t kHeadDim = 64;
constexpr double kTheta = 1e6;
const double kScale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));

at::Tensor int32_cuda(const std::vector<int32_t>& v) {
    return at::tensor(v, at::dtype(at::kInt)).to(at::kCUDA);
}

at::Tensor fp32_cuda(const std::vector<float>& v) {
    return at::tensor(v, at::dtype(at::kFloat)).to(at::kCUDA);
}

// {0, 0, -1} per sequence: every key at its own view index.
at::Tensor identity_layout(int64_t num_seqs) {
    return int32_cuda({0, 0, -1}).repeat({num_seqs, 1});
}

// A key pool of num_pages zeroed pages. Every score is q.k == 0 at any rotation, so a
// query's attention is uniform over the keys it attends and each key's share is
// exactly 1 / (keys attended). One kv head, one query head.
at::Tensor zero_pool(int64_t num_pages, at::ScalarType dt) {
    return at::zeros({num_pages, kPageSize, 1, kHeadDim}, at::device(at::kCUDA).dtype(dt));
}

at::Tensor zero_mass(int64_t num_pages) {
    return at::zeros({num_pages, 1}, at::device(at::kCUDA).dtype(at::kFloat));
}

// Every sequence's view is page_tables row s, its mass rows one view page per table
// column, so view page b of sequence s is mass row s * max_pages + b.
at::Tensor row_view_pages(const at::Tensor& page_tables) {
    const int64_t num_seqs = page_tables.size(0), max_pages = page_tables.size(1);
    return at::arange(0, (num_seqs + 1) * max_pages, max_pages, at::device(at::kCUDA).dtype(at::kInt));
}

// The ops' mass, built from each query's attention weights in each head.
class mass_reference {
    at::Tensor log_sum;  // [view pages, heads]
    at::Tensor weight_sum;  // [view pages, 1]
    int64_t page_size;

  public:
    mass_reference(int64_t view_pages, int64_t heads, int64_t page_size)
    : log_sum(at::zeros({view_pages, heads}, at::device(at::kCUDA).dtype(at::kFloat))),
      weight_sum(at::zeros({view_pages, 1}, at::device(at::kCUDA).dtype(at::kFloat))),
      page_size(page_size) {}

    // A query of mass weight w whose head h gave the keys [first_key, first_key + shares.size(0)) of the view whose
    // page 0 is mass row first_row the weights shares. Each query adds its weight to W once, with its head 0.
    void add(int64_t first_row, int64_t first_key, const at::Tensor& shares, int64_t h, double w) {
        const int64_t first_page = first_key / this->page_size;
        const int64_t pages = (first_key + shares.size(0) - 1) / this->page_size - first_page + 1;
        auto padded = at::zeros({pages * this->page_size}, shares.options());
        padded.narrow(0, first_key - first_page * this->page_size, shares.size(0)).copy_(shares);
        auto page_max = padded.view({pages, this->page_size}).amax(-1);
        this->log_sum.narrow(0, first_row + first_page, pages).select(1, h).add_(at::log(page_max) * w);
        if (h == 0) {
            this->weight_sum.narrow(0, first_row + first_page, pages).add_(w);
        }
    }

    at::Tensor mass() const {
        return at::where(
            this->weight_sum > 0,
            this->weight_sum * at::exp(this->log_sum / this->weight_sum),
            at::zeros_like(this->log_sum)
        );
    }
};

at::Tensor prefill(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& page_tables,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& seqlens_k,
    const at::Tensor& mass,
    const at::Tensor& mass_weights
) {
    return pulsarlm::attn_prefill_cuda(
        q,
        k_pool,
        v_pool,
        page_tables,
        cu_seqlens_q,
        seqlens_k,
        identity_layout(seqlens_k.size(0)),
        pulsarlm::rope_parameters{.theta = kTheta},
        kScale,
        std::nullopt,
        std::nullopt,
        mass,
        row_view_pages(page_tables),
        mass_weights
    );
}

at::Tensor decode(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& page_tables,
    const at::Tensor& context_lens,
    const at::Tensor& mass,
    const at::Tensor& mass_weights,
    int64_t num_splits = 0
) {
    return pulsarlm::attn_decode_cuda(
        q,
        k_pool,
        v_pool,
        page_tables,
        context_lens,
        identity_layout(context_lens.size(0)),
        pulsarlm::rope_parameters{.theta = kTheta},
        kScale,
        std::nullopt,
        std::nullopt,
        mass,
        row_view_pages(page_tables),
        mass_weights,
        num_splits
    );
}

}  // namespace

// Query i attends keys [0, i] uniformly, so its share of every page it reaches is 1 / (i + 1), and page b is attended
// by the queries from b * page_size on.
TEST_CASE("prefill mass is the weighted geometric mean share of the queries attending each page", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const int64_t ntok = 64, num_pages = ntok / kPageSize;
    std::vector<float> weights;
    for (int64_t i = 0; i < ntok; ++i) {
        weights.push_back(static_cast<float>(0.05 * std::pow(0.95, static_cast<double>(ntok - 1 - i))));
    }

    for (auto dt : {at::kFloat, at::kBFloat16}) {
        auto k_pool = zero_pool(num_pages, dt);
        auto v_pool = zero_pool(num_pages, dt);
        auto q = at::zeros({ntok, 1, kHeadDim}, at::device(at::kCUDA).dtype(dt));
        auto page_tables = at::arange(num_pages, at::device(at::kCUDA).dtype(at::kInt)).view({1, num_pages});
        auto cu_seqlens_q = int32_cuda({0, static_cast<int32_t>(ntok)});
        auto seqlens_k = int32_cuda({static_cast<int32_t>(ntok)});

        auto mass = zero_mass(num_pages);
        prefill(q, k_pool, v_pool, page_tables, cu_seqlens_q, seqlens_k, mass, fp32_cuda(weights));
        auto got = mass.view({-1}).to(at::kCPU);
        for (int64_t b = 0; b < num_pages; ++b) {
            double log_sum = 0.0, weight_sum = 0.0;
            for (int64_t i = b * kPageSize; i < ntok; ++i) {
                log_sum += weights[i] * std::log(1.0 / static_cast<double>(i + 1));
                weight_sum += weights[i];
            }
            const double want = weight_sum * std::exp(log_sum / weight_sum);
            REQUIRE(std::abs(got[b].item<double>() - want) < 1e-3 * want);
        }
    }
}

// Two sequences over disjoint pages, both with uniform attention, so every page holds 1 / its context of its
// sequence's one query: 16 keys for sequence 0, 128 for sequence 1.
TEST_CASE("decode mass is the query's weight times its share of each page", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const int64_t short_ctx = 16, long_ctx = 128;
    const int64_t max_pages = long_ctx / kPageSize;
    const int64_t num_pages = 2 * max_pages;
    const std::vector<float> weights{1.0f, 0.25f};

    std::vector<int32_t> tables;
    for (int32_t p = 0; p < 2 * max_pages; ++p) {
        tables.push_back(p);
    }
    auto page_tables = int32_cuda(tables).view({2, max_pages});
    auto context_lens = int32_cuda({static_cast<int32_t>(short_ctx), static_cast<int32_t>(long_ctx)});
    auto check = [&](const at::Tensor& mass) {
        auto got = mass.view({2, max_pages}).to(at::kCPU);
        REQUIRE(std::abs(got[0][0].item<double>() - weights[0] / short_ctx) < 1e-5);
        for (int64_t b = 1; b < max_pages; ++b) {
            REQUIRE(got[0][b].item<double>() == 0.0);
        }
        for (int64_t b = 0; b < max_pages; ++b) {
            REQUIRE(std::abs(got[1][b].item<double>() - weights[1] / long_ctx) < 1e-5);
        }
    };

    for (auto dt : {at::kFloat, at::kBFloat16}) {
        auto k_pool = zero_pool(num_pages, dt);
        auto v_pool = zero_pool(num_pages, dt);
        auto q = at::zeros({2, 1, kHeadDim}, at::device(at::kCUDA).dtype(dt));
        auto mass = zero_mass(num_pages);
        decode(q, k_pool, v_pool, page_tables, context_lens, mass, fp32_cuda(weights));
        check(mass);
    }

    // The split (flash-decode) path finishes the pages in its combine.
    auto k_pool = zero_pool(num_pages, at::kBFloat16);
    auto v_pool = zero_pool(num_pages, at::kBFloat16);
    auto q = at::zeros({2, 1, kHeadDim}, at::device(at::kCUDA).dtype(at::kBFloat16));
    auto split = zero_mass(num_pages);
    decode(q, k_pool, v_pool, page_tables, context_lens, split, fp32_cuda(weights), /*num_splits=*/2);
    check(split);
}

namespace {

// One sequence of the shared-line case: its view over the pool and its RoPE layout.
struct View {
    std::vector<int32_t> lines;
    int64_t ctx;
    int64_t seq_q;  // prefill queries, the last seq_q view indices
    std::array<int32_t, 3> layout;  // {n_sink, working_lo, short_offset}

    int64_t position(int64_t j) const {
        if (j < this->layout[0]) {
            return j;
        }
        if (j < this->layout[1]) {
            return this->layout[2];
        }
        return this->layout[2] + 1 + j - this->layout[1];
    }
};

// ATen reference for one view: its keys rotated to their layout positions by
// rope_rotate and rounded to the pool dtype as the kernels store them, then causal
// softmax attention for the query rows at view indices [ctx - rows, ctx). Fills o
// rows and adds each row's weights, at mass weight alpha * (1 - alpha)^(rows after it),
// to the view's mass rows from first_row on.
void reference_view(
    const View& view,
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    int64_t rows,
    int64_t q_offset,
    double alpha,
    at::Tensor& o,
    mass_reference& mass,
    int64_t first_row
) {
    const int64_t n_kv = k_pool.size(2), head_dim = k_pool.size(3);
    const int64_t group = q.size(1) / n_kv;
    auto lines = at::tensor(view.lines, at::dtype(at::kLong)).to(at::kCUDA);
    auto k = k_pool.index_select(0, lines).view({-1, n_kv, head_dim}).slice(0, 0, view.ctx).to(at::kFloat);
    auto v = v_pool.index_select(0, lines).view({-1, n_kv, head_dim}).slice(0, 0, view.ctx).to(at::kFloat);
    std::vector<int64_t> positions(view.ctx);
    for (int64_t j = 0; j < view.ctx; ++j) {
        positions[j] = view.position(j);
    }
    auto pos = at::tensor(positions, at::dtype(at::kLong)).to(at::kCUDA).view({-1, 1});
    k = pulsarlm::rope_rotate(k, pos, {.theta = kTheta}).to(k_pool.scalar_type()).to(at::kFloat);

    for (int64_t r = 0; r < rows; ++r) {
        const int64_t p = view.ctx - rows + r;
        const double graded = alpha * std::pow(1.0 - alpha, static_cast<double>(rows - 1 - r));
        for (int64_t h = 0; h < q.size(1); ++h) {
            const int64_t g = h / group;
            auto keys = k.slice(0, 0, p + 1).select(1, g);
            auto values = v.slice(0, 0, p + 1).select(1, g);
            auto w = at::softmax(at::mv(keys, q[q_offset + r][h].to(at::kFloat)) * kScale, 0);
            o[q_offset + r][h] = at::mv(values.t(), w);
            mass.add(first_row, 0, w, h, graded);
        }
    }
}

}  // namespace

// Two views share line 2: a full page of the second view, the partial tail of the
// first. The first view is laid out contiguously, the second compacted, so the shared
// line's keys sit at different positions in each and must be rotated per view, and its
// mass must land in each view's own rows. The reference rotates the unrotated pool with
// the ATen rope_rotate.
TEST_CASE("views sharing a line rotate and accumulate apart", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const int64_t n_q = 4, n_kv = 2, num_lines = 3;
    const std::vector<View> views = {
        View{{0, 2}, kPageSize + 5, 5, {0, 0, -1}},
        View{{1, 2, 0}, 2 * kPageSize + 7, 20, {4, 2 * kPageSize + 7 - 18, 6}},
    };
    const double alpha = 0.5;
    const int64_t max_pages = 3, total_view_pages = 2 * max_pages;

    std::vector<int32_t> tables(views.size() * max_pages, 0);
    std::vector<int32_t> layouts, seqlens, cu_q{0}, cu_view;
    for (size_t s = 0; s < views.size(); ++s) {
        std::copy(views[s].lines.begin(), views[s].lines.end(), tables.begin() + s * max_pages);
        layouts.insert(layouts.end(), views[s].layout.begin(), views[s].layout.end());
        seqlens.push_back(static_cast<int32_t>(views[s].ctx));
        cu_q.push_back(cu_q.back() + static_cast<int32_t>(views[s].seq_q));
        cu_view.push_back(static_cast<int32_t>(s * max_pages));
    }
    cu_view.push_back(static_cast<int32_t>(total_view_pages));
    auto page_tables = int32_cuda(tables).view({2, max_pages});
    auto rope_layout = int32_cuda(layouts).view({2, 3});
    auto seqlens_k = int32_cuda(seqlens);
    auto cu_seqlens_q = int32_cuda(cu_q);
    auto cu_view_pages = int32_cuda(cu_view);
    std::vector<float> pre_weights;
    for (const View& view : views) {
        for (int64_t r = 0; r < view.seq_q; ++r) {
            pre_weights.push_back(
                static_cast<float>(alpha * std::pow(1.0 - alpha, static_cast<double>(view.seq_q - 1 - r)))
            );
        }
    }
    auto prefill_weights = fp32_cuda(pre_weights);
    auto decode_weights = fp32_cuda({static_cast<float>(alpha), static_cast<float>(alpha)});
    auto new_mass = [&] { return at::zeros({total_view_pages, n_q}, at::device(at::kCUDA).dtype(at::kFloat)); };

    for (auto dt : {at::kFloat, at::kBFloat16}) {  // scalar then tensor-core kernels
        at::manual_seed(7);
        auto opts = at::device(at::kCUDA).dtype(dt);
        auto k_pool = at::randn({num_lines, kPageSize, n_kv, kHeadDim}, opts);
        auto v_pool = at::randn({num_lines, kPageSize, n_kv, kHeadDim}, opts);
        const double o_tol = dt == at::kFloat ? 1e-4 : 2e-2;

        auto q_pre = at::randn({cu_q.back(), n_q, kHeadDim}, opts);
        auto o_pre = at::zeros({cu_q.back(), n_q, kHeadDim}, at::device(at::kCUDA).dtype(at::kFloat));
        mass_reference reference_pre(total_view_pages, n_q, kPageSize);
        auto q_dec = at::randn({2, n_q, kHeadDim}, opts);
        auto o_dec = at::zeros({2, n_q, kHeadDim}, at::device(at::kCUDA).dtype(at::kFloat));
        mass_reference reference_dec(total_view_pages, n_q, kPageSize);
        for (size_t s = 0; s < views.size(); ++s) {
            const int64_t first_row = static_cast<int64_t>(s) * max_pages;
            reference_view(
                views[s],
                q_pre,
                k_pool,
                v_pool,
                views[s].seq_q,
                cu_q[s],
                alpha,
                o_pre,
                reference_pre,
                first_row
            );
            reference_view(views[s], q_dec, k_pool, v_pool, 1, s, alpha, o_dec, reference_dec, first_row);
        }
        auto mass_pre = reference_pre.mass();
        auto mass_dec = reference_dec.mass();

        auto mass = new_mass();
        auto o = pulsarlm::attn_prefill_cuda(
            q_pre,
            k_pool,
            v_pool,
            page_tables,
            cu_seqlens_q,
            seqlens_k,
            rope_layout,
            pulsarlm::rope_parameters{.theta = kTheta},
            kScale,
            std::nullopt,
            std::nullopt,
            mass,
            cu_view_pages,
            prefill_weights
        );
        CHECK(at::allclose(o.to(at::kFloat), o_pre, o_tol, o_tol));
        CHECK(at::allclose(mass, mass_pre, 1e-3, 1e-3));

        for (int64_t num_splits : {0, 1, 2}) {
            auto dec_mass = new_mass();
            auto dec_o = pulsarlm::attn_decode_cuda(
                q_dec,
                k_pool,
                v_pool,
                page_tables,
                seqlens_k,
                rope_layout,
                pulsarlm::rope_parameters{.theta = kTheta},
                kScale,
                std::nullopt,
                std::nullopt,
                dec_mass,
                cu_view_pages,
                decode_weights,
                num_splits
            );
            CHECK(at::allclose(dec_o.to(at::kFloat), o_dec, o_tol, o_tol));
            CHECK(at::allclose(dec_mass, mass_dec, 1e-3, 1e-3));
        }
    }
}

// The attention tests rotate their reference keys with rope_rotate, so it must agree with the rope kernel at every
// position a view reaches. An angle whose rounding scales with the position (an fp32 frequency table) drifts here,
// worst at the longest contexts.
TEST_CASE("ATen rope_rotate matches the rope kernel", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const double theta = 1e6;
    const int64_t n_heads = 3, head_dim = 128;
    // Past the trained context as well as inside it: the drift is proportional to the
    // position, so a bound that holds at 131072 holds everywhere below.
    const std::vector<int64_t> positions{0, 1, 1024, 32768, 131072};
    const int64_t seq = static_cast<int64_t>(positions.size());
    auto pos = at::tensor(positions, at::device(at::kCUDA).dtype(at::kLong));
    auto x = at::randn({n_heads, seq, head_dim}, at::device(at::kCUDA).dtype(at::kFloat));

    // Both round cos/sin to fp32 and rotate in fp32, so they agree to a few fp32 ulps
    // of a unit-scale input, not to the bit: the measured worst gap is 2.4e-7 for the
    // single rotation and 4.8e-7 for the delta, at every position alike. 1e-5 leaves
    // 20x headroom over that and still sits 130x under the 1.3e-3 an fp32 frequency
    // table put on the 32768 row.
    const double tol = 1e-5;
    auto worst = [](const at::Tensor& a, const at::Tensor& b) { return (a - b).abs().max().item<double>(); };

    CHECK(
        worst(
            pulsarlm::rope_rotate(x, pos.view({1, -1}), {.theta = theta}),
            pulsarlm::rope_cuda(x, pos, {.theta = theta})
        ) < tol
    );

    // The scorer's second rotation is a DELTA over already-roped keys, one scalar angle
    // for the whole tensor. Rotations compose, so stored + delta must land on scored.
    const int64_t stored = 128, scored = 131072;
    auto stored_pos = at::full({seq}, stored, at::device(at::kCUDA).dtype(at::kLong));
    auto delta = at::scalar_tensor(static_cast<double>(scored - stored), at::device(at::kCUDA).dtype(at::kDouble));
    CHECK(
        worst(
            pulsarlm::rope_rotate(pulsarlm::rope_cuda(x, stored_pos, {.theta = theta}), delta, {.theta = theta}),
            pulsarlm::rope_cuda(x, at::full_like(stored_pos, scored), {.theta = theta})
        ) < tol
    );
}

namespace {

struct WindowedView {
    std::vector<int32_t> lines;
    int64_t ctx;
    int64_t seq_q;
};

int64_t first_attended(int64_t query, std::optional<int64_t> window) {
    return window ? std::max<int64_t>(0, query - *window + 1) : 0;
}

// The view's keys as the kernels stage them: scaled by k_norm_weight in fp32, rotated to their view index, rounded to
// the pool dtype.
at::Tensor reference_keys(
    const WindowedView& view,
    const at::Tensor& k_pool,
    const pulsarlm::rope_parameters& rope,
    const std::optional<at::Tensor>& k_norm_weight
) {
    const int64_t n_kv = k_pool.size(2), head_dim = k_pool.size(3);
    std::vector<int32_t> readable(view.lines);
    for (auto& line : readable) {
        line = std::max<int32_t>(line, 0) % static_cast<int32_t>(k_pool.size(0));
    }
    auto lines = at::tensor(readable, at::dtype(at::kLong)).to(at::kCUDA);
    auto k = k_pool.index_select(0, lines).view({-1, n_kv, head_dim}).slice(0, 0, view.ctx).to(at::kFloat);
    if (k_norm_weight) {
        k = k * k_norm_weight->to(at::kFloat);
    }
    auto pos = at::arange(view.ctx, at::device(at::kCUDA).dtype(at::kLong)).view({-1, 1});
    return pulsarlm::rope_rotate(k, pos, rope).to(k_pool.scalar_type()).to(at::kFloat);
}

void reference_windowed(
    const WindowedView& view,
    const at::Tensor& keys,
    const at::Tensor& v_pool,
    const at::Tensor& q,
    int64_t rows,
    int64_t q_offset,
    std::optional<int64_t> window,
    double scale,
    double alpha,
    at::Tensor& o,
    mass_reference& mass,
    int64_t first_row
) {
    const int64_t n_kv = v_pool.size(2), head_dim = v_pool.size(3);
    const int64_t group = q.size(1) / n_kv;
    std::vector<int32_t> readable(view.lines);
    for (auto& line : readable) {
        line = std::max<int32_t>(line, 0) % static_cast<int32_t>(v_pool.size(0));
    }
    auto lines = at::tensor(readable, at::dtype(at::kLong)).to(at::kCUDA);
    auto v = v_pool.index_select(0, lines).view({-1, n_kv, head_dim}).slice(0, 0, view.ctx).to(at::kFloat);
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t p = view.ctx - rows + r;
        const int64_t first = first_attended(p, window);
        const double graded = alpha * std::pow(1.0 - alpha, static_cast<double>(rows - 1 - r));
        for (int64_t h = 0; h < q.size(1); ++h) {
            const int64_t g = h / group;
            auto k = keys.slice(0, first, p + 1).select(1, g);
            auto values = v.slice(0, first, p + 1).select(1, g);
            auto w = at::softmax(at::mv(k, q[q_offset + r][h].to(at::kFloat)) * scale, 0);
            o[q_offset + r][h] = at::mv(values.t(), w);
            mass.add(first_row, first, w, h, graded);
        }
    }
}

}  // namespace

// Page table entries before the first query's window hold lines past the pool, so a kernel reading one faults.
TEST_CASE(
    "attention matches the dense reference across head dims, windows, partial RoPE and scaled keys",
    "[ops][cuda]"
) {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const int64_t n_q = 4, n_kv = 2;
    const double alpha = 0.25;
    const int32_t unreadable = 1 << 20;

    for (int64_t head_dim : {64, 128, 256, 512}) {
        for (int64_t page_size : {16, 32}) {
            for (std::optional<int64_t> window : {std::optional<int64_t>{}, std::optional<int64_t>{21}}) {
                for (double factor : {1.0, 0.25}) {
                    for (bool scaled_keys : {false, true}) {
                        const pulsarlm::rope_parameters rope{.theta = 1e4, .partial_rotary_factor = factor};
                        const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
                        const int64_t num_lines = 6, max_pages = 4;
                        std::vector<WindowedView> views = {
                            WindowedView{{0, 3, 5, 1}, 3 * page_size + 5, 24},
                            WindowedView{{2, 4}, page_size + 3, 4},
                        };
                        std::vector<int32_t> tables(views.size() * max_pages, 0), seqlens, cu_q{0}, cu_view;
                        for (size_t s = 0; s < views.size(); ++s) {
                            const int64_t first_page = first_attended(views[s].ctx - views[s].seq_q, window) /
                                page_size;
                            for (int64_t b = 0; b < first_page; ++b) {
                                views[s].lines[b] = unreadable;
                            }
                            std::copy(views[s].lines.begin(), views[s].lines.end(), tables.begin() + s * max_pages);
                            seqlens.push_back(static_cast<int32_t>(views[s].ctx));
                            cu_q.push_back(cu_q.back() + static_cast<int32_t>(views[s].seq_q));
                            cu_view.push_back(static_cast<int32_t>(s * max_pages));
                        }
                        cu_view.push_back(static_cast<int32_t>(views.size() * max_pages));
                        auto page_tables = int32_cuda(tables).view({static_cast<int64_t>(views.size()), max_pages});
                        auto rope_layout = identity_layout(static_cast<int64_t>(views.size()));
                        auto seqlens_k = int32_cuda(seqlens);
                        auto cu_seqlens_q = int32_cuda(cu_q);
                        auto cu_view_pages = int32_cuda(cu_view);
                        const int64_t total_view_pages = static_cast<int64_t>(views.size()) * max_pages;
                        std::vector<float> pre_weights;
                        for (const WindowedView& view : views) {
                            for (int64_t r = 0; r < view.seq_q; ++r) {
                                pre_weights.push_back(
                                    static_cast<float>(
                                        alpha * std::pow(1.0 - alpha, static_cast<double>(view.seq_q - 1 - r))
                                    )
                                );
                            }
                        }
                        auto prefill_weights = fp32_cuda(pre_weights);
                        auto decode_weights = fp32_cuda(std::vector<float>(views.size(), static_cast<float>(alpha)));
                        auto new_mass = [&] {
                            return at::zeros({total_view_pages, n_q}, at::device(at::kCUDA).dtype(at::kFloat));
                        };

                        for (auto dt : {at::kFloat, at::kBFloat16}) {
                            for (bool head_major : {false, true}) {
                                INFO(
                                    "head_dim " << head_dim << " page_size " << page_size << " window "
                                                << window.value_or(0) << " factor " << factor << " scaled_keys "
                                                << scaled_keys << " dtype " << dt << " head_major " << head_major
                                );
                                at::manual_seed(11);
                                auto opts = at::device(at::kCUDA).dtype(dt);
                                auto lines = head_major ? at::randn({num_lines, n_kv, 2, page_size, head_dim}, opts)
                                                              .permute({0, 2, 3, 1, 4})
                                                        : at::randn({num_lines, 2, page_size, n_kv, head_dim}, opts);
                                auto k_pool = lines.select(1, 0);
                                auto v_pool = lines.select(1, 1);
                                std::optional<at::Tensor> k_norm_weight;
                                if (scaled_keys) {
                                    k_norm_weight = at::rand({head_dim}, at::device(at::kCUDA).dtype(at::kFloat)) + 0.5;
                                }
                                const double o_tol = dt == at::kFloat ? 1e-4 : 2e-2;
                                auto q_pre = at::randn({cu_q.back(), n_q, head_dim}, opts);
                                auto q_dec = at::randn({static_cast<int64_t>(views.size()), n_q, head_dim}, opts);

                                auto f32 = at::device(at::kCUDA).dtype(at::kFloat);
                                auto o_pre = at::zeros({cu_q.back(), n_q, head_dim}, f32);
                                auto o_dec = at::zeros({static_cast<int64_t>(views.size()), n_q, head_dim}, f32);
                                mass_reference reference_pre(total_view_pages, n_q, page_size);
                                mass_reference reference_dec(total_view_pages, n_q, page_size);
                                for (size_t s = 0; s < views.size(); ++s) {
                                    auto keys = reference_keys(views[s], k_pool, rope, k_norm_weight);
                                    const int64_t first_row = static_cast<int64_t>(s) * max_pages;
                                    reference_windowed(
                                        views[s],
                                        keys,
                                        v_pool,
                                        q_pre,
                                        views[s].seq_q,
                                        cu_q[s],
                                        window,
                                        scale,
                                        alpha,
                                        o_pre,
                                        reference_pre,
                                        first_row
                                    );
                                    reference_windowed(
                                        views[s],
                                        keys,
                                        v_pool,
                                        q_dec,
                                        1,
                                        static_cast<int64_t>(s),
                                        window,
                                        scale,
                                        alpha,
                                        o_dec,
                                        reference_dec,
                                        first_row
                                    );
                                }
                                auto mass_pre = reference_pre.mass();
                                auto mass_dec = reference_dec.mass();

                                auto mass = new_mass();
                                auto o = pulsarlm::attn_prefill_cuda(
                                    q_pre,
                                    k_pool,
                                    v_pool,
                                    page_tables,
                                    cu_seqlens_q,
                                    seqlens_k,
                                    rope_layout,
                                    rope,
                                    scale,
                                    window,
                                    k_norm_weight,
                                    mass,
                                    cu_view_pages,
                                    prefill_weights
                                );
                                CHECK(at::allclose(o.to(at::kFloat), o_pre, o_tol, o_tol));
                                CHECK(at::allclose(mass, mass_pre, 1e-3, 1e-3));

                                for (int64_t num_splits : {0, 1, 2}) {
                                    auto dec_mass = new_mass();
                                    auto dec_o = pulsarlm::attn_decode_cuda(
                                        q_dec,
                                        k_pool,
                                        v_pool,
                                        page_tables,
                                        seqlens_k,
                                        rope_layout,
                                        rope,
                                        scale,
                                        window,
                                        k_norm_weight,
                                        dec_mass,
                                        cu_view_pages,
                                        decode_weights,
                                        num_splits
                                    );
                                    CHECK(at::allclose(dec_o.to(at::kFloat), o_dec, o_tol, o_tol));
                                    CHECK(at::allclose(dec_mass, mass_dec, 1e-3, 1e-3));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("partial RoPE passes the unrotated pairs through", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const int64_t head_dim = 512, half = head_dim / 2, rotated = 64;
    const pulsarlm::rope_parameters rope{.theta = 1e6, .partial_rotary_factor = 0.25};
    auto x = at::randn({2, 5, head_dim}, at::device(at::kCUDA).dtype(at::kBFloat16));
    auto pos = at::tensor(std::vector<int64_t>{0, 1, 7, 4096, 100000}, at::device(at::kCUDA).dtype(at::kLong));
    auto rotated_x = pulsarlm::rope_cuda(x, pos, rope);
    for (int64_t start : {rotated, half + rotated}) {
        CHECK(
            at::equal(rotated_x.slice(-1, start, start + half - rotated), x.slice(-1, start, start + half - rotated))
        );
    }
    CHECK(!at::equal(rotated_x.slice(-1, 0, rotated), x.slice(-1, 0, rotated)));
}

TEST_CASE("bfloat16 attention of a shape the tensor-core kernels lack fails instead of running", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    REQUIRE(pulsarlm::attention_unsupported_reason(at::kFloat, 8, 32, 4, 1) == std::nullopt);
    REQUIRE(pulsarlm::attention_unsupported_reason(at::kBFloat16, 16, 128, 32, 8) == std::nullopt);
    REQUIRE(pulsarlm::attention_unsupported_reason(at::kBFloat16, 16, 96, 32, 8).has_value());
    REQUIRE(pulsarlm::attention_unsupported_reason(at::kBFloat16, 8, 128, 32, 8).has_value());
    REQUIRE(pulsarlm::attention_unsupported_reason(at::kBFloat16, 16, 128, 32, 1).has_value());

    auto opts = at::device(at::kCUDA).dtype(at::kBFloat16);
    auto pool = at::zeros({2, kPageSize, 1, 32}, opts);
    auto q = at::randn({1, 4, 32}, opts);
    REQUIRE_THROWS_WITH(
        pulsarlm::attn_decode_cuda(
            q,
            pool,
            pool,
            int32_cuda({0}).view({1, 1}),
            int32_cuda({4}),
            identity_layout(1),
            pulsarlm::rope_parameters{.theta = kTheta},
            kScale,
            std::nullopt,
            std::nullopt
        ),
        Catch::Matchers::ContainsSubstring("head dim 32")
    );
}

TEST_CASE("write_kv places each token at its view index through the page table", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const int64_t num_lines = 5, n_kv = 2, head_dim = 64;
    auto opts = at::device(at::kCUDA).dtype(at::kFloat);
    auto page_tables = int32_cuda({3, 1, 0, 4, 2, 0}).view({2, 3});
    // Sequence 0 forwards view indices [14, 20), sequence 1 [1, 3).
    auto seqlens_k = int32_cuda({20, 3});
    auto cu_seqlens_q = int32_cuda({0, 6, 8});
    auto k_new = at::randn({8, n_kv, head_dim}, opts);
    auto v_new = at::randn({8, n_kv, head_dim}, opts);

    auto k_pool = at::zeros({num_lines, kPageSize, n_kv, head_dim}, opts);
    auto v_pool = at::zeros({num_lines, kPageSize, n_kv, head_dim}, opts);
    pulsarlm::write_kv_cuda(k_pool, v_pool, k_new, v_new, page_tables, cu_seqlens_q, seqlens_k);
    for (int64_t t = 0; t < 6; ++t) {
        const int64_t index = 14 + t;
        const int64_t line = index < kPageSize ? 3 : 1;
        CHECK(at::equal(k_pool[line][index % kPageSize], k_new[t]));
        CHECK(at::equal(v_pool[line][index % kPageSize], v_new[t]));
    }
    CHECK(at::equal(k_pool[4][1], k_new[6]));
    CHECK(at::equal(k_pool[4][2], k_new[7]));

    auto lines = at::zeros({num_lines, 2, kPageSize, n_kv, head_dim}, opts);
    pulsarlm::write_kv_cuda(lines.select(1, 0), lines.select(1, 1), k_new, v_new, page_tables, cu_seqlens_q, seqlens_k);
    CHECK(at::equal(lines[3][0][14], k_new[0]));
    CHECK(at::equal(lines[3][1][14], v_new[0]));
    CHECK(at::equal(lines[4][0][2], k_new[7]));
    CHECK(at::equal(lines[4][1][2], v_new[7]));
    CHECK(lines.select(1, 1)[3][13].abs().sum().item<float>() == 0.0f);

    auto head_major = at::zeros({num_lines, n_kv, 2, kPageSize, head_dim}, opts).permute({0, 2, 3, 1, 4});
    pulsarlm::write_kv_cuda(
        head_major.select(1, 0),
        head_major.select(1, 1),
        k_new,
        v_new,
        page_tables,
        cu_seqlens_q,
        seqlens_k
    );
    CHECK(at::equal(head_major.select(1, 0)[3][14], k_new[0]));
    CHECK(at::equal(head_major.select(1, 1)[4][2], v_new[7]));

    auto pool = at::zeros({num_lines, kPageSize, n_kv, head_dim}, opts);
    auto decode_tokens = at::randn({2, n_kv, head_dim}, opts);
    pulsarlm::write_kv_cuda(pool, pool, decode_tokens, decode_tokens, page_tables, std::nullopt, seqlens_k);
    CHECK(at::equal(pool[1][3], decode_tokens[0]));
    CHECK(at::equal(pool[4][2], decode_tokens[1]));
}

TEST_CASE("gather_page_tables lays each table's rows side by side, zero past its capacity", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    std::vector<at::Tensor> tables{
        int32_cuda({1, 2, 3, 4, 5, 6}).view({2, 3}),
        int32_cuda({7, 8, 9, 10}).view({2, 2}),
    };
    auto gathered = pulsarlm::gather_page_tables_cuda(tables, 2, 3).cpu();
    auto expected = at::tensor(std::vector<int32_t>{1, 2, 3, 7, 8, 0, 4, 5, 6, 9, 10, 0}).view({2, 2, 3});
    CHECK(at::equal(gathered, expected));
}
