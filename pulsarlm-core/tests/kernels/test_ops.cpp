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
#include <limits>
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

// The mass side outputs set each view page's mass_sum per query head, the sum over the queries attending the page of
// their query weight times their max scaled score over its keys, and its weight_sum, the sum of those weights. fp32
// runs the scalar kernels, bf16 the tensor-core ones, so each case covers both.
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
// Each view page's position in a view laid out contiguously: its view index times the page size.
at::Tensor identity_positions(const at::Tensor& page_tables, int64_t page_size = kPageSize) {
    return (at::arange(page_tables.size(1), page_tables.options()) * page_size).expand_as(page_tables).contiguous();
}

// A key pool of num_pages zeroed pages. Every score is q.k == 0 at any rotation, so every
// mass_sum is 0. One kv head, one query head.
at::Tensor zero_pool(int64_t num_pages, at::ScalarType dt) {
    return at::zeros({num_pages, kPageSize, 1, kHeadDim}, at::device(at::kCUDA).dtype(dt));
}

at::Tensor zero_mass(int64_t num_pages) {
    return at::zeros({num_pages, 1}, at::device(at::kCUDA).dtype(at::kFloat));
}

at::Tensor zero_weights(int64_t num_pages) {
    return at::zeros({num_pages}, at::device(at::kCUDA).dtype(at::kFloat));
}

// Every sequence's view is page_tables row s, its mass rows one view page per table
// column, so view page b of sequence s is mass row s * max_pages + b.
at::Tensor row_view_pages(const at::Tensor& page_tables) {
    const int64_t num_seqs = page_tables.size(0), max_pages = page_tables.size(1);
    return at::arange(0, (num_seqs + 1) * max_pages, max_pages, at::device(at::kCUDA).dtype(at::kInt));
}

// The ops' mass outputs, built from each query's scaled scores in each head.
class mass_reference {
    at::Tensor mass;  // [view pages, heads]
    at::Tensor weight;  // [view pages]
    int64_t page_size;

  public:
    mass_reference(int64_t view_pages, int64_t heads, int64_t page_size)
    : mass(at::zeros({view_pages, heads}, at::device(at::kCUDA).dtype(at::kFloat))),
      weight(at::zeros({view_pages}, at::device(at::kCUDA).dtype(at::kFloat))),
      page_size(page_size) {}

    // A query of query weight w whose head h gave the keys [first_key, first_key + scores.size(0)) of the view whose
    // page 0 is mass row first_row the scaled scores scores. Each query adds its weight once, with its head 0.
    void add(int64_t first_row, int64_t first_key, const at::Tensor& scores, int64_t h, double w) {
        const int64_t first_page = first_key / this->page_size;
        const int64_t pages = (first_key + scores.size(0) - 1) / this->page_size - first_page + 1;
        auto padded = at::full({pages * this->page_size}, -std::numeric_limits<float>::infinity(), scores.options());
        padded.narrow(0, first_key - first_page * this->page_size, scores.size(0)).copy_(scores);
        auto page_max = padded.view({pages, this->page_size}).amax(-1);
        this->mass.narrow(0, first_row + first_page, pages).select(1, h).add_(page_max * w);
        if (h == 0) {
            this->weight.narrow(0, first_row + first_page, pages).add_(w);
        }
    }

    const at::Tensor& mass_sum() const {
        return this->mass;
    }

    const at::Tensor& weight_sum() const {
        return this->weight;
    }
};

at::Tensor prefill(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& page_tables,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& seqlens_k,
    const at::Tensor& mass_sum,
    const at::Tensor& weight_sum,
    const at::Tensor& query_weights
) {
    return pulsarlm::attn_prefill_cuda(
        q,
        k_pool,
        v_pool,
        page_tables,
        cu_seqlens_q,
        seqlens_k,
        identity_positions(page_tables),
        pulsarlm::rope_parameters{.theta = kTheta},
        kScale,
        std::nullopt,
        std::nullopt,
        mass_sum,
        weight_sum,
        row_view_pages(page_tables),
        query_weights
    );
}

at::Tensor decode(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& page_tables,
    const at::Tensor& context_lens,
    const at::Tensor& mass_sum,
    const at::Tensor& weight_sum,
    const at::Tensor& query_weights,
    int64_t num_splits = 0
) {
    return pulsarlm::attn_decode_cuda(
        q,
        k_pool,
        v_pool,
        page_tables,
        context_lens,
        identity_positions(page_tables),
        pulsarlm::rope_parameters{.theta = kTheta},
        kScale,
        std::nullopt,
        std::nullopt,
        mass_sum,
        weight_sum,
        row_view_pages(page_tables),
        query_weights,
        num_splits
    );
}

}  // namespace

// Page b is attended by the queries from b * page_size on.
TEST_CASE("prefill sums the weights of the queries attending each page", "[ops][cuda]") {
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

        auto mass_sum = zero_mass(num_pages);
        auto weight_sum = zero_weights(num_pages);
        prefill(q, k_pool, v_pool, page_tables, cu_seqlens_q, seqlens_k, mass_sum, weight_sum, fp32_cuda(weights));
        REQUIRE(at::equal(mass_sum, zero_mass(num_pages)));
        auto got = weight_sum.to(at::kCPU);
        for (int64_t b = 0; b < num_pages; ++b) {
            double want = 0.0;
            for (int64_t i = b * kPageSize; i < ntok; ++i) {
                want += weights[i];
            }
            REQUIRE(std::abs(got[b].item<double>() - want) < 1e-5);
        }
    }
}

// Two sequences over disjoint pages: 16 keys for sequence 0, 128 for sequence 1.
TEST_CASE("decode gives each page its query attends the query's weight", "[ops][cuda]") {
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
    auto check = [&](const at::Tensor& mass_sum, const at::Tensor& weight_sum) {
        REQUIRE(at::equal(mass_sum, zero_mass(num_pages)));
        auto got = weight_sum.view({2, max_pages}).to(at::kCPU);
        REQUIRE(got[0][0].item<float>() == weights[0]);
        for (int64_t b = 1; b < max_pages; ++b) {
            REQUIRE(got[0][b].item<float>() == 0.0f);
        }
        for (int64_t b = 0; b < max_pages; ++b) {
            REQUIRE(got[1][b].item<float>() == weights[1]);
        }
    };

    for (auto dt : {at::kFloat, at::kBFloat16}) {
        auto k_pool = zero_pool(num_pages, dt);
        auto v_pool = zero_pool(num_pages, dt);
        auto q = at::zeros({2, 1, kHeadDim}, at::device(at::kCUDA).dtype(dt));
        auto mass_sum = zero_mass(num_pages);
        auto weight_sum = zero_weights(num_pages);
        decode(q, k_pool, v_pool, page_tables, context_lens, mass_sum, weight_sum, fp32_cuda(weights));
        check(mass_sum, weight_sum);
    }

    // The split (flash-decode) path writes each slice's pages from its own CTA.
    auto k_pool = zero_pool(num_pages, at::kBFloat16);
    auto v_pool = zero_pool(num_pages, at::kBFloat16);
    auto q = at::zeros({2, 1, kHeadDim}, at::device(at::kCUDA).dtype(at::kBFloat16));
    auto mass_sum = zero_mass(num_pages);
    auto weight_sum = zero_weights(num_pages);
    decode(q, k_pool, v_pool, page_tables, context_lens, mass_sum, weight_sum, fp32_cuda(weights), /*num_splits=*/2);
    check(mass_sum, weight_sum);
}

namespace {

// One sequence of the shared-line case: its view over the pool and its pages' positions.
struct View {
    std::vector<int32_t> lines;
    int64_t ctx;
    int64_t seq_q;  // prefill queries, the last seq_q view indices
    std::vector<int32_t> page_positions;

    int64_t position(int64_t j) const {
        return this->page_positions[j / kPageSize] + j % kPageSize;
    }
};

// ATen reference for one view: its keys rotated to their page positions by
// rope_rotate and rounded to the pool dtype as the kernels store them, then causal
// softmax attention for the query rows at view indices [ctx - rows, ctx). Fills o
// rows and adds each row's scores, at query weight alpha * (1 - alpha)^(rows after it),
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
            auto scores = at::mv(keys, q[q_offset + r][h].to(at::kFloat)) * kScale;
            o[q_offset + r][h] = at::mv(values.t(), at::softmax(scores, 0));
            mass.add(first_row, 0, scores, h, graded);
        }
    }
}

}  // namespace

// Two views share line 2: a full page of the second view, the partial tail of the
// first. The first view's pages sit at contiguous positions, the second's with gaps, so
// the shared line's keys sit at different positions in each and must be rotated per view, and its
// mass must land in each view's own rows. The reference rotates the unrotated pool with
// the ATen rope_rotate.
TEST_CASE("views sharing a line rotate and accumulate apart", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const int64_t n_q = 4, n_kv = 2, num_lines = 3;
    const std::vector<View> views = {
        View{{0, 2}, kPageSize + 5, 5, {0, kPageSize, 2 * kPageSize}},
        View{{1, 2, 0}, 2 * kPageSize + 7, 20, {0, 3 * kPageSize, 5 * kPageSize}},
    };
    const double alpha = 0.5;
    const int64_t max_pages = 3, total_view_pages = 2 * max_pages;

    std::vector<int32_t> tables(views.size() * max_pages, 0);
    std::vector<int32_t> positions, seqlens, cu_q{0}, cu_view;
    for (size_t s = 0; s < views.size(); ++s) {
        std::copy(views[s].lines.begin(), views[s].lines.end(), tables.begin() + s * max_pages);
        positions.insert(positions.end(), views[s].page_positions.begin(), views[s].page_positions.end());
        seqlens.push_back(static_cast<int32_t>(views[s].ctx));
        cu_q.push_back(cu_q.back() + static_cast<int32_t>(views[s].seq_q));
        cu_view.push_back(static_cast<int32_t>(s * max_pages));
    }
    cu_view.push_back(static_cast<int32_t>(total_view_pages));
    auto page_tables = int32_cuda(tables).view({2, max_pages});
    auto page_positions = int32_cuda(positions).view({2, max_pages});
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
    auto new_weights = [&] { return at::zeros({total_view_pages}, at::device(at::kCUDA).dtype(at::kFloat)); };

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
        auto mass = new_mass();
        auto weights = new_weights();
        auto o = pulsarlm::attn_prefill_cuda(
            q_pre,
            k_pool,
            v_pool,
            page_tables,
            cu_seqlens_q,
            seqlens_k,
            page_positions,
            pulsarlm::rope_parameters{.theta = kTheta},
            kScale,
            std::nullopt,
            std::nullopt,
            mass,
            weights,
            cu_view_pages,
            prefill_weights
        );
        CHECK(at::allclose(o.to(at::kFloat), o_pre, o_tol, o_tol));
        CHECK(at::allclose(mass, reference_pre.mass_sum(), 1e-3, 1e-3));
        CHECK(at::allclose(weights, reference_pre.weight_sum(), 1e-5, 1e-5));

        for (int64_t num_splits : {0, 1, 2}) {
            auto dec_mass = new_mass();
            auto dec_weights = new_weights();
            auto dec_o = pulsarlm::attn_decode_cuda(
                q_dec,
                k_pool,
                v_pool,
                page_tables,
                seqlens_k,
                page_positions,
                pulsarlm::rope_parameters{.theta = kTheta},
                kScale,
                std::nullopt,
                std::nullopt,
                dec_mass,
                dec_weights,
                cu_view_pages,
                decode_weights,
                num_splits
            );
            CHECK(at::allclose(dec_o.to(at::kFloat), o_dec, o_tol, o_tol));
            CHECK(at::allclose(dec_mass, reference_dec.mass_sum(), 1e-3, 1e-3));
            CHECK(at::allclose(dec_weights, reference_dec.weight_sum(), 1e-5, 1e-5));
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
            auto scores = at::mv(k, q[q_offset + r][h].to(at::kFloat)) * scale;
            o[q_offset + r][h] = at::mv(values.t(), at::softmax(scores, 0));
            mass.add(first_row, first, scores, h, graded);
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
                        auto page_positions = identity_positions(page_tables, page_size);
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
                        auto new_weights = [&] {
                            return at::zeros({total_view_pages}, at::device(at::kCUDA).dtype(at::kFloat));
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
                                auto mass = new_mass();
                                auto weights = new_weights();
                                auto o = pulsarlm::attn_prefill_cuda(
                                    q_pre,
                                    k_pool,
                                    v_pool,
                                    page_tables,
                                    cu_seqlens_q,
                                    seqlens_k,
                                    page_positions,
                                    rope,
                                    scale,
                                    window,
                                    k_norm_weight,
                                    mass,
                                    weights,
                                    cu_view_pages,
                                    prefill_weights
                                );
                                CHECK(at::allclose(o.to(at::kFloat), o_pre, o_tol, o_tol));
                                CHECK(at::allclose(mass, reference_pre.mass_sum(), 1e-3, 1e-3));
                                CHECK(at::allclose(weights, reference_pre.weight_sum(), 1e-5, 1e-5));

                                for (int64_t num_splits : {0, 1, 2}) {
                                    auto dec_mass = new_mass();
                                    auto dec_weights = new_weights();
                                    auto dec_o = pulsarlm::attn_decode_cuda(
                                        q_dec,
                                        k_pool,
                                        v_pool,
                                        page_tables,
                                        seqlens_k,
                                        page_positions,
                                        rope,
                                        scale,
                                        window,
                                        k_norm_weight,
                                        dec_mass,
                                        dec_weights,
                                        cu_view_pages,
                                        decode_weights,
                                        num_splits
                                    );
                                    CHECK(at::allclose(dec_o.to(at::kFloat), o_dec, o_tol, o_tol));
                                    CHECK(at::allclose(dec_mass, reference_dec.mass_sum(), 1e-3, 1e-3));
                                    CHECK(at::allclose(dec_weights, reference_dec.weight_sum(), 1e-5, 1e-5));
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
            identity_positions(int32_cuda({0}).view({1, 1})),
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
