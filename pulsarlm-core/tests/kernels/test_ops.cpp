#include "pulsarlm/ops/normalization.hpp"
#include "pulsarlm/ops/page_tables.hpp"
#include "pulsarlm/ops/paged_attention.hpp"
#include "pulsarlm/ops/rope.hpp"

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

    auto got = pulsarlm::ops::cuda::rmsnorm(x, w, eps);

    auto ms = x.pow(2).mean(-1, /*keepdim=*/true);
    auto ref = x * at::rsqrt(ms + eps) * w;

    REQUIRE(at::allclose(got, ref, /*rtol=*/1e-5, /*atol=*/1e-5));

    // eps is what keeps an all-zero row finite; on unit-variance input it shifts the
    // result far below any tolerance, so nothing above would notice it being dropped.
    auto zeros = at::zeros({4, 512}, at::device(at::kCUDA).dtype(at::kFloat));
    REQUIRE(at::allclose(pulsarlm::ops::cuda::rmsnorm(zeros, w, eps), zeros));
}

// The mass side outputs set each view page's mass_sum per query head, the sum over the queries attending the page of
// their query weight times their max scaled score over its keys, and its weight_sum, the sum of those weights. fp32
// runs the scalar kernels, bf16 the tensor-core ones, so each case covers both.
namespace {

constexpr int64_t pool_page_size = 16;
constexpr int64_t pool_head_dim = 64;
constexpr double rope_theta = 1e6;
const double attention_scale = 1.0 / std::sqrt(static_cast<double>(pool_head_dim));

at::Tensor int32_cuda(const std::vector<int32_t>& v) {
    return at::tensor(v, at::dtype(at::kInt)).to(at::kCUDA);
}

at::Tensor fp32_cuda(const std::vector<float>& v) {
    return at::tensor(v, at::dtype(at::kFloat)).to(at::kCUDA);
}

// {0, 0, -1} per sequence: every key at its own view index.
// Each view page's position in a view laid out contiguously: its view index times the page size.
at::Tensor identity_positions(const at::Tensor& page_tables, int64_t page_size = pool_page_size) {
    return (at::arange(page_tables.size(1), page_tables.options()) * page_size).expand_as(page_tables).contiguous();
}

// A key pool of num_pages zeroed pages. Every score is q.k == 0 at any rotation, so every
// mass_sum is 0. One kv head, one query head.
at::Tensor zero_pool(int64_t num_pages, at::ScalarType dt) {
    return at::zeros({num_pages, pool_page_size, 1, pool_head_dim}, at::device(at::kCUDA).dtype(dt));
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
    const at::Tensor& key_counts,
    const at::Tensor& mass_sum,
    const at::Tensor& weight_sum,
    const at::Tensor& query_weights
) {
    return pulsarlm::ops::cuda::attn_prefill(
        q,
        k_pool,
        v_pool,
        page_tables,
        identity_positions(page_tables),
        key_counts,
        cu_seqlens_q,
        rope_theta,
        1.0,
        std::nullopt,
        attention_scale,
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
    const at::Tensor& key_counts,
    const at::Tensor& mass_sum,
    const at::Tensor& weight_sum,
    const at::Tensor& query_weights,
    int64_t num_splits = 0
) {
    return pulsarlm::ops::cuda::attn_decode(
        q,
        k_pool,
        v_pool,
        page_tables,
        identity_positions(page_tables),
        key_counts,
        rope_theta,
        1.0,
        std::nullopt,
        attention_scale,
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
    const int64_t ntok = 64, num_pages = ntok / pool_page_size;
    std::vector<float> weights;
    for (int64_t i = 0; i < ntok; ++i) {
        weights.push_back(static_cast<float>(0.05 * std::pow(0.95, static_cast<double>(ntok - 1 - i))));
    }

    for (auto dt : {at::kFloat, at::kBFloat16}) {
        auto k_pool = zero_pool(num_pages, dt);
        auto v_pool = zero_pool(num_pages, dt);
        auto q = at::zeros({ntok, 1, pool_head_dim}, at::device(at::kCUDA).dtype(dt));
        auto page_tables = at::arange(num_pages, at::device(at::kCUDA).dtype(at::kInt)).view({1, num_pages});
        auto cu_seqlens_q = int32_cuda({0, static_cast<int32_t>(ntok)});
        auto key_counts = int32_cuda({static_cast<int32_t>(ntok)});

        auto mass_sum = zero_mass(num_pages);
        auto weight_sum = zero_weights(num_pages);
        prefill(q, k_pool, v_pool, page_tables, cu_seqlens_q, key_counts, mass_sum, weight_sum, fp32_cuda(weights));
        REQUIRE(at::equal(mass_sum, zero_mass(num_pages)));
        auto got = weight_sum.to(at::kCPU);
        for (int64_t b = 0; b < num_pages; ++b) {
            double want = 0.0;
            for (int64_t i = b * pool_page_size; i < ntok; ++i) {
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
    const int64_t max_pages = long_ctx / pool_page_size;
    const int64_t num_pages = 2 * max_pages;
    const std::vector<float> weights{1.0f, 0.25f};

    std::vector<int32_t> tables;
    for (int32_t p = 0; p < 2 * max_pages; ++p) {
        tables.push_back(p);
    }
    auto page_tables = int32_cuda(tables).view({2, max_pages});
    auto key_counts = int32_cuda({static_cast<int32_t>(short_ctx), static_cast<int32_t>(long_ctx)});
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
        auto q = at::zeros({2, 1, pool_head_dim}, at::device(at::kCUDA).dtype(dt));
        auto mass_sum = zero_mass(num_pages);
        auto weight_sum = zero_weights(num_pages);
        decode(q, k_pool, v_pool, page_tables, key_counts, mass_sum, weight_sum, fp32_cuda(weights));
        check(mass_sum, weight_sum);
    }

    // The split (flash-decode) path writes each slice's pages from its own CTA.
    auto k_pool = zero_pool(num_pages, at::kBFloat16);
    auto v_pool = zero_pool(num_pages, at::kBFloat16);
    auto q = at::zeros({2, 1, pool_head_dim}, at::device(at::kCUDA).dtype(at::kBFloat16));
    auto mass_sum = zero_mass(num_pages);
    auto weight_sum = zero_weights(num_pages);
    decode(q, k_pool, v_pool, page_tables, key_counts, mass_sum, weight_sum, fp32_cuda(weights), /*num_splits=*/2);
    check(mass_sum, weight_sum);
}

namespace {

// One sequence of the shared-line case: its view over the pool and its pages' positions.
struct sequence_view {
    std::vector<int32_t> lines;
    int64_t ctx;
    int64_t seq_q;  // prefill queries, the last seq_q view indices
    std::vector<int32_t> page_positions;

    int64_t position(int64_t j) const {
        return this->page_positions[j / pool_page_size] + j % pool_page_size;
    }
};

// ATen reference for one view: its keys rotated to their page positions by
// ops::rope_rotate and rounded to the pool dtype as the kernels store them, then causal
// softmax attention for the query rows at view indices [ctx - rows, ctx). Fills o
// rows and adds each row's scores, at query weight alpha * (1 - alpha)^(rows after it),
// to the view's mass rows from first_row on.
void reference_view(
    const sequence_view& view,
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
    k = pulsarlm::ops::rope_rotate(k, pos, rope_theta, 1.0, std::nullopt).to(k_pool.scalar_type()).to(at::kFloat);

    for (int64_t r = 0; r < rows; ++r) {
        const int64_t p = view.ctx - rows + r;
        const double graded = alpha * std::pow(1.0 - alpha, static_cast<double>(rows - 1 - r));
        for (int64_t h = 0; h < q.size(1); ++h) {
            const int64_t g = h / group;
            auto keys = k.slice(0, 0, p + 1).select(1, g);
            auto values = v.slice(0, 0, p + 1).select(1, g);
            auto scores = at::mv(keys, q[q_offset + r][h].to(at::kFloat)) * attention_scale;
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
    const std::vector<sequence_view> views = {
        sequence_view{{0, 2}, pool_page_size + 5, 5, {0, pool_page_size, 2 * pool_page_size}},
        sequence_view{{1, 2, 0}, 2 * pool_page_size + 7, 20, {0, 3 * pool_page_size, 5 * pool_page_size}},
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
    auto key_counts = int32_cuda(seqlens);
    auto cu_seqlens_q = int32_cuda(cu_q);
    auto cu_view_pages = int32_cuda(cu_view);
    std::vector<float> pre_weights;
    for (const sequence_view& view : views) {
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
        auto k_pool = at::randn({num_lines, pool_page_size, n_kv, pool_head_dim}, opts);
        auto v_pool = at::randn({num_lines, pool_page_size, n_kv, pool_head_dim}, opts);
        const double o_tol = dt == at::kFloat ? 1e-4 : 2e-2;

        auto q_pre = at::randn({cu_q.back(), n_q, pool_head_dim}, opts);
        auto o_pre = at::zeros({cu_q.back(), n_q, pool_head_dim}, at::device(at::kCUDA).dtype(at::kFloat));
        mass_reference reference_pre(total_view_pages, n_q, pool_page_size);
        auto q_dec = at::randn({2, n_q, pool_head_dim}, opts);
        auto o_dec = at::zeros({2, n_q, pool_head_dim}, at::device(at::kCUDA).dtype(at::kFloat));
        mass_reference reference_dec(total_view_pages, n_q, pool_page_size);
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
        auto o = pulsarlm::ops::cuda::attn_prefill(
            q_pre,
            k_pool,
            v_pool,
            page_tables,
            page_positions,
            key_counts,
            cu_seqlens_q,
            rope_theta,
            1.0,
            std::nullopt,
            attention_scale,
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
            auto dec_o = pulsarlm::ops::cuda::attn_decode(
                q_dec,
                k_pool,
                v_pool,
                page_tables,
                page_positions,
                key_counts,
                rope_theta,
                1.0,
                std::nullopt,
                attention_scale,
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
TEST_CASE("ATen ops::rope_rotate matches the rope kernel", "[ops][cuda]") {
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
            pulsarlm::ops::rope_rotate(x, pos.view({1, -1}), theta, 1.0, std::nullopt),
            pulsarlm::ops::cuda::rope(x, pos, theta, 1.0, std::nullopt)
        ) < tol
    );

    // The scorer's second rotation is a DELTA over already-roped keys, one scalar angle
    // for the whole tensor. Rotations compose, so stored + delta must land on scored.
    const int64_t stored = 128, scored = 131072;
    auto stored_pos = at::full({seq}, stored, at::device(at::kCUDA).dtype(at::kLong));
    auto delta = at::scalar_tensor(static_cast<double>(scored - stored), at::device(at::kCUDA).dtype(at::kDouble));
    CHECK(
        worst(
            pulsarlm::ops::rope_rotate(
                pulsarlm::ops::cuda::rope(x, stored_pos, theta, 1.0, std::nullopt),
                delta,
                theta,
                1.0,
                std::nullopt
            ),
            pulsarlm::ops::cuda::rope(x, at::full_like(stored_pos, scored), theta, 1.0, std::nullopt)
        ) < tol
    );
}

namespace {

struct windowed_view {
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
    const windowed_view& view,
    const at::Tensor& k_pool,
    const pulsarlm::ops::rope_parameters& rope,
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
    return pulsarlm::ops::rope_rotate(k, pos, rope.theta, rope.partial_rotary_factor, rope.llama3_scaling)
        .to(k_pool.scalar_type())
        .to(at::kFloat);
}

void reference_windowed(
    const windowed_view& view,
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
                        const pulsarlm::ops::rope_parameters rope{.theta = 1e4, .partial_rotary_factor = factor};
                        const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
                        const int64_t num_lines = 6, max_pages = 4;
                        std::vector<windowed_view> views = {
                            windowed_view{{0, 3, 5, 1}, 3 * page_size + 5, 24},
                            windowed_view{{2, 4}, page_size + 3, 4},
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
                        auto key_counts = int32_cuda(seqlens);
                        auto cu_seqlens_q = int32_cuda(cu_q);
                        auto cu_view_pages = int32_cuda(cu_view);
                        const int64_t total_view_pages = static_cast<int64_t>(views.size()) * max_pages;
                        std::vector<float> pre_weights;
                        for (const windowed_view& view : views) {
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
                                auto o = pulsarlm::ops::cuda::attn_prefill(
                                    q_pre,
                                    k_pool,
                                    v_pool,
                                    page_tables,
                                    page_positions,
                                    key_counts,
                                    cu_seqlens_q,
                                    rope.theta,
                                    rope.partial_rotary_factor,
                                    rope.llama3_scaling,
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

                                // Measured per query, each query's views are its sequence's pages: they sum to the
                                // sequence's mass, and a query's equal the mass of a step weighing only that query.
                                const int64_t total_q = cu_q.back();
                                std::vector<int32_t> cu_query_view{0};
                                for (const windowed_view& view : views) {
                                    for (int64_t r = 0; r < view.seq_q; ++r) {
                                        cu_query_view.push_back(cu_query_view.back() + static_cast<int32_t>(max_pages));
                                    }
                                }
                                auto query_mass = at::zeros({total_q * max_pages, n_q}, mass.options());
                                auto query_weights_sum = at::zeros({total_q * max_pages}, mass.options());
                                auto verified = pulsarlm::ops::cuda::attn_prefill(
                                    q_pre,
                                    k_pool,
                                    v_pool,
                                    page_tables,
                                    page_positions,
                                    key_counts,
                                    cu_seqlens_q,
                                    rope.theta,
                                    rope.partial_rotary_factor,
                                    rope.llama3_scaling,
                                    scale,
                                    window,
                                    k_norm_weight,
                                    query_mass,
                                    query_weights_sum,
                                    int32_cuda(cu_query_view),
                                    prefill_weights,
                                    static_cast<int64_t>(views.size())
                                );
                                CHECK(at::equal(verified, o));
                                at::Tensor per_page = query_mass.view({total_q, max_pages, n_q});
                                at::Tensor per_page_weight = query_weights_sum.view({total_q, max_pages});
                                std::vector<at::Tensor> sequence_mass;
                                std::vector<at::Tensor> sequence_weight;
                                for (size_t s = 0; s < views.size(); ++s) {
                                    sequence_mass.push_back(per_page.narrow(0, cu_q[s], views[s].seq_q).sum(0));
                                    sequence_weight.push_back(
                                        per_page_weight.narrow(0, cu_q[s], views[s].seq_q).sum(0)
                                    );
                                }
                                CHECK(at::allclose(at::cat(sequence_mass), reference_pre.mass_sum(), 1e-3, 1e-3));
                                CHECK(at::allclose(at::cat(sequence_weight), reference_pre.weight_sum(), 1e-5, 1e-5));
                                for (int64_t row : {int64_t{0}, int64_t{11}, total_q - 1}) {
                                    INFO("query " << row);
                                    auto one_hot = at::zeros_like(prefill_weights);
                                    one_hot[row] = prefill_weights[row];
                                    auto alone = new_mass();
                                    auto alone_weights = new_weights();
                                    pulsarlm::ops::cuda::attn_prefill(
                                        q_pre,
                                        k_pool,
                                        v_pool,
                                        page_tables,
                                        page_positions,
                                        key_counts,
                                        cu_seqlens_q,
                                        rope.theta,
                                        rope.partial_rotary_factor,
                                        rope.llama3_scaling,
                                        scale,
                                        window,
                                        k_norm_weight,
                                        alone,
                                        alone_weights,
                                        cu_view_pages,
                                        one_hot
                                    );
                                    const int64_t s = row < cu_q[1] ? 0 : 1;
                                    CHECK(
                                        at::allclose(
                                            per_page[row],
                                            alone.narrow(0, s * max_pages, max_pages),
                                            1e-4,
                                            1e-4
                                        )
                                    );
                                }

                                for (int64_t num_splits : {0, 1, 2}) {
                                    auto dec_mass = new_mass();
                                    auto dec_weights = new_weights();
                                    auto dec_o = pulsarlm::ops::cuda::attn_decode(
                                        q_dec,
                                        k_pool,
                                        v_pool,
                                        page_tables,
                                        page_positions,
                                        key_counts,
                                        rope.theta,
                                        rope.partial_rotary_factor,
                                        rope.llama3_scaling,
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

namespace {

// Four sequences of {3, 2, 2, 1} queries over two 16-key pages of lines of their own, with random query weights. A
// sequence's view spans at most two pages and the last two sequences hold at most two queries each, so every output
// and mass sum adds at most two nonzero terms, which round alike in either order.
struct split_attention_case {
    std::vector<int32_t> query_counts{3, 2, 2, 1};
    std::vector<int32_t> key_counts{20, 31, 18, 7};
    int64_t max_pages = 2;
    int64_t n_q = 4;
    int64_t n_kv = 2;
    at::Tensor k_pool;
    at::Tensor v_pool;
    at::Tensor q;
    at::Tensor query_weights;

    explicit split_attention_case(at::ScalarType dtype) {
        auto opts = at::device(at::kCUDA).dtype(dtype);
        const auto lines = static_cast<int64_t>(this->query_counts.size()) * this->max_pages;
        this->k_pool = at::randn({lines, pool_page_size, this->n_kv, pool_head_dim}, opts);
        this->v_pool = at::randn({lines, pool_page_size, this->n_kv, pool_head_dim}, opts);
        this->q = at::randn({this->query_total(0, 4), this->n_q, pool_head_dim}, opts);
        this->query_weights = at::rand({this->query_total(0, 4)}, opts.dtype(at::kFloat)) + 0.1;
    }

    int64_t query_total(size_t first, size_t end) const {
        int64_t total = 0;
        for (size_t s = first; s < end; ++s) {
            total += this->query_counts[s];
        }
        return total;
    }

    // The views of sequences [first, end), sequence s over lines 2 s and 2 s + 1.
    pulsarlm::ops::page_views views(size_t first, size_t end) const {
        std::vector<int32_t> tables;
        std::vector<int32_t> counts;
        for (size_t s = first; s < end; ++s) {
            tables.push_back(static_cast<int32_t>(2 * s));
            tables.push_back(static_cast<int32_t>(2 * s + 1));
            counts.push_back(this->key_counts[s]);
        }
        at::Tensor page_tables = int32_cuda(tables).view({static_cast<int64_t>(end - first), this->max_pages});
        return {
            .page_tables = page_tables,
            .page_positions = identity_positions(page_tables),
            .key_counts = int32_cuda(counts),
        };
    }

    at::Tensor cu_seqlens_q(size_t first, size_t end) const {
        std::vector<int32_t> starts{0};
        for (size_t s = first; s < end; ++s) {
            starts.push_back(starts.back() + this->query_counts[s]);
        }
        return int32_cuda(starts);
    }

    at::Tensor queries(size_t first, size_t end) const {
        return this->q.narrow(0, this->query_total(0, first), this->query_total(first, end));
    }

    // A zeroed mass of mass_views views of max_pages pages each, weighing the queries of sequences [first, end).
    pulsarlm::ops::page_mass mass(int64_t mass_views, size_t first, size_t end) const {
        auto f32 = at::device(at::kCUDA).dtype(at::kFloat);
        return {
            .mass_sum = at::zeros({mass_views * this->max_pages, this->n_q}, f32),
            .weight_sum = at::zeros({mass_views * this->max_pages}, f32),
            .cu_view_pages = at::arange(mass_views + 1, f32.dtype(at::kInt)) * this->max_pages,
            .query_weights = this->query_weights.narrow(0, this->query_total(0, first), this->query_total(first, end)),
        };
    }
};

}  // namespace

TEST_CASE(
    "prefill measures the first sequences' mass per query and the others' per sequence, as a call on each part does",
    "[ops][cuda]"
) {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    for (auto dt : {at::kFloat, at::kBFloat16}) {
        INFO("dtype " << dt);
        at::manual_seed(17);
        const split_attention_case c(dt);
        // The first two sequences' 5 queries measure a view each, the last two sequences one view each.
        const pulsarlm::ops::page_mass together_mass = c.mass(7, 0, 4);
        at::Tensor together = pulsarlm::ops::cuda::attn_prefill(
            c.q,
            c.k_pool,
            c.v_pool,
            c.views(0, 4).page_tables,
            c.views(0, 4).page_positions,
            c.views(0, 4).key_counts,
            c.cu_seqlens_q(0, 4),
            rope_theta,
            1.0,
            std::nullopt,
            attention_scale,
            std::nullopt,
            std::nullopt,
            together_mass.mass_sum,
            together_mass.weight_sum,
            together_mass.cu_view_pages,
            together_mass.query_weights,
            2
        );
        const pulsarlm::ops::page_mass per_query_mass = c.mass(5, 0, 2);
        at::Tensor per_query = pulsarlm::ops::cuda::attn_prefill(
            c.queries(0, 2),
            c.k_pool,
            c.v_pool,
            c.views(0, 2).page_tables,
            c.views(0, 2).page_positions,
            c.views(0, 2).key_counts,
            c.cu_seqlens_q(0, 2),
            rope_theta,
            1.0,
            std::nullopt,
            attention_scale,
            std::nullopt,
            std::nullopt,
            per_query_mass.mass_sum,
            per_query_mass.weight_sum,
            per_query_mass.cu_view_pages,
            per_query_mass.query_weights,
            2
        );
        const pulsarlm::ops::page_mass per_sequence_mass = c.mass(2, 2, 4);
        at::Tensor per_sequence = pulsarlm::ops::cuda::attn_prefill(
            c.queries(2, 4),
            c.k_pool,
            c.v_pool,
            c.views(2, 4).page_tables,
            c.views(2, 4).page_positions,
            c.views(2, 4).key_counts,
            c.cu_seqlens_q(2, 4),
            rope_theta,
            1.0,
            std::nullopt,
            attention_scale,
            std::nullopt,
            std::nullopt,
            per_sequence_mass.mass_sum,
            per_sequence_mass.weight_sum,
            per_sequence_mass.cu_view_pages,
            per_sequence_mass.query_weights,
            0
        );

        CHECK(at::equal(together, at::cat({per_query, per_sequence})));
        CHECK(at::equal(together_mass.mass_sum, at::cat({per_query_mass.mass_sum, per_sequence_mass.mass_sum})));
        CHECK(at::equal(together_mass.weight_sum, at::cat({per_query_mass.weight_sum, per_sequence_mass.weight_sum})));
        CHECK_THROWS(
            pulsarlm::ops::cuda::attn_prefill(
                c.q,
                c.k_pool,
                c.v_pool,
                c.views(0, 4).page_tables,
                c.views(0, 4).page_positions,
                c.views(0, 4).key_counts,
                c.cu_seqlens_q(0, 4),
                rope_theta,
                1.0,
                std::nullopt,
                attention_scale,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                2
            )
        );
        CHECK_THROWS(
            pulsarlm::ops::cuda::attn_prefill(
                c.q,
                c.k_pool,
                c.v_pool,
                c.views(0, 4).page_tables,
                c.views(0, 4).page_positions,
                c.views(0, 4).key_counts,
                c.cu_seqlens_q(0, 4),
                rope_theta,
                1.0,
                std::nullopt,
                attention_scale,
                std::nullopt,
                std::nullopt,
                c.mass(7, 0, 4).mass_sum,
                c.mass(7, 0, 4).weight_sum,
                c.mass(7, 0, 4).cu_view_pages,
                c.mass(7, 0, 4).query_weights,
                4
            )
        );
    }
}

TEST_CASE("decode and prefill attention write into a given out, a slice of a larger tensor included", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    for (auto dt : {at::kFloat, at::kBFloat16}) {
        INFO("dtype " << dt);
        at::manual_seed(19);
        const split_attention_case c(dt);
        auto check_out = [&](const auto& attend, int64_t tokens) {
            at::Tensor alone = attend(std::nullopt);
            at::Tensor larger = at::randn({tokens + 5, c.n_q, pool_head_dim}, c.q.options());
            at::Tensor larger_before = larger.clone();
            at::Tensor slice = larger.narrow(0, 2, tokens);
            at::Tensor into = attend(slice);
            CHECK(into.data_ptr() == slice.data_ptr());
            CHECK(at::equal(slice, alone));
            CHECK(at::equal(larger.narrow(0, 0, 2), larger_before.narrow(0, 0, 2)));
            CHECK(at::equal(larger.narrow(0, 2 + tokens, 3), larger_before.narrow(0, 2 + tokens, 3)));
            CHECK_THROWS(attend(larger.narrow(0, 0, tokens + 1)));
            CHECK_THROWS(attend(at::empty({tokens, c.n_q, pool_head_dim}, c.q.options().dtype(at::kHalf))));
        };
        at::Tensor decode_q = c.q.narrow(0, 0, 4).contiguous();
        for (int64_t num_splits : {0, 2}) {
            INFO("num_splits " << num_splits);
            check_out(
                [&](const std::optional<at::Tensor>& out) {
                    return pulsarlm::ops::cuda::attn_decode(
                        decode_q,
                        c.k_pool,
                        c.v_pool,
                        c.views(0, 4).page_tables,
                        c.views(0, 4).page_positions,
                        c.views(0, 4).key_counts,
                        rope_theta,
                        1.0,
                        std::nullopt,
                        attention_scale,
                        std::nullopt,
                        std::nullopt,
                        std::nullopt,
                        std::nullopt,
                        std::nullopt,
                        std::nullopt,
                        num_splits,
                        out
                    );
                },
                4
            );
        }
        at::Tensor alone_mass;
        at::Tensor into_mass;
        check_out(
            [&](const std::optional<at::Tensor>& out) {
                const pulsarlm::ops::page_mass mass = c.mass(7, 0, 4);
                at::Tensor o = pulsarlm::ops::cuda::attn_prefill(
                    c.q,
                    c.k_pool,
                    c.v_pool,
                    c.views(0, 4).page_tables,
                    c.views(0, 4).page_positions,
                    c.views(0, 4).key_counts,
                    c.cu_seqlens_q(0, 4),
                    rope_theta,
                    1.0,
                    std::nullopt,
                    attention_scale,
                    std::nullopt,
                    std::nullopt,
                    mass.mass_sum,
                    mass.weight_sum,
                    mass.cu_view_pages,
                    mass.query_weights,
                    2,
                    out
                );
                (out ? into_mass : alone_mass) = at::cat({mass.mass_sum.flatten(), mass.weight_sum});
                return o;
            },
            c.query_total(0, 4)
        );
        CHECK(at::equal(into_mass, alone_mass));
    }
}

TEST_CASE("partial RoPE passes the unrotated pairs through", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const int64_t head_dim = 512, half = head_dim / 2, rotated = 64;
    const pulsarlm::ops::rope_parameters rope{.theta = 1e6, .partial_rotary_factor = 0.25};
    auto x = at::randn({2, 5, head_dim}, at::device(at::kCUDA).dtype(at::kBFloat16));
    auto pos = at::tensor(std::vector<int64_t>{0, 1, 7, 4096, 100000}, at::device(at::kCUDA).dtype(at::kLong));
    auto rotated_x = pulsarlm::ops::cuda::rope(x, pos, rope.theta, rope.partial_rotary_factor, rope.llama3_scaling);
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
    REQUIRE(pulsarlm::ops::cuda::attention_unsupported_reason(at::kFloat, 8, 32, 4, 1) == std::nullopt);
    REQUIRE(pulsarlm::ops::cuda::attention_unsupported_reason(at::kBFloat16, 16, 128, 32, 8) == std::nullopt);
    REQUIRE(pulsarlm::ops::cuda::attention_unsupported_reason(at::kBFloat16, 16, 96, 32, 8).has_value());
    REQUIRE(pulsarlm::ops::cuda::attention_unsupported_reason(at::kBFloat16, 8, 128, 32, 8).has_value());
    REQUIRE(pulsarlm::ops::cuda::attention_unsupported_reason(at::kBFloat16, 16, 128, 32, 1).has_value());

    auto opts = at::device(at::kCUDA).dtype(at::kBFloat16);
    auto pool = at::zeros({2, pool_page_size, 1, 32}, opts);
    auto q = at::randn({1, 4, 32}, opts);
    REQUIRE_THROWS_WITH(
        pulsarlm::ops::cuda::attn_decode(
            q,
            pool,
            pool,
            int32_cuda({0}).view({1, 1}),
            identity_positions(int32_cuda({0}).view({1, 1})),
            int32_cuda({4}),
            rope_theta,
            1.0,
            std::nullopt,
            attention_scale,
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
    auto key_counts = int32_cuda({20, 3});
    auto cu_seqlens_q = int32_cuda({0, 6, 8});
    auto k_new = at::randn({8, n_kv, head_dim}, opts);
    auto v_new = at::randn({8, n_kv, head_dim}, opts);

    auto k_pool = at::zeros({num_lines, pool_page_size, n_kv, head_dim}, opts);
    auto v_pool = at::zeros({num_lines, pool_page_size, n_kv, head_dim}, opts);
    pulsarlm::ops::cuda::write_kv(k_pool, v_pool, k_new, v_new, page_tables, key_counts, cu_seqlens_q);
    for (int64_t t = 0; t < 6; ++t) {
        const int64_t index = 14 + t;
        const int64_t line = index < pool_page_size ? 3 : 1;
        CHECK(at::equal(k_pool[line][index % pool_page_size], k_new[t]));
        CHECK(at::equal(v_pool[line][index % pool_page_size], v_new[t]));
    }
    CHECK(at::equal(k_pool[4][1], k_new[6]));
    CHECK(at::equal(k_pool[4][2], k_new[7]));

    auto lines = at::zeros({num_lines, 2, pool_page_size, n_kv, head_dim}, opts);
    pulsarlm::ops::cuda::write_kv(
        lines.select(1, 0),
        lines.select(1, 1),
        k_new,
        v_new,
        page_tables,
        key_counts,
        cu_seqlens_q
    );
    CHECK(at::equal(lines[3][0][14], k_new[0]));
    CHECK(at::equal(lines[3][1][14], v_new[0]));
    CHECK(at::equal(lines[4][0][2], k_new[7]));
    CHECK(at::equal(lines[4][1][2], v_new[7]));
    CHECK(lines.select(1, 1)[3][13].abs().sum().item<float>() == 0.0f);

    auto head_major = at::zeros({num_lines, n_kv, 2, pool_page_size, head_dim}, opts).permute({0, 2, 3, 1, 4});
    pulsarlm::ops::cuda::write_kv(
        head_major.select(1, 0),
        head_major.select(1, 1),
        k_new,
        v_new,
        page_tables,
        key_counts,
        cu_seqlens_q
    );
    CHECK(at::equal(head_major.select(1, 0)[3][14], k_new[0]));
    CHECK(at::equal(head_major.select(1, 1)[4][2], v_new[7]));

    auto pool = at::zeros({num_lines, pool_page_size, n_kv, head_dim}, opts);
    auto decode_tokens = at::randn({2, n_kv, head_dim}, opts);
    pulsarlm::ops::cuda::write_kv(pool, pool, decode_tokens, decode_tokens, page_tables, key_counts, std::nullopt);
    CHECK(at::equal(pool[1][3], decode_tokens[0]));
    CHECK(at::equal(pool[4][2], decode_tokens[1]));
}

TEST_CASE("gather_page_tables lays each table's rows side by side up to its page count", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    std::vector<at::Tensor> tables{
        int32_cuda({1, 2, 3, 4, 5, 6}).view({2, 3}),
        int32_cuda({7, 8, 9, 10}).view({2, 2}),
    };
    std::vector<std::size_t> page_counts{3, 2};
    auto gathered = pulsarlm::ops::cuda::gather_page_tables(tables, page_counts, 2, 3).cpu();
    CHECK(at::equal(gathered.select(1, 0), at::tensor(std::vector<int32_t>{1, 2, 3, 4, 5, 6}).view({2, 3})));
    CHECK(at::equal(gathered.select(1, 1).narrow(1, 0, 2), at::tensor(std::vector<int32_t>{7, 8, 9, 10}).view({2, 2})));
}

TEST_CASE("gather_page_tables leaves out a table's entries past its page count", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    constexpr int32_t not_a_line = 1065353216;
    std::vector<at::Tensor> tables{
        int32_cuda({1, 2, 3, 4, 5, 6}).view({2, 3}),
        int32_cuda({7, not_a_line, not_a_line, not_a_line, 8, not_a_line, not_a_line, not_a_line}).view({2, 4}),
    };
    std::vector<std::size_t> page_counts{3, 1};
    auto gathered = pulsarlm::ops::cuda::gather_page_tables(tables, page_counts, 2, 3).cpu();
    CHECK(at::equal(gathered.select(1, 0), at::tensor(std::vector<int32_t>{1, 2, 3, 4, 5, 6}).view({2, 3})));
    CHECK(at::equal(gathered.select(1, 1).narrow(1, 0, 1), at::tensor(std::vector<int32_t>{7, 8}).view({2, 1})));
}
