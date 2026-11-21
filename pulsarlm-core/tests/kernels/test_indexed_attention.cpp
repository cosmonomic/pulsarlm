#include "pulsarlm/ops/indexed_attention.hpp"
#include "pulsarlm/ops/paged_attention.hpp"
#include "pulsarlm/ops/rope.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAFunctions.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

// ops::cuda::attn_indexed against a dense ATen reference over each query's selected keys, and against
// ops::cuda::attn_prefill when every block is selected. fp32 runs the scalar kernel, bf16 the tensor-core one.

namespace {

constexpr int64_t block_size = 4;
constexpr int64_t max_pages = 4;
constexpr double alpha = 0.25;

at::Tensor int32_cuda(const std::vector<int32_t>& v) {
    return at::tensor(v, at::dtype(at::kInt)).to(at::kCUDA);
}

struct sequence_view {
    std::vector<int32_t> lines;
    int64_t ctx;
    int64_t seq_q;
};

struct view_batch {
    std::vector<sequence_view> views;
    at::Tensor page_tables;
    at::Tensor cu_seqlens_q;
    at::Tensor key_counts;
    at::Tensor page_positions;
    at::Tensor cu_view_pages;
    // Query row r of a view of seq_q queries weighs alpha * (1 - alpha)^(seq_q - 1 - r).
    std::vector<double> weights;
    at::Tensor query_weights;
    int64_t total_q = 0;
};

// A view of first_ctx keys whose last first_queries are queries, and a view of one query.
view_batch make_batch(int64_t page_size, int64_t first_ctx, int64_t first_queries) {
    view_batch b{
        .views = {sequence_view{{0, 3, 5, 1}, first_ctx, first_queries}, sequence_view{{2, 4}, page_size + 3, 1}}
    };
    std::vector<int32_t> tables(b.views.size() * max_pages, 0), seqlens, cu_q{0}, cu_view;
    for (size_t s = 0; s < b.views.size(); ++s) {
        std::copy(b.views[s].lines.begin(), b.views[s].lines.end(), tables.begin() + s * max_pages);
        seqlens.push_back(static_cast<int32_t>(b.views[s].ctx));
        cu_q.push_back(cu_q.back() + static_cast<int32_t>(b.views[s].seq_q));
        cu_view.push_back(static_cast<int32_t>(s * max_pages));
    }
    cu_view.push_back(static_cast<int32_t>(b.views.size() * max_pages));
    const auto num_seqs = static_cast<int64_t>(b.views.size());
    b.page_tables = int32_cuda(tables).view({num_seqs, max_pages});
    b.cu_seqlens_q = int32_cuda(cu_q);
    b.key_counts = int32_cuda(seqlens);
    b.page_positions = (at::arange(max_pages, b.page_tables.options()) * page_size)
                           .expand_as(b.page_tables)
                           .contiguous();
    b.cu_view_pages = int32_cuda(cu_view);
    for (const sequence_view& view : b.views) {
        for (int64_t r = 0; r < view.seq_q; ++r) {
            b.weights.push_back(alpha * std::pow(1.0 - alpha, static_cast<double>(view.seq_q - 1 - r)));
        }
    }
    b.query_weights = at::tensor(std::vector<float>(b.weights.begin(), b.weights.end())).to(at::kCUDA);
    b.total_q = cu_q.back();
    return b;
}

// Each query's blocks at random, its own block always among them.
at::Tensor random_selection(const view_batch& b, int64_t page_size, int64_t words) {
    at::Tensor bits = at::zeros({b.total_q, words * 32}, at::kBool);
    int64_t row = 0;
    for (const sequence_view& view : b.views) {
        for (int64_t r = 0; r < view.seq_q; ++r, ++row) {
            const int64_t p = view.ctx - view.seq_q + r;
            const int64_t blocks = p / block_size + 1;
            bits[row].narrow(0, 0, blocks).copy_(at::rand({blocks}) < 0.4);
            bits[row][p / block_size] = true;
        }
    }
    at::Tensor weights = at::pow(2, at::arange(32, at::kLong));
    return (bits.view({b.total_q, words, 32}).to(at::kLong) * weights).sum(-1).to(at::kInt).to(at::kCUDA);
}

at::Tensor full_selection(const view_batch& b, int64_t words) {
    return at::full({b.total_q, words}, -1, at::device(at::kCUDA).dtype(at::kInt));
}

bool selected(const at::Tensor& host_bits, int64_t row, int64_t key) {
    const int64_t block = key / block_size;
    return (static_cast<uint32_t>(host_bits[row][block / 32].item<int32_t>()) >> (block % 32)) & 1u;
}

// The dense reference: each query's keys j <= p of its selected blocks, the keys rotated to their view index and
// rounded to the pool dtype as the kernels stage them. mass_sum is [view pages, heads]: per page and head, the sum over
// the queries attending one of the page's keys of their weight times their max scaled score over its keys they attend;
// weight_sum is [view pages], the sum of those weights. A sequence's queries share its view pages, or with per_query
// each query has its own, max_pages per query row.
void reference(
    const view_batch& b,
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& selection,
    const pulsarlm::ops::rope_parameters& rope,
    double scale,
    at::Tensor& o,
    at::Tensor& mass_sum,
    at::Tensor& weight_sum,
    bool per_query
) {
    const int64_t n_kv = k_pool.size(2), head_dim = k_pool.size(3), group = q.size(1) / n_kv;
    at::Tensor host_bits = selection.cpu();
    at::Tensor host_mass = at::zeros(mass_sum.sizes(), at::kDouble);
    at::Tensor host_weight = at::zeros(weight_sum.sizes(), at::kDouble);
    int64_t row = 0;
    for (size_t s = 0; s < b.views.size(); ++s) {
        const sequence_view& view = b.views[s];
        auto lines = at::tensor(std::vector<int64_t>(view.lines.begin(), view.lines.end())).to(at::kCUDA);
        auto k = k_pool.index_select(0, lines).view({-1, n_kv, head_dim}).slice(0, 0, view.ctx).to(at::kFloat);
        auto pos = at::arange(view.ctx, at::device(at::kCUDA).dtype(at::kLong)).view({-1, 1});
        k = pulsarlm::ops::rope_rotate(k, pos, rope.theta, rope.partial_rotary_factor, rope.llama3_scaling)
                .to(k_pool.scalar_type())
                .to(at::kFloat);
        auto v = v_pool.index_select(0, lines).view({-1, n_kv, head_dim}).slice(0, 0, view.ctx).to(at::kFloat);
        const int64_t page_size = k_pool.size(1);
        for (int64_t r = 0; r < view.seq_q; ++r, ++row) {
            const int64_t p = view.ctx - view.seq_q + r;
            std::vector<int64_t> keys;
            for (int64_t j = 0; j <= p; ++j) {
                if (selected(host_bits, row, j)) {
                    keys.push_back(j);
                }
            }
            auto attended = at::tensor(keys).to(at::kCUDA);
            const double weight = b.weights[row];
            for (int64_t h = 0; h < q.size(1); ++h) {
                const int64_t g = h / group;
                auto scores = at::mv(k.index_select(0, attended).select(1, g), q[row][h].to(at::kFloat)) * scale;
                o[row][h] = at::mv(v.index_select(0, attended).select(1, g).t(), at::softmax(scores, 0));
                at::Tensor host_scores = scores.cpu();
                std::vector<std::optional<double>> page_max(max_pages);
                for (size_t i = 0; i < keys.size(); ++i) {
                    std::optional<double>& max = page_max[keys[i] / page_size];
                    auto score = static_cast<double>(host_scores[static_cast<int64_t>(i)].item<float>());
                    max = std::max(max.value_or(score), score);
                }
                for (int64_t page = 0; page < max_pages; ++page) {
                    if (page_max[page]) {
                        const int64_t view = per_query ? row : static_cast<int64_t>(s);
                        const int64_t view_page = view * max_pages + page;
                        host_mass[view_page][h] += weight * *page_max[page];
                        if (h == 0) {
                            host_weight[view_page] += weight;
                        }
                    }
                }
            }
        }
    }
    mass_sum.copy_(host_mass);
    weight_sum.copy_(host_weight);
}

}  // namespace

TEST_CASE("indexed attention attends each query's selected keys", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const auto dtype = GENERATE(at::kFloat, at::kBFloat16);
    const int64_t head_dim = GENERATE(64, 256);
    const int64_t page_size = GENERATE(16, 32);
    const int64_t n_q = 4, n_kv = 2, num_lines = 6;
    const pulsarlm::ops::rope_parameters rope{.theta = 1e4, .partial_rotary_factor = 0.25};
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    at::manual_seed(7);
    // On an RTX 5090 (170 SMs) or an H200 (132 SMs), 64 queries of the first view run one split per (query token, KV
    // group), 24 run 2 and 1 runs 16.
    const int64_t first_queries = GENERATE(64, 24, 1);
    view_batch b = make_batch(page_size, first_queries == 64 ? 4 * page_size : 3 * page_size + 5, first_queries);
    auto opts = at::device(at::kCUDA).dtype(dtype);
    auto k_pool = at::randn({num_lines, page_size, n_kv, head_dim}, opts);
    auto v_pool = at::randn({num_lines, page_size, n_kv, head_dim}, opts);
    auto q = at::randn({b.total_q, n_q, head_dim}, opts);
    const int64_t words = (max_pages * page_size / block_size + 31) / 32;
    at::Tensor selection = random_selection(b, page_size, words);
    auto mass_of = [&] {
        return at::zeros({static_cast<int64_t>(b.views.size()) * max_pages, n_q}, opts.dtype(at::kFloat));
    };
    auto weights_of = [&] {
        return at::zeros({static_cast<int64_t>(b.views.size()) * max_pages}, opts.dtype(at::kFloat));
    };

    at::Tensor mass = mass_of();
    at::Tensor weights = weights_of();
    at::Tensor o = pulsarlm::ops::cuda::attn_indexed(
        q,
        k_pool,
        v_pool,
        b.page_tables,
        b.page_positions,
        b.key_counts,
        b.cu_seqlens_q,
        rope.theta,
        rope.partial_rotary_factor,
        rope.llama3_scaling,
        scale,
        selection,
        block_size,
        max_pages * page_size / block_size,
        mass,
        weights,
        b.cu_view_pages,
        b.query_weights,
        std::nullopt
    );
    at::Tensor expected_o = at::zeros({b.total_q, n_q, head_dim}, opts.dtype(at::kFloat));
    at::Tensor expected_mass = mass_of();
    at::Tensor expected_weights = weights_of();
    reference(b, q, k_pool, v_pool, selection, rope, scale, expected_o, expected_mass, expected_weights, false);
    const double tolerance = dtype == at::kFloat ? 1e-4 : 2e-2;
    CHECK(at::allclose(o.to(at::kFloat), expected_o, tolerance, tolerance));
    CHECK(at::allclose(mass, expected_mass, tolerance, tolerance));
    CHECK(at::allclose(weights, expected_weights, 1e-5, 1e-6));

    at::Tensor dense_mass = mass_of();
    at::Tensor dense_weights = weights_of();
    at::Tensor dense = pulsarlm::ops::cuda::attn_prefill(
        q,
        k_pool,
        v_pool,
        b.page_tables,
        b.page_positions,
        b.key_counts,
        b.cu_seqlens_q,
        rope.theta,
        rope.partial_rotary_factor,
        rope.llama3_scaling,
        scale,
        std::nullopt,
        std::nullopt,
        dense_mass,
        dense_weights,
        b.cu_view_pages,
        b.query_weights
    );
    at::Tensor all_mass = mass_of();
    at::Tensor all_weights = weights_of();
    at::Tensor all = pulsarlm::ops::cuda::attn_indexed(
        q,
        k_pool,
        v_pool,
        b.page_tables,
        b.page_positions,
        b.key_counts,
        b.cu_seqlens_q,
        rope.theta,
        rope.partial_rotary_factor,
        rope.llama3_scaling,
        scale,
        full_selection(b, words),
        block_size,
        max_pages * page_size / block_size,
        all_mass,
        all_weights,
        b.cu_view_pages,
        b.query_weights,
        std::nullopt
    );
    // The scalar kernel sums each output across threads with atomicAdd, and both ops add the mass with atomicAdd, in no
    // fixed order. Joined splits round differently from one pass over the keys.
    if (dtype == at::kFloat) {
        CHECK(at::allclose(all, dense, 1e-6, 1e-6));
    } else if (first_queries == 64) {
        CHECK(at::equal(all, dense));
    } else {
        CHECK(at::allclose(all.to(at::kFloat), dense.to(at::kFloat), 2e-2, 2e-2));
    }
    CHECK(at::allclose(all_mass, dense_mass, 1e-5, 1e-6));
    CHECK(at::allclose(all_weights, dense_weights, 1e-5, 1e-6));
}

TEST_CASE("verify indexed attention measures each query's mass in its own view", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const auto dtype = GENERATE(at::kFloat, at::kBFloat16);
    const int64_t first_queries = GENERATE(3, 24);
    const int64_t head_dim = 64;
    const int64_t page_size = 16;
    const int64_t n_q = 4, n_kv = 2, num_lines = 6;
    const pulsarlm::ops::rope_parameters rope{.theta = 1e4, .partial_rotary_factor = 0.25};
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    at::manual_seed(11);
    view_batch b = make_batch(page_size, 3 * page_size + 5, first_queries);
    auto opts = at::device(at::kCUDA).dtype(dtype);
    auto k_pool = at::randn({num_lines, page_size, n_kv, head_dim}, opts);
    auto v_pool = at::randn({num_lines, page_size, n_kv, head_dim}, opts);
    auto q = at::randn({b.total_q, n_q, head_dim}, opts);
    const int64_t words = (max_pages * page_size / block_size + 31) / 32;
    at::Tensor selection = random_selection(b, page_size, words);
    at::Tensor cu_query_pages = at::arange(b.total_q + 1, b.cu_view_pages.options()) * max_pages;
    at::Tensor mass = at::zeros({b.total_q * max_pages, n_q}, opts.dtype(at::kFloat));
    at::Tensor weights = at::zeros({b.total_q * max_pages}, opts.dtype(at::kFloat));

    at::Tensor o = pulsarlm::ops::cuda::attn_indexed_verify(
        q,
        k_pool,
        v_pool,
        b.page_tables,
        b.page_positions,
        b.key_counts,
        b.cu_seqlens_q,
        rope.theta,
        rope.partial_rotary_factor,
        rope.llama3_scaling,
        scale,
        selection,
        block_size,
        max_pages * page_size / block_size,
        mass,
        weights,
        cu_query_pages,
        b.query_weights,
        std::nullopt
    );

    at::Tensor expected_o = at::zeros({b.total_q, n_q, head_dim}, opts.dtype(at::kFloat));
    at::Tensor expected_mass = at::zeros_like(mass);
    at::Tensor expected_weights = at::zeros_like(weights);
    reference(b, q, k_pool, v_pool, selection, rope, scale, expected_o, expected_mass, expected_weights, true);
    const double tolerance = dtype == at::kFloat ? 1e-4 : 2e-2;
    CHECK(at::allclose(o.to(at::kFloat), expected_o, tolerance, tolerance));
    CHECK(at::allclose(mass, expected_mass, tolerance, tolerance));
    CHECK(at::allclose(weights, expected_weights, 1e-5, 1e-6));
}

TEST_CASE("the indexed attention ops write into a given out, a slice of a larger tensor included", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const auto dtype = GENERATE(at::kFloat, at::kBFloat16);
    const bool verify = GENERATE(false, true);
    const int64_t head_dim = 64;
    // Each view spans two pages and holds at most two queries, so every output and mass sum adds at most two nonzero
    // terms, which round alike in either order.
    const int64_t page_size = 32;
    const int64_t n_q = 4, n_kv = 2, num_lines = 6;
    const pulsarlm::ops::rope_parameters rope{.theta = 1e4, .partial_rotary_factor = 0.25};
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    at::manual_seed(13);
    view_batch b = make_batch(page_size, page_size + 20, 2);
    auto opts = at::device(at::kCUDA).dtype(dtype);
    auto k_pool = at::randn({num_lines, page_size, n_kv, head_dim}, opts);
    auto v_pool = at::randn({num_lines, page_size, n_kv, head_dim}, opts);
    auto q = at::randn({b.total_q, n_q, head_dim}, opts);
    const int64_t words = (max_pages * page_size / block_size + 31) / 32;
    const at::Tensor selected = random_selection(b, page_size, words);
    const int64_t max_selected_blocks = max_pages * page_size / block_size;
    const int64_t views = verify ? b.total_q : static_cast<int64_t>(b.views.size());
    at::Tensor cu_view_pages = at::arange(views + 1, b.cu_view_pages.options()) * max_pages;
    auto attend = [&](const std::optional<at::Tensor>& out, at::Tensor& mass, at::Tensor& weights) {
        mass = at::zeros({views * max_pages, n_q}, opts.dtype(at::kFloat));
        weights = at::zeros({views * max_pages}, opts.dtype(at::kFloat));
        return verify ? pulsarlm::ops::cuda::attn_indexed_verify(
                            q,
                            k_pool,
                            v_pool,
                            b.page_tables,
                            b.page_positions,
                            b.key_counts,
                            b.cu_seqlens_q,
                            rope.theta,
                            rope.partial_rotary_factor,
                            rope.llama3_scaling,
                            scale,
                            selected,
                            block_size,
                            max_selected_blocks,
                            mass,
                            weights,
                            cu_view_pages,
                            b.query_weights,
                            out
                        )
                      : pulsarlm::ops::cuda::attn_indexed(
                            q,
                            k_pool,
                            v_pool,
                            b.page_tables,
                            b.page_positions,
                            b.key_counts,
                            b.cu_seqlens_q,
                            rope.theta,
                            rope.partial_rotary_factor,
                            rope.llama3_scaling,
                            scale,
                            selected,
                            block_size,
                            max_selected_blocks,
                            mass,
                            weights,
                            cu_view_pages,
                            b.query_weights,
                            out
                        );
    };

    at::Tensor alone_mass, alone_weights;
    at::Tensor alone = attend(std::nullopt, alone_mass, alone_weights);
    at::Tensor larger = at::randn({b.total_q + 5, n_q, head_dim}, opts);
    at::Tensor larger_before = larger.clone();
    at::Tensor slice = larger.narrow(0, 2, b.total_q);
    at::Tensor into_mass, into_weights;
    at::Tensor into = attend(slice, into_mass, into_weights);

    CHECK(into.data_ptr() == slice.data_ptr());
    CHECK(at::equal(slice, alone));
    CHECK(at::equal(larger.narrow(0, 0, 2), larger_before.narrow(0, 0, 2)));
    CHECK(at::equal(larger.narrow(0, 2 + b.total_q, 3), larger_before.narrow(0, 2 + b.total_q, 3)));
    CHECK(at::equal(into_mass, alone_mass));
    CHECK(at::equal(into_weights, alone_weights));
    at::Tensor unused_mass, unused_weights;
    CHECK_THROWS(attend(larger.narrow(0, 0, b.total_q + 1), unused_mass, unused_weights));
}
