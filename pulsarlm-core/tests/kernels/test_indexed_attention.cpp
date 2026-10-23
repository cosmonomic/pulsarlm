#include "pulsarlm/ops.hpp"
#include "pulsarlm/rope.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAFunctions.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

// attn_indexed_cuda against a dense ATen reference over each query's selected keys, and against attn_prefill_cuda when
// every block is selected. fp32 runs the scalar kernel, bf16 the tensor-core one.

namespace {

constexpr int64_t kBlockSize = 4;
constexpr int64_t kMaxPages = 4;
constexpr double kAlpha = 0.25;

at::Tensor int32_cuda(const std::vector<int32_t>& v) {
    return at::tensor(v, at::dtype(at::kInt)).to(at::kCUDA);
}

struct View {
    std::vector<int32_t> lines;
    int64_t ctx;
    int64_t seq_q;
};

struct Batch {
    std::vector<View> views;
    at::Tensor page_tables;
    at::Tensor cu_seqlens_q;
    at::Tensor seqlens_k;
    at::Tensor page_positions;
    at::Tensor cu_view_pages;
    // Query row r of a view of seq_q queries weighs kAlpha * (1 - kAlpha)^(seq_q - 1 - r).
    std::vector<double> weights;
    at::Tensor query_weights;
    int64_t total_q = 0;
};

Batch make_batch(int64_t page_size) {
    Batch b{.views = {View{{0, 3, 5, 1}, 3 * page_size + 5, 24}, View{{2, 4}, page_size + 3, 1}}};
    std::vector<int32_t> tables(b.views.size() * kMaxPages, 0), seqlens, cu_q{0}, cu_view;
    for (size_t s = 0; s < b.views.size(); ++s) {
        std::copy(b.views[s].lines.begin(), b.views[s].lines.end(), tables.begin() + s * kMaxPages);
        seqlens.push_back(static_cast<int32_t>(b.views[s].ctx));
        cu_q.push_back(cu_q.back() + static_cast<int32_t>(b.views[s].seq_q));
        cu_view.push_back(static_cast<int32_t>(s * kMaxPages));
    }
    cu_view.push_back(static_cast<int32_t>(b.views.size() * kMaxPages));
    const auto num_seqs = static_cast<int64_t>(b.views.size());
    b.page_tables = int32_cuda(tables).view({num_seqs, kMaxPages});
    b.cu_seqlens_q = int32_cuda(cu_q);
    b.seqlens_k = int32_cuda(seqlens);
    b.page_positions = (at::arange(kMaxPages, b.page_tables.options()) * page_size)
                           .expand_as(b.page_tables)
                           .contiguous();
    b.cu_view_pages = int32_cuda(cu_view);
    for (const View& view : b.views) {
        for (int64_t r = 0; r < view.seq_q; ++r) {
            b.weights.push_back(kAlpha * std::pow(1.0 - kAlpha, static_cast<double>(view.seq_q - 1 - r)));
        }
    }
    b.query_weights = at::tensor(std::vector<float>(b.weights.begin(), b.weights.end())).to(at::kCUDA);
    b.total_q = cu_q.back();
    return b;
}

// Each query's blocks at random, its own block always among them.
at::Tensor random_selection(const Batch& b, int64_t page_size, int64_t words) {
    at::Tensor bits = at::zeros({b.total_q, words * 32}, at::kBool);
    int64_t row = 0;
    for (const View& view : b.views) {
        for (int64_t r = 0; r < view.seq_q; ++r, ++row) {
            const int64_t p = view.ctx - view.seq_q + r;
            const int64_t blocks = p / kBlockSize + 1;
            bits[row].narrow(0, 0, blocks).copy_(at::rand({blocks}) < 0.4);
            bits[row][p / kBlockSize] = true;
        }
    }
    at::Tensor weights = at::pow(2, at::arange(32, at::kLong));
    return (bits.view({b.total_q, words, 32}).to(at::kLong) * weights).sum(-1).to(at::kInt).to(at::kCUDA);
}

at::Tensor full_selection(const Batch& b, int64_t words) {
    return at::full({b.total_q, words}, -1, at::device(at::kCUDA).dtype(at::kInt));
}

bool selected(const at::Tensor& host_bits, int64_t row, int64_t key) {
    const int64_t block = key / kBlockSize;
    return (static_cast<uint32_t>(host_bits[row][block / 32].item<int32_t>()) >> (block % 32)) & 1u;
}

// The dense reference: each query's keys j <= p of its selected blocks, the keys rotated to their view index and
// rounded to the pool dtype as the kernels stage them. mass_sum is [view pages, heads]: per page and head, the sum over
// the queries attending one of the page's keys of their weight times their max scaled score over its keys they attend;
// weight_sum is [view pages], the sum of those weights.
void reference(
    const Batch& b,
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& selection,
    const pulsarlm::rope_parameters& rope,
    double scale,
    at::Tensor& o,
    at::Tensor& mass_sum,
    at::Tensor& weight_sum
) {
    const int64_t n_kv = k_pool.size(2), head_dim = k_pool.size(3), group = q.size(1) / n_kv;
    at::Tensor host_bits = selection.cpu();
    at::Tensor host_mass = at::zeros(mass_sum.sizes(), at::kDouble);
    at::Tensor host_weight = at::zeros(weight_sum.sizes(), at::kDouble);
    int64_t row = 0;
    for (size_t s = 0; s < b.views.size(); ++s) {
        const View& view = b.views[s];
        auto lines = at::tensor(std::vector<int64_t>(view.lines.begin(), view.lines.end())).to(at::kCUDA);
        auto k = k_pool.index_select(0, lines).view({-1, n_kv, head_dim}).slice(0, 0, view.ctx).to(at::kFloat);
        auto pos = at::arange(view.ctx, at::device(at::kCUDA).dtype(at::kLong)).view({-1, 1});
        k = pulsarlm::rope_rotate(k, pos, rope).to(k_pool.scalar_type()).to(at::kFloat);
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
                std::vector<std::optional<double>> page_max(kMaxPages);
                for (size_t i = 0; i < keys.size(); ++i) {
                    std::optional<double>& max = page_max[keys[i] / page_size];
                    auto score = static_cast<double>(host_scores[static_cast<int64_t>(i)].item<float>());
                    max = std::max(max.value_or(score), score);
                }
                for (int64_t page = 0; page < kMaxPages; ++page) {
                    if (page_max[page]) {
                        const int64_t view_page = static_cast<int64_t>(s) * kMaxPages + page;
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
    const pulsarlm::rope_parameters rope{.theta = 1e4, .partial_rotary_factor = 0.25};
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    at::manual_seed(7);
    Batch b = make_batch(page_size);
    auto opts = at::device(at::kCUDA).dtype(dtype);
    auto k_pool = at::randn({num_lines, page_size, n_kv, head_dim}, opts);
    auto v_pool = at::randn({num_lines, page_size, n_kv, head_dim}, opts);
    auto q = at::randn({b.total_q, n_q, head_dim}, opts);
    const int64_t words = (kMaxPages * page_size / kBlockSize + 31) / 32;
    at::Tensor selection = random_selection(b, page_size, words);
    auto mass_of = [&] {
        return at::zeros({static_cast<int64_t>(b.views.size()) * kMaxPages, n_q}, opts.dtype(at::kFloat));
    };
    auto weights_of = [&] {
        return at::zeros({static_cast<int64_t>(b.views.size()) * kMaxPages}, opts.dtype(at::kFloat));
    };

    at::Tensor mass = mass_of();
    at::Tensor weights = weights_of();
    at::Tensor o = pulsarlm::attn_indexed_cuda(
        q,
        k_pool,
        v_pool,
        b.page_tables,
        b.cu_seqlens_q,
        b.seqlens_k,
        b.page_positions,
        rope,
        scale,
        selection,
        kBlockSize,
        mass,
        weights,
        b.cu_view_pages,
        b.query_weights
    );
    at::Tensor expected_o = at::zeros({b.total_q, n_q, head_dim}, opts.dtype(at::kFloat));
    at::Tensor expected_mass = mass_of();
    at::Tensor expected_weights = weights_of();
    reference(b, q, k_pool, v_pool, selection, rope, scale, expected_o, expected_mass, expected_weights);
    const double tolerance = dtype == at::kFloat ? 1e-4 : 2e-2;
    CHECK(at::allclose(o.to(at::kFloat), expected_o, tolerance, tolerance));
    CHECK(at::allclose(mass, expected_mass, tolerance, tolerance));
    CHECK(at::allclose(weights, expected_weights, 1e-5, 1e-6));

    at::Tensor dense_mass = mass_of();
    at::Tensor dense_weights = weights_of();
    at::Tensor dense = pulsarlm::attn_prefill_cuda(
        q,
        k_pool,
        v_pool,
        b.page_tables,
        b.cu_seqlens_q,
        b.seqlens_k,
        b.page_positions,
        rope,
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
    at::Tensor all = pulsarlm::attn_indexed_cuda(
        q,
        k_pool,
        v_pool,
        b.page_tables,
        b.cu_seqlens_q,
        b.seqlens_k,
        b.page_positions,
        rope,
        scale,
        full_selection(b, words),
        kBlockSize,
        all_mass,
        all_weights,
        b.cu_view_pages,
        b.query_weights
    );
    // The scalar kernel sums each output across threads with atomicAdd, and both ops add the mass with atomicAdd, in no
    // fixed order.
    if (dtype == at::kFloat) {
        CHECK(at::allclose(all, dense, 1e-6, 1e-6));
    } else {
        CHECK(at::equal(all, dense));
    }
    CHECK(at::allclose(all_mass, dense_mass, 1e-5, 1e-6));
    CHECK(at::allclose(all_weights, dense_weights, 1e-5, 1e-6));
}
