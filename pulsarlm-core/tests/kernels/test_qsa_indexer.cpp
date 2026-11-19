#include "pulsarlm/kernels/indexed_attention.hpp"
#include "pulsarlm/kernels/normalization.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAFunctions.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

// QSA's indexer ops against ATen references: index_queries_cuda against rmsnorm_cuda and the reference RoPE,
// index_blocks_cuda against the pooled, normed and roped view blocks, index_scores_cuda against the relu sums of each
// row's heads, and select_blocks_cuda against a stable descending sort.

namespace {

constexpr int64_t pool_page_size = 16;
constexpr int64_t ratio = 4;
constexpr int64_t index_dim = 32;
constexpr int64_t rotary_dim = 16;
constexpr double rope_theta = 1e6;
constexpr double eps = 1e-6;

at::Tensor int32_cuda(const std::vector<int32_t>& v) {
    return at::tensor(v, at::dtype(at::kInt)).to(at::kCUDA);
}

at::Tensor inv_freq() {
    return (1.0 / at::pow(rope_theta, at::arange(0, rotary_dim, 2, at::kLong).to(at::kFloat) / rotary_dim))
        .to(at::kCUDA);
}

at::Tensor reference_rope(const at::Tensor& x, const at::Tensor& positions) {
    auto freqs = positions.to(at::kFloat).unsqueeze(-1) * inv_freq();
    auto angles = at::cat({freqs, freqs}, -1);
    auto cos = angles.cos().to(x.scalar_type());
    auto sin = angles.sin().to(x.scalar_type());
    auto rotated = x.narrow(-1, 0, rotary_dim);
    const int64_t half = rotary_dim / 2;
    auto rotated_half = at::cat({-rotated.narrow(-1, half, half), rotated.narrow(-1, 0, half)}, -1);
    return at::cat({rotated * cos + rotated_half * sin, x.narrow(-1, rotary_dim, index_dim - rotary_dim)}, -1);
}

// Packs bool [rows, words * 32] into selection words.
at::Tensor packed(const at::Tensor& bits, int64_t words) {
    auto weights = at::pow(2, at::arange(32, at::device(at::kCUDA).dtype(at::kLong)));
    auto sums = (bits.view({bits.size(0), words, 32}).to(at::kLong) * weights).sum(-1);
    return at::where(sums >= (int64_t{1} << 31), sums - (int64_t{1} << 32), sums).to(at::kInt);
}

// Each row's sequence, complete blocks and view index for views of key_counts keys, sequence s holding rows
// [cu_seqlens[s], cu_seqlens[s + 1]), or row s alone with cu_seqlens empty, as int64 CUDA tensors.
std::tuple<at::Tensor, at::Tensor, at::Tensor>
row_layout(const std::vector<int32_t>& key_counts, const std::vector<int32_t>& cu_seqlens) {
    std::vector<int64_t> sequence, blocks, view;
    for (size_t s = 0; s < key_counts.size(); s++) {
        const int64_t first = cu_seqlens.empty() ? static_cast<int64_t>(s) : cu_seqlens[s];
        const int64_t count = cu_seqlens.empty() ? 1 : cu_seqlens[s + 1] - first;
        for (int64_t r = 0; r < count; r++) {
            const int64_t index = key_counts[s] - count + r;
            sequence.push_back(static_cast<int64_t>(s));
            view.push_back(index);
            blocks.push_back((index + 1) / ratio);
        }
    }
    auto cuda = [](const std::vector<int64_t>& v) { return at::tensor(v).to(at::kCUDA); };
    return {cuda(sequence), cuda(blocks), cuda(view)};
}

}  // namespace

TEST_CASE("index_blocks_cuda pools, norms and ropes each view block", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const auto dtype = GENERATE(at::kFloat, at::kBFloat16);
    at::manual_seed(3);
    auto opts = at::device(at::kCUDA).dtype(dtype);
    const int64_t lines = 6, max_pages = 3;
    auto index_keys = at::randn({lines, pool_page_size, index_dim}, opts);
    auto page_tables = int32_cuda({4, 1, 5, 2, 0, 3}).view({2, max_pages});
    // The second view's pages sit at positions with gaps between them.
    auto page_positions = int32_cuda({0, 16, 32, 0, 48, 112}).view({2, max_pages});
    auto norm_weight = (1.0 + 0.1 * at::randn({index_dim})).to(at::kCUDA);

    auto key_counts = int32_cuda({max_pages * pool_page_size, max_pages * pool_page_size});

    at::Tensor blocks = pulsarlm::index_blocks_cuda(
        index_keys,
        {.page_tables = page_tables, .page_positions = page_positions, .key_counts = key_counts},
        norm_weight,
        inv_freq(),
        ratio,
        eps
    );

    const int64_t max_blocks = max_pages * pool_page_size / ratio;
    auto keys = index_keys.index_select(0, page_tables.flatten().to(at::kLong)).view({2, max_blocks, ratio, index_dim});
    auto pooled = keys.to(at::kFloat).mean(2).to(dtype).to(at::kFloat);
    auto normed = (pooled * at::rsqrt(pooled.pow(2).mean(-1, true) + eps) * norm_weight).to(dtype);
    auto first_keys = at::arange(max_blocks, at::device(at::kCUDA).dtype(at::kLong)) * ratio;
    auto positions = page_positions.to(at::kLong).index_select(1, at::floor_divide(first_keys, pool_page_size)) +
        at::remainder(first_keys, pool_page_size);
    auto expected = reference_rope(normed, positions);
    const double tolerance = dtype == at::kFloat ? 1e-5 : 1e-2;
    CHECK(at::allclose(blocks.to(at::kFloat), expected.to(at::kFloat), tolerance, tolerance));
}

TEST_CASE("index_blocks_cuda zeros the blocks ending past a view's keys without reading them", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    at::manual_seed(5);
    auto opts = at::device(at::kCUDA).dtype(at::kBFloat16);
    const int64_t lines = 6, max_pages = 3;
    constexpr int32_t not_a_line = 1065353216;
    auto index_keys = at::randn({lines, pool_page_size, index_dim}, opts);
    auto page_tables = int32_cuda({4, 1, 5, 2, not_a_line, not_a_line}).view({2, max_pages});
    auto page_positions = int32_cuda({0, 16, 32, 0, 0, 0}).view({2, max_pages});
    const int64_t complete_blocks = 3;
    auto key_counts = int32_cuda({max_pages * pool_page_size, complete_blocks * ratio + 2});
    auto norm_weight = (1.0 + 0.1 * at::randn({index_dim})).to(at::kCUDA);

    at::Tensor blocks = pulsarlm::index_blocks_cuda(
        index_keys,
        {.page_tables = page_tables, .page_positions = page_positions, .key_counts = key_counts},
        norm_weight,
        inv_freq(),
        ratio,
        eps
    );
    c10::cuda::device_synchronize();

    CHECK(blocks[1].narrow(0, 0, complete_blocks).abs().sum().item<float>() > 0.0f);
}

TEST_CASE("index_scores_cuda sums the relu of each head's score over a row's complete blocks", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    // fp32 runs the scalar kernel, bf16 the tensor-core one, whose CTAs hold four rows here, of both sequences.
    const auto dtype = GENERATE(at::kFloat, at::kBFloat16);
    // Two sequences of 2 and 3 rows, or one row each.
    const bool ragged = GENERATE(true, false);
    at::manual_seed(5);
    const int64_t heads = 4, sequences = 2, max_blocks = 300, dim = 128;
    const std::vector<int32_t> key_counts = ragged ? std::vector<int32_t>{30, 1201} : std::vector<int32_t>{3, 517};
    const std::vector<int32_t> cu_seqlens{0, 2, 5};
    const int64_t rows = ragged ? 5 : 2;
    auto [row_sequence, row_blocks, view_index] = row_layout(key_counts, ragged ? cu_seqlens : std::vector<int32_t>{});
    auto queries = at::randn({rows, heads, dim}, at::device(at::kCUDA).dtype(dtype));
    auto blocks = at::randn({sequences, max_blocks, dim}, at::device(at::kCUDA).dtype(dtype));
    auto longs = at::device(at::kCUDA).dtype(at::kLong);
    const double scale = 0.25;

    at::Tensor scores = pulsarlm::index_scores_cuda(
        queries,
        blocks,
        int32_cuda(key_counts),
        ragged ? std::optional<at::Tensor>(int32_cuda(cu_seqlens)) : std::nullopt,
        0,
        ratio,
        scale
    );

    auto dots = at::matmul(
        queries.to(at::kFloat).unsqueeze(1),
        blocks.to(at::kFloat).index_select(0, row_sequence).transpose(1, 2).unsqueeze(1)
    );
    auto expected = at::relu(dots.squeeze(1)).sum(1) * scale;
    auto visible = at::arange(max_blocks, longs).unsqueeze(0) < row_blocks.unsqueeze(1);
    CHECK(at::allclose(at::where(visible, scores, 0.0), at::where(visible, expected, 0.0), 1e-4, 1e-4));
}

TEST_CASE("select_blocks_cuda takes a stable descending sort's top blocks and the tail block", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    at::manual_seed(9);
    const int64_t rows = 6, max_blocks = 700, top = 64;
    const int64_t words = max_blocks / 32 + 1;
    auto longs = at::device(at::kCUDA).dtype(at::kLong);
    // One query per sequence, at view indices 2, 39, 256, 261, 1603 and 2803: 0, 10, 64, 65, 401 and 701 complete
    // blocks, the last past the scores.
    const std::vector<int32_t> key_counts{3, 40, 257, 262, 1604, 2804};
    auto [row_sequence, row_blocks, view_index] = row_layout(key_counts, {});
    // Scores of few distinct values, so that a row's 64th block ties with many others.
    auto scores = at::floor(at::rand({rows, max_blocks}, at::device(at::kCUDA)) * 8.0) / 4.0;
    auto visible = at::arange(max_blocks, longs).unsqueeze(0) < row_blocks.unsqueeze(1);
    scores = at::where(visible, scores, -INFINITY).contiguous();

    at::Tensor
        selected = pulsarlm::select_blocks_cuda(scores, int32_cuda(key_counts), std::nullopt, 0, top, ratio, words);

    auto [sorted, order] = at::sort(scores, /*stable=*/true, /*dim=*/-1, /*descending=*/true);
    auto bits = at::zeros({rows, words * 32}, at::device(at::kCUDA).dtype(at::kBool));
    bits.scatter_(1, order.narrow(1, 0, top), sorted.narrow(1, 0, top) > -INFINITY);
    auto tail = row_blocks.unsqueeze(1);
    bits.scatter_(1, tail, (tail * ratio <= view_index.unsqueeze(1)) | bits.gather(1, tail));
    CHECK(at::equal(selected, packed(bits, words)));
}

TEST_CASE("index_queries_cuda norms and ropes each head of the index projection", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const auto dtype = GENERATE(at::kFloat, at::kBFloat16);
    at::manual_seed(11);
    const int64_t tokens = 5, heads = 3, extra_columns = 24;
    auto index_qk = at::randn({tokens, heads * index_dim + extra_columns}, at::device(at::kCUDA).dtype(dtype));
    auto norm_weight = (1.0 + 0.1 * at::randn({index_dim})).to(at::kCUDA);
    auto positions = at::tensor(std::vector<int64_t>{0, 7, 300, 4095, 131071}).to(at::kCUDA);

    at::Tensor queries = pulsarlm::index_queries_cuda(index_qk, norm_weight, inv_freq(), positions, heads, eps);

    auto normed = pulsarlm::rmsnorm_cuda(
        index_qk.narrow(1, 0, heads * index_dim).view({tokens, heads, index_dim}),
        norm_weight,
        eps
    );
    auto expected = reference_rope(normed, positions.unsqueeze(1));
    const double tolerance = dtype == at::kFloat ? 1e-5 : 1e-2;
    CHECK(at::allclose(queries.to(at::kFloat), expected.to(at::kFloat), tolerance, tolerance));
}
