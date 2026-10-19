#include "pulsarlm/ops.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAFunctions.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

// QSA's indexer ops against ATen references: index_blocks_cuda against the pooled, normed and roped view blocks,
// index_scores_cuda against the relu sums of each row's heads, and select_blocks_cuda against a stable descending sort.

namespace {

constexpr int64_t kPageSize = 16;
constexpr int64_t kRatio = 4;
constexpr int64_t kDim = 32;
constexpr int64_t kRotaryDim = 16;
constexpr double kTheta = 1e6;
constexpr double kEps = 1e-6;

at::Tensor int32_cuda(const std::vector<int32_t>& v) {
    return at::tensor(v, at::dtype(at::kInt)).to(at::kCUDA);
}

at::Tensor inv_freq() {
    return (1.0 / at::pow(kTheta, at::arange(0, kRotaryDim, 2, at::kLong).to(at::kFloat) / kRotaryDim)).to(at::kCUDA);
}

at::Tensor reference_rope(const at::Tensor& x, const at::Tensor& positions) {
    auto freqs = positions.to(at::kFloat).unsqueeze(-1) * inv_freq();
    auto angles = at::cat({freqs, freqs}, -1);
    auto cos = angles.cos().to(x.scalar_type());
    auto sin = angles.sin().to(x.scalar_type());
    auto rotated = x.narrow(-1, 0, kRotaryDim);
    const int64_t half = kRotaryDim / 2;
    auto rotated_half = at::cat({-rotated.narrow(-1, half, half), rotated.narrow(-1, 0, half)}, -1);
    return at::cat({rotated * cos + rotated_half * sin, x.narrow(-1, kRotaryDim, kDim - kRotaryDim)}, -1);
}

// Packs bool [rows, words * 32] into selection words.
at::Tensor packed(const at::Tensor& bits, int64_t words) {
    auto weights = at::pow(2, at::arange(32, at::device(at::kCUDA).dtype(at::kLong)));
    auto sums = (bits.view({bits.size(0), words, 32}).to(at::kLong) * weights).sum(-1);
    return at::where(sums >= (int64_t{1} << 31), sums - (int64_t{1} << 32), sums).to(at::kInt);
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
    auto index_keys = at::randn({lines, kPageSize, kDim}, opts);
    auto page_tables = int32_cuda({4, 1, 5, 2, 0, 3}).view({2, max_pages});
    // The second view's pages sit at positions with gaps between them.
    auto page_positions = int32_cuda({0, 16, 32, 0, 48, 112}).view({2, max_pages});
    auto norm_weight = (1.0 + 0.1 * at::randn({kDim})).to(at::kCUDA);

    at::Tensor blocks = pulsarlm::index_blocks_cuda(
        index_keys,
        page_tables,
        page_positions,
        norm_weight,
        inv_freq(),
        kRatio,
        kEps
    );

    const int64_t max_blocks = max_pages * kPageSize / kRatio;
    auto keys = index_keys.index_select(0, page_tables.flatten().to(at::kLong)).view({2, max_blocks, kRatio, kDim});
    auto pooled = keys.to(at::kFloat).mean(2).to(dtype).to(at::kFloat);
    auto normed = (pooled * at::rsqrt(pooled.pow(2).mean(-1, true) + kEps) * norm_weight).to(dtype);
    auto first_keys = at::arange(max_blocks, at::device(at::kCUDA).dtype(at::kLong)) * kRatio;
    auto positions = page_positions.to(at::kLong).index_select(1, at::floor_divide(first_keys, kPageSize)) +
        at::remainder(first_keys, kPageSize);
    auto expected = reference_rope(normed, positions);
    const double tolerance = dtype == at::kFloat ? 1e-5 : 1e-2;
    CHECK(at::allclose(blocks.to(at::kFloat), expected.to(at::kFloat), tolerance, tolerance));
}

TEST_CASE("index_scores_cuda sums the relu of each head's score over a row's complete blocks", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    // fp32 runs the scalar kernel, bf16 the tensor-core one, whose CTAs hold four rows here, of both sequences.
    const auto dtype = GENERATE(at::kFloat, at::kBFloat16);
    at::manual_seed(5);
    const int64_t rows = 5, heads = 4, sequences = 2, max_blocks = 300, dim = 128;
    auto queries = at::randn({rows, heads, dim}, at::device(at::kCUDA).dtype(dtype));
    auto blocks = at::randn({sequences, max_blocks, dim}, at::device(at::kCUDA).dtype(dtype));
    auto longs = at::device(at::kCUDA).dtype(at::kLong);
    auto row_sequence = at::tensor(std::vector<int64_t>{0, 0, 1, 1, 1}).to(at::kCUDA);
    auto row_blocks = at::tensor(std::vector<int64_t>{0, 7, 300, 129, 1}).to(at::kCUDA);
    const double scale = 0.25;

    at::Tensor scores = pulsarlm::index_scores_cuda(queries, blocks, row_sequence, row_blocks, scale);

    auto dots = at::matmul(
        queries.to(at::kFloat).unsqueeze(1),
        blocks.to(at::kFloat).index_select(0, row_sequence).transpose(1, 2).unsqueeze(1)
    );
    auto expected = at::relu(dots.squeeze(1)).sum(1) * scale;
    auto visible = at::arange(max_blocks, longs).unsqueeze(0) < row_blocks.unsqueeze(1);
    expected = at::where(visible, expected, -INFINITY);
    CHECK(at::allclose(scores, expected, 1e-4, 1e-4));
}

TEST_CASE("select_blocks_cuda takes a stable descending sort's top blocks and the tail block", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    at::manual_seed(9);
    const int64_t rows = 6, max_blocks = 700, top = 64;
    const int64_t words = max_blocks / 32 + 1;
    auto longs = at::device(at::kCUDA).dtype(at::kLong);
    // Scores of few distinct values, so that a row's 64th block ties with many others.
    auto scores = at::floor(at::rand({rows, max_blocks}, at::device(at::kCUDA)) * 8.0) / 4.0;
    auto row_blocks = at::tensor(std::vector<int64_t>{0, 10, 64, 65, 400, 700}).to(at::kCUDA);
    auto view_index = row_blocks * kRatio + at::tensor(std::vector<int64_t>{2, 3, 0, 1, 3, 3}).to(at::kCUDA);
    view_index[1] = 10 * kRatio - 1;
    auto visible = at::arange(max_blocks, longs).unsqueeze(0) < row_blocks.unsqueeze(1);
    scores = at::where(visible, scores, -INFINITY).contiguous();

    at::Tensor selected = pulsarlm::select_blocks_cuda(scores, row_blocks, view_index, top, kRatio, words);

    auto [sorted, order] = at::sort(scores, /*stable=*/true, /*dim=*/-1, /*descending=*/true);
    auto bits = at::zeros({rows, words * 32}, at::device(at::kCUDA).dtype(at::kBool));
    bits.scatter_(1, order.narrow(1, 0, top), sorted.narrow(1, 0, top) > -INFINITY);
    auto tail = row_blocks.unsqueeze(1);
    bits.scatter_(1, tail, (tail * kRatio <= view_index.unsqueeze(1)) | bits.gather(1, tail));
    CHECK(at::equal(selected, packed(bits, words)));
}
