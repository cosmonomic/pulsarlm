#include "pulsarlm/ops.hpp"

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <math_constants.h>

#include <cstdint>

namespace pulsarlm {
namespace {

constexpr int kBlocksPerCta = 128;

// One CTA per (query row, run of kBlocksPerCta blocks), one thread per block; the row's query heads sit in shared memory.
template <typename scalar_t>
__global__ void index_scores_kernel(
    const scalar_t* __restrict__ queries,
    const scalar_t* __restrict__ blocks,
    const int64_t* __restrict__ row_sequence,
    const int64_t* __restrict__ row_blocks,
    int heads,
    int dim,
    int64_t max_blocks,
    float scale,
    float* __restrict__ scores
) {
    extern __shared__ float query[];
    const int64_t row = blockIdx.x;
    for (int i = threadIdx.x; i < heads * dim; i += blockDim.x) {
        query[i] = static_cast<float>(queries[row * heads * dim + i]);
    }
    __syncthreads();
    const int64_t block = static_cast<int64_t>(blockIdx.y) * kBlocksPerCta + threadIdx.x;
    if (block >= max_blocks) {
        return;
    }
    float score = -CUDART_INF_F;
    if (block < row_blocks[row]) {
        const scalar_t* key = blocks + (row_sequence[row] * max_blocks + block) * dim;
        score = 0.0f;
        for (int h = 0; h < heads; ++h) {
            float dot = 0.0f;
            for (int d = 0; d < dim; ++d) {
                dot += query[h * dim + d] * static_cast<float>(key[d]);
            }
            score += fmaxf(dot, 0.0f);
        }
        score *= scale;
    }
    scores[row * max_blocks + block] = score;
}

}  // namespace

at::Tensor index_scores_cuda(
    const at::Tensor& queries,
    const at::Tensor& blocks,
    const at::Tensor& row_sequence,
    const at::Tensor& row_blocks,
    double scale
) {
    TORCH_CHECK(
        queries.is_cuda() && queries.dim() == 3 && queries.is_contiguous(),
        "index_scores: queries must be contiguous CUDA [rows, heads, dim]"
    );
    TORCH_CHECK(
        blocks.is_cuda() && blocks.dim() == 3 && blocks.is_contiguous() && blocks.size(2) == queries.size(2) &&
            blocks.scalar_type() == queries.scalar_type(),
        "index_scores: blocks must be contiguous CUDA [sequences, max_blocks, dim] in the queries' dtype"
    );
    const int64_t rows = queries.size(0);
    for (const at::Tensor* per_row : {&row_sequence, &row_blocks}) {
        TORCH_CHECK(
            per_row->is_cuda() && per_row->scalar_type() == at::kLong && per_row->dim() == 1 &&
                per_row->size(0) == rows && per_row->is_contiguous(),
            "index_scores: row_sequence and row_blocks must be contiguous int64 CUDA [rows]"
        );
    }
    const int64_t max_blocks = blocks.size(1);
    at::Tensor scores = at::empty({rows, max_blocks}, queries.options().dtype(at::kFloat));
    if (rows == 0 || max_blocks == 0) {
        return scores;
    }
    const auto heads = static_cast<int>(queries.size(1));
    const auto dim = static_cast<int>(queries.size(2));
    const dim3 grid(
        static_cast<unsigned>(rows),
        static_cast<unsigned>((max_blocks + kBlocksPerCta - 1) / kBlocksPerCta)
    );
    const size_t shared = static_cast<size_t>(heads) * dim * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, queries.scalar_type(), "index_scores_cuda", [&] {
        index_scores_kernel<scalar_t><<<grid, kBlocksPerCta, shared, stream>>>(
            queries.const_data_ptr<scalar_t>(),
            blocks.const_data_ptr<scalar_t>(),
            row_sequence.const_data_ptr<int64_t>(),
            row_blocks.const_data_ptr<int64_t>(),
            heads,
            dim,
            max_blocks,
            static_cast<float>(scale),
            scores.mutable_data_ptr<float>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return scores;
}

}  // namespace pulsarlm
