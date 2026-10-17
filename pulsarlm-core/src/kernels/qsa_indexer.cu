#include "attn_common.cuh"

#include "attn_tiles.cuh"
#include "pulsarlm/ops.hpp"

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <math_constants.h>

#include <algorithm>
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

// One CTA per (sequence, block), one thread per index dim: the mean of the block's raw keys in fp32, rounded, then
// rmsnorm_cuda's RMSNorm and index_rope's rotation at the block's first key under the sequence's layout, each product
// and sum rounded as the ATen ops round them.
template <typename scalar_t>
__global__ void index_blocks_kernel(
    const scalar_t* __restrict__ index_keys,
    int64_t line_stride,
    const int32_t* __restrict__ page_tables,
    int max_pages,
    int page_size,
    const int32_t* __restrict__ rope_layout,
    const float* __restrict__ norm_weight,
    const float* __restrict__ inv_freq,
    int rotary_dim,
    int ratio,
    int max_blocks,
    float eps,
    scalar_t* __restrict__ blocks
) {
    extern __shared__ float pooled[];
    __shared__ float total;
    const int s = blockIdx.x / max_blocks;
    const int block = blockIdx.x % max_blocks;
    const int dim = blockDim.x;
    const int d = threadIdx.x;
    const int first_key = block * ratio;
    const int line = page_tables[static_cast<int64_t>(s) * max_pages + first_key / page_size];
    const scalar_t* keys = index_keys + line * line_stride + static_cast<int64_t>(first_key % page_size) * dim;
    float sum = 0.0f;
    for (int k = 0; k < ratio; ++k) {
        sum += static_cast<float>(keys[static_cast<int64_t>(k) * dim + d]);
    }
    const float mean = static_cast<float>(static_cast<scalar_t>(sum / static_cast<float>(ratio)));
    pooled[d] = mean * mean;
    __syncthreads();
    if (d == 0) {
        float squares = 0.0f;
        for (int i = 0; i < dim; ++i) {
            squares += pooled[i];
        }
        total = squares;
    }
    __syncthreads();
    const float inv_rms = rsqrtf(total / static_cast<float>(dim) + eps);
    const float normed = static_cast<float>(static_cast<scalar_t>(mean * inv_rms * norm_weight[d]));
    __syncthreads();
    pooled[d] = normed;
    __syncthreads();

    const int32_t* layout = rope_layout + 3 * static_cast<int64_t>(s);
    const int position = first_key < layout[0] ? first_key
        : first_key < layout[1]                ? layout[2]
                                               : layout[2] + 1 + first_key - layout[1];
    scalar_t out = static_cast<scalar_t>(normed);
    if (d < rotary_dim) {
        const int half = rotary_dim / 2;
        const float angle = static_cast<float>(position) * inv_freq[d % half];
        const float cos_a = static_cast<float>(static_cast<scalar_t>(cosf(angle)));
        const float sin_a = static_cast<float>(static_cast<scalar_t>(sinf(angle)));
        const float pair = d < half ? -pooled[d + half] : pooled[d - half];
        const float cosine_part = static_cast<float>(static_cast<scalar_t>(normed * cos_a));
        const float sine_part = static_cast<float>(static_cast<scalar_t>(pair * sin_a));
        out = static_cast<scalar_t>(cosine_part + sine_part);
    }
    blocks[static_cast<int64_t>(blockIdx.x) * dim + d] = out;
}

constexpr int kSelectThreads = 256;

// A score's place in descending order as an unsigned key: scores are -inf or non-negative, and the bits of a
// non-negative float order as its value; -inf takes 0, below every block that can be chosen.
__device__ __forceinline__ uint32_t score_key(float score) {
    return score == -CUDART_INF_F ? 0u : __float_as_uint(score) + 1u;
}

// One CTA per row. The k-th largest key comes from four radix passes of 8 bits over the row; the blocks above it are
// chosen, then the lowest-indexed blocks equal to it, which is a stable descending sort's first k.
__global__ void select_blocks_kernel(
    const float* __restrict__ scores,
    int64_t max_blocks,
    const int64_t* __restrict__ row_blocks,
    const int64_t* __restrict__ view_index,
    int top,
    int ratio,
    int words,
    int32_t* __restrict__ selected
) {
    __shared__ uint32_t histogram[256];
    __shared__ uint32_t prefix;
    __shared__ int remaining;
    __shared__ int tie_counts[kSelectThreads];
    const int64_t row = blockIdx.x;
    const float* row_scores = scores + row * max_blocks;
    uint32_t* row_bits = reinterpret_cast<uint32_t*>(selected) + row * words;
    const int tid = threadIdx.x;
    for (int w = tid; w < words; w += kSelectThreads) {
        row_bits[w] = 0;
    }
    if (tid == 0) {
        prefix = 0;
        remaining = top;
    }
    __syncthreads();
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = tid; i < 256; i += kSelectThreads) {
            histogram[i] = 0;
        }
        __syncthreads();
        const uint32_t high_mask = shift == 24 ? 0u : ~0u << (shift + 8);
        for (int64_t b = tid; b < max_blocks; b += kSelectThreads) {
            const uint32_t key = score_key(row_scores[b]);
            if ((key & high_mask) == prefix) {
                atomicAdd(&histogram[(key >> shift) & 255u], 1u);
            }
        }
        __syncthreads();
        if (tid == 0) {
            int above = 0;
            int digit = 255;
            for (; digit > 0 && above + static_cast<int>(histogram[digit]) < remaining; --digit) {
                above += static_cast<int>(histogram[digit]);
            }
            remaining -= above;
            prefix |= static_cast<uint32_t>(digit) << shift;
        }
        __syncthreads();
    }
    // prefix is the k-th largest key; remaining, the blocks equal to it among the first k.
    const uint32_t threshold = prefix;
    const int64_t chunk = (max_blocks + kSelectThreads - 1) / kSelectThreads;
    const int64_t first = tid * chunk;
    const int64_t end = min(first + chunk, max_blocks);
    int ties = 0;
    for (int64_t b = first; b < end; ++b) {
        ties += score_key(row_scores[b]) == threshold;
    }
    tie_counts[tid] = ties;
    __syncthreads();
    int tie_rank = 0;
    for (int t = 0; t < tid; ++t) {
        tie_rank += tie_counts[t];
    }
    for (int64_t b = first; b < end; ++b) {
        const uint32_t key = score_key(row_scores[b]);
        bool chosen = key > threshold;
        if (key == threshold) {
            chosen = tie_rank < remaining;
            ++tie_rank;
        }
        if (chosen && key > 0) {
            atomicOr(&row_bits[b / 32], 1u << (b % 32));
        }
    }
    __syncthreads();
    if (tid == 0) {
        const int64_t tail = row_blocks[row];
        if (tail * ratio <= view_index[row]) {
            atomicOr(&row_bits[tail / 32], 1u << (tail % 32));
        }
    }
}

constexpr int kScoreTileBlocks = 64;
constexpr int kScoreChunkTiles = 4;
constexpr int kScoreWarps = 4;

// One CTA per (16 / heads consecutive rows, run of kScoreChunkTiles * kScoreTileBlocks blocks). The MMA rows are the
// rows' (row, head) pairs, row-major; rows of different sequences take one pass each with the others' rows zeroed.
// The dots are attn::tile_qkt's, fp32 accumulated, and a row's score sums its heads' relu in head order.
template <typename scalar_t, bool BF16, int DIM>
__global__ void __launch_bounds__(kScoreWarps * 32) index_scores_tc_kernel(
    const scalar_t* __restrict__ queries,
    const scalar_t* __restrict__ blocks,
    const int64_t* __restrict__ row_sequence,
    const int64_t* __restrict__ row_blocks,
    int64_t rows,
    int heads,
    int64_t max_blocks,
    float scale,
    float* __restrict__ scores
) {
    constexpr int SCORE_STRIDE = kScoreTileBlocks + 1;
    constexpr int kChunkElems = 16 / sizeof(scalar_t);
    constexpr int kRowChunks = DIM / kChunkElems;
    __shared__ __align__(16) scalar_t sQ[16 * DIM];
    __shared__ __align__(16) scalar_t sK[kScoreTileBlocks * DIM];
    __shared__ float sS[16 * SCORE_STRIDE];
    __shared__ int64_t sequences[16];
    __shared__ int sequence_count;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int tile_rows = 16 / heads;
    const int64_t first_row = static_cast<int64_t>(blockIdx.x) * tile_rows;
    const int rows_here = rows - first_row < tile_rows ? static_cast<int>(rows - first_row) : tile_rows;
    const int64_t first_block = static_cast<int64_t>(blockIdx.y) * kScoreChunkTiles * kScoreTileBlocks;
    if (tid == 0) {
        int count = 0;
        for (int r = 0; r < rows_here; ++r) {
            const int64_t sequence = row_sequence[first_row + r];
            bool seen = false;
            for (int i = 0; i < count; ++i) {
                seen |= sequences[i] == sequence;
            }
            if (!seen) {
                sequences[count++] = sequence;
            }
        }
        sequence_count = count;
    }
    __syncthreads();

    for (int pass = 0; pass < sequence_count; ++pass) {
        const int64_t sequence = sequences[pass];
        for (int i = tid; i < 16 * DIM; i += kScoreWarps * 32) {
            const int m = i / DIM, d = i % DIM;
            const int r = m / heads;
            scalar_t value = static_cast<scalar_t>(0);
            if (r < rows_here && row_sequence[first_row + r] == sequence) {
                value = queries[((first_row + r) * heads + m % heads) * DIM + d];
            }
            sQ[attn::swizzled_offset<DIM>(m, d)] = value;
        }
        for (int t = 0; t < kScoreChunkTiles; ++t) {
            const int64_t tile_first = first_block + static_cast<int64_t>(t) * kScoreTileBlocks;
            if (tile_first >= max_blocks) {
                break;
            }
            for (int c = tid; c < kScoreTileBlocks * kRowChunks; c += kScoreWarps * 32) {
                const int n = c / kRowChunks;
                const int d = (c % kRowChunks) * kChunkElems;
                const int64_t block = min(tile_first + n, max_blocks - 1);
                attn::cp_async_16(
                    sK + attn::swizzled_offset<DIM>(n, d),
                    blocks + (sequence * max_blocks + block) * DIM + d,
                    tile_first + n < max_blocks ? 16 : 0
                );
            }
            attn::cp_async_commit();
            attn::cp_async_wait<0>();
            __syncthreads();
            attn::tile_qkt<scalar_t, kScoreTileBlocks, DIM, BF16, SCORE_STRIDE, true>(sQ, sK, sS, warp, kScoreWarps);
            __syncthreads();
            for (int i = tid; i < rows_here * kScoreTileBlocks; i += kScoreWarps * 32) {
                const int r = i / kScoreTileBlocks, n = i % kScoreTileBlocks;
                const int64_t row = first_row + r;
                const int64_t block = tile_first + n;
                if (row_sequence[row] != sequence || block >= max_blocks) {
                    continue;
                }
                float score = -CUDART_INF_F;
                if (block < row_blocks[row]) {
                    score = 0.0f;
                    for (int h = 0; h < heads; ++h) {
                        score += fmaxf(sS[(r * heads + h) * SCORE_STRIDE + n], 0.0f);
                    }
                    score *= scale;
                }
                scores[row * max_blocks + block] = score;
            }
            __syncthreads();
        }
    }
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
    const bool tensor_cores = queries.scalar_type() != at::kFloat && heads <= 16 && 16 % heads == 0 &&
        (dim == 64 || dim == 128 || dim == 256);
    if (tensor_cores) {
        const int tile_rows = 16 / heads;
        const dim3 tc_grid(
            static_cast<unsigned>((rows + tile_rows - 1) / tile_rows),
            static_cast<unsigned>(
                (max_blocks + kScoreChunkTiles * kScoreTileBlocks - 1) / (kScoreChunkTiles * kScoreTileBlocks)
            )
        );
        auto tc_stream = at::cuda::getCurrentCUDAStream();
        attn::dispatch_tensor_core_dtype(queries.scalar_type(), [&](auto type_tag, auto bf16_tag) {
            using scalar_t = typename decltype(type_tag)::type;
            auto launch = [&](auto dim_tag) {
                index_scores_tc_kernel<scalar_t, decltype(bf16_tag)::value, decltype(dim_tag)::value>
                    <<<tc_grid, kScoreWarps * 32, 0, tc_stream>>>(
                        queries.const_data_ptr<scalar_t>(),
                        blocks.const_data_ptr<scalar_t>(),
                        row_sequence.const_data_ptr<int64_t>(),
                        row_blocks.const_data_ptr<int64_t>(),
                        rows,
                        heads,
                        max_blocks,
                        static_cast<float>(scale),
                        scores.mutable_data_ptr<float>()
                    );
            };
            if (dim == 64) {
                launch(std::integral_constant<int, 64>{});
            } else if (dim == 128) {
                launch(std::integral_constant<int, 128>{});
            } else {
                launch(std::integral_constant<int, 256>{});
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
        return scores;
    }
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

at::Tensor index_blocks_cuda(
    const at::Tensor& index_keys,
    const at::Tensor& page_tables,
    const at::Tensor& rope_layout,
    const at::Tensor& norm_weight,
    const at::Tensor& inv_freq,
    int64_t ratio,
    double eps
) {
    TORCH_CHECK(
        index_keys.is_cuda() && index_keys.dim() == 3 && index_keys.stride(2) == 1 &&
            index_keys.stride(1) == index_keys.size(2),
        "index_blocks: index_keys must be CUDA [lines, page_size, dim], each line packed"
    );
    const int64_t page_size = index_keys.size(1);
    const int64_t dim = index_keys.size(2);
    TORCH_CHECK(ratio >= 1 && page_size % ratio == 0, "index_blocks: ratio must divide the page size");
    TORCH_CHECK(dim <= 1024, "index_blocks: the index dim must be at most 1024");
    TORCH_CHECK(
        page_tables.is_cuda() && page_tables.scalar_type() == at::kInt && page_tables.dim() == 2 &&
            page_tables.is_contiguous(),
        "index_blocks: page_tables must be contiguous int32 CUDA [sequences, max_pages]"
    );
    TORCH_CHECK(
        rope_layout.is_cuda() && rope_layout.scalar_type() == at::kInt && rope_layout.is_contiguous() &&
            rope_layout.dim() == 2 && rope_layout.size(0) == page_tables.size(0) && rope_layout.size(1) == 3,
        "index_blocks: rope_layout must be contiguous int32 CUDA [sequences, 3]"
    );
    TORCH_CHECK(
        norm_weight.is_cuda() && norm_weight.scalar_type() == at::kFloat && norm_weight.numel() == dim &&
            norm_weight.is_contiguous(),
        "index_blocks: norm_weight must be contiguous fp32 CUDA [dim]"
    );
    TORCH_CHECK(
        inv_freq.is_cuda() && inv_freq.scalar_type() == at::kFloat && inv_freq.dim() == 1 &&
            2 * inv_freq.size(0) <= dim && inv_freq.is_contiguous(),
        "index_blocks: inv_freq must be contiguous fp32 CUDA [rotary_dim / 2], rotary_dim at most the index dim"
    );
    const int64_t sequences = page_tables.size(0);
    const int64_t max_blocks = page_tables.size(1) * page_size / ratio;
    at::Tensor blocks = at::empty({sequences, max_blocks, dim}, index_keys.options());
    if (blocks.numel() == 0) {
        return blocks;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, index_keys.scalar_type(), "index_blocks_cuda", [&] {
        index_blocks_kernel<scalar_t>
            <<<static_cast<int>(sequences * max_blocks), static_cast<int>(dim), dim * sizeof(float), stream>>>(
                index_keys.const_data_ptr<scalar_t>(),
                index_keys.stride(0),
                page_tables.const_data_ptr<int32_t>(),
                static_cast<int>(page_tables.size(1)),
                static_cast<int>(page_size),
                rope_layout.const_data_ptr<int32_t>(),
                norm_weight.const_data_ptr<float>(),
                inv_freq.const_data_ptr<float>(),
                static_cast<int>(2 * inv_freq.size(0)),
                static_cast<int>(ratio),
                static_cast<int>(max_blocks),
                static_cast<float>(eps),
                blocks.mutable_data_ptr<scalar_t>()
            );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return blocks;
}

at::Tensor select_blocks_cuda(
    const at::Tensor& scores,
    const at::Tensor& row_blocks,
    const at::Tensor& view_index,
    int64_t top,
    int64_t ratio,
    int64_t words
) {
    TORCH_CHECK(
        scores.is_cuda() && scores.scalar_type() == at::kFloat && scores.dim() == 2 && scores.is_contiguous(),
        "select_blocks: scores must be contiguous fp32 CUDA [rows, max_blocks]"
    );
    const int64_t rows = scores.size(0);
    for (const at::Tensor* per_row : {&row_blocks, &view_index}) {
        TORCH_CHECK(
            per_row->is_cuda() && per_row->scalar_type() == at::kLong && per_row->dim() == 1 &&
                per_row->size(0) == rows && per_row->is_contiguous(),
            "select_blocks: row_blocks and view_index must be contiguous int64 CUDA [rows]"
        );
    }
    TORCH_CHECK(
        top >= 1 && words * 32 > scores.size(1),
        "select_blocks: top must be positive and words cover a block past the scores"
    );
    at::Tensor selected = at::empty({rows, words}, scores.options().dtype(at::kInt));
    if (rows == 0) {
        return selected;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    select_blocks_kernel<<<static_cast<int>(rows), kSelectThreads, 0, stream>>>(
        scores.const_data_ptr<float>(),
        scores.size(1),
        row_blocks.const_data_ptr<int64_t>(),
        view_index.const_data_ptr<int64_t>(),
        static_cast<int>(std::min(top, scores.size(1))),
        static_cast<int>(ratio),
        static_cast<int>(words),
        selected.mutable_data_ptr<int32_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return selected;
}

}  // namespace pulsarlm
