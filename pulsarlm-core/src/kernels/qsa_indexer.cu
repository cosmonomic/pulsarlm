#include "pulsarlm/kernels/dependent_launch.cuh"
#include "pulsarlm/kernels/attn_common.cuh"

#include "pulsarlm/kernels/attn_tiles.cuh"
#include "pulsarlm/kernels/device.hpp"
#include "pulsarlm/kernels/indexed_attention.hpp"

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <math_constants.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>

namespace pulsarlm {
namespace {

constexpr int blocks_per_cta = 128;

// Where query row r, counted from first_row, of a ragged batch lies: sequence s holds rows [cu_seqlens_q[s],
// cu_seqlens_q[s + 1]), or row s alone without cu_seqlens_q, and its rows are the last of its view of key_counts[s]
// keys.
struct query_rows {
    const int32_t* key_counts;
    const int32_t* cu_seqlens_q;
    int num_seqs;
    int64_t first_row;
    int ratio;

    __device__ int sequence(int64_t r) const {
        const int64_t row = first_row + r;
        if (cu_seqlens_q == nullptr) {
            return static_cast<int>(row);
        }
        int lo = 0, hi = num_seqs;
        while (hi - lo > 1) {
            const int mid = (lo + hi) / 2;
            if (cu_seqlens_q[mid] <= row) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    __device__ int64_t view_index(int64_t r) const {
        const int s = sequence(r);
        const int64_t first = cu_seqlens_q == nullptr ? s : cu_seqlens_q[s];
        const int64_t count = cu_seqlens_q == nullptr ? 1 : cu_seqlens_q[s + 1] - first;
        return key_counts[s] - count + (first_row + r - first);
    }

    // The blocks of the view that end at or before the row's key.
    __device__ int64_t complete_blocks(int64_t r) const {
        return (view_index(r) + 1) / ratio;
    }
};

// One CTA per (query row, run of blocks_per_cta blocks), one thread per block; the row's query heads sit in shared memory.
template <typename scalar_t>
__global__ void index_scores_kernel(
    const scalar_t* __restrict__ queries,
    const scalar_t* __restrict__ blocks,
    const query_rows layout,
    int heads,
    int dim,
    int64_t max_blocks,
    float scale,
    float* __restrict__ scores
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    extern __shared__ float query[];
    const int64_t row = blockIdx.x;
    for (int i = threadIdx.x; i < heads * dim; i += blockDim.x) {
        query[i] = static_cast<float>(queries[row * heads * dim + i]);
    }
    __syncthreads();
    const int64_t block = static_cast<int64_t>(blockIdx.y) * blocks_per_cta + threadIdx.x;
    if (block >= max_blocks) {
        return;
    }
    if (block >= layout.complete_blocks(row)) {
        return;
    }
    float score = 0.0f;
    {
        const scalar_t* key = blocks + (layout.sequence(row) * max_blocks + block) * dim;
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

// One CTA per (sequence, run of blocks striding by gridDim.y), one thread per index dim: the mean of the block's raw keys in fp32, rounded, then
// rmsnorm_cuda's RMSNorm and index_rope's rotation at the block's first key under the sequence's layout, each product
// and sum rounded as the ATen ops round them.
template <typename scalar_t>
__global__ void index_blocks_kernel(
    const scalar_t* __restrict__ index_keys,
    int64_t line_stride,
    const int32_t* __restrict__ page_tables,
    int max_pages,
    int page_size,
    const int32_t* __restrict__ page_positions,
    const int32_t* __restrict__ key_counts,
    const float* __restrict__ norm_weight,
    const float* __restrict__ inv_freq,
    int rotary_dim,
    int ratio,
    int max_blocks,
    float eps,
    scalar_t* __restrict__ blocks
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    extern __shared__ float pooled[];
    __shared__ float total;
    const int s = blockIdx.x;
    const int dim = blockDim.x;
    const int d = threadIdx.x;
    const int complete_blocks = key_counts[s] / ratio;
    for (int block = blockIdx.y; block < complete_blocks; block += gridDim.y) {
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

        const int position = page_positions[static_cast<int64_t>(s) * max_pages + first_key / page_size] +
            first_key % page_size;
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
        blocks[(static_cast<int64_t>(s) * max_blocks + block) * dim + d] = out;
        __syncthreads();
    }
}

// One CTA per (token, head), one thread per dim, a power of two: rmsnorm_cuda's RMSNorm of the head's columns of
// index_qk, then the rotation index_blocks_kernel gives a block's first key, at the token's position.
template <typename scalar_t>
__global__ void index_queries_kernel(
    const scalar_t* __restrict__ index_qk,
    int64_t row_stride,
    const float* __restrict__ norm_weight,
    const float* __restrict__ inv_freq,
    int rotary_dim,
    const int64_t* __restrict__ positions,
    int heads,
    float eps,
    scalar_t* __restrict__ queries
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    extern __shared__ float pooled[];
    const int token = blockIdx.x / heads;
    const int head = blockIdx.x % heads;
    const int dim = blockDim.x;
    const int d = threadIdx.x;
    const float value = static_cast<float>(index_qk[token * row_stride + head * dim + d]);
    pooled[d] = value * value;
    __syncthreads();
    for (int stride = dim / 2; stride > 0; stride >>= 1) {
        if (d < stride) {
            pooled[d] += pooled[d + stride];
        }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(pooled[0] / static_cast<float>(dim) + eps);
    __syncthreads();
    const float normed = static_cast<float>(static_cast<scalar_t>(value * inv_rms * norm_weight[d]));
    pooled[d] = normed;
    __syncthreads();
    scalar_t out = static_cast<scalar_t>(normed);
    if (d < rotary_dim) {
        const int half = rotary_dim / 2;
        const float angle = static_cast<float>(positions[token]) * inv_freq[d % half];
        const float cos_a = static_cast<float>(static_cast<scalar_t>(cosf(angle)));
        const float sin_a = static_cast<float>(static_cast<scalar_t>(sinf(angle)));
        const float pair = d < half ? -pooled[d + half] : pooled[d - half];
        const float cosine_part = static_cast<float>(static_cast<scalar_t>(normed * cos_a));
        const float sine_part = static_cast<float>(static_cast<scalar_t>(pair * sin_a));
        out = static_cast<scalar_t>(cosine_part + sine_part);
    }
    queries[static_cast<int64_t>(blockIdx.x) * dim + d] = out;
}

constexpr int select_threads = 1024;
// The CTAs along a sequence's blocks in index_blocks_kernel.
constexpr int64_t block_runs = 1024;

// A score's place in descending order as an unsigned key: scores are -inf or non-negative, and the bits of a
// non-negative float order as its value; -inf takes 0, below every block that can be chosen.
__device__ __forceinline__ uint32_t score_key(float score) {
    return score == -CUDART_INF_F ? 0u : __float_as_uint(score) + 1u;
}

// The sum of value over threads [0, threadIdx.x) of the block; sums holds select_threads / 32 entries.
__device__ int block_exclusive_sum(int value, int* sums) {
    static_assert(select_threads % 32 == 0 && select_threads / 32 <= 32, "one warp scans the warp sums");
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    int inclusive = value;
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
        if (lane >= offset) {
            inclusive += other;
        }
    }
    if (lane == 31) {
        sums[warp] = inclusive;
    }
    __syncthreads();
    if (warp == 0) {
        int warp_sum = lane < select_threads / 32 ? sums[lane] : 0;
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int other = __shfl_up_sync(0xffffffffu, warp_sum, offset);
            if (lane >= offset) {
                warp_sum += other;
            }
        }
        if (lane < select_threads / 32) {
            sums[lane] = warp_sum;
        }
    }
    __syncthreads();
    const int result = inclusive - value + (warp > 0 ? sums[warp - 1] : 0);
    __syncthreads();
    return result;
}

// One CTA per row. The k-th largest key comes from four radix passes of 8 bits over the row; the blocks above it are
// chosen, then the lowest-indexed blocks equal to it, which is a stable descending sort's first k.
__global__ void select_blocks_kernel(
    const float* __restrict__ scores,
    int64_t max_blocks,
    const query_rows layout,
    int top,
    int ratio,
    int words,
    int32_t* __restrict__ selected
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    __shared__ uint32_t histogram[256];
    __shared__ uint32_t prefix;
    __shared__ int remaining;
    __shared__ int warp_sums[select_threads / 32];
    static_assert(select_threads >= 256, "a thread per radix digit");
    const int64_t row = blockIdx.x;
    const float* row_scores = scores + row * max_blocks;
    const int64_t scored_blocks = min(layout.complete_blocks(row), max_blocks);
    uint32_t* row_bits = reinterpret_cast<uint32_t*>(selected) + row * words;
    const int tid = threadIdx.x;
    for (int w = tid; w < words; w += select_threads) {
        row_bits[w] = 0;
    }
    if (tid == 0) {
        prefix = 0;
        remaining = top;
    }
    __syncthreads();
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = tid; i < 256; i += select_threads) {
            histogram[i] = 0;
        }
        __syncthreads();
        const uint32_t high_mask = shift == 24 ? 0u : ~0u << (shift + 8);
        for (int64_t b = tid; b < scored_blocks; b += select_threads) {
            const uint32_t key = score_key(row_scores[b]);
            if ((key & high_mask) == prefix) {
                atomicAdd(&histogram[(key >> shift) & 255u], 1u);
            }
        }
        __syncthreads();
        // Thread t below 256 holds digit 255 - t, so the digits run from the highest.
        const int digit = 255 - tid;
        const int count = tid < 256 ? static_cast<int>(histogram[digit]) : 0;
        const int wanted = remaining;
        const int above = block_exclusive_sum(count, warp_sums);
        if (tid < 256 && (above + count >= wanted || digit == 0) && above < wanted) {
            remaining = wanted - above;
            prefix |= static_cast<uint32_t>(digit) << shift;
        }
        __syncthreads();
    }
    // prefix is the k-th largest key; remaining, the blocks equal to it among the first k.
    const uint32_t threshold = prefix;
    const int64_t chunk = (scored_blocks + select_threads - 1) / select_threads;
    const int64_t first = tid * chunk;
    const int64_t end = min(first + chunk, scored_blocks);
    int ties = 0;
    for (int64_t b = first; b < end; ++b) {
        ties += score_key(row_scores[b]) == threshold;
    }
    int tie_rank = block_exclusive_sum(ties, warp_sums);
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
        const int64_t tail = layout.complete_blocks(row);
        if (tail * ratio <= layout.view_index(row)) {
            atomicOr(&row_bits[tail / 32], 1u << (tail % 32));
        }
    }
}

constexpr int score_tile_blocks = 64;
// The CTAs along a row's blocks; each strides over the runs of score_chunk_tiles tiles.
constexpr int64_t score_chunk_runs = 64;
constexpr int score_chunk_tiles = 4;
constexpr int score_warps = 4;

// One CTA per (16 / heads consecutive rows, run of score_chunk_tiles * score_tile_blocks blocks). The MMA rows are the
// rows' (row, head) pairs, row-major; rows of different sequences take one pass each with the others' rows zeroed.
// The dots are attn::tile_qkt's, fp32 accumulated, and a row's score sums its heads' relu in head order.
template <typename scalar_t, bool bf16, int dim>
__global__ void __launch_bounds__(score_warps * 32) index_scores_tc_kernel(
    const scalar_t* __restrict__ queries,
    const scalar_t* __restrict__ blocks,
    const query_rows layout,
    int64_t rows,
    int heads,
    int64_t max_blocks,
    float scale,
    float* __restrict__ scores
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    constexpr int score_stride = score_tile_blocks + 1;
    constexpr int chunk_elems = 16 / sizeof(scalar_t);
    constexpr int row_chunks = dim / chunk_elems;
    __shared__ __align__(16) scalar_t sQ[16 * dim];
    __shared__ __align__(16) scalar_t sK[score_tile_blocks * dim];
    __shared__ float sS[16 * score_stride];
    __shared__ int64_t sequences[16];
    __shared__ int sequence_count;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int tile_rows = 16 / heads;
    const int64_t first_row = static_cast<int64_t>(blockIdx.x) * tile_rows;
    const int rows_here = rows - first_row < tile_rows ? static_cast<int>(rows - first_row) : tile_rows;
    __shared__ int64_t scored_blocks;
    if (tid == 0) {
        int64_t most = 0;
        for (int r = 0; r < rows_here; ++r) {
            most = max(most, min(layout.complete_blocks(first_row + r), max_blocks));
        }
        scored_blocks = most;
        int count = 0;
        for (int r = 0; r < rows_here; ++r) {
            const int64_t sequence = layout.sequence(first_row + r);
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

    for (int64_t first_block = static_cast<int64_t>(blockIdx.y) * score_chunk_tiles * score_tile_blocks;
         first_block < scored_blocks;
         first_block += static_cast<int64_t>(gridDim.y) * score_chunk_tiles * score_tile_blocks) {
        for (int pass = 0; pass < sequence_count; ++pass) {
            const int64_t sequence = sequences[pass];
            for (int i = tid; i < 16 * dim; i += score_warps * 32) {
                const int m = i / dim, d = i % dim;
                const int r = m / heads;
                scalar_t value = static_cast<scalar_t>(0);
                if (r < rows_here && layout.sequence(first_row + r) == sequence) {
                    value = queries[((first_row + r) * heads + m % heads) * dim + d];
                }
                sQ[attn::swizzled_offset<dim>(m, d)] = value;
            }
            for (int t = 0; t < score_chunk_tiles; ++t) {
                const int64_t tile_first = first_block + static_cast<int64_t>(t) * score_tile_blocks;
                if (tile_first >= scored_blocks) {
                    break;
                }
                for (int c = tid; c < score_tile_blocks * row_chunks; c += score_warps * 32) {
                    const int n = c / row_chunks;
                    const int d = (c % row_chunks) * chunk_elems;
                    const int64_t block = min(tile_first + n, max_blocks - 1);
                    attn::cp_async_16(
                        sK + attn::swizzled_offset<dim>(n, d),
                        blocks + (sequence * max_blocks + block) * dim + d,
                        tile_first + n < max_blocks ? 16 : 0
                    );
                }
                attn::cp_async_commit();
                attn::cp_async_wait<0>();
                __syncthreads();
                attn::tile_qkt<scalar_t, score_tile_blocks, dim, bf16, score_stride, true>(
                    sQ,
                    sK,
                    sS,
                    warp,
                    score_warps
                );
                __syncthreads();
                for (int i = tid; i < rows_here * score_tile_blocks; i += score_warps * 32) {
                    const int r = i / score_tile_blocks, n = i % score_tile_blocks;
                    const int64_t row = first_row + r;
                    const int64_t block = tile_first + n;
                    if (layout.sequence(row) != sequence || block >= min(layout.complete_blocks(row), max_blocks)) {
                        continue;
                    }
                    float score = 0.0f;
                    for (int h = 0; h < heads; ++h) {
                        score += fmaxf(sS[(r * heads + h) * score_stride + n], 0.0f);
                    }
                    scores[row * max_blocks + block] = score * scale;
                }
                __syncthreads();
            }
        }
    }
}
// The layout of rows of a batch whose views hold key_counts keys, checked.
query_rows make_query_rows(
    const char* op,
    const at::Tensor& key_counts,
    const std::optional<at::Tensor>& cu_seqlens_q,
    int64_t first_row,
    int64_t ratio
) {
    TORCH_CHECK(
        key_counts.is_cuda() && key_counts.scalar_type() == at::kInt && key_counts.dim() == 1 &&
            key_counts.is_contiguous(),
        op,
        ": key_counts must be contiguous int32 CUDA [sequences]"
    );
    TORCH_CHECK(
        !cu_seqlens_q ||
            (cu_seqlens_q->is_cuda() && cu_seqlens_q->scalar_type() == at::kInt && cu_seqlens_q->is_contiguous() &&
             cu_seqlens_q->dim() == 1 && cu_seqlens_q->size(0) == key_counts.size(0) + 1),
        op,
        ": cu_seqlens_q must be contiguous int32 CUDA [sequences + 1]"
    );
    TORCH_CHECK(ratio >= 1 && first_row >= 0, op, ": ratio must be positive and first_row not negative");
    check_on_current_device(op, {key_counts, cu_seqlens_q.value_or(at::Tensor())});
    TORCH_CHECK(
        key_counts.size(0) <= std::numeric_limits<int>::max() && ratio <= std::numeric_limits<int>::max(),
        op,
        ": the sequences and ratio must each fit a 32-bit index"
    );
    return query_rows{
        .key_counts = key_counts.const_data_ptr<int32_t>(),
        .cu_seqlens_q = cu_seqlens_q ? cu_seqlens_q->const_data_ptr<int32_t>() : nullptr,
        .num_seqs = static_cast<int>(key_counts.size(0)),
        .first_row = first_row,
        .ratio = static_cast<int>(ratio),
    };
}

}  // namespace

at::Tensor index_scores_cuda(
    const at::Tensor& queries,
    const at::Tensor& blocks,
    const at::Tensor& key_counts,
    const std::optional<at::Tensor>& cu_seqlens_q,
    int64_t first_row,
    int64_t ratio,
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
    const query_rows layout = make_query_rows("index_scores", key_counts, cu_seqlens_q, first_row, ratio);
    check_on_current_device("index_scores", {queries, blocks});
    TORCH_CHECK(
        blocks.size(0) == key_counts.size(0),
        "index_scores: blocks and key_counts must hold the same sequences"
    );
    TORCH_CHECK(
        cu_seqlens_q || first_row + rows <= key_counts.size(0),
        "index_scores: without cu_seqlens_q, first_row + rows must be at most the sequences"
    );
    TORCH_CHECK(
        rows <= std::numeric_limits<int>::max() && queries.size(1) * queries.size(2) <= std::numeric_limits<int>::max(),
        "index_scores: the rows and a row's elements must each fit a 32-bit index"
    );
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
        TORCH_CHECK(
            reinterpret_cast<std::uintptr_t>(blocks.data_ptr()) % 16 == 0,
            "index_scores: blocks must start at a multiple of 16 bytes"
        );
        const int tile_rows = 16 / heads;
        const dim3 tc_grid(
            static_cast<unsigned>((rows + tile_rows - 1) / tile_rows),
            static_cast<unsigned>(std::min<int64_t>(
                (max_blocks + score_chunk_tiles * score_tile_blocks - 1) / (score_chunk_tiles * score_tile_blocks),
                score_chunk_runs
            ))
        );
        auto tc_stream = at::cuda::getCurrentCUDAStream();
        attn::dispatch_tensor_core_dtype(queries.scalar_type(), [&](auto type_tag, auto bf16_tag) {
            using scalar_t = typename decltype(type_tag)::type;
            auto launch = [&](auto dim_tag) {
                launch_dependent(
                    index_scores_tc_kernel<scalar_t, decltype(bf16_tag)::value, decltype(dim_tag)::value>,
                    tc_grid,
                    score_warps * 32,
                    0,
                    tc_stream,
                    queries.const_data_ptr<scalar_t>(),
                    blocks.const_data_ptr<scalar_t>(),
                    layout,
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
        static_cast<unsigned>((max_blocks + blocks_per_cta - 1) / blocks_per_cta)
    );
    const size_t shared = static_cast<size_t>(heads) * dim * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, queries.scalar_type(), "index_scores_cuda", [&] {
        launch_dependent(
            index_scores_kernel<scalar_t>,
            grid,
            blocks_per_cta,
            shared,
            stream,
            queries.const_data_ptr<scalar_t>(),
            blocks.const_data_ptr<scalar_t>(),
            layout,
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

at::Tensor index_queries_cuda(
    const at::Tensor& index_qk,
    const at::Tensor& norm_weight,
    const at::Tensor& inv_freq,
    const at::Tensor& positions,
    int64_t n_heads,
    double eps
) {
    const int64_t dim = norm_weight.numel();
    TORCH_CHECK(
        index_qk.is_cuda() && index_qk.dim() == 2 && index_qk.stride(1) == 1 && index_qk.size(1) >= n_heads * dim,
        "index_queries: index_qk must be CUDA [tokens, >= n_heads * dim], each row contiguous"
    );
    TORCH_CHECK(
        norm_weight.is_cuda() && norm_weight.scalar_type() == at::kFloat && norm_weight.is_contiguous() && dim >= 1 &&
            dim <= 1024 && (dim & (dim - 1)) == 0,
        "index_queries: norm_weight must be contiguous fp32 CUDA [dim], dim a power of two up to 1024"
    );
    TORCH_CHECK(
        inv_freq.is_cuda() && inv_freq.scalar_type() == at::kFloat && inv_freq.dim() == 1 &&
            2 * inv_freq.size(0) <= dim && inv_freq.is_contiguous(),
        "index_queries: inv_freq must be contiguous fp32 CUDA [rotary_dim / 2], rotary_dim at most the dim"
    );
    TORCH_CHECK(
        positions.is_cuda() && positions.scalar_type() == at::kLong && positions.dim() == 1 &&
            positions.size(0) == index_qk.size(0) && positions.is_contiguous(),
        "index_queries: positions must be contiguous int64 CUDA [tokens]"
    );
    check_on_current_device("index_queries", {index_qk, norm_weight, inv_freq, positions});
    TORCH_CHECK(
        n_heads <= std::numeric_limits<int>::max() && index_qk.size(0) * n_heads <= std::numeric_limits<int>::max(),
        "index_queries: the tokens times n_heads must fit a 32-bit index"
    );
    at::Tensor queries = at::empty({index_qk.size(0), n_heads, dim}, index_qk.options());
    if (queries.numel() == 0) {
        return queries;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, index_qk.scalar_type(), "index_queries_cuda", [&] {
        launch_dependent(
            index_queries_kernel<scalar_t>,
            static_cast<int>(index_qk.size(0) * n_heads),
            static_cast<int>(dim),
            dim * sizeof(float),
            stream,
            index_qk.const_data_ptr<scalar_t>(),
            index_qk.stride(0),
            norm_weight.const_data_ptr<float>(),
            inv_freq.const_data_ptr<float>(),
            static_cast<int>(2 * inv_freq.size(0)),
            positions.const_data_ptr<int64_t>(),
            static_cast<int>(n_heads),
            static_cast<float>(eps),
            queries.mutable_data_ptr<scalar_t>()
        );
    });
    return queries;
}

at::Tensor index_blocks_cuda(
    const at::Tensor& index_keys,
    const page_views& views,
    const at::Tensor& norm_weight,
    const at::Tensor& inv_freq,
    int64_t ratio,
    double eps
) {
    const at::Tensor& page_tables = views.page_tables;
    const at::Tensor& page_positions = views.page_positions;
    const at::Tensor& key_counts = views.key_counts;
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
        page_positions.is_cuda() && page_positions.scalar_type() == at::kInt && page_positions.is_contiguous() &&
            page_positions.sizes() == page_tables.sizes(),
        "index_blocks: page_positions must be contiguous int32 CUDA [sequences, max_pages], as page_tables"
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
    TORCH_CHECK(
        key_counts.is_cuda() && key_counts.scalar_type() == at::kInt && key_counts.dim() == 1 &&
            key_counts.size(0) == sequences && key_counts.is_contiguous(),
        "index_blocks: key_counts must be contiguous int32 CUDA [sequences]"
    );
    check_on_current_device(
        "index_blocks",
        {index_keys, page_tables, page_positions, key_counts, norm_weight, inv_freq}
    );
    TORCH_CHECK(
        sequences <= std::numeric_limits<int>::max() &&
            page_tables.size(1) * page_size <= std::numeric_limits<int>::max(),
        "index_blocks: the sequences and the keys of a view must each fit a 32-bit index"
    );
    const int64_t max_blocks = page_tables.size(1) * page_size / ratio;
    at::Tensor blocks = at::empty({sequences, max_blocks, dim}, index_keys.options());
    if (blocks.numel() == 0) {
        return blocks;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, index_keys.scalar_type(), "index_blocks_cuda", [&] {
        launch_dependent(
            index_blocks_kernel<scalar_t>,
            dim3(static_cast<unsigned>(sequences), static_cast<unsigned>(std::min<int64_t>(max_blocks, block_runs))),
            static_cast<int>(dim),
            dim * sizeof(float),
            stream,
            index_keys.const_data_ptr<scalar_t>(),
            index_keys.stride(0),
            page_tables.const_data_ptr<int32_t>(),
            static_cast<int>(page_tables.size(1)),
            static_cast<int>(page_size),
            page_positions.const_data_ptr<int32_t>(),
            key_counts.const_data_ptr<int32_t>(),
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
    const at::Tensor& key_counts,
    const std::optional<at::Tensor>& cu_seqlens_q,
    int64_t first_row,
    int64_t top,
    int64_t ratio,
    int64_t words
) {
    TORCH_CHECK(
        scores.is_cuda() && scores.scalar_type() == at::kFloat && scores.dim() == 2 && scores.is_contiguous(),
        "select_blocks: scores must be contiguous fp32 CUDA [rows, max_blocks]"
    );
    const int64_t rows = scores.size(0);
    const query_rows layout = make_query_rows("select_blocks", key_counts, cu_seqlens_q, first_row, ratio);
    assert(cu_seqlens_q || first_row + rows <= key_counts.size(0));
    TORCH_CHECK(
        top >= 1 && words * 32 > scores.size(1),
        "select_blocks: top must be positive and words cover a block past the scores"
    );
    check_on_current_device("select_blocks", {scores});
    TORCH_CHECK(
        rows <= std::numeric_limits<int>::max() && words <= std::numeric_limits<int>::max(),
        "select_blocks: the rows and words must each fit a 32-bit index"
    );
    at::Tensor selected = at::empty({rows, words}, scores.options().dtype(at::kInt));
    if (rows == 0) {
        return selected;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_dependent(
        select_blocks_kernel,
        static_cast<int>(rows),
        select_threads,
        0,
        stream,
        scores.const_data_ptr<float>(),
        scores.size(1),
        layout,
        static_cast<int>(std::min(top, scores.size(1))),
        static_cast<int>(ratio),
        static_cast<int>(words),
        selected.mutable_data_ptr<int32_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return selected;
}

}  // namespace pulsarlm
