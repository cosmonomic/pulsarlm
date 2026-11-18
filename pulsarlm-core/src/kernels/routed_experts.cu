#include "pulsarlm/kernels/arrival_counters.cuh"
#include "pulsarlm/kernels/dependent_launch.cuh"
#include "pulsarlm/kernels/mma.cuh"
#include "pulsarlm/kernels/ops.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>

namespace pulsarlm {
namespace {

constexpr int sort_threads = 1024;
constexpr int sort_warps = sort_threads / 32;
constexpr int64_t max_experts = 4096;
constexpr int64_t max_top_k = 64;
constexpr int max_tile_rows = 16;
constexpr int tile_weight_rows = 64;
constexpr int chunk_k = 64;
constexpr int chunk_vectors = chunk_k / 8;
constexpr int stages = 3;
constexpr int gemm_threads = 128;
constexpr int element_threads = 256;
// The gated product splits k until its grid reaches this many CTAs per SM.
constexpr int64_t split_target_per_sm = 2;

template <typename scalar_t> __device__ __forceinline__ float rounded(float value) {
    return static_cast<float>(static_cast<scalar_t>(value));
}

// at::silu and at::gelu(x, "tanh") in fp32.
__device__ __forceinline__ float activated(float x, expert_activation activation) {
    if (activation == expert_activation::silu) {
        return x / (1.0f + expf(-x));
    }
    constexpr float beta = M_SQRT2 * M_2_SQRTPI * 0.5f;
    constexpr float kappa = 0.044715f;
    const float cube = x * x * x;
    return 0.5f * x * (1.0f + tanhf(beta * (x + kappa * cube)));
}

__device__ void warp_argmax(float& value, int& index) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_xor_sync(0xffffffffu, value, offset);
        const int other_index = __shfl_xor_sync(0xffffffffu, index, offset);
        if (other > value || (other == value && other_index < index)) {
            value = other;
            index = other_index;
        }
    }
}

// The CTA's maximum of value, the lowest index among equal values; every thread receives it.
__device__ void block_argmax(float& value, int& index) {
    __shared__ float values[sort_warps];
    __shared__ int indices[sort_warps];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    warp_argmax(value, index);
    if (lane == 0) {
        values[warp] = value;
        indices[warp] = index;
    }
    __syncthreads();
    value = lane < sort_warps ? values[lane] : -INFINITY;
    index = lane < sort_warps ? indices[lane] : INT_MAX;
    warp_argmax(value, index);
    __syncthreads();
}

__device__ float block_sum(float value) {
    __shared__ float values[sort_warps];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) {
        values[warp] = value;
    }
    __syncthreads();
    value = lane < sort_warps ? values[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    __syncthreads();
    return value;
}

// Token blockIdx.x's top k experts by the softmax of its logits row in fp32, in descending order with the lower expert
// first among equal probabilities, into chosen [tokens, k], and their probabilities divided by their sum into weights
// [tokens, k]. probabilities holds experts floats.
template <typename scalar_t>
__device__ void route_token(
    const scalar_t* __restrict__ row,
    int experts,
    int k,
    float* probabilities,
    float* __restrict__ weights,
    int64_t* chosen
) {
    __shared__ float top[max_top_k];
    float max_logit = -INFINITY;
    int unused = 0;
    for (int e = threadIdx.x; e < experts; e += sort_threads) {
        max_logit = fmaxf(max_logit, static_cast<float>(row[e]));
    }
    block_argmax(max_logit, unused);
    float local = 0.0f;
    for (int e = threadIdx.x; e < experts; e += sort_threads) {
        const float shifted = expf(static_cast<float>(row[e]) - max_logit);
        probabilities[e] = shifted;
        local += shifted;
    }
    const float total = block_sum(local);
    // A NaN probability ranks below every other, as -inf, so that the top k stay among the experts.
    for (int e = threadIdx.x; e < experts; e += sort_threads) {
        const float probability = probabilities[e] / total;
        probabilities[e] = isnan(probability) ? -INFINITY : probability;
    }
    __syncthreads();

    for (int j = 0; j < k; j++) {
        float best = -INFINITY;
        int best_index = experts;
        for (int e = threadIdx.x; e < experts; e += sort_threads) {
            if (probabilities[e] > best || (probabilities[e] == best && e < best_index)) {
                best = probabilities[e];
                best_index = e;
            }
        }
        block_argmax(best, best_index);
        if (threadIdx.x == 0) {
            top[j] = best;
            chosen[static_cast<int64_t>(blockIdx.x) * k + j] = best_index;
            probabilities[best_index] = -INFINITY;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int j = 0; j < k; j++) {
            sum += top[j];
        }
        for (int j = 0; j < k; j++) {
            weights[static_cast<int64_t>(blockIdx.x) * k + j] = top[j] / sum;
        }
    }
}

// Run by one CTA, over the assignments to the held experts [first_expert, first_expert + experts), numbered from 0
// here. position_of[a] is the row of assignment a among them sorted by expert, -1 for an assignment to another expert;
// sorted_token and sorted_expert are the token and held expert of each row, sorted_expert -1 for the rows past them;
// tiles [3, max_tiles] holds each tile's held expert, first row and rows, at most max_tile_rows rows of one expert. The
// tiles past the experts' have no rows. The rows of one expert are in no fixed order. expert_counters holds 2 *
// experts entries.
__device__ void sort_assignments(
    const int64_t* assigned_experts,
    int assignments,
    int top_k,
    int first_expert,
    int experts,
    int max_tiles,
    int32_t* expert_counters,
    int32_t* __restrict__ position_of,
    int32_t* __restrict__ sorted_token,
    int32_t* __restrict__ sorted_expert,
    int32_t* __restrict__ tiles
) {
    int32_t* tile_expert = tiles;
    int32_t* tile_first_row = tiles + max_tiles;
    int32_t* tile_rows = tiles + 2 * max_tiles;
    int32_t* counts = expert_counters;
    int32_t* cursors = expert_counters + experts;
    __shared__ int32_t segment_rows[sort_threads];
    __shared__ int32_t segment_tiles[sort_threads];
    const int tid = threadIdx.x;
    for (int e = tid; e < experts; e += sort_threads) {
        counts[e] = 0;
    }
    __syncthreads();
    auto held = [&](int a) { return static_cast<int>(assigned_experts[a] - first_expert); };
    for (int a = tid; a < assignments; a += sort_threads) {
        const int e = held(a);
        if (e >= 0 && e < experts) {
            atomicAdd(&counts[e], 1);
        }
    }
    __syncthreads();

    const int segment = (experts + sort_threads - 1) / sort_threads;
    const int segment_first = min(tid * segment, experts);
    const int segment_end = min(segment_first + segment, experts);
    int rows = 0;
    int tile_count = 0;
    for (int e = segment_first; e < segment_end; e++) {
        rows += counts[e];
        tile_count += (counts[e] + max_tile_rows - 1) / max_tile_rows;
    }
    segment_rows[tid] = rows;
    segment_tiles[tid] = tile_count;
    __syncthreads();
    for (int offset = 1; offset < sort_threads; offset <<= 1) {
        const int earlier_rows = tid >= offset ? segment_rows[tid - offset] : 0;
        const int earlier_tiles = tid >= offset ? segment_tiles[tid - offset] : 0;
        __syncthreads();
        segment_rows[tid] += earlier_rows;
        segment_tiles[tid] += earlier_tiles;
        __syncthreads();
    }
    int row = segment_rows[tid] - rows;
    int tile = segment_tiles[tid] - tile_count;
    for (int e = segment_first; e < segment_end; e++) {
        cursors[e] = row;
        for (int first = 0; first < counts[e]; first += max_tile_rows) {
            tile_expert[tile] = e;
            tile_first_row[tile] = row + first;
            tile_rows[tile] = min(max_tile_rows, counts[e] - first);
            tile++;
        }
        row += counts[e];
    }
    for (int t = segment_tiles[sort_threads - 1] + tid; t < max_tiles; t += sort_threads) {
        tile_expert[t] = 0;
        tile_first_row[t] = 0;
        tile_rows[t] = 0;
    }
    for (int r = segment_rows[sort_threads - 1] + tid; r < assignments; r += sort_threads) {
        sorted_token[r] = 0;
        sorted_expert[r] = -1;
    }
    __syncthreads();
    for (int a = tid; a < assignments; a += sort_threads) {
        const int e = held(a);
        if (e < 0 || e >= experts) {
            position_of[a] = -1;
            continue;
        }
        const int position = atomicAdd(&cursors[e], 1);
        position_of[a] = position;
        sorted_token[position] = a / top_k;
        sorted_expert[position] = e;
    }
}

// One CTA, sort_assignments of the assignments top_experts [tokens, top_k] holds.
__global__ void __launch_bounds__(sort_threads) sort_assignments_kernel(
    const int64_t* __restrict__ top_experts,
    int assignments,
    int top_k,
    int experts,
    int max_tiles,
    int32_t* __restrict__ position_of,
    int32_t* __restrict__ sorted_token,
    int32_t* __restrict__ sorted_expert,
    int32_t* __restrict__ tiles
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    extern __shared__ int32_t expert_counters[];
    sort_assignments(
        top_experts,
        assignments,
        top_k,
        0,
        experts,
        max_tiles,
        expert_counters,
        position_of,
        sorted_token,
        sorted_expert,
        tiles
    );
}

// One CTA per token, route_token of its row of logits [tokens, experts], then sort_assignments of every token's
// assignments to the held experts [first_expert, first_expert + held_experts) in the last CTA to finish, found through
// arrivals.
template <typename scalar_t>
__global__ void __launch_bounds__(sort_threads) route_and_sort_kernel(
    const scalar_t* __restrict__ logits,
    int64_t logits_row_stride,
    int experts,
    int top_k,
    int first_expert,
    int held_experts,
    int max_tiles,
    float* __restrict__ top_weights,
    int64_t* top_experts,
    int32_t* __restrict__ arrivals,
    int32_t* __restrict__ position_of,
    int32_t* __restrict__ sorted_token,
    int32_t* __restrict__ sorted_expert,
    int32_t* __restrict__ tiles
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    extern __shared__ int32_t shared_words[];
    route_token(
        logits + static_cast<int64_t>(blockIdx.x) * logits_row_stride,
        experts,
        top_k,
        reinterpret_cast<float*>(shared_words),
        top_weights,
        top_experts
    );
    if (last_to_arrive(arrivals, static_cast<int>(gridDim.x))) {
        sort_assignments(
            top_experts,
            static_cast<int>(gridDim.x) * top_k,
            top_k,
            first_expert,
            held_experts,
            max_tiles,
            shared_words,
            position_of,
            sorted_token,
            sorted_expert,
            tiles
        );
    }
}

// The element offset of 8-element vector index vector of row row in a staged tile of chunk_k columns, permuted so that
// the 8 rows an ldmatrix quadrant reads fall in distinct banks.
__device__ __forceinline__ int staged_offset(int row, int vector) {
    return row * chunk_k + ((vector ^ (row & 7)) * 8);
}

// Multiplies the tile's rows of x by tile_weight_rows weight rows of the tile's expert over k, one CTA per (tile of
// sort_assignments_kernel's tiles, block of weight rows). With gated, the block is the gate rows [n0, n0 + 32) and the
// up rows [n + n0, n + n0 + 32), x's rows are those of the tile's sorted tokens, and out [rows, n] holds
// activation(gate) * up. Otherwise the block is the rows [n0, n0 + 64), x's rows are the tile's rows, and out
// [rows, n] holds the products. With partial, split blockIdx.z of the gated product covers chunks_per_split chunks of
// k and writes its fp32 gate and up sums into partial [gridDim.z, rows of every tile, 2 * n]; the last split of a
// block to finish, found through block_counters [gridDim.x * gridDim.y], sums the splits in split order and writes
// out.
template <typename scalar_t, bool bf16, bool gated>
__global__ void __launch_bounds__(gemm_threads) expert_tile_gemm_kernel(
    const scalar_t* __restrict__ x,
    int64_t x_row_stride,
    const int32_t* __restrict__ sorted_token,
    const scalar_t* __restrict__ weights,
    int64_t expert_stride,
    int64_t weight_row_stride,
    int n,
    int k,
    const int32_t* __restrict__ tiles,
    expert_activation activation,
    scalar_t* __restrict__ out,
    int chunks_per_split,
    float* partial,
    int64_t partial_rows,
    int32_t* __restrict__ block_counters
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    const int rows = tiles[2 * gridDim.x + blockIdx.x];
    if (rows == 0) {
        return;
    }
    const int first_row = tiles[gridDim.x + blockIdx.x];
    const scalar_t* expert_weights = weights + static_cast<int64_t>(tiles[blockIdx.x]) * expert_stride;
    constexpr int block_columns = gated ? tile_weight_rows / 2 : tile_weight_rows;
    const int n0 = static_cast<int>(blockIdx.y) * block_columns;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    __shared__ __align__(16) scalar_t staged_x[stages][max_tile_rows * chunk_k];
    __shared__ __align__(16) scalar_t staged_weights[stages][tile_weight_rows * chunk_k];

    const int x_row = tid / chunk_vectors;
    const int x_vector = tid % chunk_vectors;
    const scalar_t* x_source = x;
    if (x_row < rows) {
        const int64_t source_row = gated ? sorted_token[first_row + x_row] : first_row + x_row;
        x_source = x + source_row * x_row_stride;
    }
    constexpr int weight_vectors_per_thread = tile_weight_rows * chunk_vectors / gemm_threads;
    const scalar_t* weight_sources[weight_vectors_per_thread];
    bool weight_valid[weight_vectors_per_thread];
#pragma unroll
    for (int i = 0; i < weight_vectors_per_thread; i++) {
        const int row = (tid + i * gemm_threads) / chunk_vectors;
        int column;
        int64_t weight_row;
        if constexpr (gated) {
            column = n0 + 8 * (row / 16) + row % 8;
            weight_row = (row % 16) < 8 ? column : n + column;
        } else {
            column = n0 + row;
            weight_row = column;
        }
        weight_valid[i] = column < n;
        weight_sources[i] = weight_valid[i] ? expert_weights + weight_row * weight_row_stride : expert_weights;
    }

    auto stage = [&](int slot, int chunk) {
        const int k0 = chunk * chunk_k;
        const int x_column = k0 + x_vector * 8;
        const bool x_valid = x_row < rows && x_column < k;
        __pipeline_memcpy_async(
            &staged_x[slot][staged_offset(x_row, x_vector)],
            x_valid ? x_source + x_column : x,
            16,
            x_valid ? 0 : 16
        );
#pragma unroll
        for (int i = 0; i < weight_vectors_per_thread; i++) {
            const int index = tid + i * gemm_threads;
            const int row = index / chunk_vectors;
            const int vector = index % chunk_vectors;
            const int column = k0 + vector * 8;
            const bool valid = weight_valid[i] && column < k;
            __pipeline_memcpy_async(
                &staged_weights[slot][staged_offset(row, vector)],
                valid ? weight_sources[i] + column : expert_weights,
                16,
                valid ? 0 : 16
            );
        }
    };

    float first_fragment[4] = {};
    float second_fragment[4] = {};
    const int first_chunk = static_cast<int>(blockIdx.z) * chunks_per_split;
    const int chunks = min((k + chunk_k - 1) / chunk_k - first_chunk, chunks_per_split);
    for (int chunk = 0; chunk < stages - 1; chunk++) {
        if (chunk < chunks) {
            stage(chunk, first_chunk + chunk);
        }
        __pipeline_commit();
    }
    for (int chunk = 0; chunk < chunks; chunk++) {
        __pipeline_wait_prior(stages - 2);
        __syncthreads();
        const int next = chunk + stages - 1;
        if (next < chunks) {
            stage(next % stages, first_chunk + next);
        }
        __pipeline_commit();
        const scalar_t* x_tile = staged_x[chunk % stages];
        const scalar_t* weight_tile = staged_weights[chunk % stages];
#pragma unroll
        for (int step = 0; step < chunk_k / 16; step++) {
            const int vector = 2 * step + lane / 16;
            uint32_t a[4];
            uint32_t b[4];
            mma::ldmatrix_x4(a, x_tile + staged_offset(lane % 16, vector));
            mma::ldmatrix_x4(b, weight_tile + staged_offset(16 * warp + lane % 16, vector));
            const uint32_t first_b[2] = {b[0], b[2]};
            const uint32_t second_b[2] = {b[1], b[3]};
            mma::mma_m16n8k16<bf16>(first_fragment, a, first_b, first_fragment);
            mma::mma_m16n8k16<bf16>(second_fragment, a, second_b, second_fragment);
        }
    }

    const int pair = 2 * (lane % 4);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int row = lane / 4 + (i < 2 ? 0 : 8);
        if (row >= rows) {
            continue;
        }
        scalar_t* out_row = out + static_cast<int64_t>(first_row + row) * n;
        if constexpr (gated) {
            const int column = n0 + 8 * warp + pair + i % 2;
            if (column < n && partial != nullptr) {
                float* partial_row = partial + (blockIdx.z * partial_rows + first_row + row) * 2 * n;
                partial_row[column] = first_fragment[i];
                partial_row[n + column] = second_fragment[i];
            } else if (column < n) {
                const float gate = rounded<scalar_t>(first_fragment[i]);
                const float up = rounded<scalar_t>(second_fragment[i]);
                out_row[column] = static_cast<scalar_t>(__fmul_rn(rounded<scalar_t>(activated(gate, activation)), up));
            }
        } else {
            const int column = n0 + 16 * warp + pair + i % 2;
            if (column < n) {
                out_row[column] = static_cast<scalar_t>(first_fragment[i]);
            }
            if (column + 8 < n) {
                out_row[column + 8] = static_cast<scalar_t>(second_fragment[i]);
            }
        }
    }
    if constexpr (gated) {
        if (partial == nullptr ||
            !last_to_arrive(block_counters + blockIdx.x * gridDim.y + blockIdx.y, static_cast<int>(gridDim.z))) {
            return;
        }
        for (int index = tid; index < rows * block_columns; index += gemm_threads) {
            const int64_t row = first_row + index / block_columns;
            const int column = n0 + index % block_columns;
            if (column >= n) {
                continue;
            }
            float gate = 0.0f;
            float up = 0.0f;
            for (int split = 0; split < static_cast<int>(gridDim.z); split++) {
                const float* partial_row = partial + (split * partial_rows + row) * 2 * n;
                gate += partial_row[column];
                up += partial_row[n + column];
            }
            out[row * n + column] = static_cast<scalar_t>(
                __fmul_rn(rounded<scalar_t>(activated(rounded<scalar_t>(gate), activation)), rounded<scalar_t>(up))
            );
        }
    }
}

// One thread per (sorted row, column), the fp32 form of expert_tile_gemm_kernel.
template <bool gated>
__global__ void expert_rows_scalar_kernel(
    const float* __restrict__ x,
    int64_t x_row_stride,
    const int32_t* __restrict__ sorted_token,
    const int32_t* __restrict__ sorted_expert,
    const float* __restrict__ weights,
    int64_t expert_stride,
    int64_t weight_row_stride,
    int64_t rows,
    int n,
    int k,
    expert_activation activation,
    float* __restrict__ out
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * n) {
        return;
    }
    const int64_t row = index / n;
    const int column = static_cast<int>(index % n);
    if (sorted_expert[row] < 0) {
        return;
    }
    const float* x_row = x + (gated ? sorted_token[row] : row) * x_row_stride;
    const float* expert_weights = weights + static_cast<int64_t>(sorted_expert[row]) * expert_stride;
    auto dot = [&](int64_t weight_row) {
        const float* w = expert_weights + weight_row * weight_row_stride;
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += x_row[i] * w[i];
        }
        return sum;
    };
    out[index] = gated ? activated(dot(column), activation) * dot(n + column) : dot(column);
}

// Sums each token's expert outputs weighted by top_weights in fp32 in top_k order, one thread per (token, column),
// skipping the assignments without a row.
template <typename scalar_t>
__global__ void combine_experts_kernel(
    const scalar_t* __restrict__ expert_out,
    const float* __restrict__ top_weights,
    const int32_t* __restrict__ position_of,
    int64_t tokens,
    int top_k,
    int hidden,
    scalar_t* __restrict__ out
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= tokens * hidden) {
        return;
    }
    const int64_t token = index / hidden;
    const int column = static_cast<int>(index % hidden);
    float sum = 0.0f;
    for (int j = 0; j < top_k; j++) {
        const int64_t assignment = token * top_k + j;
        const int32_t position = position_of[assignment];
        if (position >= 0) {
            sum += top_weights[assignment] *
                static_cast<float>(expert_out[static_cast<int64_t>(position) * hidden + column]);
        }
    }
    out[index] = static_cast<scalar_t>(sum);
}

unsigned element_blocks(int64_t elements) {
    return static_cast<unsigned>((elements + element_threads - 1) / element_threads);
}

int64_t max_tiles_of(int64_t assignments, int64_t experts) {
    return std::min(assignments, assignments / max_tile_rows + std::min(experts, assignments));
}

int64_t gate_up_column_blocks(int64_t intermediate) {
    return (intermediate + tile_weight_rows / 2 - 1) / (tile_weight_rows / 2);
}

// The splits the gated product takes along k, and the chunks of k each covers.
std::pair<int64_t, int64_t> gate_up_splits(int64_t max_tiles, int64_t intermediate, int64_t hidden) {
    const int64_t gate_up_blocks = max_tiles * gate_up_column_blocks(intermediate);
    const int64_t chunks = (hidden + chunk_k - 1) / chunk_k;
    const int64_t sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int64_t splits = std::clamp<int64_t>(split_target_per_sm * sms / gate_up_blocks, 1, chunks);
    const int64_t chunks_per_split = (chunks + splits - 1) / splits;
    return {(chunks + chunks_per_split - 1) / chunks_per_split, chunks_per_split};
}

template <typename scalar_t, bool bf16>
void launch_tile_gemms(
    const at::Tensor& h,
    const at::Tensor& gate_up_proj,
    const at::Tensor& down_proj,
    expert_activation activation,
    const int32_t* sorted_token,
    const int32_t* tiles,
    int64_t max_tiles,
    at::Tensor& activations,
    at::Tensor& expert_out,
    const at::Tensor& counters
) {
    const int64_t hidden = h.size(1);
    const int64_t intermediate = down_proj.size(2);
    const int64_t assignments = activations.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t chunks = (hidden + chunk_k - 1) / chunk_k;
    const auto [used_splits, chunks_per_split] = gate_up_splits(max_tiles, intermediate, hidden);
    at::Tensor partial;
    if (used_splits > 1) {
        partial = at::empty({used_splits, assignments, 2 * intermediate}, h.options().dtype(at::kFloat));
    }
    const dim3 gate_up_grid(
        static_cast<unsigned>(max_tiles),
        static_cast<unsigned>(gate_up_column_blocks(intermediate)),
        static_cast<unsigned>(used_splits)
    );
    launch_dependent(
        expert_tile_gemm_kernel<scalar_t, bf16, true>,
        gate_up_grid,
        gemm_threads,
        0,
        stream,
        h.const_data_ptr<scalar_t>(),
        h.stride(0),
        sorted_token,
        gate_up_proj.const_data_ptr<scalar_t>(),
        gate_up_proj.stride(0),
        gate_up_proj.stride(1),
        static_cast<int>(intermediate),
        static_cast<int>(hidden),
        tiles,
        activation,
        activations.mutable_data_ptr<scalar_t>(),
        static_cast<int>(chunks_per_split),
        used_splits > 1 ? partial.mutable_data_ptr<float>() : nullptr,
        assignments,
        used_splits > 1 ? counters.mutable_data_ptr<int32_t>() : nullptr
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const dim3 down_grid(
        static_cast<unsigned>(max_tiles),
        static_cast<unsigned>((hidden + tile_weight_rows - 1) / tile_weight_rows)
    );
    launch_dependent(
        expert_tile_gemm_kernel<scalar_t, bf16, false>,
        down_grid,
        gemm_threads,
        0,
        stream,
        activations.const_data_ptr<scalar_t>(),
        intermediate,
        sorted_token,
        down_proj.const_data_ptr<scalar_t>(),
        down_proj.stride(0),
        down_proj.stride(1),
        static_cast<int>(hidden),
        static_cast<int>(intermediate),
        tiles,
        activation,
        expert_out.mutable_data_ptr<scalar_t>(),
        static_cast<int>((intermediate + chunk_k - 1) / chunk_k),
        nullptr,
        assignments,
        nullptr
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

bool aligned_rows(const at::Tensor& t, int64_t row_stride) {
    return reinterpret_cast<uintptr_t>(t.const_data_ptr()) % 16 == 0 && row_stride * t.element_size() % 16 == 0;
}

void check_counters(const at::Tensor& counters, const at::Tensor& h, int64_t needed) {
    TORCH_CHECK(
        counters.is_cuda() && counters.device() == h.device() && counters.scalar_type() == at::kInt &&
            counters.is_contiguous() && counters.numel() >= needed,
        "routed_experts: counters must be contiguous int32 on h's device, at least routed_experts_counters of them"
    );
}

void check_experts(const at::Tensor& h, const at::Tensor& gate_up_proj, const at::Tensor& down_proj) {
    TORCH_CHECK(
        h.is_cuda() && h.dim() == 2 && h.stride(1) == 1,
        "routed_experts: h must be CUDA [tokens, hidden], each row contiguous"
    );
    const int64_t hidden = h.size(1);
    TORCH_CHECK(
        gate_up_proj.dim() == 3 && down_proj.dim() == 3 && gate_up_proj.size(0) == down_proj.size(0) &&
            gate_up_proj.size(2) == hidden && down_proj.size(1) == hidden &&
            gate_up_proj.size(1) == 2 * down_proj.size(2) && gate_up_proj.stride(2) == 1 && down_proj.stride(2) == 1 &&
            gate_up_proj.scalar_type() == h.scalar_type() && down_proj.scalar_type() == h.scalar_type() &&
            gate_up_proj.device() == h.device() && down_proj.device() == h.device(),
        "routed_experts: gate_up_proj must be [experts, 2 * intermediate, hidden] and down_proj [experts, hidden, "
        "intermediate], each row contiguous, in h's dtype on h's device"
    );
    const int64_t experts = gate_up_proj.size(0);
    TORCH_CHECK(experts >= 1 && experts <= max_experts, "routed_experts: experts must be between 1 and ", max_experts);
}

// The routed experts' output for top_k assignments per token of h. sort(position_of, sorted_token, sorted_expert,
// tiles, max_tiles) launches the kernel that sorts the assignments as sort_assignments does and leaves their weights in
// top_weights [tokens, top_k].
template <typename Sort>
at::Tensor run_experts(
    const at::Tensor& h,
    int64_t top_k,
    const at::Tensor& top_weights,
    const at::Tensor& gate_up_proj,
    const at::Tensor& down_proj,
    expert_activation activation,
    const at::Tensor& counters,
    Sort&& sort
) {
    const int64_t tokens = h.size(0);
    const int64_t hidden = h.size(1);
    const int64_t experts = gate_up_proj.size(0);
    const int64_t intermediate = down_proj.size(2);
    at::Tensor out = at::empty({tokens, hidden}, h.options());
    const int64_t assignments = tokens * top_k;
    if (assignments == 0) {
        return out.zero_();
    }
    TORCH_CHECK(assignments <= std::numeric_limits<int32_t>::max() / max_tile_rows, "routed_experts: too many assignments");

    const int64_t max_tiles = max_tiles_of(assignments, experts);
    at::Tensor indices = at::empty({3 * assignments + 3 * max_tiles}, h.options().dtype(at::kInt));
    int32_t* position_of = indices.mutable_data_ptr<int32_t>();
    int32_t* sorted_token = position_of + assignments;
    int32_t* sorted_expert = sorted_token + assignments;
    int32_t* tiles = sorted_expert + assignments;
    sort(position_of, sorted_token, sorted_expert, tiles, max_tiles);
    auto stream = at::cuda::getCurrentCUDAStream();

    at::Tensor activations = at::empty({assignments, intermediate}, h.options());
    at::Tensor expert_out = at::empty({assignments, hidden}, h.options());
    if (h.scalar_type() == at::kFloat) {
        launch_dependent(
            expert_rows_scalar_kernel<true>,
            element_blocks(assignments * intermediate),
            element_threads,
            0,
            stream,
            h.const_data_ptr<float>(),
            h.stride(0),
            sorted_token,
            sorted_expert,
            gate_up_proj.const_data_ptr<float>(),
            gate_up_proj.stride(0),
            gate_up_proj.stride(1),
            assignments,
            static_cast<int>(intermediate),
            static_cast<int>(hidden),
            activation,
            activations.mutable_data_ptr<float>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        launch_dependent(
            expert_rows_scalar_kernel<false>,
            element_blocks(assignments * hidden),
            element_threads,
            0,
            stream,
            activations.const_data_ptr<float>(),
            intermediate,
            sorted_token,
            sorted_expert,
            down_proj.const_data_ptr<float>(),
            down_proj.stride(0),
            down_proj.stride(1),
            assignments,
            static_cast<int>(hidden),
            static_cast<int>(intermediate),
            activation,
            expert_out.mutable_data_ptr<float>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        TORCH_CHECK(
            h.scalar_type() == at::kBFloat16 || h.scalar_type() == at::kHalf,
            "routed_experts: h must be fp32, fp16 or bf16"
        );
        TORCH_CHECK(
            hidden % 8 == 0 && intermediate % 8 == 0 && aligned_rows(h, h.stride(0)) &&
                aligned_rows(gate_up_proj, gate_up_proj.stride(1)) && aligned_rows(down_proj, down_proj.stride(1)) &&
                gate_up_proj.stride(0) % 8 == 0 && down_proj.stride(0) % 8 == 0,
            "routed_experts: hidden and intermediate must be multiples of 8 and the rows 16-byte aligned"
        );
        if (h.scalar_type() == at::kBFloat16) {
            launch_tile_gemms<at::BFloat16, true>(
                h,
                gate_up_proj,
                down_proj,
                activation,
                sorted_token,
                tiles,
                max_tiles,
                activations,
                expert_out,
                counters
            );
        } else {
            launch_tile_gemms<at::Half, false>(
                h,
                gate_up_proj,
                down_proj,
                activation,
                sorted_token,
                tiles,
                max_tiles,
                activations,
                expert_out,
                counters
            );
        }
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, h.scalar_type(), "routed_experts_cuda", [&] {
        launch_dependent(
            combine_experts_kernel<scalar_t>,
            element_blocks(tokens * hidden),
            element_threads,
            0,
            stream,
            expert_out.const_data_ptr<scalar_t>(),
            top_weights.const_data_ptr<float>(),
            position_of,
            tokens,
            static_cast<int>(top_k),
            static_cast<int>(hidden),
            out.mutable_data_ptr<scalar_t>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return out;
}

}  // namespace

// The route-and-sort kernel's one counter, then the split gated product's one per block, used by kernels in turn.
int64_t
routed_experts_counters(int64_t tokens, int64_t top_k, int64_t held_experts, int64_t intermediate, int64_t hidden) {
    const int64_t assignments = tokens * top_k;
    if (assignments == 0) {
        return 0;
    }
    const int64_t max_tiles = max_tiles_of(assignments, held_experts);
    const bool split = gate_up_splits(max_tiles, intermediate, hidden).first > 1;
    return std::max<int64_t>(1, split ? max_tiles * gate_up_column_blocks(intermediate) : 0);
}

at::Tensor routed_experts_cuda(
    const at::Tensor& h,
    const at::Tensor& top_experts,
    const at::Tensor& top_weights,
    const at::Tensor& gate_up_proj,
    const at::Tensor& down_proj,
    expert_activation activation,
    const at::Tensor& counters
) {
    check_experts(h, gate_up_proj, down_proj);
    TORCH_CHECK(
        top_experts.scalar_type() == at::kLong && top_weights.scalar_type() == at::kFloat && top_experts.dim() == 2 &&
            top_experts.size(0) == h.size(0) && top_experts.sizes() == top_weights.sizes() &&
            top_experts.is_contiguous() && top_weights.is_contiguous() && top_experts.device() == h.device() &&
            top_weights.device() == h.device(),
        "routed_experts: top_experts must be contiguous int64 and top_weights contiguous fp32 [tokens, top_k] on h's "
        "device"
    );
    const int64_t experts = gate_up_proj.size(0);
    const int64_t top_k = top_experts.size(1);
    check_counters(counters, h, routed_experts_counters(h.size(0), top_k, experts, down_proj.size(2), h.size(1)));
    auto sort =
        [&](int32_t* position_of, int32_t* sorted_token, int32_t* sorted_expert, int32_t* tiles, int64_t max_tiles) {
            launch_dependent(
                sort_assignments_kernel,
                1,
                sort_threads,
                2 * experts * sizeof(int32_t),
                at::cuda::getCurrentCUDAStream(),
                top_experts.const_data_ptr<int64_t>(),
                static_cast<int>(h.size(0) * top_k),
                static_cast<int>(top_k),
                static_cast<int>(experts),
                static_cast<int>(max_tiles),
                position_of,
                sorted_token,
                sorted_expert,
                tiles
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        };
    return run_experts(h, top_k, top_weights, gate_up_proj, down_proj, activation, counters, sort);
}

at::Tensor routed_experts_cuda(
    const at::Tensor& h,
    const at::Tensor& router_logits,
    int64_t top_k,
    int64_t first_expert,
    const at::Tensor& gate_up_proj,
    const at::Tensor& down_proj,
    expert_activation activation,
    const at::Tensor& counters
) {
    check_experts(h, gate_up_proj, down_proj);
    const int64_t tokens = h.size(0);
    const int64_t held_experts = gate_up_proj.size(0);
    check_counters(counters, h, routed_experts_counters(tokens, top_k, held_experts, down_proj.size(2), h.size(1)));
    TORCH_CHECK(
        router_logits.is_cuda() && router_logits.dim() == 2 && router_logits.size(0) == tokens &&
            router_logits.stride(1) == 1 && router_logits.device() == h.device(),
        "routed_experts: router_logits must be CUDA [tokens, experts] on h's device, each row contiguous"
    );
    const int64_t experts = router_logits.size(1);
    TORCH_CHECK(
        experts <= max_experts && first_expert >= 0 && first_expert + held_experts <= experts,
        "routed_experts: the held experts [first_expert, first_expert + gate_up_proj.size(0)) must lie within the "
        "router's experts, at most ",
        max_experts
    );
    TORCH_CHECK(
        top_k >= 1 && top_k <= experts && top_k <= max_top_k,
        "routed_experts: top_k must be between 1 and the experts, at most ",
        max_top_k
    );
    at::Tensor top_weights = at::empty({tokens, top_k}, h.options().dtype(at::kFloat));
    at::Tensor top_experts = at::empty({tokens, top_k}, h.options().dtype(at::kLong));
    auto sort =
        [&](int32_t* position_of, int32_t* sorted_token, int32_t* sorted_expert, int32_t* tiles, int64_t max_tiles) {
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::kHalf,
                at::kBFloat16,
                router_logits.scalar_type(),
                "routed_experts_cuda",
                [&] {
                    launch_dependent(
                        route_and_sort_kernel<scalar_t>,
                        static_cast<unsigned>(tokens),
                        sort_threads,
                        std::max(experts, 2 * held_experts) * sizeof(int32_t),
                        at::cuda::getCurrentCUDAStream(),
                        router_logits.const_data_ptr<scalar_t>(),
                        router_logits.stride(0),
                        static_cast<int>(experts),
                        static_cast<int>(top_k),
                        static_cast<int>(first_expert),
                        static_cast<int>(held_experts),
                        static_cast<int>(max_tiles),
                        top_weights.mutable_data_ptr<float>(),
                        top_experts.mutable_data_ptr<int64_t>(),
                        counters.mutable_data_ptr<int32_t>(),
                        position_of,
                        sorted_token,
                        sorted_expert,
                        tiles
                    );
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                }
            );
        };
    return run_experts(h, top_k, top_weights, gate_up_proj, down_proj, activation, counters, sort);
}

}  // namespace pulsarlm
