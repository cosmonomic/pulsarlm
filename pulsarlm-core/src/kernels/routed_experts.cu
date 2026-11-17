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
#include <cmath>
#include <cstdint>
#include <limits>

namespace pulsarlm {
namespace {

constexpr int kSortThreads = 1024;
constexpr int64_t kMaxExperts = 4096;
constexpr int kTileRows = 16;
constexpr int kTileWeightRows = 64;
constexpr int kChunkK = 64;
constexpr int kChunkVectors = kChunkK / 8;
constexpr int kStages = 3;
constexpr int kGemmThreads = 128;
constexpr int kElementThreads = 256;
// The gated product splits k until its grid reaches this many CTAs per SM.
constexpr int64_t kSplitTargetPerSm = 2;

template <typename scalar_t> __device__ __forceinline__ float rounded(float value) {
    return static_cast<float>(static_cast<scalar_t>(value));
}

// at::silu and at::gelu(x, "tanh") in fp32.
__device__ __forceinline__ float activated(float x, expert_activation activation) {
    if (activation == expert_activation::silu) {
        return x / (1.0f + expf(-x));
    }
    constexpr float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5f;
    constexpr float kKappa = 0.044715f;
    const float cube = x * x * x;
    return 0.5f * x * (1.0f + tanhf(kBeta * (x + kKappa * cube)));
}

// Runs as one CTA. position_of[a] is the row of assignment a among the assignments sorted by expert, sorted_token and
// sorted_expert are the token and expert of each row, and tiles [3, max_tiles] holds each tile's expert, first row and
// rows, at most kTileRows rows of one expert. The tiles past the experts' have no rows. The rows of one expert are in
// no fixed order.
__global__ void __launch_bounds__(kSortThreads) sort_assignments_kernel(
    const int64_t* __restrict__ assigned_experts,
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
    int32_t* tile_expert = tiles;
    int32_t* tile_first_row = tiles + max_tiles;
    int32_t* tile_rows = tiles + 2 * max_tiles;
    extern __shared__ int32_t expert_counters[];
    int32_t* counts = expert_counters;
    int32_t* cursors = expert_counters + experts;
    __shared__ int32_t segment_rows[kSortThreads];
    __shared__ int32_t segment_tiles[kSortThreads];
    const int tid = threadIdx.x;
    for (int e = tid; e < experts; e += kSortThreads) {
        counts[e] = 0;
    }
    __syncthreads();
    for (int a = tid; a < assignments; a += kSortThreads) {
        atomicAdd(&counts[assigned_experts[a]], 1);
    }
    __syncthreads();

    const int segment = (experts + kSortThreads - 1) / kSortThreads;
    const int first_expert = min(tid * segment, experts);
    const int end_expert = min(first_expert + segment, experts);
    int rows = 0;
    int tile_count = 0;
    for (int e = first_expert; e < end_expert; e++) {
        rows += counts[e];
        tile_count += (counts[e] + kTileRows - 1) / kTileRows;
    }
    segment_rows[tid] = rows;
    segment_tiles[tid] = tile_count;
    __syncthreads();
    for (int offset = 1; offset < kSortThreads; offset <<= 1) {
        const int earlier_rows = tid >= offset ? segment_rows[tid - offset] : 0;
        const int earlier_tiles = tid >= offset ? segment_tiles[tid - offset] : 0;
        __syncthreads();
        segment_rows[tid] += earlier_rows;
        segment_tiles[tid] += earlier_tiles;
        __syncthreads();
    }
    int row = segment_rows[tid] - rows;
    int tile = segment_tiles[tid] - tile_count;
    for (int e = first_expert; e < end_expert; e++) {
        cursors[e] = row;
        for (int first = 0; first < counts[e]; first += kTileRows) {
            tile_expert[tile] = e;
            tile_first_row[tile] = row + first;
            tile_rows[tile] = min(kTileRows, counts[e] - first);
            tile++;
        }
        row += counts[e];
    }
    for (int t = segment_tiles[kSortThreads - 1] + tid; t < max_tiles; t += kSortThreads) {
        tile_expert[t] = 0;
        tile_first_row[t] = 0;
        tile_rows[t] = 0;
    }
    __syncthreads();
    for (int a = tid; a < assignments; a += kSortThreads) {
        const auto e = static_cast<int>(assigned_experts[a]);
        const int position = atomicAdd(&cursors[e], 1);
        position_of[a] = position;
        sorted_token[position] = a / top_k;
        sorted_expert[position] = e;
    }
}

// The element offset of 8-element vector index vector of row row in a staged tile of kChunkK columns, permuted so that
// the 8 rows an ldmatrix quadrant reads fall in distinct banks.
__device__ __forceinline__ int staged_offset(int row, int vector) {
    return row * kChunkK + ((vector ^ (row & 7)) * 8);
}

// Multiplies the tile's rows of x by kTileWeightRows weight rows of the tile's expert over k, one CTA per (tile of
// sort_assignments_kernel's tiles, block of weight rows). With gated, the block is the gate rows [n0, n0 + 32) and the
// up rows [n + n0, n + n0 + 32), x's rows are those of the tile's sorted tokens, and out [rows, n] holds
// activation(gate) * up. Otherwise the block is the rows [n0, n0 + 64), x's rows are the tile's rows, and out
// [rows, n] holds the products. With partial, split blockIdx.z of the gated product covers chunks_per_split chunks of
// k and writes its fp32 gate and up sums into partial [gridDim.z, rows of every tile, 2 * n]; the last split of a
// block to finish, found through block_counters [gridDim.x * gridDim.y], sums the splits in split order and writes
// out.
template <typename scalar_t, bool bf16, bool gated>
__global__ void __launch_bounds__(kGemmThreads) expert_tile_gemm_kernel(
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
    constexpr int kBlockColumns = gated ? kTileWeightRows / 2 : kTileWeightRows;
    const int n0 = static_cast<int>(blockIdx.y) * kBlockColumns;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    __shared__ __align__(16) scalar_t staged_x[kStages][kTileRows * kChunkK];
    __shared__ __align__(16) scalar_t staged_weights[kStages][kTileWeightRows * kChunkK];

    const int x_row = tid / kChunkVectors;
    const int x_vector = tid % kChunkVectors;
    const scalar_t* x_source = x;
    if (x_row < rows) {
        const int64_t source_row = gated ? sorted_token[first_row + x_row] : first_row + x_row;
        x_source = x + source_row * x_row_stride;
    }
    constexpr int kWeightVectorsPerThread = kTileWeightRows * kChunkVectors / kGemmThreads;
    const scalar_t* weight_sources[kWeightVectorsPerThread];
    bool weight_valid[kWeightVectorsPerThread];
#pragma unroll
    for (int i = 0; i < kWeightVectorsPerThread; i++) {
        const int row = (tid + i * kGemmThreads) / kChunkVectors;
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
        const int k0 = chunk * kChunkK;
        const int x_column = k0 + x_vector * 8;
        const bool x_valid = x_row < rows && x_column < k;
        __pipeline_memcpy_async(
            &staged_x[slot][staged_offset(x_row, x_vector)],
            x_valid ? x_source + x_column : x,
            16,
            x_valid ? 0 : 16
        );
#pragma unroll
        for (int i = 0; i < kWeightVectorsPerThread; i++) {
            const int index = tid + i * kGemmThreads;
            const int row = index / kChunkVectors;
            const int vector = index % kChunkVectors;
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
    const int chunks = min((k + kChunkK - 1) / kChunkK - first_chunk, chunks_per_split);
    for (int chunk = 0; chunk < kStages - 1; chunk++) {
        if (chunk < chunks) {
            stage(chunk, first_chunk + chunk);
        }
        __pipeline_commit();
    }
    for (int chunk = 0; chunk < chunks; chunk++) {
        __pipeline_wait_prior(kStages - 2);
        __syncthreads();
        const int next = chunk + kStages - 1;
        if (next < chunks) {
            stage(next % kStages, first_chunk + next);
        }
        __pipeline_commit();
        const scalar_t* x_tile = staged_x[chunk % kStages];
        const scalar_t* weight_tile = staged_weights[chunk % kStages];
#pragma unroll
        for (int step = 0; step < kChunkK / 16; step++) {
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
        for (int index = tid; index < rows * kBlockColumns; index += kGemmThreads) {
            const int64_t row = first_row + index / kBlockColumns;
            const int column = n0 + index % kBlockColumns;
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

// Sums each token's expert outputs weighted by top_weights in fp32 in top_k order, one thread per (token, column).
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
        sum += top_weights[assignment] *
            static_cast<float>(expert_out[static_cast<int64_t>(position_of[assignment]) * hidden + column]);
    }
    out[index] = static_cast<scalar_t>(sum);
}

unsigned element_blocks(int64_t elements) {
    return static_cast<unsigned>((elements + kElementThreads - 1) / kElementThreads);
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
    at::Tensor& expert_out
) {
    const int64_t hidden = h.size(1);
    const int64_t intermediate = down_proj.size(2);
    const int64_t assignments = activations.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t gate_up_blocks = max_tiles * ((intermediate + kTileWeightRows / 2 - 1) / (kTileWeightRows / 2));
    const int64_t chunks = (hidden + kChunkK - 1) / kChunkK;
    const int64_t sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int64_t splits = std::clamp<int64_t>(kSplitTargetPerSm * sms / gate_up_blocks, 1, chunks);
    const int64_t chunks_per_split = (chunks + splits - 1) / splits;
    const int64_t used_splits = (chunks + chunks_per_split - 1) / chunks_per_split;
    at::Tensor partial;
    if (used_splits > 1) {
        partial = at::empty({used_splits, assignments, 2 * intermediate}, h.options().dtype(at::kFloat));
    }
    const dim3 gate_up_grid(
        static_cast<unsigned>(max_tiles),
        static_cast<unsigned>((intermediate + kTileWeightRows / 2 - 1) / (kTileWeightRows / 2)),
        static_cast<unsigned>(used_splits)
    );
    launch_dependent(
        expert_tile_gemm_kernel<scalar_t, bf16, true>,
        gate_up_grid,
        kGemmThreads,
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
        used_splits > 1 ? arrival_counters(gate_up_grid.x * gate_up_grid.y) : nullptr
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const dim3 down_grid(
        static_cast<unsigned>(max_tiles),
        static_cast<unsigned>((hidden + kTileWeightRows - 1) / kTileWeightRows)
    );
    launch_dependent(
        expert_tile_gemm_kernel<scalar_t, bf16, false>,
        down_grid,
        kGemmThreads,
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
        static_cast<int>((intermediate + kChunkK - 1) / kChunkK),
        nullptr,
        assignments,
        nullptr
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

bool aligned_rows(const at::Tensor& t, int64_t row_stride) {
    return reinterpret_cast<uintptr_t>(t.const_data_ptr()) % 16 == 0 && row_stride * t.element_size() % 16 == 0;
}

}  // namespace

at::Tensor routed_experts_cuda(
    const at::Tensor& h,
    const at::Tensor& top_experts,
    const at::Tensor& top_weights,
    const at::Tensor& gate_up_proj,
    const at::Tensor& down_proj,
    expert_activation activation
) {
    TORCH_CHECK(
        h.is_cuda() && h.dim() == 2 && h.stride(1) == 1,
        "routed_experts: h must be CUDA [tokens, hidden], each row contiguous"
    );
    const int64_t tokens = h.size(0);
    const int64_t hidden = h.size(1);
    TORCH_CHECK(
        top_experts.scalar_type() == at::kLong && top_weights.scalar_type() == at::kFloat && top_experts.dim() == 2 &&
            top_experts.size(0) == tokens && top_experts.sizes() == top_weights.sizes() &&
            top_experts.is_contiguous() && top_weights.is_contiguous() && top_experts.device() == h.device() &&
            top_weights.device() == h.device(),
        "routed_experts: top_experts must be contiguous int64 and top_weights contiguous fp32 [tokens, top_k] on h's "
        "device"
    );
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
    const int64_t intermediate = down_proj.size(2);
    const int64_t top_k = top_experts.size(1);
    TORCH_CHECK(experts >= 1 && experts <= kMaxExperts, "routed_experts: experts must be between 1 and ", kMaxExperts);
    at::Tensor out = at::empty({tokens, hidden}, h.options());
    const int64_t assignments = tokens * top_k;
    if (assignments == 0) {
        return out.zero_();
    }
    TORCH_CHECK(assignments <= std::numeric_limits<int32_t>::max() / kTileRows, "routed_experts: too many assignments");

    const int64_t max_tiles = std::min(assignments, assignments / kTileRows + std::min(experts, assignments));
    at::Tensor indices = at::empty({3 * assignments + 3 * max_tiles}, h.options().dtype(at::kInt));
    int32_t* position_of = indices.mutable_data_ptr<int32_t>();
    int32_t* sorted_token = position_of + assignments;
    int32_t* sorted_expert = sorted_token + assignments;
    int32_t* tiles = sorted_expert + assignments;
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_dependent(
        sort_assignments_kernel,
        1,
        kSortThreads,
        2 * experts * sizeof(int32_t),
        stream,
        top_experts.const_data_ptr<int64_t>(),
        static_cast<int>(assignments),
        static_cast<int>(top_k),
        static_cast<int>(experts),
        static_cast<int>(max_tiles),
        position_of,
        sorted_token,
        sorted_expert,
        tiles
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    at::Tensor activations = at::empty({assignments, intermediate}, h.options());
    at::Tensor expert_out = at::empty({assignments, hidden}, h.options());
    if (h.scalar_type() == at::kFloat) {
        launch_dependent(
            expert_rows_scalar_kernel<true>,
            element_blocks(assignments * intermediate),
            kElementThreads,
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
            kElementThreads,
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
                expert_out
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
                expert_out
            );
        }
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, h.scalar_type(), "routed_experts_cuda", [&] {
        launch_dependent(
            combine_experts_kernel<scalar_t>,
            element_blocks(tokens * hidden),
            kElementThreads,
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

}  // namespace pulsarlm
