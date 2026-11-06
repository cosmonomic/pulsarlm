#pragma once

#include <ATen/cuda/CUDAContext.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

// The parts int4_gemm.cu and marlin.cu share: activation staging, the m16n8k16 epilogue store and the split-K
// partial reduction. Each strategy's tile sizes reach these helpers as template parameters.

namespace pulsarlm {
namespace dequant_gemm {

// The non-split path stores into y in y's dtype; the split path stores fp32 into this CTA's partial slice at
// partial_offset, so the combine kernel sums fp32.
template <typename scalar_type, bool split>
__device__ __forceinline__ void store_out(
    scalar_type* __restrict__ y,
    float* __restrict__ y_partial,
    int64_t partial_offset,
    int row,
    int column,
    int n,
    float value
) {
    const int64_t index = static_cast<int64_t>(row) * n + column;
    if constexpr (split) {
        y_partial[partial_offset + index] = value;
    } else {
        y[index] = static_cast<scalar_type>(value);
    }
}

// Loads the x tile [tile_m][tile_k] at (m0, k0), zero past m rows or k columns.
template <int tile_m, int tile_k, int n_threads, typename scalar_type>
__device__ __forceinline__ void load_x_chunk(
    scalar_type* __restrict__ x_tile,
    const scalar_type* __restrict__ x,
    int m0,
    int k0,
    int m,
    int k,
    int thread
) {
#pragma unroll
    for (int i = thread; i < tile_m * tile_k; i += n_threads) {
        const int row = m0 + i / tile_k, column = k0 + i % tile_k;
        x_tile[i] = (row < m && column < k) ? x[static_cast<int64_t>(row) * k + column] : static_cast<scalar_type>(0);
    }
}

// Stores a warp's subtiles_per_warp x row_tiles m16n8k16 accumulators. Lane L holds rows {L/4, L/4 + 8} and
// columns {2(L%4), 2(L%4) + 1} of each 16x8 tile; warp w owns N subtiles w + i * n_warps.
template <typename scalar_type, bool split, int subtiles_per_warp, int row_tiles, int n_warps>
__device__ __forceinline__ void store_acc_tile(
    scalar_type* __restrict__ y,
    float* __restrict__ y_partial,
    int64_t partial_offset,
    const float (&acc)[subtiles_per_warp][row_tiles][4],
    int m0,
    int n0,
    int m,
    int n,
    int warp,
    int lane
) {
    const int group = lane >> 2, pair = lane & 3;
#pragma unroll
    for (int subtile = 0; subtile < subtiles_per_warp; ++subtile) {
        const int column = n0 + (warp + subtile * n_warps) * 8 + pair * 2;
#pragma unroll
        for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
            const int row0 = m0 + row_tile * 16 + group, row1 = row0 + 8;
            const float* a = acc[subtile][row_tile];
            if (row0 < m) {
                if (column < n) {
                    store_out<scalar_type, split>(y, y_partial, partial_offset, row0, column, n, a[0]);
                }
                if (column + 1 < n) {
                    store_out<scalar_type, split>(y, y_partial, partial_offset, row0, column + 1, n, a[1]);
                }
            }
            if (row1 < m) {
                if (column < n) {
                    store_out<scalar_type, split>(y, y_partial, partial_offset, row1, column, n, a[2]);
                }
                if (column + 1 < n) {
                    store_out<scalar_type, split>(y, y_partial, partial_offset, row1, column + 1, n, a[3]);
                }
            }
        }
    }
}

constexpr int combine_threads = 256;

// y_partial is [splits, m, n] fp32; one thread sums each (m, n) in split order, which keeps the sum deterministic.
template <typename scalar_type>
__global__ void
combine_partials_kernel(scalar_type* __restrict__ y, const float* __restrict__ y_partial, int64_t total, int splits) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }
    const float* partial = y_partial + index;
    float sum = 0.0f;
#pragma unroll 1
    for (int i = 0; i < splits; ++i) {
        sum += partial[static_cast<int64_t>(i) * total];
    }
    y[index] = static_cast<scalar_type>(sum);
}

template <typename scalar_type>
void launch_combine(scalar_type* y, const float* y_partial, int m, int n, int splits, cudaStream_t stream) {
    const int64_t total = static_cast<int64_t>(m) * n;
    const int64_t blocks = (total + combine_threads - 1) / combine_threads;
    combine_partials_kernel<scalar_type>
        <<<static_cast<unsigned>(blocks), combine_threads, 0, stream>>>(y, y_partial, total, splits);
}

// Past ctas_per_sm CTAs per SM, more K splits only add combine work.
constexpr int ctas_per_sm = 4;

// The K splits that fill the SMs when the non-split grid has base_ctas CTAs over n_chunks K chunks.
inline int split_k_count(int64_t base_ctas, int n_chunks) {
    const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int64_t splits = (ctas_per_sm * static_cast<int64_t>(sm_count) + base_ctas - 1) / base_ctas;
    splits = std::min<int64_t>(std::max<int64_t>(splits, 1), n_chunks);
    return static_cast<int>(splits);
}

}  // namespace dequant_gemm
}  // namespace pulsarlm
