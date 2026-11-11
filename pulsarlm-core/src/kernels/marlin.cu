#include "dequant_gemm_common.cuh"

#include "pulsarlm/kernels/ops.hpp"

#include "mma.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <type_traits>

// Marlin-style dequant-GEMM: y = x @ dequant(W)^T over weights repacked once into the m16n8k16 B fragment order and
// dequantized in registers; no dequantized weight is written to global or shared memory.
//
// Layout. For the 16-wide K subchunk at k0, lane L of a warp computing an 8-wide N subtile holds column L / 4 of the
// subtile at K offsets k0 + 2 (L % 4) + {0, 1} in b[0] and k0 + 8 + 2 (L % 4) + {0, 1} in b[1].
// weight_format::words_per_lane int32 words hold what a lane reads over a 32-wide K tile. Words are ordered by column
// block (tile_n columns), K tile, N subtile, word, then lane, so a warp reads 32 consecutive words and any K chunk of
// whole K tiles is one contiguous run per column block. N pads to tile_n and K to 32 with values that decode to 0.
//
// Scales. scale_marlin[g, b] multiplies K group g (group_size columns) of N block b (block_n rows). group_size is a
// multiple of 16, so one scale covers a lane's four values of a subchunk.

namespace pulsarlm {
namespace {

constexpr int word_span_k = 32;

struct tile_sizes {
    int m;
    int n;
    int k;
};

struct pipeline {
    int stages;
    int n_warps;
};

// tile.n is part of the weight layout, so the repack and every launch share it.
constexpr int layout_tile_n = 64;
constexpr int launch_tile_k = 64;
constexpr pipeline launch_pipeline{.stages = 3, .n_warps = 4};

__host__ __device__ __forceinline__ int64_t round_up(int64_t value, int64_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

__host__ __device__ __forceinline__ int64_t ceil_div(int64_t value, int64_t divisor) {
    return (value + divisor - 1) / divisor;
}

template <typename scalar_type> constexpr bool bf16 = std::is_same_v<scalar_type, at::BFloat16>;

template <typename scalar_type> using pair_type = std::conditional_t<bf16<scalar_type>, __nv_bfloat162, __half2>;

template <typename scalar_type> __device__ __forceinline__ pair_type<scalar_type> as_pair(uint32_t bits) {
    pair_type<scalar_type> pair;
    memcpy(&pair, &bits, 4);
    return pair;
}

template <typename scalar_type> __device__ __forceinline__ uint32_t as_bits(pair_type<scalar_type> pair) {
    uint32_t bits;
    memcpy(&bits, &pair, 4);
    return bits;
}

template <typename scalar_type> __device__ __forceinline__ pair_type<scalar_type> scale_pair(float scale) {
    if constexpr (bf16<scalar_type>) {
        return __bfloat162bfloat162(__float2bfloat16(scale));
    } else {
        return __half2half2(__float2half(scale));
    }
}

// Signed int4, 8 per int32 along K, least significant nibble first ([N, K/8]). A lane's word holds its values at K
// offsets 2 pair + {0, 8, 16, 24} in nibbles 0-3 and those plus 1 in nibbles 4-7, pair = lane % 4; the subchunk at
// offset 16 is the word shifted right by 8. Nibbles are stored as value ^ 8, which the decode bit trick needs.
struct int4 {
    using source_type = int32_t;
    static constexpr int words_per_lane = 1;

    __device__ static uint32_t
    encode_word(const int32_t* __restrict__ weight, int row, int k_base, int pair, int word, int n, int k) {
        const int offsets[8] = {
            2 * pair,
            8 + 2 * pair,
            16 + 2 * pair,
            24 + 2 * pair,
            2 * pair + 1,
            9 + 2 * pair,
            17 + 2 * pair,
            25 + 2 * pair,
        };
        uint32_t encoded = 0;
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            const int column = k_base + offsets[p];
            uint32_t nibble = 8;
            if (row < n && column < k) {
                const int32_t packed = weight[static_cast<int64_t>(row) * (k / 8) + column / 8];
                nibble = ((static_cast<uint32_t>(packed) >> (4 * (column % 8))) & 0xF) ^ 8;
            }
            encoded |= nibble << (4 * p);
        }
        return encoded;
    }

    __device__ static uint32_t subchunk_word(const int32_t* lane_words, int subchunk) {
        return static_cast<uint32_t>(lane_words[0]) >> (8 * subchunk);
    }

    template <typename scalar_type> __device__ static void decode(uint32_t b[2], uint32_t word, float scale) {
        using pair = pair_type<scalar_type>;
        const pair scales = scale_pair<scalar_type>(scale);
        if constexpr (bf16<scalar_type>) {
            // 0x4300 is 128.0: a nibble n in the low mantissa bits reads as 128 + n, so subtracting 136 gives n - 8.
            // bf16's 7 mantissa bits can't hold the nibble at bit 4, so it is shifted down first.
            const uint32_t low = (word & 0x000f000f) | 0x43004300;
            const uint32_t high = ((word >> 4) & 0x000f000f) | 0x43004300;
            b[0] = as_bits<scalar_type>(
                __hmul2(__hsub2(as_pair<scalar_type>(low), as_pair<scalar_type>(0x43084308)), scales)
            );
            b[1] = as_bits<scalar_type>(
                __hmul2(__hsub2(as_pair<scalar_type>(high), as_pair<scalar_type>(0x43084308)), scales)
            );
        } else {
            // 0x6400 is 1024.0: the nibble at bit 0 reads as 1024 + n, the one at bit 4 as 1024 + 16 n, which a fma by
            // 1/16 (0x2c00) and -72 (0xd480) turns into n - 8.
            const uint32_t low = (word & 0x000f000f) | 0x64006400;
            const uint32_t high = (word & 0x00f000f0) | 0x64006400;
            const pair values_low = __hsub2(as_pair<scalar_type>(low), as_pair<scalar_type>(0x64086408));
            const pair values_high = __hfma2(
                as_pair<scalar_type>(high),
                as_pair<scalar_type>(0x2c002c00),
                as_pair<scalar_type>(0xd480d480)
            );
            b[0] = as_bits<scalar_type>(__hmul2(values_low, scales));
            b[1] = as_bits<scalar_type>(__hmul2(values_high, scales));
        }
    }
};

// e4m3 bytes ([N, K]). Word w of a lane holds the subchunk at K offset 16 w: its bytes at K offsets 2 pair + {0, 1, 8,
// 9}, pair = lane % 4.
struct fp8 {
    using source_type = uint8_t;
    static constexpr int words_per_lane = 2;

    __device__ static uint32_t
    encode_word(const uint8_t* __restrict__ weight, int row, int k_base, int pair, int word, int n, int k) {
        const int offsets[4] = {0, 1, 8, 9};
        uint32_t encoded = 0;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int column = k_base + 16 * word + 2 * pair + offsets[p];
            const uint32_t byte = (row < n && column < k) ? weight[static_cast<int64_t>(row) * k + column] : 0;
            encoded |= byte << (8 * p);
        }
        return encoded;
    }

    __device__ static uint32_t subchunk_word(const int32_t* lane_words, int subchunk) {
        return static_cast<uint32_t>(lane_words[32 * subchunk]);
    }

    // Sign, exponent and mantissa move into a 16-bit float's fields unchanged, which divides the value by 2^(bias -
    // 7): 2^120 for bf16, 2^8 for fp16. Multiplying by that power of two (0x7b80, 0x5c00) is exact, so the weight
    // reaches the scale multiply exactly, subnormals included. The NaN codes 0x7f and 0xff are not handled.
    template <typename scalar_type> __device__ static void decode(uint32_t b[2], uint32_t word, float scale) {
        constexpr int shift = bf16<scalar_type> ? 4 : 1;
        constexpr uint32_t exponent_bias = bf16<scalar_type> ? 0x7b807b80 : 0x5c005c00;
        const pair_type<scalar_type> scales = scale_pair<scalar_type>(scale);
        const uint32_t pairs[2] = {__byte_perm(word, 0, 0x1404), __byte_perm(word, 0, 0x3424)};
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const uint32_t bits = (pairs[i] & 0x80008000) | ((pairs[i] & 0x7f007f00) >> shift);
            const pair_type<scalar_type> value = __hmul2(
                as_pair<scalar_type>(bits),
                as_pair<scalar_type>(exponent_bias)
            );
            b[i] = as_bits<scalar_type>(__hmul2(value, scales));
        }
    }
};

template <class weight_format, int tile_n>
__global__ void repack_weight_kernel(
    const typename weight_format::source_type* __restrict__ weight,
    int32_t* __restrict__ weight_marlin,
    int n,
    int k,
    int k_tiles,
    int64_t total
) {
    constexpr int n_subtiles = tile_n / 8;
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) {
        return;
    }
    int64_t rest = index;
    const int lane = static_cast<int>(rest % 32);
    rest /= 32;
    const int word = static_cast<int>(rest % weight_format::words_per_lane);
    rest /= weight_format::words_per_lane;
    const int subtile = static_cast<int>(rest % n_subtiles);
    rest /= n_subtiles;
    const int k_tile = static_cast<int>(rest % k_tiles);
    const int column_block = static_cast<int>(rest / k_tiles);

    const int row = (column_block * n_subtiles + subtile) * 8 + lane / 4;
    weight_marlin[index] = static_cast<int32_t>(
        weight_format::encode_word(weight, row, k_tile * word_span_k, lane % 4, word, n, k)
    );
}

__global__ void
transpose_scale_kernel(const float* __restrict__ scale, float* __restrict__ scale_marlin, int rows, int groups) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<int64_t>(rows) * groups) {
        return;
    }
    const int row = static_cast<int>(index / groups);
    const int group = static_cast<int>(index % groups);
    scale_marlin[static_cast<int64_t>(group) * rows + row] = scale[index];
}

template <int n_threads>
__device__ __forceinline__ void copy_weight_chunk(
    int32_t* __restrict__ tile,
    const int32_t* __restrict__ weight,
    int64_t first_word,
    int words,
    int thread
) {
    for (int word = thread * 4; word < words; word += n_threads * 4) {
        __pipeline_memcpy_async(&tile[word], &weight[first_word + word], sizeof(uint4));
    }
}

template <class weight_format, tile_sizes tile, pipeline pipe, typename scalar_type, bool split>
__global__ void __launch_bounds__(pipe.n_warps * 32) marlin_kernel(
    const scalar_type* __restrict__ x,
    const int32_t* __restrict__ weight,
    const float* __restrict__ scale,
    scalar_type* __restrict__ y,
    float* __restrict__ y_partial,
    int m,
    int n,
    int k,
    int k_tiles,
    int scale_rows,
    int block_n,
    int group_size,
    int chunks_per_split
) {
    constexpr int tile_m = tile.m, tile_n = tile.n, tile_k = tile.k;
    constexpr int stages = pipe.stages, n_warps = pipe.n_warps;
    constexpr int n_threads = n_warps * 32;
    constexpr int row_tiles = tile_m / 16;
    constexpr int n_subtiles = tile_n / 8;
    constexpr int subtiles_per_warp = n_subtiles / n_warps;
    constexpr int tiles_per_chunk = tile_k / word_span_k;
    constexpr int words_per_tile = n_subtiles * weight_format::words_per_lane * 32;
    static_assert(tile_m % 16 == 0 && n_subtiles % n_warps == 0 && tile_k % word_span_k == 0 && stages >= 2);

    const int thread = threadIdx.x;
    const int warp = thread / 32;
    const int lane = thread % 32;
    const int column_block = blockIdx.x;
    const int n0 = column_block * tile_n;
    const int m0 = blockIdx.y * tile_m;

    const int n_chunks = static_cast<int>(ceil_div(k_tiles, tiles_per_chunk));
    const int split_index = split ? static_cast<int>(blockIdx.z) : 0;
    const int chunk_begin = split ? split_index * chunks_per_split : 0;
    const int chunk_end = split ? min(chunk_begin + chunks_per_split, n_chunks) : n_chunks;
    const int local_chunks = chunk_end - chunk_begin;

    __shared__ alignas(16) scalar_type x_tiles[stages][tile_m * tile_k];
    __shared__ alignas(16) int32_t weight_tiles[stages][tiles_per_chunk * words_per_tile];

    const int group = lane / 4;
    const int quadrant = lane / 8, quadrant_row = lane % 8;
    const int a_row = (quadrant % 2) * 8 + quadrant_row, a_column = (quadrant / 2) * 8;

    float acc[subtiles_per_warp][row_tiles][4] = {};

    auto chunk_tiles = [&](int chunk) { return min(tiles_per_chunk, k_tiles - chunk * tiles_per_chunk); };
    auto fetch = [&](int chunk, int stage) {
        const int64_t first_word = (static_cast<int64_t>(column_block) * k_tiles + chunk * tiles_per_chunk) *
            words_per_tile;
        copy_weight_chunk<n_threads>(
            weight_tiles[stage],
            weight,
            first_word,
            chunk_tiles(chunk) * words_per_tile,
            thread
        );
        dequant_gemm::load_x_chunk<tile_m, tile_k, n_threads>(x_tiles[stage], x, m0, chunk * tile_k, m, k, thread);
    };

#pragma unroll
    for (int stage = 0; stage < stages - 1; ++stage) {
        if (stage < local_chunks) {
            fetch(chunk_begin + stage, stage);
        }
        __pipeline_commit();
    }

    for (int local_chunk = 0; local_chunk < local_chunks; ++local_chunk) {
        const int chunk = chunk_begin + local_chunk;
        const int stage = local_chunk % stages;
        __pipeline_wait_prior(stages - 2);
        __syncthreads();
        const int next = local_chunk + stages - 1;
        if (next < local_chunks) {
            fetch(chunk_begin + next, next % stages);
        }
        __pipeline_commit();

        const int tiles = chunk_tiles(chunk);
#pragma unroll
        for (int k_tile = 0; k_tile < tiles_per_chunk; ++k_tile) {
            if (k_tile >= tiles) {
                break;
            }
#pragma unroll
            for (int subchunk = 0; subchunk < 2; ++subchunk) {
                const int chunk_column = k_tile * word_span_k + subchunk * 16;
                const int k16 = chunk * tile_k + chunk_column;
                uint32_t a[row_tiles][4];
#pragma unroll
                for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
                    mma::ldmatrix_x4(
                        a[row_tile],
                        &x_tiles[stage][(row_tile * 16 + a_row) * tile_k + chunk_column + a_column]
                    );
                }
#pragma unroll
                for (int i = 0; i < subtiles_per_warp; ++i) {
                    const int subtile = warp + i * n_warps;
                    const int32_t* lane_words = &weight_tiles[stage]
                                                             [(k_tile * n_subtiles + subtile) *
                                                                  weight_format::words_per_lane * 32 +
                                                              lane];
                    const int column = n0 + subtile * 8 + group;
                    const float s = (column < n && k16 < k)
                        ? scale[static_cast<int64_t>(k16 / group_size) * scale_rows + column / block_n]
                        : 0.0f;
                    uint32_t b[2];
                    weight_format::template decode<scalar_type>(
                        b,
                        weight_format::subchunk_word(lane_words, subchunk),
                        s
                    );
#pragma unroll
                    for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
                        mma::mma_m16n8k16<bf16<scalar_type>>(acc[i][row_tile], a[row_tile], b, acc[i][row_tile]);
                    }
                }
            }
        }
    }

    const int64_t partial_offset = static_cast<int64_t>(split_index) * m * n;
    dequant_gemm::store_acc_tile<scalar_type, split, subtiles_per_warp, row_tiles, n_warps>(
        y,
        y_partial,
        partial_offset,
        acc,
        m0,
        n0,
        m,
        n,
        warp,
        lane
    );
}

int64_t marlin_words(int64_t n, int64_t k, int words_per_lane) {
    return round_up(n, layout_tile_n) / 8 * ceil_div(k, word_span_k) * words_per_lane * 32;
}

template <class weight_format>
std::tuple<at::Tensor, at::Tensor> repack(
    const char* op,
    const at::Tensor& weight,
    int64_t n,
    int64_t k,
    const at::Tensor& weight_scale,
    int64_t block_n,
    int64_t group_size
) {
    TORCH_CHECK(weight.is_cuda() && weight_scale.is_cuda(), op, ": inputs must be CUDA tensors");
    TORCH_CHECK(block_n > 0, op, ": block_n must be positive");
    TORCH_CHECK(group_size > 0 && group_size % 16 == 0, op, ": group_size must be a positive multiple of 16");
    TORCH_CHECK(k % group_size == 0, op, ": K must be a multiple of group_size");
    const int64_t scale_rows = ceil_div(n, block_n);
    const int64_t groups = k / group_size;
    TORCH_CHECK(
        weight_scale.dim() == 2 && weight_scale.size(0) == scale_rows && weight_scale.size(1) == groups,
        op,
        ": weight_scale must be [ceil(N / block_n), K / group_size]"
    );

    const at::Tensor source = weight.contiguous();
    const at::Tensor scale = weight_scale.to(at::kFloat).contiguous();
    const int64_t total = marlin_words(n, k, weight_format::words_per_lane);
    at::Tensor weight_marlin = at::empty({total}, weight.options().dtype(at::kInt));
    at::Tensor scale_marlin = at::empty({groups, scale_rows}, scale.options());

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int threads = 256;
    if (total > 0) {
        repack_weight_kernel<weight_format, layout_tile_n>
            <<<static_cast<unsigned>(ceil_div(total, threads)), threads, 0, stream>>>(
                static_cast<const typename weight_format::source_type*>(source.const_data_ptr()),
                weight_marlin.data_ptr<int32_t>(),
                static_cast<int>(n),
                static_cast<int>(k),
                static_cast<int>(ceil_div(k, word_span_k)),
                total
            );
    }
    if (scale.numel() > 0) {
        transpose_scale_kernel<<<static_cast<unsigned>(ceil_div(scale.numel(), threads)), threads, 0, stream>>>(
            scale.data_ptr<float>(),
            scale_marlin.data_ptr<float>(),
            static_cast<int>(scale_rows),
            static_cast<int>(groups)
        );
    }
    return {weight_marlin, scale_marlin};
}

template <class weight_format, int tile_m, typename scalar_type, bool split>
void launch(
    const dim3& grid,
    cudaStream_t stream,
    const at::Tensor& x,
    const at::Tensor& weight_marlin,
    const at::Tensor& scale_marlin,
    at::Tensor& y,
    float* y_partial,
    int n,
    int block_n,
    int group_size,
    int chunks_per_split
) {
    const int k = static_cast<int>(x.size(1));
    constexpr tile_sizes tile{.m = tile_m, .n = layout_tile_n, .k = launch_tile_k};
    marlin_kernel<weight_format, tile, launch_pipeline, scalar_type, split>
        <<<grid, launch_pipeline.n_warps * 32, 0, stream>>>(
            x.data_ptr<scalar_type>(),
            weight_marlin.data_ptr<int32_t>(),
            scale_marlin.data_ptr<float>(),
            y.data_ptr<scalar_type>(),
            y_partial,
            static_cast<int>(x.size(0)),
            n,
            k,
            static_cast<int>(ceil_div(k, word_span_k)),
            static_cast<int>(scale_marlin.size(1)),
            block_n,
            group_size,
            chunks_per_split
        );
}

template <class weight_format, int tile_m, typename scalar_type>
void run(
    const at::Tensor& x,
    const at::Tensor& weight_marlin,
    const at::Tensor& scale_marlin,
    at::Tensor& y,
    int n,
    int block_n,
    int group_size
) {
    const int64_t m = x.size(0);
    const int n_chunks = static_cast<int>(ceil_div(ceil_div(x.size(1), word_span_k), launch_tile_k / word_span_k));
    const auto grid_x = static_cast<unsigned>(round_up(n, layout_tile_n) / layout_tile_n);
    const auto grid_y = static_cast<unsigned>(ceil_div(m, tile_m));
    const int splits = dequant_gemm::split_k_count(static_cast<int64_t>(grid_x) * grid_y, n_chunks);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (splits > 1) {
        const int chunks_per_split = static_cast<int>(ceil_div(n_chunks, splits));
        at::Tensor y_partial = at::empty({splits, m, static_cast<int64_t>(n)}, x.options().dtype(at::kFloat));
        launch<weight_format, tile_m, scalar_type, true>(
            dim3(grid_x, grid_y, static_cast<unsigned>(splits)),
            stream,
            x,
            weight_marlin,
            scale_marlin,
            y,
            y_partial.data_ptr<float>(),
            n,
            block_n,
            group_size,
            chunks_per_split
        );
        dequant_gemm::launch_combine(
            y.data_ptr<scalar_type>(),
            y_partial.data_ptr<float>(),
            static_cast<int>(m),
            n,
            splits,
            stream
        );
    } else {
        launch<weight_format, tile_m, scalar_type, false>(
            dim3(grid_x, grid_y, 1),
            stream,
            x,
            weight_marlin,
            scale_marlin,
            y,
            nullptr,
            n,
            block_n,
            group_size,
            n_chunks
        );
    }
}

template <class weight_format>
at::Tensor gemm(
    const char* op,
    const at::Tensor& x,
    const at::Tensor& weight_marlin,
    const at::Tensor& scale_marlin,
    int64_t n,
    int64_t block_n,
    int64_t group_size
) {
    TORCH_CHECK(
        x.is_cuda() && weight_marlin.is_cuda() && scale_marlin.is_cuda(),
        op,
        ": all inputs must be CUDA tensors"
    );
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16, op, ": x must be fp16 or bf16");
    TORCH_CHECK(x.dim() == 2 && x.is_contiguous(), op, ": x must be a contiguous [M, K]");
    TORCH_CHECK(weight_marlin.scalar_type() == at::kInt, op, ": weight_marlin must be int32");
    TORCH_CHECK(scale_marlin.scalar_type() == at::kFloat, op, ": scale_marlin must be fp32");
    TORCH_CHECK(block_n > 0, op, ": block_n must be positive");
    TORCH_CHECK(group_size > 0 && group_size % 16 == 0, op, ": group_size must be a positive multiple of 16");
    const int64_t k = x.size(1);
    TORCH_CHECK(k % group_size == 0, op, ": K must be a multiple of group_size");
    TORCH_CHECK(
        scale_marlin.dim() == 2 && scale_marlin.size(0) == k / group_size &&
            scale_marlin.size(1) == ceil_div(n, block_n),
        op,
        ": scale_marlin must be [K / group_size, ceil(N / block_n)]"
    );
    TORCH_CHECK(
        weight_marlin.numel() == marlin_words(n, k, weight_format::words_per_lane),
        op,
        ": weight_marlin has the wrong size for this shape"
    );

    at::Tensor y = at::empty({x.size(0), n}, x.options());
    if (x.size(0) == 0 || n == 0) {
        return y;
    }
    const at::Tensor weight_words = weight_marlin.contiguous();
    const at::Tensor scales = scale_marlin.contiguous();
    const bool small_m = x.size(0) <= 16;
    const auto n32 = static_cast<int>(n);
    const auto block_n32 = static_cast<int>(block_n);
    const auto group_size32 = static_cast<int>(group_size);
    if (x.scalar_type() == at::kBFloat16) {
        if (small_m) {
            run<weight_format, 16, at::BFloat16>(x, weight_words, scales, y, n32, block_n32, group_size32);
        } else {
            run<weight_format, 64, at::BFloat16>(x, weight_words, scales, y, n32, block_n32, group_size32);
        }
    } else {
        if (small_m) {
            run<weight_format, 16, at::Half>(x, weight_words, scales, y, n32, block_n32, group_size32);
        } else {
            run<weight_format, 64, at::Half>(x, weight_words, scales, y, n32, block_n32, group_size32);
        }
    }
    return y;
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> repack_int4_marlin_cuda(
    const at::Tensor& weight_packed,
    const at::Tensor& weight_scale,
    int64_t block_n,
    int64_t group_size
) {
    TORCH_CHECK(
        weight_packed.scalar_type() == at::kInt && weight_packed.dim() == 2,
        "repack_int4_marlin: weight_packed must be int32 [N, K/8]"
    );
    return repack<int4>(
        "repack_int4_marlin",
        weight_packed,
        weight_packed.size(0),
        weight_packed.size(1) * 8,
        weight_scale,
        block_n,
        group_size
    );
}

std::tuple<at::Tensor, at::Tensor>
repack_fp8_marlin_cuda(const at::Tensor& weight, const at::Tensor& weight_scale, int64_t block_n, int64_t group_size) {
    TORCH_CHECK(
        weight.scalar_type() == at::kFloat8_e4m3fn && weight.dim() == 2,
        "repack_fp8_marlin: weight must be float8_e4m3fn [N, K]"
    );
    return repack<fp8>("repack_fp8_marlin", weight, weight.size(0), weight.size(1), weight_scale, block_n, group_size);
}

at::Tensor gemm_int4_marlin_cuda(
    const at::Tensor& x,
    const at::Tensor& weight_marlin,
    const at::Tensor& scale_marlin,
    int64_t n,
    int64_t block_n,
    int64_t group_size
) {
    return gemm<int4>("gemm_int4_marlin", x, weight_marlin, scale_marlin, n, block_n, group_size);
}

at::Tensor gemm_fp8_marlin_cuda(
    const at::Tensor& x,
    const at::Tensor& weight_marlin,
    const at::Tensor& scale_marlin,
    int64_t n,
    int64_t block_n,
    int64_t group_size
) {
    return gemm<fp8>("gemm_fp8_marlin", x, weight_marlin, scale_marlin, n, block_n, group_size);
}

}  // namespace pulsarlm
