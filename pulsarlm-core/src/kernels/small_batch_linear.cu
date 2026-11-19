#include "pulsarlm/kernels/dependent_launch.cuh"
#include "pulsarlm/kernels/device.hpp"
#include "pulsarlm/kernels/gating.hpp"
#include "pulsarlm/kernels/small_batch_linear.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>
#include <type_traits>

namespace pulsarlm {
namespace {

constexpr int cta_threads = 256;
constexpr int warps = cta_threads / 32;
constexpr int64_t max_tokens = 4;
// The weight rows a CTA takes; its warps split their columns.
constexpr int rows = 4;

template <typename scalar_t> __device__ __forceinline__ float rounded(float value) {
    return static_cast<float>(static_cast<scalar_t>(value));
}

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

// Adds to sums[t] the dot product of 8 consecutive values of row with those of token t's row of x.
template <typename scalar_t, int tokens>
__device__ __forceinline__ void
add_dots(const scalar_t* __restrict__ row, const scalar_t* __restrict__ x, int64_t x_stride, float (&sums)[tokens]) {
    const uint4 loaded = *reinterpret_cast<const uint4*>(row);
    const auto* weights = reinterpret_cast<const scalar_t*>(&loaded);
#pragma unroll
    for (int t = 0; t < tokens; t++) {
        const uint4 x_vector = *reinterpret_cast<const uint4*>(x + t * x_stride);
        const auto* values = reinterpret_cast<const scalar_t*>(&x_vector);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            sums[t] += static_cast<float>(weights[i]) * static_cast<float>(values[i]);
        }
    }
}

// Each CTA takes rows output columns. With gated, each column c also takes weight row half + c, and the CTA writes
// silu(x . row c) * (x . row half + c), each rounded as at::linear and silu_mul_cuda round; otherwise x . row c. The
// warps split the columns of x and their sums meet in shared memory.
template <typename scalar_t, int tokens, bool gated>
__global__ void __launch_bounds__(cta_threads) small_batch_linear_kernel(
    const scalar_t* __restrict__ x,
    int64_t x_stride,
    const scalar_t* __restrict__ weight,
    int64_t weight_stride,
    int64_t columns,
    int64_t half,
    int k,
    scalar_t* __restrict__ out
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    constexpr int parts = gated ? 2 : 1;
    __shared__ float warp_sums[warps][parts][rows][tokens];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int64_t first = static_cast<int64_t>(blockIdx.x) * rows;
    float sums[parts][rows][tokens] = {};
    for (int column = threadIdx.x * 8; column < k; column += cta_threads * 8) {
#pragma unroll
        for (int part = 0; part < parts; part++) {
#pragma unroll
            for (int r = 0; r < rows; r++) {
                if (first + r < columns) {
                    const int64_t row = part * half + first + r;
                    add_dots<scalar_t, tokens>(
                        weight + row * weight_stride + column,
                        x + column,
                        x_stride,
                        sums[part][r]
                    );
                }
            }
        }
    }
#pragma unroll
    for (int part = 0; part < parts; part++) {
#pragma unroll
        for (int r = 0; r < rows; r++) {
#pragma unroll
            for (int t = 0; t < tokens; t++) {
                const float sum = warp_sum(sums[part][r][t]);
                if (lane == 0) {
                    warp_sums[warp][part][r][t] = sum;
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < rows * tokens) {
        const int r = threadIdx.x / tokens;
        const int t = threadIdx.x % tokens;
        if (first + r < columns) {
            float dots[parts];
#pragma unroll
            for (int part = 0; part < parts; part++) {
                dots[part] = 0.0f;
                for (int w = 0; w < warps; w++) {
                    dots[part] += warp_sums[w][part][r][t];
                }
            }
            float value = dots[0];
            if constexpr (gated) {
                const float gate = rounded<scalar_t>(dots[0]);
                const float up = rounded<scalar_t>(dots[1]);
                value = __fmul_rn(rounded<scalar_t>(gate / (1.0f + expf(-gate))), up);
            }
            out[t * columns + first + r] = static_cast<scalar_t>(value);
        }
    }
}

template <typename Body> void dispatch_tokens(int64_t tokens, Body body) {
    switch (tokens) {
    case 1:
        body(std::integral_constant<int, 1>{});
        break;
    case 2:
        body(std::integral_constant<int, 2>{});
        break;
    case 3:
        body(std::integral_constant<int, 3>{});
        break;
    case 4:
        body(std::integral_constant<int, 4>{});
        break;
    default:
        TORCH_CHECK(false, "small_batch_linear: the kernel takes 1 to 4 tokens, got ", tokens);
    }
}

// Whether x and weight take the matrix-vector kernel.
bool fits(const at::Tensor& x, const at::Tensor& weight) {
    return x.is_cuda() && x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= max_tokens &&
        (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf) && weight.scalar_type() == x.scalar_type() &&
        weight.dim() == 2 && weight.size(1) == x.size(1) && x.size(1) % 8 == 0 && vector_rows(x, x.stride(0)) &&
        vector_rows(weight, weight.stride(0));
}

template <bool gated> at::Tensor launch(const at::Tensor& x, const at::Tensor& weight) {
    const int64_t tokens = x.size(0);
    const int64_t columns = gated ? weight.size(0) / 2 : weight.size(0);
    at::Tensor out = at::empty({tokens, columns}, x.options());
    if (columns == 0) {
        return out;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_REDUCED_FLOATING_TYPES(x.scalar_type(), "small_batch_linear_cuda", [&] {
        dispatch_tokens(tokens, [&](auto tokens_tag) {
            launch_dependent(
                small_batch_linear_kernel<scalar_t, decltype(tokens_tag)::value, gated>,
                static_cast<unsigned>((columns + rows - 1) / rows),
                cta_threads,
                0,
                stream,
                x.const_data_ptr<scalar_t>(),
                x.stride(0),
                weight.const_data_ptr<scalar_t>(),
                weight.stride(0),
                columns,
                columns,
                static_cast<int>(x.size(1)),
                out.mutable_data_ptr<scalar_t>()
            );
        });
    });
    return out;
}

}  // namespace

at::Tensor small_batch_linear_cuda(const at::Tensor& x, const at::Tensor& weight) {
    check_on_current_device("small_batch_linear", {x, weight});
    if (!fits(x, weight)) {
        return at::linear(x, weight);
    }
    return launch<false>(x, weight);
}

at::Tensor small_batch_gated_linear_cuda(const at::Tensor& x, const at::Tensor& gate_up) {
    check_on_current_device("small_batch_gated_linear", {x, gate_up});
    TORCH_CHECK(
        gate_up.dim() == 2 && gate_up.size(0) % 2 == 0,
        "small_batch_gated_linear: gate_up must be [2 * out, in], the gate rows first"
    );
    if (!fits(x, gate_up)) {
        return silu_mul_cuda(at::linear(x, gate_up));
    }
    return launch<true>(x, gate_up);
}

}  // namespace pulsarlm
