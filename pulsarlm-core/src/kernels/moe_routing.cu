#include "pulsarlm/kernels/ops.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <tuple>

namespace pulsarlm {
namespace {

constexpr int kThreads = 256;
constexpr int64_t kMaxExperts = 4096;
constexpr int64_t kMaxTopK = 64;

template <typename scalar_t> __device__ __forceinline__ float rounded(float value) {
    return static_cast<float>(static_cast<scalar_t>(value));
}

// The block's maximum of value, the lowest index among equal values; every thread receives it.
__device__ void block_argmax(float& value, int& index, float* values, int* indices) {
    values[threadIdx.x] = value;
    indices[threadIdx.x] = index;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float other = values[threadIdx.x + stride];
            const int other_index = indices[threadIdx.x + stride];
            if (other > values[threadIdx.x] || (other == values[threadIdx.x] && other_index < indices[threadIdx.x])) {
                values[threadIdx.x] = other;
                indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    value = values[0];
    index = indices[0];
    __syncthreads();
}

__device__ float block_sum(float value, float* values) {
    values[threadIdx.x] = value;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            values[threadIdx.x] += values[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float sum = values[0];
    __syncthreads();
    return sum;
}

// One CTA per token: the softmax of its logits in fp32, its top k experts by probability, the lower expert first among
// equal probabilities, in descending order, and their probabilities divided by their sum.
template <typename scalar_t>
__global__ void route_top_k_kernel(
    const scalar_t* __restrict__ logits,
    int64_t row_stride,
    int experts,
    int k,
    float* __restrict__ weights,
    int64_t* __restrict__ chosen
) {
    extern __shared__ float probabilities[];
    __shared__ float values[kThreads];
    __shared__ int indices[kThreads];
    __shared__ float top[kMaxTopK];
    const scalar_t* row = logits + static_cast<int64_t>(blockIdx.x) * row_stride;

    float max_logit = -INFINITY;
    int unused = 0;
    for (int e = threadIdx.x; e < experts; e += kThreads) {
        max_logit = fmaxf(max_logit, static_cast<float>(row[e]));
    }
    block_argmax(max_logit, unused, values, indices);
    float local = 0.0f;
    for (int e = threadIdx.x; e < experts; e += kThreads) {
        const float shifted = expf(static_cast<float>(row[e]) - max_logit);
        probabilities[e] = shifted;
        local += shifted;
    }
    const float total = block_sum(local, values);
    for (int e = threadIdx.x; e < experts; e += kThreads) {
        probabilities[e] /= total;
    }
    __syncthreads();

    for (int j = 0; j < k; j++) {
        float best = -INFINITY;
        int best_index = experts;
        for (int e = threadIdx.x; e < experts; e += kThreads) {
            if (probabilities[e] > best || (probabilities[e] == best && e < best_index)) {
                best = probabilities[e];
                best_index = e;
            }
        }
        block_argmax(best, best_index, values, indices);
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

// One thread per output element: silu of the gate half, rounded, times the up half, rounded.
template <typename scalar_t>
__global__ void silu_mul_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ y, int64_t tokens, int64_t n) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= tokens * n) {
        return;
    }
    const int64_t token = index / n;
    const int64_t i = index % n;
    const float gate = static_cast<float>(x[token * 2 * n + i]);
    const float activated = rounded<scalar_t>(gate / (1.0f + expf(-gate)));
    y[index] = static_cast<scalar_t>(__fmul_rn(activated, static_cast<float>(x[token * 2 * n + n + i])));
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> route_top_k_cuda(const at::Tensor& logits, int64_t k) {
    TORCH_CHECK(
        logits.is_cuda() && logits.dim() == 2 && logits.stride(1) == 1,
        "route_top_k: logits must be CUDA [tokens, experts], each row contiguous"
    );
    const int64_t tokens = logits.size(0);
    const int64_t experts = logits.size(1);
    TORCH_CHECK(
        experts >= 1 && experts <= kMaxExperts && k >= 1 && k <= experts && k <= kMaxTopK,
        "route_top_k: experts must be at most ",
        kMaxExperts,
        " and k between 1 and the experts, at most ",
        kMaxTopK
    );
    at::Tensor weights = at::empty({tokens, k}, logits.options().dtype(at::kFloat));
    at::Tensor chosen = at::empty({tokens, k}, logits.options().dtype(at::kLong));
    if (tokens == 0) {
        return {weights, chosen};
    }
    const size_t shared = static_cast<size_t>(experts) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, logits.scalar_type(), "route_top_k_cuda", [&] {
        route_top_k_kernel<scalar_t><<<static_cast<unsigned>(tokens), kThreads, shared, stream>>>(
            logits.const_data_ptr<scalar_t>(),
            logits.stride(0),
            static_cast<int>(experts),
            static_cast<int>(k),
            weights.mutable_data_ptr<float>(),
            chosen.mutable_data_ptr<int64_t>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return {weights, chosen};
}

at::Tensor silu_mul_cuda(const at::Tensor& x) {
    TORCH_CHECK(
        x.is_cuda() && x.dim() == 2 && x.is_contiguous() && x.size(1) % 2 == 0,
        "silu_mul: x must be contiguous CUDA [tokens, 2 * n]"
    );
    const int64_t tokens = x.size(0);
    const int64_t n = x.size(1) / 2;
    at::Tensor y = at::empty({tokens, n}, x.options());
    if (y.numel() == 0) {
        return y;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "silu_mul_cuda", [&] {
        silu_mul_kernel<scalar_t><<<static_cast<unsigned>((y.numel() + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
            x.const_data_ptr<scalar_t>(),
            y.mutable_data_ptr<scalar_t>(),
            tokens,
            n
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return y;
}

}  // namespace pulsarlm
