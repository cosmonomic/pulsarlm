#include "pulsarlm/kernels/ops.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <tuple>

namespace pulsarlm {
namespace {

constexpr int kThreads = 256;

template <typename scalar_t> __device__ __forceinline__ float rounded(float value) {
    return static_cast<float>(static_cast<scalar_t>(value));
}

__device__ __forceinline__ float sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

// One CTA per (token, stream): the stream's RMSNorm over hidden in fp32, times weight, rounded once.
template <typename scalar_t>
__global__ void stream_rmsnorm_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ weight,
    scalar_t* __restrict__ y,
    int64_t streams,
    int64_t hidden,
    float eps
) {
    __shared__ float partial[kThreads];
    const int64_t offset = static_cast<int64_t>(blockIdx.x) * hidden;
    const int64_t stream_offset = static_cast<int64_t>(blockIdx.x) % streams * hidden;
    float squares = 0.0f;
    for (int64_t i = threadIdx.x; i < hidden; i += kThreads) {
        const float v = static_cast<float>(x[offset + i]);
        squares += v * v;
    }
    partial[threadIdx.x] = squares;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(partial[0] / static_cast<float>(hidden) + eps);
    for (int64_t i = threadIdx.x; i < hidden; i += kThreads) {
        y[offset + i] = static_cast<scalar_t>(static_cast<float>(x[offset + i]) * inv_rms * weight[stream_offset + i]);
    }
}

// One thread per (token, hidden index): the mean over streams of the rounded sigmoid(gate) times the stream, each
// product rounded; threads of the first hidden indices also write the token's injection weights.
template <typename scalar_t>
__global__ void hyper_connection_mix_kernel(
    const scalar_t* __restrict__ normed,
    const scalar_t* __restrict__ gate_logits,
    const scalar_t* __restrict__ inject_logits,
    int64_t inject_row_stride,
    scalar_t* __restrict__ input,
    scalar_t* __restrict__ injection_weights,
    int64_t tokens,
    int64_t streams,
    int64_t hidden
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= tokens * hidden) {
        return;
    }
    const int64_t token = index / hidden;
    const int64_t h = index % hidden;
    const int64_t row = token * streams * hidden;
    float sum = 0.0f;
    for (int64_t s = 0; s < streams; s++) {
        const int64_t i = row + s * hidden + h;
        const float gate = rounded<scalar_t>(sigmoid(static_cast<float>(gate_logits[i])));
        sum += rounded<scalar_t>(__fmul_rn(gate, static_cast<float>(normed[i])));
    }
    const float scale = 1.0f / static_cast<float>(streams);
    input[index] = static_cast<scalar_t>(sum * scale);
    if (h < streams) {
        const float logit = rounded<scalar_t>(static_cast<float>(inject_logits[token * inject_row_stride + h]) * scale);
        injection_weights[token * streams + h] = static_cast<scalar_t>(2.0f * rounded<scalar_t>(sigmoid(logit)));
    }
}

// One thread per element of x: x plus the rounded product of the block output and the stream's injection weight.
template <typename scalar_t>
__global__ void hyper_connection_inject_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ out,
    const scalar_t* __restrict__ injection_weights,
    scalar_t* __restrict__ y,
    int64_t elements,
    int64_t streams,
    int64_t hidden
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const int64_t token = index / (streams * hidden);
    const int64_t s = index / hidden % streams;
    const int64_t h = index % hidden;
    const float injected = rounded<scalar_t>(
        __fmul_rn(static_cast<float>(out[token * hidden + h]), static_cast<float>(injection_weights[token * streams + s]))
    );
    y[index] = static_cast<scalar_t>(static_cast<float>(x[index]) + injected);
}

unsigned blocks_for(int64_t threads) {
    return static_cast<unsigned>((threads + kThreads - 1) / kThreads);
}

}  // namespace

at::Tensor stream_rmsnorm_cuda(const at::Tensor& x, const at::Tensor& weight, int64_t n_streams, double eps) {
    TORCH_CHECK(
        x.is_cuda() && x.dim() == 2 && x.is_contiguous() && n_streams >= 1 && x.size(1) % n_streams == 0,
        "stream_rmsnorm: x must be contiguous CUDA [tokens, n_streams * hidden]"
    );
    TORCH_CHECK(
        weight.is_cuda() && weight.scalar_type() == at::kFloat && weight.dim() == 1 && weight.size(0) == x.size(1) &&
            weight.is_contiguous(),
        "stream_rmsnorm: weight must be contiguous CUDA fp32 [n_streams * hidden]"
    );
    at::Tensor y = at::empty_like(x);
    if (x.numel() == 0) {
        return y;
    }
    const int64_t hidden = x.size(1) / n_streams;
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "stream_rmsnorm_cuda", [&] {
        stream_rmsnorm_kernel<scalar_t><<<static_cast<unsigned>(x.size(0) * n_streams), kThreads, 0, stream>>>(
            x.const_data_ptr<scalar_t>(),
            weight.const_data_ptr<float>(),
            y.mutable_data_ptr<scalar_t>(),
            n_streams,
            hidden,
            static_cast<float>(eps)
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return y;
}

std::tuple<at::Tensor, at::Tensor> hyper_connection_mix_cuda(
    const at::Tensor& normed,
    const at::Tensor& gate_logits,
    const at::Tensor& inject_logits
) {
    TORCH_CHECK(
        normed.is_cuda() && normed.dim() == 2 && normed.is_contiguous(),
        "hyper_connection_mix: normed must be contiguous CUDA [tokens, hc * hidden]"
    );
    TORCH_CHECK(
        gate_logits.sizes() == normed.sizes() && gate_logits.scalar_type() == normed.scalar_type() &&
            gate_logits.is_contiguous(),
        "hyper_connection_mix: gate_logits must be contiguous and shaped as normed, in its dtype"
    );
    TORCH_CHECK(
        inject_logits.is_cuda() && inject_logits.dim() == 2 && inject_logits.size(0) == normed.size(0) &&
            inject_logits.scalar_type() == normed.scalar_type() && inject_logits.stride(1) == 1,
        "hyper_connection_mix: inject_logits must be CUDA [tokens, hc] in normed's dtype, each row contiguous"
    );
    const int64_t tokens = normed.size(0);
    const int64_t streams = inject_logits.size(1);
    TORCH_CHECK(
        streams >= 1 && normed.size(1) % streams == 0,
        "hyper_connection_mix: normed must hold hc whole streams"
    );
    const int64_t hidden = normed.size(1) / streams;
    TORCH_CHECK(streams <= hidden, "hyper_connection_mix: hc must be at most the hidden size");
    at::Tensor input = at::empty({tokens, hidden}, normed.options());
    at::Tensor injection_weights = at::empty({tokens, streams}, normed.options());
    if (tokens == 0) {
        return {input, injection_weights};
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, normed.scalar_type(), "hyper_connection_mix_cuda", [&] {
        hyper_connection_mix_kernel<scalar_t><<<blocks_for(tokens * hidden), kThreads, 0, stream>>>(
            normed.const_data_ptr<scalar_t>(),
            gate_logits.const_data_ptr<scalar_t>(),
            inject_logits.const_data_ptr<scalar_t>(),
            inject_logits.stride(0),
            input.mutable_data_ptr<scalar_t>(),
            injection_weights.mutable_data_ptr<scalar_t>(),
            tokens,
            streams,
            hidden
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return {input, injection_weights};
}

at::Tensor hyper_connection_inject_cuda(
    const at::Tensor& x,
    const at::Tensor& out,
    const at::Tensor& injection_weights
) {
    TORCH_CHECK(
        x.is_cuda() && x.dim() == 2 && x.is_contiguous(),
        "hyper_connection_inject: x must be contiguous CUDA [tokens, hc * hidden]"
    );
    TORCH_CHECK(
        out.dim() == 2 && out.size(0) == x.size(0) && out.scalar_type() == x.scalar_type() && out.is_contiguous(),
        "hyper_connection_inject: out must be contiguous [tokens, hidden] in x's dtype"
    );
    TORCH_CHECK(
        injection_weights.dim() == 2 && injection_weights.size(0) == x.size(0) &&
            injection_weights.scalar_type() == x.scalar_type() && injection_weights.is_contiguous() &&
            injection_weights.size(1) * out.size(1) == x.size(1),
        "hyper_connection_inject: injection_weights must be contiguous [tokens, hc] in x's dtype, hc * hidden x's width"
    );
    at::Tensor y = at::empty_like(x);
    if (x.numel() == 0) {
        return y;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "hyper_connection_inject_cuda", [&] {
        hyper_connection_inject_kernel<scalar_t><<<blocks_for(x.numel()), kThreads, 0, stream>>>(
            x.const_data_ptr<scalar_t>(),
            out.const_data_ptr<scalar_t>(),
            injection_weights.const_data_ptr<scalar_t>(),
            y.mutable_data_ptr<scalar_t>(),
            x.numel(),
            injection_weights.size(1),
            out.size(1)
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return y;
}

}  // namespace pulsarlm
