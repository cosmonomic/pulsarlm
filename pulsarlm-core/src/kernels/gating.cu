#include "pulsarlm/kernels/dependent_launch.cuh"
#include "pulsarlm/kernels/ops.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

namespace pulsarlm {
namespace {

constexpr int kThreads = 256;

template <typename scalar_t> __device__ __forceinline__ float rounded(float value) {
    return static_cast<float>(static_cast<scalar_t>(value));
}

template <typename scalar_t> __device__ __forceinline__ float rounded_sigmoid(scalar_t logit) {
    return rounded<scalar_t>(1.0f / (1.0f + expf(-static_cast<float>(logit))));
}

// One thread per element of x: x plus sigmoid(the row's gate logit) times y, each operation rounded as ATen rounds it.
template <typename scalar_t>
__global__ void sigmoid_gated_add_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ gate_logits,
    int64_t gate_row_stride,
    const scalar_t* __restrict__ y,
    scalar_t* __restrict__ out,
    int64_t elements,
    int64_t columns
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const float gate = rounded_sigmoid(gate_logits[index / columns * gate_row_stride]);
    const float gated = rounded<scalar_t>(__fmul_rn(gate, static_cast<float>(y[index])));
    out[index] = static_cast<scalar_t>(static_cast<float>(x[index]) + gated);
}

// One thread per element of y [rows, middle, columns]: y times sigmoid(the gate logit at its position), rounded as ATen
// rounds them.
template <typename scalar_t>
__global__ void sigmoid_gated_kernel(
    const scalar_t* __restrict__ y,
    const scalar_t* __restrict__ gate_logits,
    int64_t gate_row_stride,
    int64_t gate_middle_stride,
    scalar_t* __restrict__ out,
    int64_t elements,
    int64_t middle,
    int64_t columns
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const int64_t column = index % columns;
    const int64_t position = index / columns;
    const int64_t gate_index = position / middle * gate_row_stride + position % middle * gate_middle_stride + column;
    out[index] = static_cast<scalar_t>(
        __fmul_rn(static_cast<float>(y[index]), rounded_sigmoid(gate_logits[gate_index]))
    );
}

// One thread per output element: silu of the gate half, rounded, times the up half, rounded.
template <typename scalar_t>
__global__ void silu_mul_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ y, int64_t tokens, int64_t n) {
    wait_for_prior_kernels();
    allow_next_kernel();
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

unsigned blocks_for(int64_t elements) {
    return static_cast<unsigned>((elements + kThreads - 1) / kThreads);
}

}  // namespace

at::Tensor sigmoid_gated_add_cuda(const at::Tensor& x, const at::Tensor& gate_logits, const at::Tensor& y) {
    TORCH_CHECK(
        x.is_cuda() && x.dim() == 2 && x.is_contiguous() && y.sizes() == x.sizes() && y.is_contiguous() &&
            y.scalar_type() == x.scalar_type(),
        "sigmoid_gated_add: x and y must be contiguous CUDA [rows, columns] of one dtype"
    );
    TORCH_CHECK(
        gate_logits.dim() == 2 && gate_logits.size(0) == x.size(0) && gate_logits.size(1) == 1 &&
            gate_logits.scalar_type() == x.scalar_type() && gate_logits.device() == x.device(),
        "sigmoid_gated_add: gate_logits must be [rows, 1] in x's dtype on x's device"
    );
    at::Tensor out = at::empty_like(x);
    if (x.numel() == 0) {
        return out;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "sigmoid_gated_add_cuda", [&] {
        launch_dependent(
            sigmoid_gated_add_kernel<scalar_t>,
            blocks_for(x.numel()),
            kThreads,
            0,
            stream,
            x.const_data_ptr<scalar_t>(),
            gate_logits.const_data_ptr<scalar_t>(),
            gate_logits.stride(0),
            y.const_data_ptr<scalar_t>(),
            out.mutable_data_ptr<scalar_t>(),
            x.numel(),
            x.size(1)
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return out;
}

at::Tensor sigmoid_gated_cuda(const at::Tensor& y, const at::Tensor& gate_logits) {
    TORCH_CHECK(
        y.is_cuda() && (y.dim() == 2 || y.dim() == 3) && y.is_contiguous(),
        "sigmoid_gated: y must be contiguous CUDA [rows, columns] or [rows, middle, columns]"
    );
    TORCH_CHECK(
        gate_logits.sizes() == y.sizes() && gate_logits.stride(-1) == 1 &&
            gate_logits.scalar_type() == y.scalar_type() && gate_logits.device() == y.device(),
        "sigmoid_gated: gate_logits must be shaped as y in its dtype, its last axis contiguous"
    );
    at::Tensor out = at::empty_like(y);
    if (y.numel() == 0) {
        return out;
    }
    const int64_t middle = y.dim() == 3 ? y.size(1) : 1;
    const int64_t middle_stride = y.dim() == 3 ? gate_logits.stride(1) : 0;
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, y.scalar_type(), "sigmoid_gated_cuda", [&] {
        launch_dependent(
            sigmoid_gated_kernel<scalar_t>,
            blocks_for(y.numel()),
            kThreads,
            0,
            stream,
            y.const_data_ptr<scalar_t>(),
            gate_logits.const_data_ptr<scalar_t>(),
            gate_logits.stride(0),
            middle_stride,
            out.mutable_data_ptr<scalar_t>(),
            y.numel(),
            middle,
            y.size(-1)
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return out;
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
        launch_dependent(
            silu_mul_kernel<scalar_t>,
            blocks_for(y.numel()),
            kThreads,
            0,
            stream,
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
