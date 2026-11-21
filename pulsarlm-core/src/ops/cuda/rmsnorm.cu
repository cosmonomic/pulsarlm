#include "pulsarlm/ops/cuda/dependent_launch.cuh"
#include "pulsarlm/ops/cuda/device.hpp"
#include "pulsarlm/ops/normalization.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>

// RMSNorm over the last dimension:
//   y = x / sqrt(mean(x^2) + eps) * weight
//
// One block per row, block-stride reduction.

namespace pulsarlm::ops::cuda {
namespace kernel {

template <typename scalar_t, typename weight_t>
__global__ void rmsnorm(
    const scalar_t* __restrict__ x,
    const weight_t* __restrict__ weight,
    scalar_t* __restrict__ y,
    int64_t dim,
    double eps
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    const int64_t row = blockIdx.x;
    const scalar_t* xr = x + row * dim;
    scalar_t* yr = y + row * dim;

    float local = 0.0f;
    for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = static_cast<float>(xr[i]);
        local += v * v;
    }

    // blockDim.x is a power of two (the launcher rounds it up), so the tree
    // reduction covers every thread.
    extern __shared__ float shared[];
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float inv_rms = rsqrtf(shared[0] / static_cast<float>(dim) + static_cast<float>(eps));
    for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
        yr[i] = static_cast<scalar_t>(static_cast<float>(xr[i]) * inv_rms * static_cast<float>(weight[i]));
    }
}

}  // namespace kernel

at::Tensor rmsnorm(const at::Tensor& x, const at::Tensor& weight, double eps) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "rmsnorm: inputs must be CUDA tensors");
    check_on_current_device("rmsnorm", {x, weight});
    TORCH_CHECK(
        weight.scalar_type() == x.scalar_type() || weight.scalar_type() == at::kFloat,
        "rmsnorm: the weight's dtype must be x's or float32"
    );
    TORCH_CHECK(
        weight.dim() == 1 && weight.size(0) == x.size(-1),
        "rmsnorm: weight must be [dim] matching the last axis of x"
    );
    if (x.numel() == 0) {
        return at::empty_like(x);
    }

    auto xc = x.contiguous();
    auto wc = weight.contiguous();
    auto y = at::empty_like(xc);

    const int64_t dim = xc.size(-1);
    const int64_t rows = xc.numel() / dim;
    // The kernel's tree reduction requires a power-of-two thread count.
    const int64_t max_threads = std::min<int64_t>(1024, dim);
    int threads = 1;
    while (threads < max_threads) {
        threads <<= 1;
    }
    const size_t smem = static_cast<size_t>(threads) * sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, xc.scalar_type(), "rmsnorm", [&] {
        if (wc.scalar_type() == xc.scalar_type()) {
            launch_dependent(
                kernel::rmsnorm<scalar_t, scalar_t>,
                rows,
                threads,
                smem,
                stream,
                xc.data_ptr<scalar_t>(),
                wc.data_ptr<scalar_t>(),
                y.data_ptr<scalar_t>(),
                dim,
                eps
            );
        } else {
            launch_dependent(
                kernel::rmsnorm<scalar_t, float>,
                rows,
                threads,
                smem,
                stream,
                xc.data_ptr<scalar_t>(),
                wc.data_ptr<float>(),
                y.data_ptr<scalar_t>(),
                dim,
                eps
            );
        }
    });
    return y;
}

}  // namespace pulsarlm::ops::cuda
