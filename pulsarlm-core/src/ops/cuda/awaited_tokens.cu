#include "pulsarlm/ops/cuda/dependent_launch.cuh"
#include "pulsarlm/ops/cuda/device.hpp"
#include "pulsarlm/ops/sampling.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cassert>
#include <cstdint>

namespace pulsarlm::ops::cuda {
namespace {

constexpr int cta_threads = 128;

}  // namespace

namespace kernel {

__global__ void fill_awaited_tokens(int64_t* __restrict__ tokens, int64_t rows, const int64_t* __restrict__ samples) {
    wait_for_prior_kernels();
    allow_next_kernel();
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row < rows && tokens[row] < 0) {
        tokens[row] = samples[-tokens[row] - 1];
    }
}

}  // namespace kernel

void fill_awaited_tokens(const at::Tensor& tokens, const at::Tensor& samples) {
    check_on_current_device("fill_awaited_tokens", {tokens, samples});
    TORCH_CHECK(
        tokens.is_cuda() && tokens.scalar_type() == at::kLong && tokens.dim() == 1 && tokens.is_contiguous(),
        "fill_awaited_tokens: tokens must be contiguous int64 CUDA [rows]"
    );
    TORCH_CHECK(
        samples.is_cuda() && samples.scalar_type() == at::kLong && samples.is_contiguous() &&
            samples.device() == tokens.device(),
        "fill_awaited_tokens: samples must be contiguous int64 on tokens' device"
    );
    assert(samples.dim() == 1);
    const int64_t rows = tokens.size(0);
    if (rows == 0) {
        return;
    }
    launch_dependent(
        kernel::fill_awaited_tokens,
        cta_count(rows, cta_threads),
        cta_threads,
        0,
        at::cuda::getCurrentCUDAStream(),
        tokens.mutable_data_ptr<int64_t>(),
        rows,
        samples.const_data_ptr<int64_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace pulsarlm::ops::cuda
