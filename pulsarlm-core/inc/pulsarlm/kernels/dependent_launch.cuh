#pragma once

#include "pulsarlm/kernels/device.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

// Programmatic dependent launch (compute capability 9.0 and later): a kernel launched with launch_dependent may begin
// while the kernels ahead of it in its stream are still running, so its launch overlaps their tail. Every kernel
// launched this way calls wait_for_prior_kernels before it reads or writes global memory.

namespace pulsarlm {

// Blocks until the kernels ahead in the stream have completed and their writes are visible.
__device__ __forceinline__ void wait_for_prior_kernels() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

// Lets the next kernel in the stream begin launching before this one completes.
__device__ __forceinline__ void allow_next_kernel() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

template <typename... Params, typename... Args>
void launch_dependent(
    void (*kernel)(Params...),
    dim3 grid,
    dim3 block,
    std::size_t shared_bytes,
    cudaStream_t stream,
    Args&&... args
) {
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = shared_bytes;
    config.stream = stream;
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;
    if (supports_dependent_launch(at::cuda::current_device())) {
        config.attrs = &attribute;
        config.numAttrs = 1;
    }
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, std::forward<Args>(args)...));
}

}  // namespace pulsarlm
