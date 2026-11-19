#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace pulsarlm {

// Throws c10::Error unless every defined tensor of tensors is on the current CUDA device, whose current stream the
// ops launch on. An undefined tensor is an optional argument that was not passed.
inline void check_on_current_device(const char* op, std::initializer_list<at::Tensor> tensors) {
    for (const at::Tensor& tensor : tensors) {
        TORCH_CHECK(
            !tensor.defined() || (tensor.is_cuda() && tensor.get_device() == at::cuda::current_device()),
            op,
            ": every tensor must be on the current CUDA device"
        );
    }
}

// The streaming multiprocessors of the current CUDA device.
int sm_count();

// Whether device runs a kernel launched with launch_dependent before the kernels ahead of it in its stream complete.
bool supports_dependent_launch(int device);

// The shared memory a CTA may use when its kernel was not passed to allow_shared_memory.
constexpr std::size_t shared_memory_without_opt_in = 48 * 1024;

// Whether a CTA on the current CUDA device may use bytes of shared memory.
bool fits_shared_memory(std::size_t bytes);

// Raises the dynamic shared memory a CTA of kernel may use to bytes, which must pass fits_shared_memory. A kernel's
// static shared memory counts against shared_memory_without_opt_in too, so a kernel that has some is passed here even
// when bytes is below that limit.
template <typename kernel_type> void allow_shared_memory(kernel_type kernel, std::size_t bytes) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
}

constexpr std::int64_t int32_max = std::numeric_limits<std::int32_t>::max();

// The CTAs of cta_threads threads that hold thread_count threads.
constexpr unsigned cta_count(std::int64_t thread_count, int cta_threads) {
    return static_cast<unsigned>((thread_count + cta_threads - 1) / cta_threads);
}

// Whether the rows of t, row_stride 16-bit elements apart, are contiguous and start at multiples of 16 bytes.
inline bool vector_rows(const at::Tensor& t, std::int64_t row_stride) {
    return t.stride(-1) == 1 && row_stride % 8 == 0 && reinterpret_cast<std::uintptr_t>(t.const_data_ptr()) % 16 == 0;
}

}  // namespace pulsarlm
