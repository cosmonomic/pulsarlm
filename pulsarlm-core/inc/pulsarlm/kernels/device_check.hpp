#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>

#include <initializer_list>

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

}  // namespace pulsarlm
