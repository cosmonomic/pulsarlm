#pragma once

#include <ATen/ATen.h>

namespace pulsarlm {

// An uninitialized tensor in page-locked host memory.
inline at::Tensor pinned(at::IntArrayRef sizes, at::ScalarType dtype) {
    return at::empty(sizes, at::TensorOptions().dtype(dtype).pinned_memory(true));
}

}  // namespace pulsarlm
