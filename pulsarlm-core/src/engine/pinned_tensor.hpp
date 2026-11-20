#pragma once

#include <ATen/ATen.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace pulsarlm {

// An uninitialized tensor in page-locked host memory.
inline at::Tensor pinned(at::IntArrayRef sizes, at::ScalarType dtype) {
    return at::empty(sizes, at::TensorOptions().dtype(dtype).pinned_memory(true));
}

// values in page-locked host memory, [values.size()] in value_type's dtype.
template <typename value_type> at::Tensor pinned_values(const std::vector<value_type>& values) {
    at::Tensor t = pinned({static_cast<std::int64_t>(values.size())}, c10::CppTypeToScalarType<value_type>::value);
    std::ranges::copy(values, t.mutable_data_ptr<value_type>());
    return t;
}

}  // namespace pulsarlm
