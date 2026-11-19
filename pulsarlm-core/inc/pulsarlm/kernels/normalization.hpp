#pragma once

#include <ATen/core/Tensor.h>

#include <cstdint>

namespace pulsarlm {

// RMSNorm over x's last axis, in fp32. weight is in x's dtype or fp32.
at::Tensor rmsnorm_cuda(const at::Tensor& x, const at::Tensor& weight, double eps);

// The RMSNorm of each of x's n_streams streams over its hidden entries, in fp32, times weight, in x's dtype. x is
// contiguous [tokens, n_streams * hidden], weight fp32 [n_streams * hidden].
at::Tensor stream_rmsnorm_cuda(const at::Tensor& x, const at::Tensor& weight, int64_t n_streams, double eps);

}  // namespace pulsarlm
