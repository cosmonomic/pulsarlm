#pragma once

#include <ATen/core/Tensor.h>

namespace pulsarlm::ops::cuda {

// Copy segment i of segments, pinned int64 [n, 3] of {device byte offset, host byte offset, bytes}, between the
// contiguous device byte tensor and the pinned host byte tensor. The kernel reads the pinned segments, so they stay
// alive until the stream has run it.
void copy_segments_to_host(const at::Tensor& device, const at::Tensor& host, const at::Tensor& segments);
void copy_segments_from_host(const at::Tensor& device, const at::Tensor& host, const at::Tensor& segments);

}  // namespace pulsarlm::ops::cuda
