#include "pulsarlm/ops.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>

namespace pulsarlm {
namespace {

constexpr int kCopyThreads = 256;

// One CTA per segment.
__global__ void copy_segments_kernel(
    unsigned char* __restrict__ device,
    unsigned char* __restrict__ host,
    const int64_t* __restrict__ segments,
    bool to_host
) {
    const int64_t* segment = segments + 3 * static_cast<int64_t>(blockIdx.x);
    unsigned char* device_bytes = device + segment[0];
    unsigned char* host_bytes = host + segment[1];
    const int64_t bytes = segment[2];
    const unsigned char* from = to_host ? device_bytes : host_bytes;
    unsigned char* to = to_host ? host_bytes : device_bytes;
    const bool vectors = ((reinterpret_cast<uintptr_t>(from) | reinterpret_cast<uintptr_t>(to) | bytes) & 15) == 0;
    if (vectors) {
        const auto* from_vectors = reinterpret_cast<const uint4*>(from);
        auto* to_vectors = reinterpret_cast<uint4*>(to);
        for (int64_t i = threadIdx.x; i < bytes / 16; i += kCopyThreads) {
            to_vectors[i] = from_vectors[i];
        }
    } else {
        for (int64_t i = threadIdx.x; i < bytes; i += kCopyThreads) {
            to[i] = from[i];
        }
    }
}

}  // namespace

void copy_segments_cuda(const at::Tensor& device, const at::Tensor& host, const at::Tensor& segments, bool to_host) {
    TORCH_CHECK(
        device.is_cuda() && device.scalar_type() == at::kByte && device.is_contiguous(),
        "copy_segments: device must be a contiguous CUDA byte tensor"
    );
    TORCH_CHECK(
        host.is_pinned() && host.scalar_type() == at::kByte && host.is_contiguous(),
        "copy_segments: host must be a pinned contiguous byte tensor"
    );
    TORCH_CHECK(
        segments.is_pinned() && segments.scalar_type() == at::kLong && segments.dim() == 2 && segments.size(1) == 3 &&
            segments.is_contiguous(),
        "copy_segments: segments must be pinned contiguous int64 [n, 3]"
    );
    if (segments.size(0) == 0) {
        return;
    }
    copy_segments_kernel<<<static_cast<int>(segments.size(0)), kCopyThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        device.data_ptr<unsigned char>(),
        host.data_ptr<unsigned char>(),
        segments.data_ptr<int64_t>(),
        to_host
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace pulsarlm
