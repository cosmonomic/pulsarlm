#include "pulsarlm/ops.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>

namespace pulsarlm {
namespace {

constexpr int kCopyThreads = 256;

struct segment {
    int64_t device_offset;
    int64_t host_offset;
    int64_t bytes;
};

__device__ __forceinline__ segment segment_of_block(const int64_t* segments) {
    const int64_t* row = segments + 3 * static_cast<int64_t>(blockIdx.x);
    return segment{row[0], row[1], row[2]};
}

__device__ __forceinline__ void copy_bytes(const unsigned char* from, unsigned char* to, int64_t bytes) {
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

// One CTA per segment.
__global__ void copy_segments_to_host_kernel(
    const unsigned char* __restrict__ device,
    unsigned char* __restrict__ host,
    const int64_t* __restrict__ segments
) {
    const segment s = segment_of_block(segments);
    copy_bytes(device + s.device_offset, host + s.host_offset, s.bytes);
}

// One CTA per segment.
__global__ void copy_segments_from_host_kernel(
    unsigned char* __restrict__ device,
    const unsigned char* __restrict__ host,
    const int64_t* __restrict__ segments
) {
    const segment s = segment_of_block(segments);
    copy_bytes(host + s.host_offset, device + s.device_offset, s.bytes);
}

void check_segments(const at::Tensor& device, const at::Tensor& host, const at::Tensor& segments) {
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
}

}  // namespace

void copy_segments_to_host_cuda(const at::Tensor& device, const at::Tensor& host, const at::Tensor& segments) {
    check_segments(device, host, segments);
    if (segments.size(0) == 0) {
        return;
    }
    const auto blocks = static_cast<int>(segments.size(0));
    copy_segments_to_host_kernel<<<blocks, kCopyThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        device.data_ptr<unsigned char>(),
        host.data_ptr<unsigned char>(),
        segments.data_ptr<int64_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void copy_segments_from_host_cuda(const at::Tensor& device, const at::Tensor& host, const at::Tensor& segments) {
    check_segments(device, host, segments);
    if (segments.size(0) == 0) {
        return;
    }
    const auto blocks = static_cast<int>(segments.size(0));
    copy_segments_from_host_kernel<<<blocks, kCopyThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        device.data_ptr<unsigned char>(),
        host.data_ptr<unsigned char>(),
        segments.data_ptr<int64_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace pulsarlm
