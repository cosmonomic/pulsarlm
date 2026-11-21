#include "pulsarlm/ops/cuda/dependent_launch.cuh"
#include "pulsarlm/ops/cuda/device.hpp"
#include "pulsarlm/ops/page_copy.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>
#include <limits>

namespace pulsarlm::ops::cuda {
namespace {

constexpr int copy_threads = 256;

}  // namespace

namespace kernel {

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
        for (int64_t i = threadIdx.x; i < bytes / 16; i += copy_threads) {
            to_vectors[i] = from_vectors[i];
        }
    } else {
        for (int64_t i = threadIdx.x; i < bytes; i += copy_threads) {
            to[i] = from[i];
        }
    }
}

// One CTA per segment.
__global__ void copy_segments_to_host(
    const unsigned char* __restrict__ device,
    unsigned char* __restrict__ host,
    const int64_t* __restrict__ segments
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    const segment s = segment_of_block(segments);
    copy_bytes(device + s.device_offset, host + s.host_offset, s.bytes);
}

// One CTA per segment.
__global__ void copy_segments_from_host(
    unsigned char* __restrict__ device,
    const unsigned char* __restrict__ host,
    const int64_t* __restrict__ segments
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    const segment s = segment_of_block(segments);
    copy_bytes(host + s.host_offset, device + s.device_offset, s.bytes);
}

}  // namespace kernel

namespace {

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
    check_on_current_device("copy_segments", {device});
    TORCH_CHECK(
        segments.size(0) <= std::numeric_limits<int>::max(),
        "copy_segments: the segments must fit a 32-bit index"
    );
    const int64_t* rows = segments.const_data_ptr<int64_t>();
    for (int64_t i = 0; i < segments.size(0); i++) {
        const kernel::segment s{rows[3 * i], rows[3 * i + 1], rows[3 * i + 2]};
        TORCH_CHECK(
            s.bytes >= 0 && s.device_offset >= 0 && s.host_offset >= 0 && s.bytes <= device.numel() - s.device_offset &&
                s.bytes <= host.numel() - s.host_offset,
            "copy_segments: kernel::segment ",
            i,
            " must lie within the device and host tensors"
        );
    }
}

}  // namespace

void copy_segments_to_host(const at::Tensor& device, const at::Tensor& host, const at::Tensor& segments) {
    check_segments(device, host, segments);
    if (segments.size(0) == 0) {
        return;
    }
    const auto blocks = static_cast<int>(segments.size(0));
    launch_dependent(
        kernel::copy_segments_to_host,
        blocks,
        copy_threads,
        0,
        at::cuda::getCurrentCUDAStream(),
        device.data_ptr<unsigned char>(),
        host.data_ptr<unsigned char>(),
        segments.data_ptr<int64_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void copy_segments_from_host(const at::Tensor& device, const at::Tensor& host, const at::Tensor& segments) {
    check_segments(device, host, segments);
    if (segments.size(0) == 0) {
        return;
    }
    const auto blocks = static_cast<int>(segments.size(0));
    launch_dependent(
        kernel::copy_segments_from_host,
        blocks,
        copy_threads,
        0,
        at::cuda::getCurrentCUDAStream(),
        device.data_ptr<unsigned char>(),
        host.data_ptr<unsigned char>(),
        segments.data_ptr<int64_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace pulsarlm::ops::cuda
