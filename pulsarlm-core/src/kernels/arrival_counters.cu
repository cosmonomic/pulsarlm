#include "pulsarlm/kernels/arrival_counters.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

#include <cuda_runtime.h>

#include <map>
#include <mutex>

namespace pulsarlm {

namespace {

constexpr int64_t kCounters = int64_t{1} << 16;

}  // namespace

int32_t* arrival_counters(int64_t count) {
    TORCH_CHECK(count <= kCounters, "arrival_counters: ", count, " counters requested, at most ", kCounters);
    static std::mutex guard;
    static auto* counters = new std::map<cudaStream_t, int32_t*>();
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const std::lock_guard<std::mutex> lock(guard);
    int32_t*& stream_counters = (*counters)[stream];
    if (stream_counters == nullptr) {
        C10_CUDA_CHECK(cudaMalloc(&stream_counters, kCounters * sizeof(int32_t)));
        C10_CUDA_CHECK(cudaMemset(stream_counters, 0, kCounters * sizeof(int32_t)));
    }
    return stream_counters;
}

}  // namespace pulsarlm
