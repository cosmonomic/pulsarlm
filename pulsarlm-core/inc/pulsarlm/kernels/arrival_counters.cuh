#pragma once

#include <cstdint>

// Counters by which the CTAs sharing a piece of work find the last of them to finish it, so that it can finish the work
// alone within the same kernel.

namespace pulsarlm {

// count int32 counters of the current stream, zero between the kernels that use them, each kernel waiting for the
// kernels ahead of it (wait_for_prior_kernels) before it touches them. Allocated on a stream's first call, which must
// not be within a CUDA graph capture, and never freed.
int32_t* arrival_counters(int64_t count);

// Whether this CTA is the last of arrivals CTAs to reach counter, every thread of the CTA calling it once. The global
// writes the CTAs made before arriving are visible to the last, which returns counter to zero.
__device__ __forceinline__ bool last_to_arrive(int32_t* counter, int arrivals) {
    __shared__ bool last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        last = atomicAdd(counter, 1) == arrivals - 1;
        if (last) {
            *counter = 0;
        }
    }
    __syncthreads();
    if (last) {
        __threadfence();
    }
    return last;
}

}  // namespace pulsarlm
