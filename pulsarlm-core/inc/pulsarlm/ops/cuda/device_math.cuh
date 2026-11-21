#pragma once

namespace pulsarlm::ops::cuda::kernel {

// value rounded to scalar_t, as ATen rounds an operation's fp32 result to its dtype.
template <typename scalar_t> __device__ __forceinline__ float rounded(float value) {
    return static_cast<float>(static_cast<scalar_t>(value));
}

__device__ __forceinline__ float sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

// The sum of value over the 32 lanes of a warp, every lane active.
__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

}  // namespace pulsarlm::ops::cuda::kernel
