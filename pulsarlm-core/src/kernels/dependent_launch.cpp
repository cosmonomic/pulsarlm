#include "pulsarlm/kernels/dependent_launch.cuh"

#include <ATen/cuda/CUDAContext.h>

#include <array>
#include <atomic>
#include <cstddef>

namespace pulsarlm {

bool supports_dependent_launch(int device) {
    // Per device, 0 until checked, then 1 without support and 2 with.
    static std::array<std::atomic<int>, 64> support{};
    std::atomic<int>& known = support.at(static_cast<std::size_t>(device));
    if (known == 0) {
        known = at::cuda::getDeviceProperties(device)->major >= 9 ? 2 : 1;
    }
    return known == 2;
}

}  // namespace pulsarlm
