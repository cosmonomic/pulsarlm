#include "pulsarlm/kernels/device.hpp"

#include <ATen/cuda/CUDAContext.h>

namespace pulsarlm {

int sm_count() {
    return at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
}

bool supports_dependent_launch(int device) {
    return at::cuda::getDeviceProperties(device)->major >= 9;
}

bool fits_shared_memory(std::size_t bytes) {
    return bytes <= at::cuda::getCurrentDeviceProperties()->sharedMemPerBlockOptin;
}

}  // namespace pulsarlm
