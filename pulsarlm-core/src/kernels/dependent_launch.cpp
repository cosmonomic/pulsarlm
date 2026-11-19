#include "pulsarlm/kernels/dependent_launch.cuh"

#include <ATen/cuda/CUDAContext.h>

namespace pulsarlm {

bool supports_dependent_launch(int device) {
    return at::cuda::getDeviceProperties(device)->major >= 9;
}

}  // namespace pulsarlm
