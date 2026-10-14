#!/bin/bash
# Runs in quay.io/pypa/manylinux_2_34_x86_64 or manylinux_2_34_aarch64 with the repository mounted at /io; writes the wheel to /io/dist.
set -euxo pipefail

dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/$(uname -m | sed s/aarch64/sbsa/)/cuda-rhel9.repo
dnf install -y \
    cuda-nvcc-13-4 cuda-cudart-devel-13-4 cuda-nvrtc-devel-13-4 cuda-nvtx-13-4 cuda-cupti-13-4 \
    cuda-profiler-api-13-4 libcublas-devel-13-4 libcufft-devel-13-4 libcurand-devel-13-4 libcusolver-devel-13-4 \
    libcusparse-devel-13-4 libnvjitlink-devel-13-4 openssl-devel gcc-toolset-15-gcc-c++
export PATH=/usr/local/cuda-13.4/bin:/opt/rh/gcc-toolset-15/root/usr/bin:$PATH

export XMAKE_ROOT=y
curl -fsSL https://xmake.io/shget.text | bash
export PATH=$HOME/.local/bin:$PATH

git config --global --add safe.directory /io
cd /io
PULSARLM_CUDA_ARCH=${PULSARLM_CUDA_ARCH:-sm_80,sm_86,sm_89,sm_90,sm_100,sm_120,compute_120} \
    uv build --wheel --package pulsarlm --out-dir /tmp/wheel

# torch and its CUDA runtime are loaded by `import torch` before the extension, so the wheel links them by soname.
auditwheel repair /tmp/wheel/*.whl --wheel-dir dist \
    --exclude 'libtorch*.so' --exclude 'libc10*.so' --exclude libcudart.so.13 --exclude libnvrtc.so.13
