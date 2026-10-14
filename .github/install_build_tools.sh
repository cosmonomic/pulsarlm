#!/bin/bash
# cibuildwheel's before-all in a manylinux_2_34 container: CUDA 13.4, gcc-toolset-15 and xmake.
set -euxo pipefail

dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/$(uname -m | sed s/aarch64/sbsa/)/cuda-rhel9.repo
dnf install -y \
    cuda-nvcc-13-4 cuda-cudart-devel-13-4 cuda-nvrtc-devel-13-4 cuda-nvtx-13-4 cuda-cupti-13-4 \
    cuda-profiler-api-13-4 libcublas-devel-13-4 libcufft-devel-13-4 libcurand-devel-13-4 libcusolver-devel-13-4 \
    libcusparse-devel-13-4 libnvjitlink-devel-13-4 openssl-devel gcc-toolset-15-gcc-c++

export XMAKE_ROOT=y
curl -fsSL https://xmake.io/shget.text | bash

git config --global --add safe.directory '*'
