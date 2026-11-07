#!/bin/bash
# cibuildwheel's before-all in a manylinux_2_34 container: CUDA 13.2, gcc-toolset-15 and xmake.
set -euxo pipefail

dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/$(uname -m | sed s/aarch64/sbsa/)/cuda-rhel9.repo
dnf install -y \
    cuda-nvcc-13-2 cuda-cudart-devel-13-2 cuda-nvrtc-devel-13-2 cuda-nvtx-13-2 cuda-cupti-13-2 \
    cuda-profiler-api-13-2 libcublas-devel-13-2 libcufft-devel-13-2 libcurand-devel-13-2 libcusolver-devel-13-2 \
    libcusparse-devel-13-2 libnvjitlink-devel-13-2 openssl-devel gcc-toolset-15-gcc-c++

export XMAKE_ROOT=y
curl -fsSL https://xmake.io/shget.text | bash

git config --global --add safe.directory '*'
