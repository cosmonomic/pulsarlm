#include "pulsarlm/engine/p2p_all_reduce.hpp"

#include "pulsarlm/kernels/p2p_all_reduce.hpp"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace pulsarlm {

namespace {

std::size_t flag_count(std::size_t ranks) {
    return 2 * p2p_all_reduce_max_blocks * ranks;
}

// Enables access from device to peer unless they are one device; false when it is impossible.
bool reach(const at::Device& device, const at::Device& peer) {
    if (device == peer) {
        return true;
    }
    int possible = 0;
    C10_CUDA_CHECK(cudaDeviceCanAccessPeer(&possible, device.index(), peer.index()));
    if (possible == 0) {
        return false;
    }
    c10::cuda::CUDAGuard guard(device);
    cudaError_t enabled = cudaDeviceEnablePeerAccess(peer.index(), 0);
    if (enabled == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
        return true;
    }
    C10_CUDA_CHECK(enabled);
    return true;
}

}  // namespace

p2p_all_reduce::p2p_all_reduce(std::span<const at::Device> devices, std::size_t max_bytes)
: devices(devices.begin(), devices.end()),
  max_bytes(max_bytes) {
    for (const at::Device& device : this->devices) {
        c10::cuda::CUDAGuard guard(device);
        void* buffer = nullptr;
        void* rank_flags = nullptr;
        void* rank_counters = nullptr;
        C10_CUDA_CHECK(cudaMalloc(&buffer, max_bytes));
        buffers.push_back(static_cast<std::byte*>(buffer));
        C10_CUDA_CHECK(cudaMalloc(&rank_flags, flag_count(devices.size()) * sizeof(std::uint64_t)));
        flags.push_back(static_cast<std::uint64_t*>(rank_flags));
        C10_CUDA_CHECK(cudaMalloc(&rank_counters, p2p_all_reduce_max_blocks * sizeof(std::uint64_t)));
        counters.push_back(static_cast<std::uint64_t*>(rank_counters));
        C10_CUDA_CHECK(cudaMemset(rank_flags, 0, flag_count(devices.size()) * sizeof(std::uint64_t)));
        C10_CUDA_CHECK(cudaMemset(rank_counters, 0, p2p_all_reduce_max_blocks * sizeof(std::uint64_t)));
        C10_CUDA_CHECK(cudaDeviceSynchronize());
    }
}

std::unique_ptr<p2p_all_reduce> p2p_all_reduce::make(std::span<const at::Device> devices, std::size_t max_bytes) {
    if (devices.empty() || devices.size() > p2p_all_reduce_max_ranks) {
        return nullptr;
    }
    for (const at::Device& device : devices) {
        for (const at::Device& peer : devices) {
            if (!reach(device, peer)) {
                return nullptr;
            }
        }
    }
    return std::unique_ptr<p2p_all_reduce>(new p2p_all_reduce(devices, max_bytes));
}

p2p_all_reduce::~p2p_all_reduce() {
    for (std::size_t rank = 0; rank < devices.size(); rank++) {
        c10::cuda::CUDAGuard guard(devices[rank]);
        cudaFree(buffers[rank]);
        cudaFree(flags[rank]);
        cudaFree(counters[rank]);
    }
}

bool p2p_all_reduce::fits(const at::Tensor& x) const noexcept {
    const at::ScalarType dtype = x.scalar_type();
    return x.is_cuda() && x.is_contiguous() && x.nbytes() <= max_bytes && x.nbytes() % 16 == 0 &&
        reinterpret_cast<std::uintptr_t>(x.data_ptr()) % 16 == 0 &&
        (dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16);
}

void p2p_all_reduce::all_reduce(std::size_t rank, at::Tensor& x) {
    p2p_all_reduce_cuda(x, rank, buffers, flags, counters);
}

}  // namespace pulsarlm
