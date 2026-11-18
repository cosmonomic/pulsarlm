#include "pulsarlm/kernels/p2p_all_reduce.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>

namespace pulsarlm {
namespace {

constexpr int kThreads = 512;

struct rank_pointers {
    std::byte* buffers[p2p_all_reduce_max_ranks];
    std::uint64_t* flags[p2p_all_reduce_max_ranks];
};

__device__ __forceinline__ void store_release(std::uint64_t* address, std::uint64_t value) {
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(address), "l"(value) : "memory");
}

__device__ __forceinline__ std::uint64_t load_acquire(const std::uint64_t* address) {
    std::uint64_t value;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
    return value;
}

// Thread p below ranks sets flag (phase, block, rank) of rank p to epoch and waits for its own flag (phase, block,
// p) to reach it, so every rank's block has passed the point.
__device__ void meet(
    const rank_pointers& ranks,
    int rank_count,
    int rank,
    int phase,
    std::uint64_t epoch
) {
    __syncthreads();
    const int peer = static_cast<int>(threadIdx.x);
    if (peer < rank_count) {
        const int64_t slot = (static_cast<int64_t>(phase) * p2p_all_reduce_max_blocks + blockIdx.x) * rank_count;
        store_release(ranks.flags[peer] + slot + rank, epoch);
        while (load_acquire(ranks.flags[rank] + slot + peer) < epoch) {
        }
    }
    __syncthreads();
}

template <typename scalar_t> __device__ __forceinline__ float as_float(scalar_t value) {
    return static_cast<float>(value);
}

template <> __device__ __forceinline__ float as_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <> __device__ __forceinline__ float as_float(__half value) {
    return __half2float(value);
}

template <typename scalar_t> __device__ __forceinline__ scalar_t from_float(float value) {
    return static_cast<scalar_t>(value);
}

template <> __device__ __forceinline__ __nv_bfloat16 from_float(float value) {
    return __float2bfloat16_rn(value);
}

template <> __device__ __forceinline__ __half from_float(float value) {
    return __float2half_rn(value);
}

// Block b takes the 16-byte vectors [b * per_block, (b + 1) * per_block) of x: it copies them into its rank's
// buffer, meets the other ranks' block b, sums every rank's copy in rank order into x, and meets them again so that
// no rank's next call replaces its buffer while another still reads it.
template <typename scalar_t>
__global__ void __launch_bounds__(kThreads) p2p_all_reduce_kernel(
    scalar_t* __restrict__ x,
    int64_t vectors,
    int64_t per_block,
    rank_pointers ranks,
    int rank_count,
    int rank,
    std::uint64_t* __restrict__ counters
) {
    constexpr int kPerVector = 16 / sizeof(scalar_t);
    __shared__ std::uint64_t epoch;
    if (threadIdx.x == 0) {
        epoch = ++counters[blockIdx.x];
    }
    const int64_t first = blockIdx.x * per_block;
    const int64_t last = min(first + per_block, vectors);
    const auto* in = reinterpret_cast<const uint4*>(x);
    auto* own = reinterpret_cast<uint4*>(ranks.buffers[rank]);
    for (int64_t v = first + threadIdx.x; v < last; v += kThreads) {
        own[v] = in[v];
    }
    meet(ranks, rank_count, rank, 0, epoch);
    for (int64_t v = first + threadIdx.x; v < last; v += kThreads) {
        float sums[kPerVector] = {};
        for (int p = 0; p < rank_count; p++) {
            const uint4 loaded = __ldcg(reinterpret_cast<const uint4*>(ranks.buffers[p]) + v);
            const auto* values = reinterpret_cast<const scalar_t*>(&loaded);
#pragma unroll
            for (int i = 0; i < kPerVector; i++) {
                sums[i] += as_float(values[i]);
            }
        }
        uint4 stored;
        auto* out = reinterpret_cast<scalar_t*>(&stored);
#pragma unroll
        for (int i = 0; i < kPerVector; i++) {
            out[i] = from_float<scalar_t>(sums[i]);
        }
        reinterpret_cast<uint4*>(x)[v] = stored;
    }
    meet(ranks, rank_count, rank, 1, epoch);
}

}  // namespace

void p2p_all_reduce_cuda(
    at::Tensor& x,
    std::size_t rank,
    std::span<std::byte* const> buffers,
    std::span<std::uint64_t* const> flags,
    std::span<std::uint64_t* const> counters
) {
    const std::size_t rank_count = buffers.size();
    TORCH_CHECK(
        rank_count >= 1 && rank_count <= p2p_all_reduce_max_ranks && flags.size() == rank_count &&
            counters.size() == rank_count && rank < rank_count,
        "p2p_all_reduce: a rank's buffer, flags and counters for each of at most ",
        p2p_all_reduce_max_ranks,
        " ranks"
    );
    TORCH_CHECK(
        x.is_cuda() && x.is_contiguous() && x.nbytes() % 16 == 0 &&
            reinterpret_cast<std::uintptr_t>(x.data_ptr()) % 16 == 0,
        "p2p_all_reduce: x must be a contiguous CUDA tensor, 16-byte aligned, of a multiple of 16 bytes"
    );
    const auto vectors = static_cast<int64_t>(x.nbytes() / 16);
    if (vectors == 0) {
        return;
    }
    rank_pointers ranks{};
    for (std::size_t r = 0; r < rank_count; r++) {
        ranks.buffers[r] = buffers[r];
        ranks.flags[r] = flags[r];
    }
    const int64_t blocks = std::min<int64_t>(p2p_all_reduce_max_blocks, (vectors + kThreads - 1) / kThreads);
    const int64_t per_block = (vectors + blocks - 1) / blocks;
    auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
    auto launch = [&](auto* data) {
        using scalar_t = std::remove_pointer_t<decltype(data)>;
        p2p_all_reduce_kernel<scalar_t><<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
            data,
            vectors,
            per_block,
            ranks,
            static_cast<int>(rank_count),
            static_cast<int>(rank),
            counters[rank]
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    switch (x.scalar_type()) {
    case at::kFloat:
        launch(x.data_ptr<float>());
        break;
    case at::kHalf:
        launch(reinterpret_cast<__half*>(x.data_ptr<at::Half>()));
        break;
    case at::kBFloat16:
        launch(reinterpret_cast<__nv_bfloat16*>(x.data_ptr<at::BFloat16>()));
        break;
    default:
        TORCH_CHECK(false, "p2p_all_reduce: x must be fp32, fp16 or bf16");
    }
}

}  // namespace pulsarlm
