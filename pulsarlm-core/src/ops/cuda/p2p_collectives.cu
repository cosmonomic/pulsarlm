#include "pulsarlm/ops/p2p_collectives.hpp"

#include "pulsarlm/ops/cuda/device.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <utility>

namespace pulsarlm::ops::cuda {
namespace {

constexpr int cta_threads = 512;

}  // namespace

namespace kernel {

struct rank_pointers {
    std::byte* buffers[p2p_max_ranks];
    std::uint64_t* flags[p2p_max_ranks];
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
__device__ void meet(const rank_pointers& ranks, int rank_count, int rank, int phase, std::uint64_t epoch) {
    __syncthreads();
    const int peer = static_cast<int>(threadIdx.x);
    if (peer < rank_count) {
        const int64_t slot = (static_cast<int64_t>(phase) * p2p_max_blocks + blockIdx.x) * rank_count;
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
__global__ void __launch_bounds__(cta_threads) p2p_all_reduce(
    scalar_t* __restrict__ x,
    int64_t vectors,
    int64_t per_block,
    rank_pointers ranks,
    int rank_count,
    int rank,
    std::uint64_t* __restrict__ counters
) {
    constexpr int per_vector = 16 / sizeof(scalar_t);
    __shared__ std::uint64_t epoch;
    if (threadIdx.x == 0) {
        epoch = ++counters[blockIdx.x];
    }
    __syncthreads();
    const int64_t first = blockIdx.x * per_block;
    const int64_t last = min(first + per_block, vectors);
    const auto* in = reinterpret_cast<const uint4*>(x);
    auto* own = reinterpret_cast<uint4*>(ranks.buffers[rank]);
    for (int64_t v = first + threadIdx.x; v < last; v += cta_threads) {
        own[v] = in[v];
    }
    meet(ranks, rank_count, rank, 0, epoch);
    for (int64_t v = first + threadIdx.x; v < last; v += cta_threads) {
        float sums[per_vector] = {};
        for (int p = 0; p < rank_count; p++) {
            const uint4 loaded = __ldcg(reinterpret_cast<const uint4*>(ranks.buffers[p]) + v);
            const auto* values = reinterpret_cast<const scalar_t*>(&loaded);
#pragma unroll
            for (int i = 0; i < per_vector; i++) {
                sums[i] += as_float(values[i]);
            }
        }
        uint4 stored;
        auto* out = reinterpret_cast<scalar_t*>(&stored);
#pragma unroll
        for (int i = 0; i < per_vector; i++) {
            out[i] = from_float<scalar_t>(sums[i]);
        }
        reinterpret_cast<uint4*>(x)[v] = stored;
    }
    meet(ranks, rank_count, rank, 1, epoch);
}

// Block b takes the 16-byte vectors [b * per_block, (b + 1) * per_block) of each rank's part: it copies them of x into
// every rank's buffer at its rank's part, meets the other ranks' block b, copies every rank's part from its own
// buffer into out, and meets them again so that no rank's next call replaces a buffer while its owner still reads it.
__global__ void __launch_bounds__(cta_threads) p2p_all_gather(
    const uint4* __restrict__ x,
    uint4* __restrict__ out,
    int64_t vectors,
    int64_t per_block,
    rank_pointers ranks,
    int rank_count,
    int rank,
    std::uint64_t* __restrict__ counters
) {
    __shared__ std::uint64_t epoch;
    if (threadIdx.x == 0) {
        epoch = ++counters[blockIdx.x];
    }
    __syncthreads();
    const int64_t first = blockIdx.x * per_block;
    const int64_t last = min(first + per_block, vectors);
    for (int64_t v = first + threadIdx.x; v < last; v += cta_threads) {
        const uint4 value = x[v];
        for (int p = 0; p < rank_count; p++) {
            reinterpret_cast<uint4*>(ranks.buffers[p])[rank * vectors + v] = value;
        }
    }
    meet(ranks, rank_count, rank, 0, epoch);
    const auto* own = reinterpret_cast<const uint4*>(ranks.buffers[rank]);
    for (int p = 0; p < rank_count; p++) {
        for (int64_t v = first + threadIdx.x; v < last; v += cta_threads) {
            out[p * vectors + v] = __ldcg(own + p * vectors + v);
        }
    }
    meet(ranks, rank_count, rank, 1, epoch);
}

}  // namespace kernel

namespace {

void check_ranks(
    const char* op,
    std::size_t rank,
    std::span<std::byte* const> buffers,
    std::span<std::uint64_t* const> flags,
    std::span<std::uint64_t* const> counters
) {
    const std::size_t rank_count = buffers.size();
    TORCH_CHECK(
        rank_count >= 1 && rank_count <= p2p_max_ranks && flags.size() == rank_count && counters.size() == rank_count &&
            rank < rank_count,
        op,
        ": a rank's buffer, flags and counters for each of at most ",
        p2p_max_ranks,
        " ranks"
    );
}

bool is_aligned_contiguous(const at::Tensor& x) {
    return x.is_cuda() && x.is_contiguous() && x.nbytes() % 16 == 0 &&
        reinterpret_cast<std::uintptr_t>(x.data_ptr()) % 16 == 0;
}

kernel::rank_pointers collect_pointers(std::span<std::byte* const> buffers, std::span<std::uint64_t* const> flags) {
    kernel::rank_pointers ranks{};
    for (std::size_t r = 0; r < buffers.size(); r++) {
        ranks.buffers[r] = buffers[r];
        ranks.flags[r] = flags[r];
    }
    return ranks;
}

// The blocks of a call over vectors 16-byte vectors per rank, and the vectors of each block.
std::pair<int64_t, int64_t> split_blocks(int64_t vectors) {
    const int64_t blocks = std::min<int64_t>(p2p_max_blocks, (vectors + cta_threads - 1) / cta_threads);
    return {blocks, (vectors + blocks - 1) / blocks};
}

}  // namespace

void p2p_all_gather(
    const at::Tensor& x,
    const at::Tensor& out,
    std::size_t rank,
    std::span<std::byte* const> buffers,
    std::span<std::uint64_t* const> flags,
    std::span<std::uint64_t* const> counters
) {
    check_on_current_device("p2p_all_gather", {x, out});
    check_ranks("p2p_all_gather", rank, buffers, flags, counters);
    TORCH_CHECK(
        is_aligned_contiguous(x) && is_aligned_contiguous(out),
        "p2p_all_gather: x and out must be contiguous CUDA tensors, 16-byte aligned, of a multiple of 16 bytes"
    );
    TORCH_CHECK(
        out.scalar_type() == x.scalar_type() && out.nbytes() == x.nbytes() * buffers.size(),
        "p2p_all_gather: out must hold every rank's x in x's dtype"
    );
    const auto vectors = static_cast<int64_t>(x.nbytes() / 16);
    if (vectors == 0) {
        return;
    }
    const std::pair<int64_t, int64_t> split = split_blocks(vectors);
    const int64_t blocks = split.first;
    const int64_t per_block = split.second;
    kernel::p2p_all_gather<<<
        static_cast<unsigned>(blocks),
        cta_threads,
        0,
        at::cuda::getCurrentCUDAStream(x.device().index())>>>(
        static_cast<const uint4*>(x.const_data_ptr()),
        static_cast<uint4*>(out.data_ptr()),
        vectors,
        per_block,
        collect_pointers(buffers, flags),
        static_cast<int>(buffers.size()),
        static_cast<int>(rank),
        counters[rank]
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void p2p_all_reduce(
    at::Tensor& x,
    std::size_t rank,
    std::span<std::byte* const> buffers,
    std::span<std::uint64_t* const> flags,
    std::span<std::uint64_t* const> counters
) {
    check_on_current_device("p2p_all_reduce", {x});
    check_ranks("p2p_all_reduce", rank, buffers, flags, counters);
    const std::size_t rank_count = buffers.size();
    TORCH_CHECK(
        is_aligned_contiguous(x),
        "p2p_all_reduce: x must be a contiguous CUDA tensor, 16-byte aligned, of a multiple of 16 bytes"
    );
    const auto vectors = static_cast<int64_t>(x.nbytes() / 16);
    if (vectors == 0) {
        return;
    }
    const kernel::rank_pointers ranks = collect_pointers(buffers, flags);
    const std::pair<int64_t, int64_t> split = split_blocks(vectors);
    const int64_t blocks = split.first;
    const int64_t per_block = split.second;
    auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
    auto launch = [&](auto* data) {
        using scalar_t = std::remove_pointer_t<decltype(data)>;
        kernel::p2p_all_reduce<scalar_t><<<static_cast<unsigned>(blocks), cta_threads, 0, stream>>>(
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

}  // namespace pulsarlm::ops::cuda
