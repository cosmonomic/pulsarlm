#pragma once

#include <ATen/core/Tensor.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace pulsarlm {

constexpr std::size_t p2p_all_reduce_max_ranks = 8;
constexpr std::size_t p2p_all_reduce_max_blocks = 32;

// Sums x over the ranks in place, on the current stream, every rank computing the same sum in rank order. Rank r's
// buffer, flags and counters live on its device and every rank's device can read and write them. buffers[r] holds at
// least x's bytes; flags[r] holds 2 * p2p_all_reduce_max_blocks * ranks values and counters[r]
// p2p_all_reduce_max_blocks, all zero before the first call, and every rank makes the same calls in the same order.
// x is contiguous fp32, fp16 or bf16, 16-byte aligned, of a multiple of 16 bytes.
void p2p_all_reduce_cuda(
    at::Tensor& x,
    std::size_t rank,
    std::span<std::byte* const> buffers,
    std::span<std::uint64_t* const> flags,
    std::span<std::uint64_t* const> counters
);

// Writes every rank's x into out, rank r's at out's bytes [r * x.nbytes(), (r + 1) * x.nbytes()), on the current
// stream. The buffers, flags and counters are p2p_all_reduce_cuda's, shared with it, so buffers[r] holds at least
// out's bytes, and every rank makes the same calls of both ops in the same order. x and out are contiguous, 16-byte
// aligned, of a multiple of 16 bytes, out of x's dtype and of x.nbytes() times the ranks.
void p2p_all_gather_cuda(
    const at::Tensor& x,
    const at::Tensor& out,
    std::size_t rank,
    std::span<std::byte* const> buffers,
    std::span<std::uint64_t* const> flags,
    std::span<std::uint64_t* const> counters
);

}  // namespace pulsarlm
