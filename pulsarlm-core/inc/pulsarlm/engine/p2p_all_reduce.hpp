#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace pulsarlm {

// Sums small tensors over GPUs of one process through a buffer on each GPU that the others read: every rank copies its
// tensor into its buffer, waits until every rank's copy is there, then sums all copies in rank order, so every rank
// computes the same result. A call can be captured in a CUDA graph. Every rank makes the same calls in the same order,
// each on its own thread.
class p2p_all_reduce {
    std::vector<at::Device> devices;
    std::size_t max_bytes;
    std::vector<std::byte*> buffers;
    std::vector<std::uint64_t*> flags;
    std::vector<std::uint64_t*> counters;

    p2p_all_reduce(std::span<const at::Device> devices, std::size_t max_bytes);

  public:
    // One rank per entry of devices, reducing tensors of at most max_bytes; null when two of the devices cannot reach
    // each other's memory. A device may repeat, its ranks then sharing it.
    static std::unique_ptr<p2p_all_reduce> make(std::span<const at::Device> devices, std::size_t max_bytes);

    p2p_all_reduce(const p2p_all_reduce&) = delete;
    p2p_all_reduce& operator=(const p2p_all_reduce&) = delete;
    ~p2p_all_reduce();

    // Whether x is a contiguous fp32, fp16 or bf16 tensor of at most max_bytes, 16-byte aligned, of a multiple of 16
    // bytes.
    bool fits(const at::Tensor& x) const noexcept;
    // Sums x over the ranks in place on rank's current stream; x fits.
    void all_reduce(std::size_t rank, at::Tensor& x);
};

}  // namespace pulsarlm
