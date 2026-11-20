#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace pulsarlm {

// Sums or gathers small tensors over GPUs of one process through a buffer on each GPU that the others reach. To sum,
// every rank copies its tensor into its buffer, waits until every rank's copy is there, then sums all copies in rank
// order, so every rank computes the same result. To gather, every rank copies its tensor into every rank's buffer,
// waits until every rank's copies are there, then copies its buffer out. A call can be captured in a CUDA graph. Every
// rank makes the same calls in the same order, each on its own thread.
class p2p_group {
    std::vector<at::Device> devices;
    std::size_t max_bytes;
    std::vector<std::byte*> buffers;
    std::vector<std::uint64_t*> flags;
    std::vector<std::uint64_t*> counters;

    p2p_group(std::span<const at::Device> devices, std::size_t max_bytes);

  public:
    // One rank per entry of devices, reducing tensors of at most max_bytes; null when two of the devices cannot reach
    // each other's memory. A device may repeat, its ranks then sharing it.
    static std::unique_ptr<p2p_group> make(std::span<const at::Device> devices, std::size_t max_bytes);

    p2p_group(const p2p_group&) = delete;
    p2p_group& operator=(const p2p_group&) = delete;
    ~p2p_group();

    // Whether x is a contiguous fp32, fp16 or bf16 tensor of at most max_bytes, 16-byte aligned, of a multiple of 16
    // bytes.
    bool fits(const at::Tensor& x) const noexcept;
    // Whether byte_count is at most max_bytes and a multiple of 16.
    bool fits_size(std::size_t byte_count) const noexcept;
    // Sums x over the ranks in place on rank's current stream; x fits.
    void all_reduce(std::size_t rank, at::Tensor& x);
    // Whether every rank's x, gathered, fits: x is contiguous, 16-byte aligned, of a multiple of 16 bytes, and of at
    // most max_bytes over the ranks together.
    bool fits_gather(const at::Tensor& x) const noexcept;
    // Every rank's x concatenated along dim 0 in rank order, on rank's current stream; x fits a gather.
    at::Tensor all_gather(std::size_t rank, const at::Tensor& x);
};

}  // namespace pulsarlm
