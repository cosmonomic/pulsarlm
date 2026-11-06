#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace pulsarlm {

// One GPU's member of a group of GPUs in this process that reduce and gather tensors together. Every member runs the
// same collectives in the same order, each on its own thread; a collective is queued on the member's current CUDA
// stream. Default-constructed: a group of one GPU, whose collectives do nothing.
class communicator {
  public:
    class group;

  private:
    std::shared_ptr<group> members;
    std::size_t member_rank = 0;

    communicator(std::shared_ptr<group> members, std::size_t member_rank) noexcept;

  public:
    communicator() = default;

    // Rank i on devices[i], over NCCL (ncclCommInitAll); the devices are distinct.
    static std::vector<communicator> make(std::span<const at::Device> devices);
    // size ranks on one device, without NCCL, for tests: a collective goes through device buffers the ranks share and
    // blocks each rank's thread until every rank has queued it.
    static std::vector<communicator> make_on_one_device(at::Device device, std::size_t size);

    std::size_t rank() const noexcept;
    std::size_t size() const noexcept;
    // Sums x over the ranks, in place; x is contiguous.
    void all_reduce(at::Tensor& x) const;
    // Every rank's x concatenated along dim 0 in rank order; x is contiguous and of one shape on every rank.
    at::Tensor all_gather(const at::Tensor& x) const;
};

}  // namespace pulsarlm
