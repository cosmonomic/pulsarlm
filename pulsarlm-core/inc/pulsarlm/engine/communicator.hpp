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
// stream. An all-reduce or an all-gather of a tensor that p2p_all_reduce fits goes through it when the group's GPUs
// reach each other's memory, and otherwise through the group. Default-constructed: a group of one GPU, whose
// collectives do nothing.
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
    // size ranks on one device, without NCCL, for tests: a collective that p2p_all_reduce does not fit goes through
    // device buffers the ranks share and blocks each rank's thread until every rank has queued it. The ranks'
    // p2p_all_reduce kernels spin on the one device while they wait for each other, and loading a kernel's module
    // waits for the device's running kernels; every rank's thread waits for the others before it queues one of them,
    // so the eager collectives queue no kernel lazily beside a waiting one, but a process replaying CUDA graphs of
    // the ranks loads its modules eagerly (CUDA_MODULE_LOADING=EAGER).
    static std::vector<communicator> make_on_one_device(at::Device device, std::size_t size);

    std::size_t rank() const noexcept;
    std::size_t size() const noexcept;
    // Sums x over the ranks, in place; x is contiguous.
    void all_reduce(at::Tensor& x) const;
    // Whether an all-reduce of a contiguous fp32, fp16 or bf16 tensor of byte_count bytes, allocated by ATen, can be
    // captured in a CUDA graph: with one rank, or through p2p_all_reduce.
    bool captures_all_reduce(std::size_t byte_count) const noexcept;
    // Every rank's x concatenated along dim 0 in rank order; x is contiguous and of one shape on every rank.
    at::Tensor all_gather(const at::Tensor& x) const;
};

}  // namespace pulsarlm
