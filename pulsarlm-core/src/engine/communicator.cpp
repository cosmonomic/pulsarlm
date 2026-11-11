#include "pulsarlm/engine/communicator.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>
#include <nccl.h>

#include <barrier>
#include <cstddef>
#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pulsarlm {

class communicator::group {
  public:
    virtual ~group() = default;
    virtual std::size_t size() const noexcept = 0;
    virtual void all_reduce(std::size_t rank, at::Tensor& x) = 0;
    virtual at::Tensor all_gather(std::size_t rank, const at::Tensor& x) = 0;
};

namespace {

void check_nccl(ncclResult_t result) {
    if (result != ncclSuccess) {
        throw std::runtime_error(std::format("NCCL: {}", ncclGetErrorString(result)));
    }
}

ncclDataType_t nccl_type(at::ScalarType dtype) {
    switch (dtype) {
    case at::kFloat:
        return ncclFloat32;
    case at::kHalf:
        return ncclFloat16;
    case at::kBFloat16:
        return ncclBfloat16;
    case at::kLong:
        return ncclInt64;
    case at::kInt:
        return ncclInt32;
    case at::kByte:
        return ncclUint8;
    default:
        throw std::invalid_argument(std::format("communicator: no NCCL type for {}", c10::toString(dtype)));
    }
}

void check_contiguous(const at::Tensor& x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "communicator: a collective takes a contiguous CUDA tensor");
}

std::vector<std::int64_t> gathered_sizes(const at::Tensor& x, std::size_t size) {
    TORCH_CHECK(x.dim() >= 1, "communicator: all_gather takes a tensor of at least one dimension");
    std::vector<std::int64_t> sizes = x.sizes().vec();
    sizes.front() *= static_cast<std::int64_t>(size);
    return sizes;
}

cudaStream_t current_stream(const at::Tensor& x) {
    return at::cuda::getCurrentCUDAStream(x.device().index()).stream();
}

class nccl_group final : public communicator::group {
    std::vector<ncclComm_t> comms;

  public:
    explicit nccl_group(std::span<const at::Device> devices)
    : comms(devices.size()) {
        std::vector<int> ordinals;
        for (const at::Device& device : devices) {
            TORCH_CHECK(
                device.is_cuda() && device.has_index(),
                "communicator: a rank's device is an indexed CUDA device"
            );
            ordinals.push_back(device.index());
        }
        check_nccl(ncclCommInitAll(comms.data(), static_cast<int>(comms.size()), ordinals.data()));
    }

    nccl_group(const nccl_group&) = delete;
    nccl_group& operator=(const nccl_group&) = delete;

    ~nccl_group() override {
        for (ncclComm_t comm : comms) {
            ncclCommDestroy(comm);
        }
    }

    std::size_t size() const noexcept override {
        return comms.size();
    }

    void all_reduce(std::size_t rank, at::Tensor& x) override {
        check_contiguous(x);
        check_nccl(ncclAllReduce(
            x.data_ptr(),
            x.data_ptr(),
            static_cast<std::size_t>(x.numel()),
            nccl_type(x.scalar_type()),
            ncclSum,
            comms[rank],
            current_stream(x)
        ));
    }

    at::Tensor all_gather(std::size_t rank, const at::Tensor& x) override {
        check_contiguous(x);
        at::Tensor gathered = at::empty(gathered_sizes(x, comms.size()), x.options());
        check_nccl(ncclAllGather(
            x.data_ptr(),
            gathered.data_ptr(),
            static_cast<std::size_t>(x.numel()),
            nccl_type(x.scalar_type()),
            comms[rank],
            current_stream(x)
        ));
        return gathered;
    }
};

// Each rank copies its tensor into a buffer of its own and records an event; after the first barrier, every rank
// waits for every event and combines the buffers in rank order, so all ranks compute the same result. The second
// barrier keeps a rank from replacing its buffer while another rank's thread still reads the list of buffers.
class one_device_group final : public communicator::group {
    std::vector<at::Tensor> buffers;
    std::vector<cudaEvent_t> written;
    std::barrier<> buffers_written;
    std::barrier<> buffers_read;

    std::span<const at::Tensor> exchange(std::size_t rank, const at::Tensor& x) {
        check_contiguous(x);
        cudaStream_t stream = current_stream(x);
        buffers[rank] = x.clone();
        C10_CUDA_CHECK(cudaEventRecord(written[rank], stream));
        buffers_written.arrive_and_wait();
        for (std::size_t other = 0; other < buffers.size(); other++) {
            C10_CUDA_CHECK(cudaStreamWaitEvent(stream, written[other]));
            c10::cuda::CUDACachingAllocator::recordStream(
                buffers[other].storage().data_ptr(),
                at::cuda::getCurrentCUDAStream(x.device().index())
            );
        }
        return buffers;
    }

  public:
    one_device_group(at::Device device, std::size_t size)
    : buffers(size),
      written(size),
      buffers_written(static_cast<std::ptrdiff_t>(size)),
      buffers_read(static_cast<std::ptrdiff_t>(size)) {
        c10::cuda::CUDAGuard guard(device);
        for (cudaEvent_t& event : written) {
            C10_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        }
    }

    one_device_group(const one_device_group&) = delete;
    one_device_group& operator=(const one_device_group&) = delete;

    ~one_device_group() override {
        for (cudaEvent_t event : written) {
            cudaEventDestroy(event);
        }
    }

    std::size_t size() const noexcept override {
        return buffers.size();
    }

    void all_reduce(std::size_t rank, at::Tensor& x) override {
        std::span<const at::Tensor> all = exchange(rank, x);
        x.copy_(all.front());
        for (const at::Tensor& other : all.subspan(1)) {
            x.add_(other);
        }
        buffers_read.arrive_and_wait();
    }

    at::Tensor all_gather(std::size_t rank, const at::Tensor& x) override {
        std::vector<std::int64_t> sizes = gathered_sizes(x, buffers.size());
        std::span<const at::Tensor> all = exchange(rank, x);
        at::Tensor gathered = at::cat(std::vector<at::Tensor>(all.begin(), all.end()), 0);
        buffers_read.arrive_and_wait();
        return gathered.view(sizes);
    }
};

}  // namespace

communicator::communicator(std::shared_ptr<group> members, std::size_t member_rank) noexcept
: members(std::move(members)),
  member_rank(member_rank) {}

std::vector<communicator> communicator::make(std::span<const at::Device> devices) {
    auto members = std::make_shared<nccl_group>(devices);
    std::vector<communicator> ranks;
    for (std::size_t rank = 0; rank < devices.size(); rank++) {
        ranks.push_back(communicator(members, rank));
    }
    return ranks;
}

std::vector<communicator> communicator::make_on_one_device(at::Device device, std::size_t size) {
    auto members = std::make_shared<one_device_group>(device, size);
    std::vector<communicator> ranks;
    for (std::size_t rank = 0; rank < size; rank++) {
        ranks.push_back(communicator(members, rank));
    }
    return ranks;
}

std::size_t communicator::rank() const noexcept {
    return member_rank;
}

std::size_t communicator::size() const noexcept {
    return members ? members->size() : 1;
}

void communicator::all_reduce(at::Tensor& x) const {
    if (members) {
        members->all_reduce(member_rank, x);
    }
}

at::Tensor communicator::all_gather(const at::Tensor& x) const {
    return members ? members->all_gather(member_rank, x) : x;
}

}  // namespace pulsarlm
