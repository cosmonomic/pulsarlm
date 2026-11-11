#include "pulsarlm/engine/communicator.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>

#undef CHECK
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

namespace {

using pulsarlm::communicator;

at::TensorOptions cuda(at::Device device) {
    return at::TensorOptions().dtype(at::kFloat).device(device);
}

// Rank r's tensor: r + 1 everywhere, [2, 3].
at::Tensor rank_tensor(std::size_t rank, at::Device device) {
    return at::full({2, 3}, static_cast<double>(rank + 1), cuda(device));
}

struct collective_results {
    std::vector<at::Tensor> reduced;
    std::vector<at::Tensor> gathered;
};

// Each rank on its own thread, as every rank queues its collectives.
collective_results run_collectives(const std::vector<communicator>& ranks, std::span<const at::Device> devices) {
    collective_results results{
        .reduced = std::vector<at::Tensor>(ranks.size()),
        .gathered = std::vector<at::Tensor>(ranks.size())
    };
    std::vector<std::thread> threads;
    for (std::size_t rank = 0; rank < ranks.size(); rank++) {
        threads.emplace_back([&, rank] {
            c10::cuda::CUDAGuard guard(devices[rank]);
            at::Tensor x = rank_tensor(rank, devices[rank]);
            ranks[rank].all_reduce(x);
            results.reduced[rank] = x;
            results.gathered[rank] = ranks[rank].all_gather(rank_tensor(rank, devices[rank]));
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    return results;
}

void check_results(const collective_results& results, std::size_t size) {
    double total = static_cast<double>(size * (size + 1) / 2);
    std::vector<at::Tensor> parts;
    for (std::size_t rank = 0; rank < size; rank++) {
        parts.push_back(at::full({2, 3}, static_cast<double>(rank + 1), at::TensorOptions().dtype(at::kFloat)));
    }
    at::Tensor expected_gathered = at::cat(parts, 0);
    for (std::size_t rank = 0; rank < size; rank++) {
        REQUIRE(at::equal(results.reduced[rank].cpu(), at::full({2, 3}, total, at::TensorOptions().dtype(at::kFloat))));
        REQUIRE(at::equal(results.gathered[rank].cpu(), expected_gathered));
    }
}

}  // namespace

TEST_CASE("a default communicator is one rank whose collectives do nothing", "[runtime][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    communicator single;
    REQUIRE(single.rank() == 0);
    REQUIRE(single.size() == 1);
    at::Tensor x = rank_tensor(0, at::Device(at::kCUDA, 0));
    single.all_reduce(x);
    REQUIRE(at::equal(x, rank_tensor(0, x.device())));
    REQUIRE(at::equal(single.all_gather(x), x));
}

TEST_CASE("ranks on one device reduce and gather in rank order", "[runtime][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const std::size_t size = GENERATE(std::size_t{1}, std::size_t{2}, std::size_t{3});
    const at::Device device(at::kCUDA, 0);
    std::vector<communicator> ranks = communicator::make_on_one_device(device, size);
    REQUIRE(ranks.size() == size);
    for (std::size_t rank = 0; rank < size; rank++) {
        REQUIRE(ranks[rank].rank() == rank);
        REQUIRE(ranks[rank].size() == size);
    }
    std::vector<at::Device> devices(size, device);
    for (int round = 0; round < 3; round++) {
        check_results(run_collectives(ranks, devices), size);
    }
}

TEST_CASE("NCCL ranks on distinct devices reduce and gather in rank order", "[runtime][cuda]") {
    if (c10::cuda::device_count() < 2) {
        SKIP("fewer than two CUDA devices");
    }
    std::vector<at::Device> devices;
    for (int index = 0; index < c10::cuda::device_count(); index++) {
        devices.emplace_back(at::kCUDA, static_cast<at::DeviceIndex>(index));
    }
    std::vector<communicator> ranks = communicator::make(devices);
    check_results(run_collectives(ranks, devices), devices.size());
}
