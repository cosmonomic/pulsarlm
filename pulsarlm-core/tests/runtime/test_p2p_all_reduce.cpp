#include "pulsarlm/engine/p2p_all_reduce.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

// p2p_all_reduce with every rank on one device, each on its own thread and stream, against the fp32 sum in rank order.

namespace {

constexpr std::size_t ranks = 4;
constexpr std::size_t max_bytes = 2 * 1024 * 1024;

// Complete before returning: the ranks read them on streams that do not wait for the calling thread's.
std::vector<at::Tensor> rank_inputs(std::int64_t elements, at::ScalarType dtype, std::int64_t seed) {
    std::vector<at::Tensor> inputs;
    for (std::size_t rank = 0; rank < ranks; rank++) {
        at::manual_seed(seed + static_cast<std::int64_t>(rank));
        inputs.push_back(at::randn({elements}, at::device(at::kCUDA).dtype(dtype)));
    }
    c10::cuda::device_synchronize();
    return inputs;
}

at::Tensor rank_order_sum(const std::vector<at::Tensor>& inputs) {
    at::Tensor sum = at::zeros_like(inputs.front(), at::kFloat);
    for (const at::Tensor& input : inputs) {
        sum = sum + input.to(at::kFloat);
    }
    return sum.to(inputs.front().scalar_type());
}

// Runs body(rank, stream) for every rank at once, each on its own thread and stream. Catch2's assertions are not safe
// to make from these threads.
template <typename Body> void on_every_rank(Body body) {
    std::vector<std::thread> threads;
    for (std::size_t rank = 0; rank < ranks; rank++) {
        threads.emplace_back([&body, rank] {
            c10::cuda::CUDAStream stream = c10::cuda::getStreamFromPool();
            c10::cuda::CUDAStreamGuard guard(stream);
            body(rank, stream);
            stream.synchronize();
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
}

std::unique_ptr<pulsarlm::p2p_all_reduce> make_ranks() {
    std::vector<at::Device> devices(ranks, at::Device(at::kCUDA, 0));
    return pulsarlm::p2p_all_reduce::make(devices, max_bytes);
}

}  // namespace

TEST_CASE("p2p_all_reduce sums every rank's tensor in rank order on every rank", "[runtime][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    std::unique_ptr<pulsarlm::p2p_all_reduce> all_reduce = make_ranks();
    REQUIRE(all_reduce != nullptr);
    const at::ScalarType dtype = GENERATE(at::kFloat, at::kBFloat16, at::kHalf);
    // One vector, one block, and more vectors than all blocks hold at once.
    const std::int64_t bytes = GENERATE(16, 5120, 1024 * 1024);
    const std::int64_t elements = bytes / static_cast<std::int64_t>(c10::elementSize(dtype));
    for (std::int64_t call = 0; call < 3; call++) {
        std::vector<at::Tensor> inputs = rank_inputs(elements, dtype, 10 * call);
        std::vector<at::Tensor> outputs(ranks);
        for (const at::Tensor& input : inputs) {
            REQUIRE(all_reduce->fits(input));
        }
        on_every_rank([&](std::size_t rank, c10::cuda::CUDAStream) {
            outputs[rank] = inputs[rank].clone();
            all_reduce->all_reduce(rank, outputs[rank]);
        });
        at::Tensor expected = rank_order_sum(inputs);
        for (const at::Tensor& output : outputs) {
            CHECK(at::equal(output, expected));
        }
    }
}

TEST_CASE("p2p_all_reduce replays from CUDA graphs with new inputs", "[runtime][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    std::unique_ptr<pulsarlm::p2p_all_reduce> all_reduce = make_ranks();
    REQUIRE(all_reduce != nullptr);
    const std::int64_t elements = 2560;
    std::vector<at::Tensor> fixed = rank_inputs(elements, at::kBFloat16, 1);
    std::vector<std::unique_ptr<at::cuda::CUDAGraph>> graphs(ranks);
    on_every_rank([&](std::size_t rank, c10::cuda::CUDAStream) {
        graphs[rank] = std::make_unique<at::cuda::CUDAGraph>();
        graphs[rank]->capture_begin({0, 0}, cudaStreamCaptureModeThreadLocal);
        all_reduce->all_reduce(rank, fixed[rank]);
        graphs[rank]->capture_end();
    });
    for (std::int64_t replay = 0; replay < 3; replay++) {
        std::vector<at::Tensor> inputs = rank_inputs(elements, at::kBFloat16, 100 + replay);
        on_every_rank([&](std::size_t rank, c10::cuda::CUDAStream) {
            fixed[rank].copy_(inputs[rank]);
            graphs[rank]->replay();
        });
        at::Tensor expected = rank_order_sum(inputs);
        for (const at::Tensor& output : fixed) {
            CHECK(at::equal(output, expected));
        }
    }
}

TEST_CASE("p2p_all_reduce fits only small aligned float tensors", "[runtime][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    std::unique_ptr<pulsarlm::p2p_all_reduce> all_reduce = make_ranks();
    REQUIRE(all_reduce != nullptr);
    auto cuda = at::device(at::kCUDA);
    CHECK(all_reduce->fits(at::zeros({8}, cuda.dtype(at::kBFloat16))));
    CHECK_FALSE(all_reduce->fits(at::zeros({7}, cuda.dtype(at::kBFloat16))));
    CHECK_FALSE(all_reduce->fits(at::zeros({16}, cuda.dtype(at::kBFloat16)).narrow(0, 1, 8)));
    CHECK_FALSE(all_reduce->fits(at::zeros({4}, cuda.dtype(at::kLong))));
    CHECK_FALSE(all_reduce->fits(at::zeros({static_cast<std::int64_t>(max_bytes)}, cuda.dtype(at::kHalf))));
}
