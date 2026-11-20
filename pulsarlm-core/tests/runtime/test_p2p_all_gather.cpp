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

// p2p_all_reduce's gather with every rank on one device, each on its own thread and stream, against at::cat of the
// ranks' tensors, compared byte for byte.

namespace {

constexpr std::size_t ranks = 4;
constexpr std::size_t max_bytes = 2 * 1024 * 1024;

// Random bit patterns of dtype, among them negative zeros and NaNs, rank r's part [length, 8] starting with -0.0.
// Complete before returning: the ranks read them on streams that do not wait for the calling thread's.
std::vector<at::Tensor> rank_inputs(std::int64_t length, at::ScalarType dtype, std::int64_t seed) {
    std::vector<at::Tensor> inputs;
    const auto element_bytes = static_cast<std::int64_t>(c10::elementSize(dtype));
    for (std::size_t rank = 0; rank < ranks; rank++) {
        at::manual_seed(seed + static_cast<std::int64_t>(rank));
        at::Tensor bytes = at::randint(0, 256, {length * 8 * element_bytes}, at::device(at::kCUDA).dtype(at::kByte));
        at::Tensor input = bytes.view(dtype).view({length, 8});
        input.view(-1).narrow(0, 0, 1).fill_(-0.0);
        inputs.push_back(input);
    }
    c10::cuda::device_synchronize();
    return inputs;
}

at::Tensor raw_bytes(const at::Tensor& x) {
    return x.contiguous().view(-1).view(at::kByte).cpu();
}

// Runs body(rank) for every rank at once, each on its own thread and stream. Catch2's assertions are not safe to make
// from these threads.
template <typename body_type> void on_every_rank(body_type body) {
    std::vector<std::thread> threads;
    for (std::size_t rank = 0; rank < ranks; rank++) {
        threads.emplace_back([&body, rank] {
            c10::cuda::CUDAStream stream = c10::cuda::getStreamFromPool();
            c10::cuda::CUDAStreamGuard guard(stream);
            body(rank);
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

TEST_CASE("p2p_all_reduce gathers every rank's tensor in rank order, byte for byte", "[runtime][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    std::unique_ptr<pulsarlm::p2p_all_reduce> group = make_ranks();
    REQUIRE(group != nullptr);
    const at::ScalarType dtype = GENERATE(at::kFloat, at::kBFloat16, at::kHalf, at::kLong);
    // One row of int64 or four vectors, one block, and more vectors than all blocks hold at once.
    const std::int64_t rank_bytes = GENERATE(64, 5120, 256 * 1024);
    const std::int64_t length = rank_bytes / (8 * static_cast<std::int64_t>(c10::elementSize(dtype)));
    for (std::int64_t call = 0; call < 3; call++) {
        std::vector<at::Tensor> inputs = rank_inputs(length, dtype, 10 * call);
        std::vector<at::Tensor> outputs(ranks);
        for (const at::Tensor& input : inputs) {
            REQUIRE(group->fits_gather(input));
        }
        on_every_rank([&](std::size_t rank) { outputs[rank] = group->all_gather(rank, inputs[rank]); });
        const at::Tensor expected = raw_bytes(at::cat(inputs, 0));
        for (const at::Tensor& output : outputs) {
            CHECK(output.sizes() == at::IntArrayRef({length * static_cast<std::int64_t>(ranks), 8}));
            CHECK(at::equal(raw_bytes(output), expected));
        }
    }
}

TEST_CASE("p2p_all_reduce gathers and sums in turn on its shared buffers", "[runtime][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    std::unique_ptr<pulsarlm::p2p_all_reduce> group = make_ranks();
    REQUIRE(group != nullptr);
    for (std::int64_t call = 0; call < 4; call++) {
        std::vector<at::Tensor> gathered_inputs = rank_inputs(640, at::kBFloat16, 20 * call);
        std::vector<at::Tensor> summed_inputs;
        for (std::size_t rank = 0; rank < ranks; rank++) {
            at::manual_seed(1000 + 20 * call + static_cast<std::int64_t>(rank));
            summed_inputs.push_back(at::randn({4096}, at::device(at::kCUDA).dtype(at::kFloat)));
        }
        c10::cuda::device_synchronize();
        std::vector<at::Tensor> gathered(ranks);
        std::vector<at::Tensor> summed(ranks);
        on_every_rank([&](std::size_t rank) {
            gathered[rank] = group->all_gather(rank, gathered_inputs[rank]);
            summed[rank] = summed_inputs[rank].clone();
            group->all_reduce(rank, summed[rank]);
        });
        at::Tensor expected_sum = at::zeros_like(summed_inputs.front());
        for (const at::Tensor& input : summed_inputs) {
            expected_sum = expected_sum + input;
        }
        const at::Tensor expected_gather = raw_bytes(at::cat(gathered_inputs, 0));
        for (std::size_t rank = 0; rank < ranks; rank++) {
            CHECK(at::equal(raw_bytes(gathered[rank]), expected_gather));
            CHECK(at::equal(summed[rank], expected_sum));
        }
    }
}

TEST_CASE("p2p_all_reduce gathers from CUDA graphs with new inputs", "[runtime][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    std::unique_ptr<pulsarlm::p2p_all_reduce> group = make_ranks();
    REQUIRE(group != nullptr);
    const std::int64_t length = 320;
    std::vector<at::Tensor> fixed = rank_inputs(length, at::kBFloat16, 1);
    std::vector<at::Tensor> outputs(ranks);
    std::vector<std::unique_ptr<at::cuda::CUDAGraph>> graphs(ranks);
    on_every_rank([&](std::size_t rank) {
        graphs[rank] = std::make_unique<at::cuda::CUDAGraph>();
        graphs[rank]->capture_begin({0, 0}, cudaStreamCaptureModeThreadLocal);
        outputs[rank] = group->all_gather(rank, fixed[rank]);
        graphs[rank]->capture_end();
    });
    for (std::int64_t replay = 0; replay < 3; replay++) {
        std::vector<at::Tensor> inputs = rank_inputs(length, at::kBFloat16, 100 + replay);
        on_every_rank([&](std::size_t rank) {
            fixed[rank].copy_(inputs[rank]);
            graphs[rank]->replay();
        });
        const at::Tensor expected = raw_bytes(at::cat(inputs, 0));
        for (const at::Tensor& output : outputs) {
            CHECK(at::equal(raw_bytes(output), expected));
        }
    }
}

TEST_CASE("p2p_all_reduce gathers only aligned tensors whose gather fits its buffers", "[runtime][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    std::unique_ptr<pulsarlm::p2p_all_reduce> group = make_ranks();
    REQUIRE(group != nullptr);
    auto cuda = at::device(at::kCUDA);
    const auto per_rank_halves = static_cast<std::int64_t>(max_bytes / ranks / 2);
    CHECK(group->fits_gather(at::zeros({8}, cuda.dtype(at::kBFloat16))));
    CHECK(group->fits_gather(at::zeros({2}, cuda.dtype(at::kLong))));
    CHECK(group->fits_gather(at::zeros({per_rank_halves}, cuda.dtype(at::kHalf))));
    CHECK_FALSE(group->fits_gather(at::zeros({per_rank_halves + 8}, cuda.dtype(at::kHalf))));
    CHECK_FALSE(group->fits_gather(at::zeros({7}, cuda.dtype(at::kBFloat16))));
    CHECK_FALSE(group->fits_gather(at::zeros({16}, cuda.dtype(at::kBFloat16)).narrow(0, 1, 8)));
}
