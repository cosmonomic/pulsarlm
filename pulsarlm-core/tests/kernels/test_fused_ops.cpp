#include "pulsarlm/kernels/ops.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAFunctions.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cstdint>
#include <tuple>

// The fused hyper-connection and MoE routing ops against the ATen ops they replace.

namespace {

constexpr int64_t kTokens = 5;
constexpr int64_t kStreams = 4;
constexpr int64_t kHidden = 96;

double tolerance(at::ScalarType dtype) {
    return dtype == at::kFloat ? 1e-5 : 1e-2;
}

}  // namespace

TEST_CASE("stream_rmsnorm normalizes each stream over its hidden entries", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const auto dtype = GENERATE(at::kFloat, at::kBFloat16);
    const double eps = 1e-6;
    at::Tensor x = at::randn({kTokens, kStreams * kHidden}, at::device(at::kCUDA).dtype(dtype));
    at::Tensor weight = (1.0 + 0.1 * at::randn({kStreams * kHidden})).to(at::kCUDA);

    at::Tensor got = pulsarlm::stream_rmsnorm_cuda(x, weight, kStreams, eps);

    at::Tensor streams = x.to(at::kFloat).view({kTokens, kStreams, kHidden});
    at::Tensor normed = streams * at::rsqrt(streams.pow(2).mean(-1, true) + eps);
    at::Tensor expected = (normed.flatten(1) * weight).to(dtype);
    CHECK(at::allclose(got.to(at::kFloat), expected.to(at::kFloat), tolerance(dtype), tolerance(dtype)));
}

TEST_CASE("hyper_connection_mix gates and averages the streams and weighs the injection", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const auto dtype = GENERATE(at::kFloat, at::kBFloat16);
    auto opts = at::device(at::kCUDA).dtype(dtype);
    at::Tensor normed = at::randn({kTokens, kStreams * kHidden}, opts);
    at::Tensor gate_logits = at::randn({kTokens, kStreams * kHidden}, opts);
    at::Tensor inject_logits = at::randn({kTokens, 16 + kStreams}, opts).narrow(1, 16, kStreams);

    auto [input, injection_weights] = pulsarlm::hyper_connection_mix_cuda(normed, gate_logits, inject_logits);

    at::Tensor gates = at::sigmoid(gate_logits).view({kTokens, kStreams, kHidden});
    at::Tensor expected_input = (gates * normed.view({kTokens, kStreams, kHidden})).mean(-2);
    at::Tensor expected_injection = 2 * at::sigmoid(inject_logits / static_cast<double>(kStreams));
    CHECK(at::allclose(input.to(at::kFloat), expected_input.to(at::kFloat), tolerance(dtype), tolerance(dtype)));
    CHECK(at::allclose(
        injection_weights.to(at::kFloat),
        expected_injection.to(at::kFloat),
        tolerance(dtype),
        tolerance(dtype)
    ));
}

TEST_CASE("hyper_connection_inject adds the block output to each stream by its weight", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const auto dtype = GENERATE(at::kFloat, at::kBFloat16);
    auto opts = at::device(at::kCUDA).dtype(dtype);
    at::Tensor x = at::randn({kTokens, kStreams * kHidden}, opts);
    at::Tensor out = at::randn({kTokens, kHidden}, opts);
    at::Tensor injection_weights = at::randn({kTokens, kStreams}, opts);

    at::Tensor got = pulsarlm::hyper_connection_inject_cuda(x, out, injection_weights);

    at::Tensor expected = x + (out.unsqueeze(1) * injection_weights.unsqueeze(-1)).flatten(1);
    CHECK(at::equal(got, expected));
}

TEST_CASE("route_top_k picks each token's most probable experts and renormalizes them", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const auto dtype = GENERATE(at::kFloat, at::kBFloat16);
    const int64_t experts = GENERATE(16, 512);
    const int64_t k = 10;
    at::Tensor logits = at::randn({kTokens, experts + 1}, at::device(at::kCUDA).dtype(dtype)).narrow(1, 0, experts);

    auto [weights, chosen] = pulsarlm::route_top_k_cuda(logits, k);

    at::Tensor probabilities = at::softmax(logits, -1, at::kFloat);
    at::Tensor top = std::get<0>(at::topk(probabilities, k, -1));
    CHECK(at::allclose(probabilities.gather(1, chosen), top, 1e-6, 0.0));
    CHECK(at::allclose(weights, top / top.sum(-1, true), 1e-6, 1e-6));
}

TEST_CASE("silu_mul multiplies silu of the gate half by the up half", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const auto dtype = GENERATE(at::kFloat, at::kBFloat16);
    at::Tensor x = at::randn({kTokens, 2 * kHidden}, at::device(at::kCUDA).dtype(dtype));

    at::Tensor got = pulsarlm::silu_mul_cuda(x);

    at::Tensor expected = at::silu(x.narrow(1, 0, kHidden)) * x.narrow(1, kHidden, kHidden);
    CHECK(at::allclose(got.to(at::kFloat), expected.to(at::kFloat), tolerance(dtype), tolerance(dtype)));
}
