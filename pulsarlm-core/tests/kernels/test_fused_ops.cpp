#include "pulsarlm/ops/gating.hpp"
#include "pulsarlm/ops/hyper_connection.hpp"
#include "pulsarlm/ops/normalization.hpp"
#include "pulsarlm/ops/small_batch_linear.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAFunctions.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cstdint>
#include <tuple>

// The fused hyper-connection and MoE routing ops against the ATen ops they replace.

namespace {

constexpr int64_t token_count = 5;
constexpr int64_t stream_count = 4;
constexpr int64_t hidden_size = 96;

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
    at::Tensor x = at::randn({token_count, stream_count * hidden_size}, at::device(at::kCUDA).dtype(dtype));
    at::Tensor weight = (1.0 + 0.1 * at::randn({stream_count * hidden_size})).to(at::kCUDA);

    at::Tensor got = pulsarlm::ops::cuda::stream_rmsnorm(x, weight, stream_count, eps);

    at::Tensor streams = x.to(at::kFloat).view({token_count, stream_count, hidden_size});
    at::Tensor normed = streams * at::rsqrt(streams.pow(2).mean(-1, true) + eps);
    at::Tensor expected = (normed.flatten(1) * weight).to(dtype);
    CHECK(at::allclose(got.to(at::kFloat), expected.to(at::kFloat), tolerance(dtype), tolerance(dtype)));
}

TEST_CASE("stream_rmsnorm_linear normalizes the streams and projects them", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const auto dtype = GENERATE(at::kFloat, at::kBFloat16);
    // Up to 4 tokens run as one kernel, 5 through stream_rmsnorm and cuBLAS.
    const int64_t tokens = GENERATE(1, 4, 5);
    const double eps = 1e-6;
    auto opts = at::device(at::kCUDA).dtype(dtype);
    at::Tensor x = at::randn({tokens, stream_count * hidden_size}, opts);
    at::Tensor norm_weight = (1.0 + 0.1 * at::randn({stream_count * hidden_size})).to(at::kCUDA);
    at::Tensor weight = at::randn({68, stream_count * hidden_size}, opts) / 20.0;

    auto [normed, projected] = pulsarlm::ops::cuda::stream_rmsnorm_linear(x, norm_weight, stream_count, eps, weight);

    at::Tensor expected_normed = pulsarlm::ops::cuda::stream_rmsnorm(x, norm_weight, stream_count, eps);
    CHECK(at::allclose(normed.to(at::kFloat), expected_normed.to(at::kFloat), tolerance(dtype), tolerance(dtype)));
    at::Tensor expected = at::linear(expected_normed.to(at::kFloat), weight.to(at::kFloat));
    CHECK(at::allclose(projected.to(at::kFloat), expected, tolerance(dtype), tolerance(dtype)));
}

TEST_CASE("hyper_connection_mix gates and averages the streams and weighs the injection", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const auto dtype = GENERATE(at::kFloat, at::kBFloat16);
    // Up to 4 tokens run as one kernel, 5 through cuBLAS and the mix kernel.
    const int64_t tokens = GENERATE(1, 4, 5);
    constexpr int64_t rank = 40;
    auto opts = at::device(at::kCUDA).dtype(dtype);
    at::Tensor normed = at::randn({tokens, stream_count * hidden_size}, opts);
    at::Tensor projected = at::randn({tokens, rank + stream_count}, opts);
    at::Tensor up_weight = at::randn({stream_count * hidden_size, rank}, opts) / 4.0;

    auto [input, injection_weights] = pulsarlm::ops::cuda::hyper_connection_mix(
        normed,
        projected.narrow(1, 0, rank),
        up_weight,
        projected.narrow(1, rank, stream_count)
    );

    at::Tensor mix_weights = at::silu(projected.narrow(1, 0, rank) / static_cast<double>(stream_count));
    at::Tensor gates = at::sigmoid(at::linear(mix_weights, up_weight)).view({tokens, stream_count, hidden_size});
    at::Tensor expected_input = (gates * normed.view({tokens, stream_count, hidden_size})).mean(-2);
    at::Tensor expected_injection = 2 *
        at::sigmoid(projected.narrow(1, rank, stream_count) / static_cast<double>(stream_count));
    CHECK(at::allclose(input.to(at::kFloat), expected_input.to(at::kFloat), tolerance(dtype), tolerance(dtype)));
    CHECK(at::equal(injection_weights, expected_injection));
}

TEST_CASE("sigmoid_gated_add and sigmoid_gated round as the ATen ops do", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const auto dtype = GENERATE(at::kFloat, at::kBFloat16);
    auto opts = at::device(at::kCUDA).dtype(dtype);
    at::Tensor x = at::randn({token_count, hidden_size}, opts);
    at::Tensor y = at::randn({token_count, hidden_size}, opts);
    at::Tensor logits = at::randn({token_count, 3}, opts);
    CHECK(
        at::equal(
            pulsarlm::ops::cuda::sigmoid_gated_add(x, logits.narrow(1, 2, 1), y),
            x + at::sigmoid(logits.narrow(1, 2, 1)) * y
        )
    );

    at::Tensor heads = at::randn({token_count, 3, 2 * hidden_size}, opts);
    at::Tensor o = at::randn({token_count, 3, hidden_size}, opts);
    at::Tensor gate = heads.narrow(-1, hidden_size, hidden_size);
    CHECK(at::equal(pulsarlm::ops::cuda::sigmoid_gated(o, gate), o * at::sigmoid(gate)));
    CHECK(
        at::equal(
            pulsarlm::ops::cuda::sigmoid_gated(o.view({token_count, -1}), gate.reshape({token_count, -1})),
            o.view({token_count, -1}) * at::sigmoid(gate.reshape({token_count, -1}))
        )
    );
}

TEST_CASE("hyper_connection_inject adds the block output to each stream by its weight", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const auto dtype = GENERATE(at::kFloat, at::kBFloat16);
    auto opts = at::device(at::kCUDA).dtype(dtype);
    at::Tensor x = at::randn({token_count, stream_count * hidden_size}, opts);
    at::Tensor out = at::randn({token_count, hidden_size}, opts);
    at::Tensor injection_weights = at::randn({token_count, stream_count}, opts);

    at::Tensor got = pulsarlm::ops::cuda::hyper_connection_inject(x, out, injection_weights);

    at::Tensor expected = x + (out.unsqueeze(1) * injection_weights.unsqueeze(-1)).flatten(1);
    CHECK(at::equal(got, expected));
}

TEST_CASE("silu_mul multiplies silu of the gate half by the up half", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const auto dtype = GENERATE(at::kFloat, at::kBFloat16);
    at::Tensor x = at::randn({token_count, 2 * hidden_size}, at::device(at::kCUDA).dtype(dtype));

    at::Tensor got = pulsarlm::ops::cuda::silu_mul(x);

    at::Tensor expected = at::silu(x.narrow(1, 0, hidden_size)) * x.narrow(1, hidden_size, hidden_size);
    CHECK(at::allclose(got.to(at::kFloat), expected.to(at::kFloat), tolerance(dtype), tolerance(dtype)));
}

TEST_CASE("small_batch_linear and small_batch_gated_linear match at::linear", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const auto dtype = GENERATE(at::kFloat, at::kBFloat16);
    // Up to 4 tokens run as one kernel, 5 through cuBLAS.
    const int64_t tokens = GENERATE(1, 3, 4, 5);
    auto opts = at::device(at::kCUDA).dtype(dtype);
    at::Tensor x = at::randn({tokens, 2 * stream_count * hidden_size}, opts);
    // 70 columns leave a CTA's last rows past the weight.
    at::Tensor weight = at::randn({70, 2 * stream_count * hidden_size}, opts) / 20.0;
    at::Tensor gate_up = at::randn({2 * 70, 2 * stream_count * hidden_size}, opts) / 20.0;

    at::Tensor expected = at::linear(x.to(at::kFloat), weight.to(at::kFloat));
    CHECK(
        at::allclose(
            pulsarlm::ops::cuda::small_batch_linear(x, weight).to(at::kFloat),
            expected,
            tolerance(dtype),
            tolerance(dtype)
        )
    );
    at::Tensor projected = at::linear(x.to(at::kFloat), gate_up.to(at::kFloat)).to(dtype);
    CHECK(
        at::allclose(
            pulsarlm::ops::cuda::small_batch_gated_linear(x, gate_up).to(at::kFloat),
            pulsarlm::ops::cuda::silu_mul(projected).to(at::kFloat),
            tolerance(dtype),
            tolerance(dtype)
        )
    );
}

TEST_CASE("injected_stream_rmsnorm_linear injects, normalizes and projects the streams", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const auto dtype = GENERATE(at::kFloat, at::kBFloat16);
    // Up to 4 tokens run as one kernel, 5 through the separate ops.
    const int64_t tokens = GENERATE(1, 4, 5);
    const double eps = 1e-6;
    auto opts = at::device(at::kCUDA).dtype(dtype);
    at::Tensor x = at::randn({tokens, stream_count * hidden_size}, opts);
    at::Tensor out = at::randn({tokens, hidden_size}, opts);
    at::Tensor injection_weights = at::randn({tokens, stream_count}, opts);
    at::Tensor norm_weight = (1.0 + 0.1 * at::randn({stream_count * hidden_size})).to(at::kCUDA);
    at::Tensor weight = at::randn({68, stream_count * hidden_size}, opts) / 20.0;

    auto [summed, normed, projected] = pulsarlm::ops::cuda::injected_stream_rmsnorm_linear(
        x,
        out,
        injection_weights,
        norm_weight,
        stream_count,
        eps,
        weight
    );

    at::Tensor expected_summed = pulsarlm::ops::cuda::hyper_connection_inject(x, out, injection_weights);
    CHECK(at::equal(summed, expected_summed));
    at::Tensor expected_normed = pulsarlm::ops::cuda::stream_rmsnorm(expected_summed, norm_weight, stream_count, eps);
    CHECK(at::allclose(normed.to(at::kFloat), expected_normed.to(at::kFloat), tolerance(dtype), tolerance(dtype)));
    at::Tensor expected = at::linear(expected_normed.to(at::kFloat), weight.to(at::kFloat));
    CHECK(at::allclose(projected.to(at::kFloat), expected, tolerance(dtype), tolerance(dtype)));
}
