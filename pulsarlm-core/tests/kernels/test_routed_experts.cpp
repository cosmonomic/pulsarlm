#include "pulsarlm/ops/routed_experts.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <cstdint>
#include <tuple>

// Checks ops::cuda::routed_experts against a dense ATen reference that rounds where the op rounds. Gate, up, the
// activation, the product and each expert's output are rounded to the dtype, and the weighted sum is rounded once.

namespace {

constexpr int64_t expert_count = 16;
// The last experts no token is routed to.
constexpr int64_t unrouted_experts = 4;

at::TensorOptions cuda(at::ScalarType dtype) {
    return at::device(at::kCUDA).dtype(dtype);
}

at::Tensor activated(const at::Tensor& x, pulsarlm::ops::expert_activation activation) {
    return activation == pulsarlm::ops::expert_activation::silu ? at::silu(x) : at::gelu(x, "tanh");
}

at::Tensor reference(
    const at::Tensor& h,
    const at::Tensor& top_experts,
    const at::Tensor& top_weights,
    const at::Tensor& gate_up_proj,
    const at::Tensor& down_proj,
    pulsarlm::ops::expert_activation activation
) {
    const at::ScalarType dtype = h.scalar_type();
    const int64_t tokens = h.size(0);
    const int64_t top_k = top_experts.size(1);
    const int64_t intermediate = down_proj.size(2);
    at::Tensor experts = top_experts.flatten();
    at::Tensor rows = h.to(at::kFloat).repeat_interleave(top_k, 0).unsqueeze(2);
    at::Tensor gate_up = at::bmm(gate_up_proj.index_select(0, experts).to(at::kFloat), rows).squeeze(2).to(dtype);
    at::Tensor gate = activated(gate_up.narrow(1, 0, intermediate), activation);
    at::Tensor product = (gate.to(at::kFloat) * gate_up.narrow(1, intermediate, intermediate).to(at::kFloat)).to(dtype);
    at::Tensor down = at::bmm(down_proj.index_select(0, experts).to(at::kFloat), product.to(at::kFloat).unsqueeze(2))
                          .squeeze(2);
    at::Tensor weighted = down.to(dtype).to(at::kFloat) * top_weights.flatten().unsqueeze(1);
    return weighted.view({tokens, top_k, -1}).sum(1).to(dtype);
}

std::tuple<at::Tensor, at::Tensor> routing(int64_t tokens, int64_t top_k) {
    at::Tensor scores = at::rand({tokens, expert_count - unrouted_experts}, cuda(at::kFloat));
    at::Tensor top_experts = std::get<1>(at::topk(scores, top_k, -1)).contiguous();
    at::Tensor top_weights = at::rand({tokens, top_k}, cuda(at::kFloat));
    return {top_experts, top_weights};
}

// The zeroed arrival counters ops::cuda::routed_experts needs for h routed to top_k of gate_up_proj's experts.
at::Tensor
counters_for(const at::Tensor& h, int64_t top_k, const at::Tensor& gate_up_proj, const at::Tensor& down_proj) {
    return at::zeros(
        {pulsarlm::ops::cuda::routed_experts_counters(
            h.size(0),
            top_k,
            gate_up_proj.size(0),
            down_proj.size(2),
            h.size(1)
        )},
        cuda(at::kInt)
    );
}

double max_relative_error(const at::Tensor& got, const at::Tensor& expected) {
    return ((got.to(at::kFloat) - expected.to(at::kFloat)).abs().max() / expected.to(at::kFloat).abs().max())
        .item<double>();
}

}  // namespace

TEST_CASE("routed_experts matches a dense reference", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const auto dtype = GENERATE(at::kFloat, at::kBFloat16, at::kHalf);
    const auto activation = GENERATE(
        pulsarlm::ops::expert_activation::silu,
        pulsarlm::ops::expert_activation::gelu_tanh
    );
    const int64_t tokens = GENERATE(1, 4, 300);
    const int64_t top_k = GENERATE(1, 8);
    const auto [hidden, intermediate] = GENERATE(
        std::tuple<int64_t, int64_t>{96, 40},
        std::tuple<int64_t, int64_t>{256, 128}
    );
    INFO("dtype " << dtype << " tokens " << tokens << " top_k " << top_k << " hidden " << hidden);
    at::manual_seed(3);
    at::Tensor h = at::randn({tokens, hidden}, cuda(dtype));
    at::Tensor gate_up_proj = at::randn({expert_count, 2 * intermediate, hidden}, cuda(dtype)) / std::sqrt(hidden);
    at::Tensor down_proj = at::randn({expert_count, hidden, intermediate}, cuda(dtype)) / std::sqrt(intermediate);
    auto [top_experts, top_weights] = routing(tokens, top_k);

    at::Tensor got = pulsarlm::ops::cuda::routed_experts(
        h,
        top_experts,
        top_weights,
        gate_up_proj,
        down_proj,
        activation,
        counters_for(h, top_experts.size(1), gate_up_proj, down_proj)
    );
    at::Tensor expected = reference(h, top_experts, top_weights, gate_up_proj, down_proj, activation);
    CHECK(got.scalar_type() == dtype);
    CHECK(max_relative_error(got, expected) < (dtype == at::kFloat ? 1e-5 : 2e-2));
}

TEST_CASE("routed_experts routes each token to its most probable experts by its router logits", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const auto dtype = GENERATE(at::kFloat, at::kBFloat16);
    const int64_t tokens = GENERATE(1, 4, 300);
    const int64_t top_k = GENERATE(1, 8);
    const int64_t hidden = 96;
    const int64_t intermediate = 40;
    INFO("dtype " << dtype << " tokens " << tokens << " top_k " << top_k);
    at::manual_seed(7);
    at::Tensor h = at::randn({tokens, hidden}, cuda(dtype));
    at::Tensor gate_up_proj = at::randn({expert_count, 2 * intermediate, hidden}, cuda(dtype)) / std::sqrt(hidden);
    at::Tensor down_proj = at::randn({expert_count, hidden, intermediate}, cuda(dtype)) / std::sqrt(intermediate);
    // Router logits whose rows lie apart.
    at::Tensor router_logits = at::randn({tokens, expert_count + 3}, cuda(dtype)).narrow(1, 0, expert_count);
    const auto activation = pulsarlm::ops::expert_activation::silu;

    at::Tensor got = pulsarlm::ops::cuda::routed_experts(
        h,
        router_logits,
        top_k,
        0,
        gate_up_proj,
        down_proj,
        activation,
        counters_for(h, top_k, gate_up_proj, down_proj)
    );

    at::Tensor probabilities = at::softmax(router_logits, -1, at::kFloat);
    auto [top_probabilities, top_experts] = at::topk(probabilities, top_k, -1);
    at::Tensor top_weights = top_probabilities / top_probabilities.sum(-1, true);
    at::Tensor expected = reference(h, top_experts, top_weights, gate_up_proj, down_proj, activation);
    CHECK(max_relative_error(got, expected) < (dtype == at::kFloat ? 1e-5 : 2e-2));
}

// Logits of four distinct values, so that a token's k-th expert ties with others. Each token takes the experts of a
// stable descending sort of its probabilities, the lower expert first among equals, weighted by their renormalized
// probabilities, as the top_experts form given those does.
TEST_CASE("routed_experts takes the lower expert among tied router logits", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const int64_t tokens = 64;
    const int64_t top_k = GENERATE(1, 3, 8);
    // 2048 experts give each routing thread two of them.
    const int64_t experts = GENERATE(expert_count, int64_t{2048});
    const int64_t hidden = 96;
    const int64_t intermediate = 40;
    INFO("top_k " << top_k << " experts " << experts);
    at::manual_seed(13);
    at::Tensor h = at::randn({tokens, hidden}, cuda(at::kFloat));
    at::Tensor gate_up_proj = at::randn({experts, 2 * intermediate, hidden}, cuda(at::kFloat)) / std::sqrt(hidden);
    at::Tensor down_proj = at::randn({experts, hidden, intermediate}, cuda(at::kFloat)) / std::sqrt(intermediate);
    at::Tensor router_logits = at::floor(at::rand({tokens, experts}, cuda(at::kFloat)) * 4.0);
    const auto activation = pulsarlm::ops::expert_activation::silu;

    at::Tensor got = pulsarlm::ops::cuda::routed_experts(
        h,
        router_logits,
        top_k,
        0,
        gate_up_proj,
        down_proj,
        activation,
        counters_for(h, top_k, gate_up_proj, down_proj)
    );

    at::Tensor probabilities = at::softmax(router_logits, -1, at::kFloat);
    auto [sorted, order] = at::sort(probabilities, /*stable=*/true, /*dim=*/-1, /*descending=*/true);
    at::Tensor top_experts = order.narrow(1, 0, top_k).contiguous();
    at::Tensor top_probabilities = sorted.narrow(1, 0, top_k);
    at::Tensor top_weights = (top_probabilities / top_probabilities.sum(-1, true)).contiguous();
    at::Tensor expected = pulsarlm::ops::cuda::routed_experts(
        h,
        top_experts,
        top_weights,
        gate_up_proj,
        down_proj,
        activation,
        counters_for(h, top_experts.size(1), gate_up_proj, down_proj)
    );
    CHECK(max_relative_error(got, expected) < 1e-6);
}

TEST_CASE("routed_experts over each GPU's held experts sums to routed_experts over all of them", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const auto dtype = GENERATE(at::kFloat, at::kBFloat16);
    const int64_t tokens = GENERATE(1, 4, 300);
    const int64_t top_k = 8;
    const int64_t hidden = 96;
    const int64_t intermediate = 40;
    const int64_t gpus = 4;
    const int64_t held = expert_count / gpus;
    INFO("dtype " << dtype << " tokens " << tokens);
    at::manual_seed(11);
    at::Tensor h = at::randn({tokens, hidden}, cuda(dtype));
    at::Tensor gate_up_proj = at::randn({expert_count, 2 * intermediate, hidden}, cuda(dtype)) / std::sqrt(hidden);
    at::Tensor down_proj = at::randn({expert_count, hidden, intermediate}, cuda(dtype)) / std::sqrt(intermediate);
    at::Tensor router_logits = at::randn({tokens, expert_count}, cuda(dtype));
    const auto activation = pulsarlm::ops::expert_activation::silu;

    at::Tensor whole = pulsarlm::ops::cuda::routed_experts(
        h,
        router_logits,
        top_k,
        0,
        gate_up_proj,
        down_proj,
        activation,
        counters_for(h, top_k, gate_up_proj, down_proj)
    );
    at::Tensor summed = at::zeros({tokens, hidden}, cuda(at::kFloat));
    for (int64_t gpu = 0; gpu < gpus; gpu++) {
        summed += pulsarlm::ops::cuda::routed_experts(
                      h,
                      router_logits,
                      top_k,
                      gpu * held,
                      gate_up_proj.narrow(0, gpu * held, held),
                      down_proj.narrow(0, gpu * held, held),
                      activation,
                      counters_for(h, top_k, gate_up_proj.narrow(0, gpu * held, held), down_proj)
        )
                      .to(at::kFloat);
    }

    CHECK(max_relative_error(summed, whole) < (dtype == at::kFloat ? 1e-5 : 2e-2));
}

TEST_CASE("routed_experts replays from a CUDA graph with new inputs", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const int64_t tokens = 4;
    const int64_t top_k = 8;
    const int64_t hidden = 256;
    const int64_t intermediate = 128;
    const auto activation = pulsarlm::ops::expert_activation::silu;
    at::manual_seed(5);
    at::Tensor gate_up_proj = at::randn({expert_count, 2 * intermediate, hidden}, cuda(at::kBFloat16)) / 16.0;
    at::Tensor down_proj = at::randn({expert_count, hidden, intermediate}, cuda(at::kBFloat16)) / 16.0;
    at::Tensor h = at::randn({tokens, hidden}, cuda(at::kBFloat16));
    auto [top_experts, top_weights] = routing(tokens, top_k);

    c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromPool());
    pulsarlm::ops::cuda::routed_experts(
        h,
        top_experts,
        top_weights,
        gate_up_proj,
        down_proj,
        activation,
        counters_for(h, top_experts.size(1), gate_up_proj, down_proj)
    );
    at::cuda::CUDAGraph graph;
    at::Tensor graph_counters = counters_for(h, top_k, gate_up_proj, down_proj);
    graph.capture_begin();
    at::Tensor captured = pulsarlm::ops::cuda::routed_experts(
        h,
        top_experts,
        top_weights,
        gate_up_proj,
        down_proj,
        activation,
        graph_counters
    );
    graph.capture_end();

    auto [next_experts, next_weights] = routing(tokens, top_k);
    at::Tensor next_h = at::randn({tokens, hidden}, cuda(at::kBFloat16));
    h.copy_(next_h);
    top_experts.copy_(next_experts);
    top_weights.copy_(next_weights);
    graph.replay();
    at::Tensor eager = pulsarlm::ops::cuda::routed_experts(
        next_h,
        next_experts,
        next_weights,
        gate_up_proj,
        down_proj,
        activation,
        counters_for(next_h, top_k, gate_up_proj, down_proj)
    );
    CHECK(at::equal(captured, eager));
}
