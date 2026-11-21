#pragma once

#include "pulsarlm/ops/expert_activation.hpp"

#include <ATen/core/Tensor.h>

#include <cstdint>

namespace pulsarlm::ops::cuda {

// Each token's sum over its top_k experts of top_weights times down_proj(activation(gate) * up) of its row of h
// [tokens, hidden], with top_experts (int64) and top_weights (fp32) [tokens, top_k], gate_up_proj [experts, 2 *
// intermediate, hidden], gate rows first, and down_proj [experts, hidden, intermediate]. gate, up and each expert's
// output are rounded to the dtype, the activation before the product with up, and the weighted outputs are summed in
// fp32 in top_k order. Nothing is read back to the host.
at::Tensor routed_experts(
    const at::Tensor& h,
    const at::Tensor& top_experts,
    const at::Tensor& top_weights,
    const at::Tensor& gate_up_proj,
    const at::Tensor& down_proj,
    expert_activation activation,
    const at::Tensor& counters
);

// ops::cuda::routed_experts with each token's top_k experts by the softmax in fp32 of its router_logits [tokens,
// experts], whose rows may lie apart, in descending order with the lower expert first among equal probabilities,
// weighted by their probabilities divided by their sum. gate_up_proj and down_proj hold the experts [first_expert,
// first_expert + gate_up_proj.size(0)); the assignments to the other experts are left out of the sums.
at::Tensor routed_experts(
    const at::Tensor& h,
    const at::Tensor& router_logits,
    int64_t top_k,
    int64_t first_expert,
    const at::Tensor& gate_up_proj,
    const at::Tensor& down_proj,
    expert_activation activation,
    const at::Tensor& counters
);

// The arrival counters (arrival_counters.cuh) ops::cuda::routed_experts needs for tokens rows of h, each routed to
// top_k of the held experts, as both forms use them.
int64_t
routed_experts_counters(int64_t tokens, int64_t top_k, int64_t held_experts, int64_t intermediate, int64_t hidden);

}  // namespace pulsarlm::ops::cuda
