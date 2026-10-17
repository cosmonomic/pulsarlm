#pragma once

#include "pulsarlm/model/interface.hpp"
#include "pulsarlm/rope.hpp"

#include <ATen/core/Tensor.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>

// The steps of a paged decoder forward that every model takes the same way.

namespace pulsarlm {

// The projected queries q [ntok, n_q_heads * head_dim] rotated to their positions pos, as [ntok, n_q_heads, head_dim].
at::Tensor
rope_q(const at::Tensor& q, const at::Tensor& pos, int64_t n_q_heads, int64_t head_dim, const rope_parameters& rope);

// Writes the group's new keys k and values v, [ntok, n_kv_heads, head_dim] with k unrotated, into KV layer kv_layer's
// lines, then attends them with the rotated queries q [ntok, n_q_heads, head_dim] and returns o [ntok, n_q_heads,
// head_dim]. Undefined k and v leave the lines as they are, for a layer reading another layer's KV. sliding_window and
// k_norm_weight are as attn_prefill_cuda takes them. When g.page_mass[kv_layer] is defined, the layer maxes the mass
// attn_prefill_cuda measures into it.
at::Tensor paged_attention(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const kv& lines,
    std::size_t kv_layer,
    const GroupBatch& g,
    double scale,
    const rope_parameters& rope,
    std::optional<int64_t> sliding_window = std::nullopt,
    const std::optional<at::Tensor>& k_norm_weight = std::nullopt
);

// Each token's sum over its top_k experts of top_weights times down_proj(activation(gate) * up) of its row of h
// [ntok, hidden], with top_experts and top_weights (fp32) [ntok, top_k], gate_up_proj
// [num_experts, 2 * intermediate, hidden] and down_proj [num_experts, hidden, intermediate]. The assignments are sorted
// by expert and counted on the device, so routing never waits for the host.
at::Tensor routed_experts(
    const at::Tensor& h,
    const at::Tensor& top_experts,
    const at::Tensor& top_weights,
    const at::Tensor& gate_up_proj,
    const at::Tensor& down_proj,
    const std::function<at::Tensor(const at::Tensor&)>& activation
);

// The rows of x whose logits g asks for.
at::Tensor logit_inputs(const at::Tensor& x, const GroupBatch& g);

// The logits group_forward returns for each present group of batch.
StepLogits paged_forward(const StepBatch& batch, const std::function<at::Tensor(const GroupBatch&)>& group_forward);

}  // namespace pulsarlm
