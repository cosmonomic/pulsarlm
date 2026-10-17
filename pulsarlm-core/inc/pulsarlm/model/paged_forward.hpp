#pragma once

#include "pulsarlm/model/interface.hpp"  // GroupBatch, StepBatch, StepLogits
#include "pulsarlm/rope.hpp"

#include <ATen/core/Tensor.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>

// Model-agnostic primitives of the paged transformer forward, shared by every
// native `*Model`. The per-layer arithmetic (norm placement, projection bias, MLP
// shape, lm_head tying) stays model-specific in the concrete `*Model`.

namespace pulsarlm {

// Apply RoPE (rotate_half over head_dim) to a group's q, the projection output
// [ntok, n_q_heads * head_dim]; returns it roped and reshaped to
// [ntok, n_q_heads, head_dim], ready for paged attention.
at::Tensor
rope_q(const at::Tensor& q, const at::Tensor& pos, int64_t n_q_heads, int64_t head_dim, const rope_parameters& rope);

// write_kv the group's new K/V into lines, KV layer kv_layer's, then run the group's
// paged attention (prefill vs decode by g.is_prefill()) and return o
// [ntok, n_q_heads, head_dim]. q/k/v are already projected and q roped in the
// group's [ntok, heads, head_dim] layout; k is unrotated. Undefined k and v leave the
// lines as they are, for a layer reading another layer's KV. sliding_window and
// k_norm_weight are as attn_prefill_cuda takes them. When g.page_mass[kv_layer] is
// defined, the layer maxes its attention mass into it.
// lse_capture, when defined, receives this group's per-(query row, query head)
// logsumexp -- the softmax denominator the attention used. fp32 [ntok, n_q_heads].
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
    const std::optional<at::Tensor>& k_norm_weight = std::nullopt,
    const at::Tensor& lse_capture = {}
);

// The rows of x whose logits g asks for.
at::Tensor logit_inputs(const at::Tensor& x, const GroupBatch& g);

// Forward skeleton for one group: embed lookup -> n_layers decoder blocks ->
// head. layer_fn(x, layer) is the model-specific decoder block; head_fn(x) is the
// model-specific final norm + lm_head.
at::Tensor paged_group_forward(
    const at::Tensor& embed,
    const GroupBatch& g,
    int64_t n_layers,
    const std::function<at::Tensor(at::Tensor, int64_t)>& layer_fn,
    const std::function<at::Tensor(at::Tensor)>& head_fn
);

// Dispatch a step's present groups through group_fn, packing per-group StepLogits.
// group_fn is the model's group forward.
StepLogits paged_forward(const StepBatch& batch, const std::function<at::Tensor(const GroupBatch&)>& group_fn);

}  // namespace pulsarlm
