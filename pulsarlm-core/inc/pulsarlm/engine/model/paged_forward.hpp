#pragma once

#include "pulsarlm/engine/model/interface.hpp"
#include "pulsarlm/kernels/ops.hpp"
#include "pulsarlm/kernels/rope.hpp"

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
// k_norm_weight are as attn_prefill_cuda takes them. When g.page_mass_sum[kv_layer] is defined, the layer writes the
// mass_sum attn_prefill_cuda measures into reader's columns of it, reader being the layer's index among those reading
// kv_layer, and reader 0 writes the weight_sum into g.page_weight_sum[kv_layer].
at::Tensor paged_attention(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const kv& lines,
    std::size_t kv_layer,
    std::size_t reader,
    const group_batch& g,
    double scale,
    const rope_parameters& rope,
    std::optional<int64_t> sliding_window = std::nullopt,
    const std::optional<at::Tensor>& k_norm_weight = std::nullopt
);

// The state rows of g of KV layer kv_layer, a recurrent layer keeping a state.
state_rows state_layer_rows(const group_batch& g, std::size_t kv_layer);
// The state rows of g of KV layer kv_layer, a recurrent layer keeping a window of inputs.
state_rows window_layer_rows(const group_batch& g, std::size_t kv_layer);

// The rows of x whose logits g asks for.
at::Tensor logit_inputs(const at::Tensor& x, const group_batch& g);

// The logits group_forward returns for each present group of batch.
step_logits paged_forward(const step_batch& batch, const std::function<at::Tensor(const group_batch&)>& group_forward);

}  // namespace pulsarlm
