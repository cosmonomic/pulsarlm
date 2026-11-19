#pragma once

#include "pulsarlm/engine/model/interface.hpp"
#include "pulsarlm/kernels/gated_deltanet.hpp"
#include "pulsarlm/kernels/rope.hpp"

#include <ATen/core/Tensor.h>

#include <cstddef>
#include <cstdint>
#include <optional>

// The steps of a paged decoder forward that every model takes the same way.

namespace pulsarlm {

// The projected queries q [ntok, n_q_heads * head_dim] rotated to their positions pos, as [ntok, n_q_heads, head_dim].
at::Tensor
rope_q(const at::Tensor& q, const at::Tensor& pos, int64_t n_q_heads, int64_t head_dim, const rope_parameters& rope);

// Writes the batch's new keys k and values v, [ntok, n_kv_heads, head_dim] with k unrotated, into KV layer kv_layer's
// lines, then attends them with the rotated queries q [ntok, n_q_heads, head_dim] and returns o [ntok, n_q_heads,
// head_dim]. Undefined k and v leave the lines as they are, for a layer reading another layer's KV. sliding_window and
// k_norm_weight are as attention_layer holds them. When g.page_mass_sum[kv_layer] is defined, the layer writes the
// mass_sum attn_prefill_cuda measures into reader's columns of it, reader being the layer's index among those reading
// kv_layer, and reader 0 writes the weight_sum into g.page_weight_sum[kv_layer]. In a verify batch each query measures
// its mass in a view of its own (attn_prefill_verify_cuda).
at::Tensor paged_attention(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const kv& lines,
    std::size_t kv_layer,
    std::size_t reader,
    const model::batch& g,
    double scale,
    const rope_parameters& rope,
    std::optional<int64_t> sliding_window = std::nullopt,
    const std::optional<at::Tensor>& k_norm_weight = std::nullopt
);

// Writes the hidden rows [rows of g, hidden] at each sequence's last row of g and at each of its rows ending a page into
// the last_hidden of line, KV layer kv_layer's, so a line keeps the hidden row at the latest position written into it.
void keep_last_hidden(const model::batch& g, const at::Tensor& hidden, const kv& line, std::size_t kv_layer);

// The state rows of g of KV layer kv_layer, a recurrent layer keeping a state.
state_rows state_layer_rows(const model::batch& g, std::size_t kv_layer);
// The state rows of g of KV layer kv_layer, a recurrent layer keeping a window of inputs.
state_rows window_layer_rows(const model::batch& g, std::size_t kv_layer);

// The projected rows of a Gated DeltaNet sublayer.
struct gated_deltanet_inputs {
    at::Tensor qkv;
    at::Tensor a;
    at::Tensor b;
    at::Tensor z;
};

// The weights of a Gated DeltaNet sublayer that its kernels read.
struct gated_deltanet_weights {
    at::Tensor conv;
    at::Tensor dt_bias;
    at::Tensor neg_a;
    at::Tensor norm;
};

// The gated outputs of a Gated DeltaNet sublayer for the projected rows of b. Writes the layer's state and
// convolution window into their lines.
at::Tensor gated_deltanet(
    const model::batch& b,
    const gated_deltanet_inputs& projected,
    const gated_deltanet_weights& weights,
    const kv& state_lines,
    const kv& conv_lines,
    std::size_t state_kv_layer,
    std::size_t conv_kv_layer,
    std::int64_t key_heads,
    double rms_norm_eps,
    gate_activation activation
);

// The rows of x whose logits g asks for.
at::Tensor logit_inputs(const at::Tensor& x, const model::batch& g);

}  // namespace pulsarlm
