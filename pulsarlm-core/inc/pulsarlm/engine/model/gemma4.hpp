#pragma once

#include "pulsarlm/engine/model/interface.hpp"
#include "pulsarlm/engine/model/gemma4_config.hpp"
#include "pulsarlm/kernels/rope.hpp"

#include <ATen/core/Dict.h>
#include <ATen/core/Tensor.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

// Gemma 4 text decoder, gemma4 (E2B, E4B) and gemma4_unified (12B), over the paged CUDA ops. Weights carry Hugging
// Face's names under model.language_model.

namespace pulsarlm {

class gemma4_model : public model {
    struct layer_weights {
        at::Tensor input_layernorm;
        at::Tensor post_attention_layernorm;
        at::Tensor pre_feedforward_layernorm;
        at::Tensor post_feedforward_layernorm;
        at::Tensor q_proj;
        at::Tensor q_norm;
        // Undefined in a layer reading another layer's KV.
        at::Tensor k_proj;
        at::Tensor k_norm;
        // Undefined where K=V or the layer reads another layer's KV.
        at::Tensor v_proj;
        at::Tensor o_proj;
        at::Tensor gate_proj;
        at::Tensor up_proj;
        at::Tensor down_proj;
        at::Tensor layer_scalar;
        // Defined with per-layer inputs.
        at::Tensor per_layer_input_gate;
        at::Tensor per_layer_projection;
        at::Tensor post_per_layer_input_norm;
        // Defined with MoE blocks.
        at::Tensor router_proj;
        at::Tensor router_scale;
        at::Tensor router_per_expert_scale;
        at::Tensor experts_gate_up_proj;  // [num_experts, 2 * moe_intermediate_size, hidden_size]
        at::Tensor experts_down_proj;  // [num_experts, hidden_size, moe_intermediate_size]
        at::Tensor pre_feedforward_layernorm_2;
        at::Tensor post_feedforward_layernorm_1;
        at::Tensor post_feedforward_layernorm_2;
    };

    gemma4_config config;
    std::int64_t n_layers;
    // The KV layer each layer writes or reads. A layer keeping KV is KV layer of its own index.
    std::vector<std::size_t> kv_layer_of;
    // Each layer's index among the layers reading its KV layer, the one keeping it first.
    std::vector<std::size_t> reader_of;
    std::vector<kv_layer> kv_specs;
    at::Tensor embed_tokens;
    at::Tensor norm;
    at::Tensor embed_tokens_per_layer;
    at::Tensor per_layer_model_projection;
    at::Tensor per_layer_projection_norm;
    std::vector<layer_weights> layers;
    // For the weightless RMSNorms: on V, one per head dim, and in the router.
    at::Tensor sliding_ones;
    at::Tensor full_ones;
    at::Tensor hidden_ones;

    bool sliding(std::int64_t layer) const;
    at::Tensor per_layer_inputs(const at::Tensor& tokens, const at::Tensor& embedded) const;
    at::Tensor block(
        at::Tensor x,
        std::int64_t layer,
        const at::Tensor& per_layer_input,
        const group_batch& g,
        std::span<const kv> lines
    ) const;
    at::Tensor
    attention(const at::Tensor& h, std::int64_t layer, const group_batch& g, std::span<const kv> lines) const;
    // The experts' output for the hidden states x a layer's MLP block starts from.
    at::Tensor experts(const at::Tensor& x, std::int64_t layer) const;
    at::Tensor head(const at::Tensor& x) const;
    at::Tensor group_forward(const group_batch& g, std::span<const kv> lines) const;

  public:
    // weights are on the lines' device and dtype.
    gemma4_model(gemma4_config config, const c10::Dict<std::string, at::Tensor>& weights);

    std::vector<kv_layer> kv_layers() const override;
    step_logits forward(const step_batch& batch, std::span<const kv> lines) override;
};

}  // namespace pulsarlm
