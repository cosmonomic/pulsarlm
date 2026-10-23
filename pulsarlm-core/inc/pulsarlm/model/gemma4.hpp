#pragma once

#include "pulsarlm/model/interface.hpp"
#include "pulsarlm/rope.hpp"

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

enum class gemma4_layer_type { sliding_attention, full_attention };

// Hugging Face's text config. global_n_kv_heads is the KV heads of a full-attention layer. The last
// num_kv_shared_layers layers keep no KV: each reads the KV of the last earlier layer of its type. A
// hidden_size_per_layer_input of 0 means no per-layer inputs. With enable_moe_block, every layer adds the output of
// its top_k_experts experts to its MLP's.
struct gemma4_config {
    std::int64_t vocab_size;
    std::int64_t hidden_size;
    std::int64_t n_heads;
    std::int64_t n_kv_heads;
    std::int64_t head_dim;
    std::int64_t global_n_kv_heads;
    std::int64_t global_head_dim;
    bool attention_k_eq_v;
    std::vector<gemma4_layer_type> layer_types;
    std::int64_t sliding_window;
    rope_parameters sliding_rope;
    rope_parameters full_rope;
    double rms_norm_eps;
    std::optional<double> final_logit_softcapping;
    std::int64_t num_kv_shared_layers;
    std::int64_t hidden_size_per_layer_input;
    bool enable_moe_block = false;
    std::int64_t top_k_experts = 0;
};

class Gemma4Model : public Model {
    struct LayerWeights {
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
    std::vector<LayerWeights> layers;
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
        const GroupBatch& g,
        std::span<const kv> lines
    ) const;
    at::Tensor attention(const at::Tensor& h, std::int64_t layer, const GroupBatch& g, std::span<const kv> lines) const;
    // The experts' output for the hidden states x a layer's MLP block starts from.
    at::Tensor experts(const at::Tensor& x, std::int64_t layer) const;
    at::Tensor head(const at::Tensor& x) const;
    at::Tensor group_forward(const GroupBatch& g, std::span<const kv> lines) const;

  public:
    // weights are on the lines' device and dtype.
    Gemma4Model(gemma4_config config, const c10::Dict<std::string, at::Tensor>& weights);

    std::vector<kv_layer> kv_layers() const override;
    StepLogits forward(const StepBatch& batch, std::span<const kv> lines) override;
};

}  // namespace pulsarlm
