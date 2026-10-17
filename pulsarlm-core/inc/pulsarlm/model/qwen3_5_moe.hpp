#pragma once

#include "pulsarlm/model/interface.hpp"
#include "pulsarlm/model/qwen3_5.hpp"
#include "pulsarlm/rope.hpp"

#include <ATen/core/Dict.h>
#include <ATen/core/Tensor.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

// Qwen3.5-MoE text decoder (qwen3_5_moe), over the paged CUDA ops: Qwen3.5's Gated DeltaNet and gated full-attention
// layers, each followed by a sparse block of each token's top num_experts_per_tok routed experts plus a shared expert
// scaled by its sigmoid gate. Weights carry Hugging Face's names under model.language_model., the untied head
// lm_head.weight.

namespace pulsarlm {

// Hugging Face's text config. The first partial_rotary_factor * head_dim dims of a query or key head rotate, in
// rotate_half pairs among themselves at the frequencies of a head of that many dims; the rest pass through.
struct qwen3_5_moe_config {
    std::int64_t vocab_size;
    std::int64_t hidden_size;
    std::vector<qwen3_5_layer_type> layer_types;
    std::int64_t n_heads;
    std::int64_t n_kv_heads;
    std::int64_t head_dim;
    double rope_theta;
    double partial_rotary_factor;
    std::int64_t linear_n_key_heads;
    std::int64_t linear_n_value_heads;
    std::int64_t linear_key_head_dim;
    std::int64_t linear_value_head_dim;
    std::int64_t linear_conv_kernel;
    double rms_norm_eps;
    std::int64_t num_experts_per_tok;
};

// A Gated DeltaNet layer keeps two recurrent KV layers: its delta rule state S, fp32 [value heads, key head dim, value
// head dim], then its convolution's last linear_conv_kernel - 1 inputs, [conv channels, linear_conv_kernel - 1] in the
// weights' dtype.
class Qwen3_5MoeModel : public Model {
    // Query and key heads are stored with their dims permuted as Qwen3_5Model stores them.
    struct AttentionWeights {
        at::Tensor q_proj;  // [n_heads * 2 * head_dim, hidden], each head's query then gate
        at::Tensor k_proj;
        at::Tensor v_proj;
        at::Tensor o_proj;
        at::Tensor q_norm;  // fp32 1 + weight
        at::Tensor k_norm;
    };

    struct DeltaNetWeights {
        at::Tensor in_proj_qkv;
        at::Tensor in_proj_z;
        at::Tensor in_proj_b;
        at::Tensor in_proj_a;
        at::Tensor conv;  // [conv channels, 1, linear_conv_kernel]
        at::Tensor dt_bias;  // fp32
        at::Tensor neg_a;  // fp32 -exp(A_log)
        at::Tensor norm;
        at::Tensor out_proj;
    };

    struct MoeWeights {
        at::Tensor router;  // [num_experts, hidden]
        at::Tensor experts_gate_up_proj;  // [num_experts, 2 * moe_intermediate_size, hidden]
        at::Tensor experts_down_proj;  // [num_experts, hidden, moe_intermediate_size]
        at::Tensor shared_gate_proj;
        at::Tensor shared_up_proj;
        at::Tensor shared_down_proj;
        at::Tensor shared_expert_gate;  // [1, hidden]
    };

    struct LayerWeights {
        at::Tensor input_layernorm;  // fp32 1 + weight
        at::Tensor post_attention_layernorm;
        MoeWeights moe;
        std::optional<AttentionWeights> attention;
        std::optional<DeltaNetWeights> deltanet;
    };

    qwen3_5_moe_config config;
    std::int64_t n_layers;
    rope_parameters rope;
    // The first KV layer of each layer.
    std::vector<std::size_t> kv_layer_of;
    std::vector<kv_layer> kv_specs;
    at::Tensor embed_tokens;
    at::Tensor lm_head;
    at::Tensor norm;  // fp32 1 + weight
    std::vector<LayerWeights> layers;

    bool full_attention(std::int64_t layer) const;
    at::Tensor attention(const at::Tensor& h, std::int64_t layer, const GroupBatch& g, std::span<const kv> lines) const;
    at::Tensor deltanet(const at::Tensor& h, std::int64_t layer, const GroupBatch& g, std::span<const kv> lines) const;
    at::Tensor moe(const at::Tensor& h, std::int64_t layer) const;
    at::Tensor group_forward(const GroupBatch& g, std::span<const kv> lines) const;

  public:
    // weights are on the lines' device, in the attention lines' dtype.
    Qwen3_5MoeModel(qwen3_5_moe_config config, const c10::Dict<std::string, at::Tensor>& weights);

    std::vector<kv_layer> kv_layers() const override;
    StepLogits forward(const StepBatch& batch, std::span<const kv> lines) override;
};

}  // namespace pulsarlm
