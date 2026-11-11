#pragma once

#include "pulsarlm/engine/model/interface.hpp"
#include "pulsarlm/engine/model/qwen3_5_moe_config.hpp"
#include "pulsarlm/engine/model/qwen3_5.hpp"
#include "pulsarlm/kernels/rope.hpp"

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

// A Gated DeltaNet layer keeps two recurrent KV layers: its delta rule state S, fp32 [value heads, key head dim, value
// head dim], then its convolution's last linear_conv_kernel - 1 inputs, [conv channels, linear_conv_kernel - 1] in the
// weights' dtype.
class qwen3_5_moe_model : public model {
    // Query and key heads are stored with their dims permuted as qwen3_5_model stores them.
    struct attention_weights {
        at::Tensor q_proj;  // [n_heads * 2 * head_dim, hidden], each head's query then gate
        at::Tensor k_proj;
        at::Tensor v_proj;
        at::Tensor o_proj;
        at::Tensor q_norm;  // fp32 1 + weight
        at::Tensor k_norm;
    };

    struct deltanet_weights {
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

    struct moe_weights {
        at::Tensor router;  // [num_experts, hidden]
        at::Tensor experts_gate_up_proj;  // [num_experts, 2 * moe_intermediate_size, hidden]
        at::Tensor experts_down_proj;  // [num_experts, hidden, moe_intermediate_size]
        at::Tensor shared_gate_proj;
        at::Tensor shared_up_proj;
        at::Tensor shared_down_proj;
        at::Tensor shared_expert_gate;  // [1, hidden]
    };

    struct layer_weights {
        at::Tensor input_layernorm;  // fp32 1 + weight
        at::Tensor post_attention_layernorm;
        moe_weights moe;
        std::optional<attention_weights> attention;
        std::optional<deltanet_weights> deltanet;
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
    std::vector<layer_weights> layers;

    bool full_attention(std::int64_t layer) const;
    at::Tensor
    attention(const at::Tensor& h, std::int64_t layer, const group_batch& g, std::span<const kv> lines) const;
    at::Tensor deltanet(const at::Tensor& h, std::int64_t layer, const group_batch& g, std::span<const kv> lines) const;
    at::Tensor moe(const at::Tensor& h, std::int64_t layer) const;
    at::Tensor group_forward(const group_batch& g, std::span<const kv> lines) const;

  public:
    // weights are on the lines' device, in the attention lines' dtype.
    qwen3_5_moe_model(qwen3_5_moe_config config, const c10::Dict<std::string, at::Tensor>& weights);

    std::vector<kv_layer> kv_layers() const override;
    step_logits forward(const step_batch& batch, std::span<const kv> lines) override;
};

}  // namespace pulsarlm
