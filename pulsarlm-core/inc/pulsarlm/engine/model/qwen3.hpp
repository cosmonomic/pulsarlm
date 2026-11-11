#pragma once

#include "pulsarlm/engine/model/dense_config.hpp"
#include "pulsarlm/engine/model/interface.hpp"

#include <ATen/core/Dict.h>
#include <ATen/core/Tensor.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

// Qwen3 decoder over the paged CUDA ops: pre-norm RMSNorm, bias-free attention projections with an RMSNorm over each
// query and key head before RoPE, and a SwiGLU MLP. The head is lm_head.weight when the checkpoint has one, else
// model.embed_tokens.weight.

namespace pulsarlm {

class qwen3_model : public model {
    struct layer_weights {
        at::Tensor input_layernorm;
        at::Tensor post_attention_layernorm;
        at::Tensor q_proj;
        at::Tensor k_proj;
        at::Tensor v_proj;
        at::Tensor o_proj;
        at::Tensor q_norm;
        at::Tensor k_norm;
        at::Tensor gate_proj;
        at::Tensor up_proj;
        at::Tensor down_proj;
    };

    dense_config config;
    at::Tensor embed_tokens;
    at::Tensor lm_head;
    at::Tensor norm;
    std::vector<layer_weights> layers;

    at::Tensor
    attention(const at::Tensor& h, std::int64_t layer, const group_batch& g, std::span<const kv> lines) const;
    at::Tensor group_forward(const group_batch& g, std::span<const kv> lines) const;

  public:
    // weights are on the lines' device and dtype.
    qwen3_model(dense_config config, const c10::Dict<std::string, at::Tensor>& weights);

    std::vector<kv_layer> kv_layers() const override;
    step_logits forward(const step_batch& batch, std::span<const kv> lines) override;
};

}  // namespace pulsarlm
