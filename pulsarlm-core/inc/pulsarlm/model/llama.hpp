#pragma once

#include "pulsarlm/model/dense_config.hpp"
#include "pulsarlm/model/interface.hpp"

#include <ATen/core/Dict.h>
#include <ATen/core/Tensor.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

// Llama decoder over the paged CUDA ops: pre-norm RMSNorm, bias-free attention projections and a SwiGLU MLP. The head
// is lm_head.weight when the checkpoint has one, else model.embed_tokens.weight.

namespace pulsarlm {

class LlamaModel : public Model {
    struct LayerWeights {
        at::Tensor input_layernorm;
        at::Tensor post_attention_layernorm;
        at::Tensor q_proj;
        at::Tensor k_proj;
        at::Tensor v_proj;
        at::Tensor o_proj;
        at::Tensor gate_proj;
        at::Tensor up_proj;
        at::Tensor down_proj;
    };

    dense_config config;
    at::Tensor embed_tokens;
    at::Tensor lm_head;
    at::Tensor norm;
    std::vector<LayerWeights> layers;

    at::Tensor attention(const at::Tensor& h, std::int64_t layer, const GroupBatch& g, std::span<const kv> lines) const;
    at::Tensor group_forward(const GroupBatch& g, std::span<const kv> lines) const;

  public:
    // weights are on the lines' device and dtype.
    LlamaModel(dense_config config, const c10::Dict<std::string, at::Tensor>& weights);

    std::vector<kv_layer> kv_layers() const override;
    StepLogits forward(const StepBatch& batch, std::span<const kv> lines) override;
};

}  // namespace pulsarlm
