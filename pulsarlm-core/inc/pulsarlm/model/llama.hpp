#pragma once

#include "pulsarlm/model/interface.hpp"
#include "pulsarlm/rope.hpp"

#include <ATen/core/Tensor.h>

#include <ATen/core/Dict.h>

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

// Llama transformer forward over the paged CUDA ops: pre-norm RMSNorm, bias-free QKV projection, SwiGLU MLP, and an
// lm_head that is lm_head.weight when the checkpoint ships one, else tied to model.embed_tokens.weight.

namespace pulsarlm {

struct LlamaModel : Model {
    LlamaModel(
        int64_t n_layers,
        int64_t n_heads,
        int64_t n_kv_heads,
        int64_t head_dim,
        int64_t hidden,
        int64_t intermediate,
        int64_t vocab,
        rope_parameters rope,
        double rms_eps,
        c10::Dict<std::string, at::Tensor> weights
    );

    std::vector<kv_layer> kv_layers() const override;
    StepLogits forward(const StepBatch& batch, std::span<const kv> lines) override;

  private:
    struct LayerWeights {
        at::Tensor input_layernorm;
        at::Tensor post_attention_layernorm;
        at::Tensor q_proj_weight;
        at::Tensor k_proj_weight;
        at::Tensor v_proj_weight;
        at::Tensor o_proj_weight;
        at::Tensor gate_proj_weight;
        at::Tensor up_proj_weight;
        at::Tensor down_proj_weight;
    };

    at::Tensor block(at::Tensor x, int64_t layer, const GroupBatch& g, std::span<const kv> lines);
    at::Tensor head(at::Tensor x);

    int64_t n_layers;
    int64_t n_heads;
    int64_t n_kv_heads;
    int64_t head_dim;
    int64_t hidden;
    int64_t intermediate;
    int64_t vocab;
    rope_parameters rope;
    double rms_eps;
    double scale;  // 1 / sqrt(head_dim)
    at::Tensor embed_tokens;  // [vocab, hidden]
    std::optional<at::Tensor> lm_head;
    at::Tensor norm_weight;
    std::vector<LayerWeights> layers;
};

}  // namespace pulsarlm
