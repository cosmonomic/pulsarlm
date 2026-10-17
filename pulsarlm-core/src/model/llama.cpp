#include "pulsarlm/model/llama.hpp"

#include "pulsarlm/ops.hpp"
#include "pulsarlm/model/paged_forward.hpp"

#include <ATen/ATen.h>
#include <ATen/Functions.h>

#include <cmath>
#include <optional>
#include <string>
#include <utility>

namespace pulsarlm {

namespace {

at::Tensor need(const c10::Dict<std::string, at::Tensor>& w, const std::string& name) {
    TORCH_CHECK(w.contains(name), "LlamaModel: missing weight '", name, "'");
    return w.at(name);
}

}  // namespace

LlamaModel::LlamaModel(
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
)
: n_layers(n_layers),
  n_heads(n_heads),
  n_kv_heads(n_kv_heads),
  head_dim(head_dim),
  hidden(hidden),
  intermediate(intermediate),
  vocab(vocab),
  rope(rope),
  rms_eps(rms_eps),
  scale(1.0 / std::sqrt(static_cast<double>(head_dim))),
  embed_tokens(need(weights, "model.embed_tokens.weight")),
  norm_weight(need(weights, "model.norm.weight")) {
    if (weights.contains("lm_head.weight")) {
        this->lm_head = weights.at("lm_head.weight");
    }
    this->layers.reserve(n_layers);
    for (int64_t i = 0; i < n_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        this->layers.push_back(
            LayerWeights{
                need(weights, p + "input_layernorm.weight"),
                need(weights, p + "post_attention_layernorm.weight"),
                need(weights, p + "self_attn.q_proj.weight"),
                need(weights, p + "self_attn.k_proj.weight"),
                need(weights, p + "self_attn.v_proj.weight"),
                need(weights, p + "self_attn.o_proj.weight"),
                need(weights, p + "mlp.gate_proj.weight"),
                need(weights, p + "mlp.up_proj.weight"),
                need(weights, p + "mlp.down_proj.weight")
            }
        );
    }
}

at::Tensor LlamaModel::block(at::Tensor x, int64_t layer, const GroupBatch& g, std::span<const kv> lines) {
    const LayerWeights& lw = this->layers[layer];
    const int64_t ntok = x.size(0);

    auto h = rmsnorm_cuda(x, lw.input_layernorm, this->rms_eps);
    auto q = at::linear(h, lw.q_proj_weight);
    auto k = at::linear(h, lw.k_proj_weight);
    auto v = at::linear(h, lw.v_proj_weight);

    q = rope_q(q, g.pos.select(0, layer), this->n_heads, this->head_dim, this->rope);
    // The lines store K and V unrotated, as the paged ops' [tokens, n_kv_heads, head_dim].
    k = k.view({ntok, this->n_kv_heads, this->head_dim}).contiguous();
    v = v.view({ntok, this->n_kv_heads, this->head_dim}).contiguous();
    auto o = paged_attention(q, k, v, lines[layer], static_cast<std::size_t>(layer), g, this->scale, this->rope);

    o = o.reshape({ntok, this->n_heads * this->head_dim});
    x = x + at::linear(o, lw.o_proj_weight);

    auto h2 = rmsnorm_cuda(x, lw.post_attention_layernorm, this->rms_eps);
    auto gate = at::linear(h2, lw.gate_proj_weight);
    auto up = at::linear(h2, lw.up_proj_weight);
    x = x + at::linear(at::silu(gate) * up, lw.down_proj_weight);
    return x;
}

at::Tensor LlamaModel::head(at::Tensor x) {
    x = rmsnorm_cuda(x, this->norm_weight, this->rms_eps);
    const at::Tensor& w = this->lm_head.has_value() ? *this->lm_head : this->embed_tokens;
    return at::matmul(x, w.t());
}

std::vector<kv_layer> LlamaModel::kv_layers() const {
    return std::vector<kv_layer>(
        this->n_layers,
        kv_layer{
            .n_kv_heads = static_cast<std::size_t>(this->n_kv_heads),
            .head_dim = static_cast<std::size_t>(this->head_dim),
            .n_q_heads = static_cast<std::size_t>(this->n_heads)
        }
    );
}

StepLogits LlamaModel::forward(const StepBatch& batch, std::span<const kv> lines) {
    return paged_forward(batch, [&](const GroupBatch& g) {
        return paged_group_forward(
            this->embed_tokens,
            g,
            this->n_layers,
            [&](at::Tensor x, int64_t layer) { return this->block(x, layer, g, lines); },
            [&](at::Tensor x) { return this->head(x); }
        );
    });
}

}  // namespace pulsarlm
