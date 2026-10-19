#include "pulsarlm/model/qwen3.hpp"

#include "pulsarlm/model/paged_forward.hpp"
#include "pulsarlm/ops.hpp"

#include <ATen/ATen.h>

#include <cmath>
#include <format>
#include <stdexcept>
#include <string>
#include <utility>

namespace pulsarlm {

namespace {

at::Tensor need(const c10::Dict<std::string, at::Tensor>& weights, const std::string& name) {
    if (!weights.contains(name)) {
        throw std::invalid_argument(std::format("Qwen3Model: missing weight {:?}", name));
    }
    return weights.at(name);
}

}  // namespace

Qwen3Model::Qwen3Model(dense_config config, const c10::Dict<std::string, at::Tensor>& weights)
: config(std::move(config)),
  embed_tokens(need(weights, "model.embed_tokens.weight")),
  norm(need(weights, "model.norm.weight")) {
    lm_head = weights.contains("lm_head.weight") ? weights.at("lm_head.weight") : embed_tokens;
    for (std::int64_t layer = 0; layer < this->config.n_layers; layer++) {
        const std::string p = "model.layers." + std::to_string(layer) + ".";
        layers.push_back(
            LayerWeights{
                .input_layernorm = need(weights, p + "input_layernorm.weight"),
                .post_attention_layernorm = need(weights, p + "post_attention_layernorm.weight"),
                .q_proj = need(weights, p + "self_attn.q_proj.weight"),
                .k_proj = need(weights, p + "self_attn.k_proj.weight"),
                .v_proj = need(weights, p + "self_attn.v_proj.weight"),
                .o_proj = need(weights, p + "self_attn.o_proj.weight"),
                .q_norm = need(weights, p + "self_attn.q_norm.weight"),
                .k_norm = need(weights, p + "self_attn.k_norm.weight"),
                .gate_proj = need(weights, p + "mlp.gate_proj.weight"),
                .up_proj = need(weights, p + "mlp.up_proj.weight"),
                .down_proj = need(weights, p + "mlp.down_proj.weight"),
            }
        );
    }
}

std::vector<kv_layer> Qwen3Model::kv_layers() const {
    return std::vector<kv_layer>(
        config.n_layers,
        kv_layer{
            .n_kv_heads = static_cast<std::size_t>(config.n_kv_heads),
            .head_dim = static_cast<std::size_t>(config.head_dim),
            .n_q_heads = static_cast<std::size_t>(config.n_heads),
        }
    );
}

at::Tensor
Qwen3Model::attention(const at::Tensor& h, std::int64_t layer, const GroupBatch& g, std::span<const kv> lines) const {
    const LayerWeights& w = layers[layer];
    const dense_config& c = config;
    const std::int64_t ntok = h.size(0);

    at::Tensor q = rmsnorm_cuda(at::linear(h, w.q_proj).view({ntok, c.n_heads, c.head_dim}), w.q_norm, c.rms_norm_eps);
    q = rope_q(q.view({ntok, c.n_heads * c.head_dim}), g.pos, c.n_heads, c.head_dim, c.rope);
    at::Tensor k = rmsnorm_cuda(
        at::linear(h, w.k_proj).view({ntok, c.n_kv_heads, c.head_dim}),
        w.k_norm,
        c.rms_norm_eps
    );
    at::Tensor v = at::linear(h, w.v_proj).view({ntok, c.n_kv_heads, c.head_dim}).contiguous();
    at::Tensor o = paged_attention(
        q,
        k,
        v,
        lines[layer],
        static_cast<std::size_t>(layer),
        g,
        1.0 / std::sqrt(static_cast<double>(c.head_dim)),
        c.rope
    );
    return at::linear(o.reshape({ntok, c.n_heads * c.head_dim}), w.o_proj);
}

at::Tensor Qwen3Model::group_forward(const GroupBatch& g, std::span<const kv> lines) const {
    const double eps = config.rms_norm_eps;
    at::Tensor x = embed_tokens.index_select(0, g.tokens);
    for (std::int64_t layer = 0; layer < config.n_layers; layer++) {
        const LayerWeights& w = layers[layer];
        x = x + attention(rmsnorm_cuda(x, w.input_layernorm, eps), layer, g, lines);
        at::Tensor h = rmsnorm_cuda(x, w.post_attention_layernorm, eps);
        x = x + at::linear(at::silu(at::linear(h, w.gate_proj)) * at::linear(h, w.up_proj), w.down_proj);
    }
    return at::matmul(rmsnorm_cuda(logit_inputs(x, g), norm, eps), lm_head.t());
}

StepLogits Qwen3Model::forward(const StepBatch& batch, std::span<const kv> lines) {
    return paged_forward(batch, [&](const GroupBatch& g) { return group_forward(g, lines); });
}

}  // namespace pulsarlm
