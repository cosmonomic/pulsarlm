#include "pulsarlm/engine/model/qwen2.hpp"

#include "pulsarlm/engine/model/paged_forward.hpp"
#include "pulsarlm/kernels/ops.hpp"

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
        throw std::invalid_argument(std::format("qwen2_model: missing weight {:?}", name));
    }
    return weights.at(name);
}

void check_size(const at::Tensor& tensor, std::int64_t dim, std::int64_t size, const std::string& name) {
    if (tensor.dim() <= dim || tensor.size(dim) != size) {
        throw std::invalid_argument(std::format("qwen2_model: {:?} must have {} in dim {}", name, size, dim));
    }
}

}  // namespace

qwen2_model::qwen2_model(dense_config config, const c10::Dict<std::string, at::Tensor>& weights)
: config(std::move(config)),
  embed_tokens(need(weights, "model.embed_tokens.weight")),
  norm(need(weights, "model.norm.weight")) {
    if (embed_tokens.dim() != 2) {
        throw std::invalid_argument("qwen2_model: \"model.embed_tokens.weight\" must have 2 dims");
    }
    for (std::int64_t layer = 0; layer < this->config.n_layers; layer++) {
        const std::string p = "model.layers." + std::to_string(layer) + ".";
        const std::string a = p + "self_attn.";
        const dense_config& c = this->config;
        const std::int64_t hidden = embed_tokens.size(1);
        check_size(need(weights, a + "q_proj.weight"), 0, c.n_heads * c.head_dim, a + "q_proj.weight");
        check_size(need(weights, a + "q_proj.weight"), 1, hidden, a + "q_proj.weight");
        check_size(need(weights, a + "k_proj.weight"), 0, c.n_kv_heads * c.head_dim, a + "k_proj.weight");
        check_size(need(weights, a + "k_proj.weight"), 1, hidden, a + "k_proj.weight");
        check_size(need(weights, a + "v_proj.weight"), 0, c.n_kv_heads * c.head_dim, a + "v_proj.weight");
        check_size(need(weights, a + "v_proj.weight"), 1, hidden, a + "v_proj.weight");
        check_size(need(weights, a + "o_proj.weight"), 0, hidden, a + "o_proj.weight");
        check_size(need(weights, a + "o_proj.weight"), 1, c.n_heads * c.head_dim, a + "o_proj.weight");
        check_size(need(weights, a + "q_proj.bias"), 0, c.n_heads * c.head_dim, a + "q_proj.bias");
        check_size(need(weights, a + "k_proj.bias"), 0, c.n_kv_heads * c.head_dim, a + "k_proj.bias");
        check_size(need(weights, a + "v_proj.bias"), 0, c.n_kv_heads * c.head_dim, a + "v_proj.bias");
        layers.push_back(
            layer_weights{
                .input_layernorm = need(weights, p + "input_layernorm.weight"),
                .post_attention_layernorm = need(weights, p + "post_attention_layernorm.weight"),
                .q_proj = need(weights, p + "self_attn.q_proj.weight"),
                .q_proj_bias = need(weights, p + "self_attn.q_proj.bias"),
                .k_proj = need(weights, p + "self_attn.k_proj.weight"),
                .k_proj_bias = need(weights, p + "self_attn.k_proj.bias"),
                .v_proj = need(weights, p + "self_attn.v_proj.weight"),
                .v_proj_bias = need(weights, p + "self_attn.v_proj.bias"),
                .o_proj = need(weights, p + "self_attn.o_proj.weight"),
                .gate_proj = need(weights, p + "mlp.gate_proj.weight"),
                .up_proj = need(weights, p + "mlp.up_proj.weight"),
                .down_proj = need(weights, p + "mlp.down_proj.weight"),
            }
        );
    }
}

std::vector<kv_layer> qwen2_model::kv_layers() const {
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
qwen2_model::attention(const at::Tensor& h, std::int64_t layer, const group_batch& g, std::span<const kv> lines) const {
    const layer_weights& w = layers[layer];
    const dense_config& c = config;
    const std::int64_t ntok = h.size(0);

    at::Tensor q = rope_q(at::linear(h, w.q_proj, w.q_proj_bias), g.pos, c.n_heads, c.head_dim, c.rope);
    at::Tensor k = at::linear(h, w.k_proj, w.k_proj_bias).view({ntok, c.n_kv_heads, c.head_dim}).contiguous();
    at::Tensor v = at::linear(h, w.v_proj, w.v_proj_bias).view({ntok, c.n_kv_heads, c.head_dim}).contiguous();
    at::Tensor o = paged_attention(
        q,
        k,
        v,
        lines[layer],
        static_cast<std::size_t>(layer),
        0,
        g,
        1.0 / std::sqrt(static_cast<double>(c.head_dim)),
        c.rope
    );
    return at::linear(o.reshape({ntok, c.n_heads * c.head_dim}), w.o_proj);
}

at::Tensor qwen2_model::group_forward(const group_batch& g, std::span<const kv> lines) const {
    const double eps = config.rms_norm_eps;
    at::Tensor x = embed_tokens.index_select(0, g.tokens);
    for (std::int64_t layer = 0; layer < config.n_layers; layer++) {
        const layer_weights& w = layers[layer];
        x = x + attention(rmsnorm_cuda(x, w.input_layernorm, eps), layer, g, lines);
        at::Tensor h = rmsnorm_cuda(x, w.post_attention_layernorm, eps);
        x = x + at::linear(at::silu(at::linear(h, w.gate_proj)) * at::linear(h, w.up_proj), w.down_proj);
    }
    return at::matmul(rmsnorm_cuda(logit_inputs(x, g), norm, eps), embed_tokens.t());
}

step_logits qwen2_model::forward(const step_batch& batch, std::span<const kv> lines) {
    return paged_forward(batch, [&](const group_batch& g) { return group_forward(g, lines); });
}

}  // namespace pulsarlm
