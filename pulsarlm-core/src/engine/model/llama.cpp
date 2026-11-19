#include "pulsarlm/engine/model/llama.hpp"

#include "pulsarlm/engine/model/paged_forward.hpp"
#include "pulsarlm/kernels/normalization.hpp"
#include "weights.hpp"

#include <ATen/ATen.h>

#include <cassert>
#include <cmath>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace pulsarlm {

namespace {

constexpr std::string_view model_name = "llama_model";

void check_size(const at::Tensor& tensor, std::int64_t dim, std::int64_t size, const std::string& name) {
    if (tensor.dim() <= dim || tensor.size(dim) != size) {
        throw std::invalid_argument(std::format("llama_model: {:?} must have {} in dim {}", name, size, dim));
    }
}

}  // namespace

llama_model::llama_model(dense_config config, const c10::Dict<std::string, at::Tensor>& weights)
: config(std::move(config)),
  embed_tokens(required_weight(weights, model_name, "model.embed_tokens.weight")),
  norm(required_weight(weights, model_name, "model.norm.weight")) {
    if (embed_tokens.dim() != 2) {
        throw std::invalid_argument("llama_model: \"model.embed_tokens.weight\" must have 2 dims");
    }
    lm_head = weights.contains("lm_head.weight") ? weights.at("lm_head.weight") : embed_tokens;
    for (std::int64_t layer = 0; layer < this->config.n_layers; layer++) {
        const std::string p = "model.layers." + std::to_string(layer) + ".";
        const std::string a = p + "self_attn.";
        const dense_config& c = this->config;
        const std::int64_t hidden = embed_tokens.size(1);
        check_size(
            required_weight(weights, model_name, a + "q_proj.weight"),
            0,
            c.n_heads * c.head_dim,
            a + "q_proj.weight"
        );
        check_size(required_weight(weights, model_name, a + "q_proj.weight"), 1, hidden, a + "q_proj.weight");
        check_size(
            required_weight(weights, model_name, a + "k_proj.weight"),
            0,
            c.n_kv_heads * c.head_dim,
            a + "k_proj.weight"
        );
        check_size(required_weight(weights, model_name, a + "k_proj.weight"), 1, hidden, a + "k_proj.weight");
        check_size(
            required_weight(weights, model_name, a + "v_proj.weight"),
            0,
            c.n_kv_heads * c.head_dim,
            a + "v_proj.weight"
        );
        check_size(required_weight(weights, model_name, a + "v_proj.weight"), 1, hidden, a + "v_proj.weight");
        check_size(required_weight(weights, model_name, a + "o_proj.weight"), 0, hidden, a + "o_proj.weight");
        check_size(
            required_weight(weights, model_name, a + "o_proj.weight"),
            1,
            c.n_heads * c.head_dim,
            a + "o_proj.weight"
        );
        layers.push_back(
            layer_weights{
                .input_layernorm = required_weight(weights, model_name, p + "input_layernorm.weight"),
                .post_attention_layernorm = required_weight(weights, model_name, p + "post_attention_layernorm.weight"),
                .q_proj = required_weight(weights, model_name, p + "self_attn.q_proj.weight"),
                .k_proj = required_weight(weights, model_name, p + "self_attn.k_proj.weight"),
                .v_proj = required_weight(weights, model_name, p + "self_attn.v_proj.weight"),
                .o_proj = required_weight(weights, model_name, p + "self_attn.o_proj.weight"),
                .gate_proj = required_weight(weights, model_name, p + "mlp.gate_proj.weight"),
                .up_proj = required_weight(weights, model_name, p + "mlp.up_proj.weight"),
                .down_proj = required_weight(weights, model_name, p + "mlp.down_proj.weight"),
            }
        );
    }
}

std::vector<kv_layer> llama_model::kv_layers() const {
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
llama_model::attention(const at::Tensor& h, std::int64_t layer, const batch& g, std::span<const kv> lines) const {
    const layer_weights& w = layers[layer];
    const dense_config& c = config;
    const std::int64_t ntok = h.size(0);

    at::Tensor q = rope_q(at::linear(h, w.q_proj), g.pos, c.n_heads, c.head_dim, c.rope);
    at::Tensor k = at::linear(h, w.k_proj).view({ntok, c.n_kv_heads, c.head_dim}).contiguous();
    at::Tensor v = at::linear(h, w.v_proj).view({ntok, c.n_kv_heads, c.head_dim}).contiguous();
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

model::output llama_model::forward(const batch& b, std::span<const kv> lines) {
    check_batch(b);
    assert(b.kind != batch_kind::verify);
    const double eps = config.rms_norm_eps;
    at::Tensor x = embed_tokens.index_select(0, b.tokens);
    for (std::int64_t layer = 0; layer < config.n_layers; layer++) {
        const layer_weights& w = layers[layer];
        x = x + attention(rmsnorm_cuda(x, w.input_layernorm, eps), layer, b, lines);
        at::Tensor h = rmsnorm_cuda(x, w.post_attention_layernorm, eps);
        x = x + at::linear(at::silu(at::linear(h, w.gate_proj)) * at::linear(h, w.up_proj), w.down_proj);
    }
    return output{.logits = at::matmul(rmsnorm_cuda(logit_inputs(x, b), norm, eps), lm_head.t())};
}

}  // namespace pulsarlm
