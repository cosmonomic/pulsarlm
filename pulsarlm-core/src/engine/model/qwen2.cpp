#include "pulsarlm/engine/model/qwen2.hpp"

#include "pulsarlm/engine/model/paged_forward.hpp"
#include "pulsarlm/ops/normalization.hpp"
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

constexpr std::string_view model_name = "qwen2_model";

void check_size(const at::Tensor& tensor, std::int64_t dim, std::int64_t size, const std::string& name) {
    if (tensor.dim() <= dim || tensor.size(dim) != size) {
        throw std::invalid_argument(std::format("qwen2_model: {:?} must have {} in dim {}", name, size, dim));
    }
}

}  // namespace

qwen2_model::qwen2_model(dense_config config, const c10::Dict<std::string, at::Tensor>& weights)
: config(std::move(config)),
  embed_tokens(required_weight(weights, model_name, "model.embed_tokens.weight")),
  norm(required_weight(weights, model_name, "model.norm.weight")) {
    if (embed_tokens.dim() != 2) {
        throw std::invalid_argument("qwen2_model: \"model.embed_tokens.weight\" must have 2 dims");
    }
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
        check_size(
            required_weight(weights, model_name, a + "q_proj.bias"),
            0,
            c.n_heads * c.head_dim,
            a + "q_proj.bias"
        );
        check_size(
            required_weight(weights, model_name, a + "k_proj.bias"),
            0,
            c.n_kv_heads * c.head_dim,
            a + "k_proj.bias"
        );
        check_size(
            required_weight(weights, model_name, a + "v_proj.bias"),
            0,
            c.n_kv_heads * c.head_dim,
            a + "v_proj.bias"
        );
        layers.push_back(
            layer_weights{
                .input_layernorm = required_weight(weights, model_name, p + "input_layernorm.weight"),
                .post_attention_layernorm = required_weight(weights, model_name, p + "post_attention_layernorm.weight"),
                .q_proj = required_weight(weights, model_name, p + "self_attn.q_proj.weight"),
                .q_proj_bias = required_weight(weights, model_name, p + "self_attn.q_proj.bias"),
                .k_proj = required_weight(weights, model_name, p + "self_attn.k_proj.weight"),
                .k_proj_bias = required_weight(weights, model_name, p + "self_attn.k_proj.bias"),
                .v_proj = required_weight(weights, model_name, p + "self_attn.v_proj.weight"),
                .v_proj_bias = required_weight(weights, model_name, p + "self_attn.v_proj.bias"),
                .o_proj = required_weight(weights, model_name, p + "self_attn.o_proj.weight"),
                .gate_proj = required_weight(weights, model_name, p + "mlp.gate_proj.weight"),
                .up_proj = required_weight(weights, model_name, p + "mlp.up_proj.weight"),
                .down_proj = required_weight(weights, model_name, p + "mlp.down_proj.weight"),
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

at::Tensor qwen2_model::attention(
    const at::Tensor& h,
    std::int64_t layer,
    const batch& g,
    std::span<const kv> lines,
    graph_segments& segments
) const {
    const layer_weights& w = layers[layer];
    const dense_config& c = config;
    const std::int64_t ntok = h.size(0);

    at::Tensor q = rope_q(at::linear(h, w.q_proj, w.q_proj_bias), g.pos, c.n_heads, c.head_dim, c.rope);
    at::Tensor k = at::linear(h, w.k_proj, w.k_proj_bias).view({ntok, c.n_kv_heads, c.head_dim}).contiguous();
    at::Tensor v = at::linear(h, w.v_proj, w.v_proj_bias).view({ntok, c.n_kv_heads, c.head_dim}).contiguous();
    at::Tensor o = at::empty_like(q);
    segments.run_between_segments(
        [this, &g, lines, layer](
            const at::Tensor& queries,
            const at::Tensor& keys,
            const at::Tensor& values,
            const at::Tensor& attended
        ) {
            paged_attention(
                queries,
                keys,
                values,
                attended,
                lines[layer],
                static_cast<std::size_t>(layer),
                0,
                g,
                1.0 / std::sqrt(static_cast<double>(config.head_dim)),
                config.rope
            );
        },
        q,
        k,
        v,
        o
    );
    return at::linear(o.reshape({ntok, c.n_heads * c.head_dim}), w.o_proj);
}

model::output qwen2_model::forward(const batch& b, std::span<const kv> lines, graph_segments& segments) {
    check_batch(b);
    assert(b.per_query_state_request_count == 0);
    const double eps = config.rms_norm_eps;
    at::Tensor x = embed_tokens.index_select(0, b.tokens);
    for (std::int64_t layer = 0; layer < config.n_layers; layer++) {
        const layer_weights& w = layers[layer];
        x = x + attention(ops::cuda::rmsnorm(x, w.input_layernorm, eps), layer, b, lines, segments);
        at::Tensor h = ops::cuda::rmsnorm(x, w.post_attention_layernorm, eps);
        x = x + at::linear(at::silu(at::linear(h, w.gate_proj)) * at::linear(h, w.up_proj), w.down_proj);
    }
    at::Tensor logits = at::empty({b.logit_tokens.size(0), embed_tokens.size(0)}, x.options());
    segments.run_between_segments(
        [this, &b](const at::Tensor& hidden_states, const at::Tensor& out) {
            at::Tensor written = out.narrow(0, 0, b.logit_tokens.size(0));
            at::matmul_out(
                written,
                ops::cuda::rmsnorm(logit_inputs(hidden_states, b), norm, config.rms_norm_eps),
                embed_tokens.t()
            );
        },
        x,
        logits
    );
    return output{.logits = logits};
}

}  // namespace pulsarlm
