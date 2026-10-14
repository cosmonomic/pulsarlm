#include "pulsarlm/model/gemma4.hpp"

#include "pulsarlm/model/paged_forward.hpp"
#include "pulsarlm/ops.hpp"

#include <ATen/ATen.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace pulsarlm {

namespace {

const std::string prefix = "model.language_model.";

at::Tensor need(const c10::Dict<std::string, at::Tensor>& weights, const std::string& name) {
    if (!weights.contains(prefix + name)) {
        throw std::invalid_argument(std::format("Gemma4Model: missing weight {:?}", prefix + name));
    }
    return weights.at(prefix + name);
}

at::Tensor gelu_tanh(const at::Tensor& x) {
    return at::gelu(x, "tanh");
}

// Hugging Face rounds an embedding scale to the weights' dtype before multiplying.
at::Tensor scaled(const at::Tensor& x, double scale) {
    return x * at::scalar_tensor(scale, x.options());
}

}  // namespace

Gemma4Model::Gemma4Model(gemma4_config config, const c10::Dict<std::string, at::Tensor>& weights)
: config(std::move(config)),
  n_layers(static_cast<std::int64_t>(this->config.layer_types.size())),
  embed_tokens(need(weights, "embed_tokens.weight")),
  norm(need(weights, "norm.weight")) {
    const gemma4_config& c = this->config;
    if (c.layer_types.empty() || c.layer_types.back() != gemma4_layer_type::full_attention) {
        throw std::invalid_argument("Gemma4Model: the last layer must be a full-attention layer");
    }
    if (c.num_kv_shared_layers < 0 || c.num_kv_shared_layers >= n_layers) {
        throw std::invalid_argument("Gemma4Model: num_kv_shared_layers must leave a layer keeping KV");
    }
    std::int64_t first_shared = n_layers - c.num_kv_shared_layers;
    for (std::int64_t layer = 0; layer < n_layers; layer++) {
        if (layer < first_shared) {
            kv_layer_of.push_back(kv_specs.size());
            bool full = !sliding(layer);
            kv_specs.push_back(
                kv_layer{
                    .n_kv_heads = static_cast<std::size_t>(full ? c.global_n_kv_heads : c.n_kv_heads),
                    .head_dim = static_cast<std::size_t>(full ? c.global_head_dim : c.head_dim),
                    .values_only = full && c.attention_k_eq_v,
                    .sliding_window = full ? std::nullopt
                                           : std::optional<std::size_t>(static_cast<std::size_t>(c.sliding_window)),
                }
            );
            continue;
        }
        std::int64_t source = first_shared - 1;
        while (source >= 0 && sliding(source) != sliding(layer)) {
            source--;
        }
        if (source < 0) {
            throw std::invalid_argument(
                std::format("Gemma4Model: KV-shared layer {} has no earlier layer of its type", layer)
            );
        }
        kv_layer_of.push_back(kv_layer_of[source]);
    }

    if (c.hidden_size_per_layer_input > 0) {
        embed_tokens_per_layer = need(weights, "embed_tokens_per_layer.weight");
        per_layer_model_projection = need(weights, "per_layer_model_projection.weight");
        per_layer_projection_norm = need(weights, "per_layer_projection_norm.weight");
    }
    for (std::int64_t layer = 0; layer < n_layers; layer++) {
        const std::string p = "layers." + std::to_string(layer) + ".";
        LayerWeights w{
            .input_layernorm = need(weights, p + "input_layernorm.weight"),
            .post_attention_layernorm = need(weights, p + "post_attention_layernorm.weight"),
            .pre_feedforward_layernorm = need(weights, p + "pre_feedforward_layernorm.weight"),
            .post_feedforward_layernorm = need(weights, p + "post_feedforward_layernorm.weight"),
            .q_proj = need(weights, p + "self_attn.q_proj.weight"),
            .q_norm = need(weights, p + "self_attn.q_norm.weight"),
            .o_proj = need(weights, p + "self_attn.o_proj.weight"),
            .gate_proj = need(weights, p + "mlp.gate_proj.weight"),
            .up_proj = need(weights, p + "mlp.up_proj.weight"),
            .down_proj = need(weights, p + "mlp.down_proj.weight"),
            .layer_scalar = need(weights, p + "layer_scalar"),
        };
        if (layer < first_shared) {
            w.k_proj = need(weights, p + "self_attn.k_proj.weight");
            w.k_norm = need(weights, p + "self_attn.k_norm.weight");
            if (!kv_specs[kv_layer_of[layer]].values_only) {
                w.v_proj = need(weights, p + "self_attn.v_proj.weight");
            }
        }
        if (c.hidden_size_per_layer_input > 0) {
            w.per_layer_input_gate = need(weights, p + "per_layer_input_gate.weight");
            w.per_layer_projection = need(weights, p + "per_layer_projection.weight");
            w.post_per_layer_input_norm = need(weights, p + "post_per_layer_input_norm.weight");
        }
        layers.push_back(std::move(w));
    }
    sliding_ones = at::ones({c.head_dim}, embed_tokens.options());
    full_ones = at::ones({c.global_head_dim}, embed_tokens.options());
    for (std::int64_t layer = 3 * n_layers / 4; layer < n_layers; layer++) {
        if (!sliding(layer)) {
            relevance_layers.insert(layer);
        }
    }
}

bool Gemma4Model::sliding(std::int64_t layer) const {
    return config.layer_types[layer] == gemma4_layer_type::sliding_attention;
}

std::vector<kv_layer> Gemma4Model::kv_layers() const {
    return kv_specs;
}

void Gemma4Model::set_relevance_layers(const std::vector<std::int64_t>& layers) {
    for (std::int64_t layer : layers) {
        if (layer < 0 || layer >= n_layers || sliding(layer)) {
            throw std::invalid_argument(std::format("Gemma4Model: layer {} is not a full-attention layer", layer));
        }
    }
    relevance_layers = std::unordered_set<std::int64_t>(layers.begin(), layers.end());
}

// [ntok, n_layers, hidden_size_per_layer_input].
at::Tensor Gemma4Model::per_layer_inputs(const at::Tensor& tokens, const at::Tensor& embedded) const {
    const std::int64_t ntok = tokens.size(0);
    const std::int64_t width = config.hidden_size_per_layer_input;
    at::Tensor by_token = scaled(embed_tokens_per_layer.index_select(0, tokens), std::sqrt(static_cast<double>(width)))
                              .view({ntok, n_layers, width});
    at::Tensor projected = (at::linear(embedded, per_layer_model_projection) *
                            std::pow(static_cast<double>(config.hidden_size), -0.5))
                               .view({ntok, n_layers, width});
    projected = rmsnorm_cuda(projected, per_layer_projection_norm, config.rms_norm_eps);
    return (projected + by_token) * std::pow(2.0, -0.5);
}

at::Tensor
Gemma4Model::attention(const at::Tensor& h, std::int64_t layer, const GroupBatch& g, std::span<const kv> lines) const {
    const LayerWeights& w = layers[layer];
    const std::int64_t ntok = h.size(0);
    const std::size_t kv_index = kv_layer_of[layer];
    const kv_layer& spec = kv_specs[kv_index];
    const bool full = !sliding(layer);
    const auto heads = static_cast<std::int64_t>(spec.n_kv_heads);
    const auto head_dim = static_cast<std::int64_t>(spec.head_dim);
    const rope_parameters& rope = full ? config.full_rope : config.sliding_rope;
    const at::Tensor& ones = full ? full_ones : sliding_ones;

    at::Tensor q = rmsnorm_cuda(
        at::linear(h, w.q_proj).view({ntok, config.n_heads, head_dim}),
        w.q_norm,
        config.rms_norm_eps
    );
    q = rope_q(q.view({ntok, config.n_heads * head_dim}), g.pos, config.n_heads, head_dim, rope);

    at::Tensor k, v;
    std::optional<at::Tensor> k_norm_weight;
    if (w.k_proj.defined()) {
        at::Tensor raw_k = at::linear(h, w.k_proj).view({ntok, heads, head_dim});
        if (spec.values_only) {
            v = rmsnorm_cuda(raw_k, ones, config.rms_norm_eps);
            k = v;
        } else {
            k = rmsnorm_cuda(raw_k, w.k_norm, config.rms_norm_eps);
            v = rmsnorm_cuda(at::linear(h, w.v_proj).view({ntok, heads, head_dim}), ones, config.rms_norm_eps);
        }
    }
    if (spec.values_only) {
        k_norm_weight = layers[static_cast<std::int64_t>(kv_index)].k_norm;
    }
    std::optional<std::int64_t> window;
    if (spec.sliding_window) {
        window = static_cast<std::int64_t>(*spec.sliding_window);
    }
    at::Tensor o = paged_attention(
        q,
        k,
        v,
        lines[kv_index],
        kv_index,
        g,
        relevance_layers.contains(layer),
        1.0,
        rope,
        window,
        k_norm_weight
    );
    return at::linear(o.reshape({ntok, config.n_heads * head_dim}), w.o_proj);
}

at::Tensor Gemma4Model::block(
    at::Tensor x,
    std::int64_t layer,
    const at::Tensor& per_layer_input,
    const GroupBatch& g,
    std::span<const kv> lines
) const {
    const LayerWeights& w = layers[layer];
    const double eps = config.rms_norm_eps;
    at::Tensor attended = attention(rmsnorm_cuda(x, w.input_layernorm, eps), layer, g, lines);
    x = x + rmsnorm_cuda(attended, w.post_attention_layernorm, eps);

    at::Tensor h = rmsnorm_cuda(x, w.pre_feedforward_layernorm, eps);
    at::Tensor mlp = at::linear(gelu_tanh(at::linear(h, w.gate_proj)) * at::linear(h, w.up_proj), w.down_proj);
    x = x + rmsnorm_cuda(mlp, w.post_feedforward_layernorm, eps);

    if (per_layer_input.defined()) {
        at::Tensor gated = gelu_tanh(at::linear(x, w.per_layer_input_gate)) * per_layer_input;
        x = x + rmsnorm_cuda(at::linear(gated, w.per_layer_projection), w.post_per_layer_input_norm, eps);
    }
    return x * w.layer_scalar;
}

at::Tensor Gemma4Model::head(const at::Tensor& x) const {
    at::Tensor logits = at::matmul(rmsnorm_cuda(x, norm, config.rms_norm_eps), embed_tokens.t());
    if (config.final_logit_softcapping) {
        logits = at::tanh(logits / *config.final_logit_softcapping) * *config.final_logit_softcapping;
    }
    return logits;
}

at::Tensor Gemma4Model::group_forward(const GroupBatch& g, std::span<const kv> lines) const {
    at::Tensor x = scaled(embed_tokens.index_select(0, g.tokens), std::sqrt(static_cast<double>(config.hidden_size)));
    at::Tensor per_layer;
    if (config.hidden_size_per_layer_input > 0) {
        per_layer = per_layer_inputs(g.tokens, x);
    }
    for (std::int64_t layer = 0; layer < n_layers; layer++) {
        x = block(x, layer, per_layer.defined() ? per_layer.select(1, layer) : at::Tensor(), g, lines);
    }
    return head(x);
}

StepLogits Gemma4Model::forward(const StepBatch& batch, std::span<const kv> lines) {
    return paged_forward(batch, [&](const GroupBatch& g) { return group_forward(g, lines); });
}

}  // namespace pulsarlm
