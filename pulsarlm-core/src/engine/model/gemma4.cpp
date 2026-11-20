#include "pulsarlm/engine/model/gemma4.hpp"

#include "pulsarlm/engine/model/paged_forward.hpp"
#include "pulsarlm/kernels/normalization.hpp"
#include "pulsarlm/kernels/routed_experts.hpp"
#include "weights.hpp"

#include <ATen/ATen.h>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace pulsarlm {

namespace {

const std::string prefix = "model.language_model.";

constexpr std::string_view model_name = "gemma4_model";

at::Tensor text_weight(const c10::Dict<std::string, at::Tensor>& weights, const std::string& name) {
    return required_weight(weights, model_name, prefix + name);
}

void check_size(const at::Tensor& tensor, std::int64_t dim, std::int64_t size, const std::string& name) {
    if (tensor.dim() <= dim || tensor.size(dim) != size) {
        throw std::invalid_argument(std::format("gemma4_model: {:?} must have {} in dim {}", prefix + name, size, dim));
    }
}

at::Tensor gelu_tanh(const at::Tensor& x) {
    return at::gelu(x, "tanh");
}

// Hugging Face rounds an embedding scale to the weights' dtype before multiplying.
at::Tensor scaled(const at::Tensor& x, double scale) {
    return x * at::scalar_tensor(scale, x.options());
}

}  // namespace

gemma4_model::gemma4_model(gemma4_config config, const c10::Dict<std::string, at::Tensor>& weights)
: config(std::move(config)),
  n_layers(static_cast<std::int64_t>(this->config.layer_types.size())),
  embed_tokens(text_weight(weights, "embed_tokens.weight")),
  norm(text_weight(weights, "norm.weight")) {
    const gemma4_config& c = this->config;
    if (c.layer_types.empty() || c.layer_types.back() != gemma4_layer_type::full_attention) {
        throw std::invalid_argument("gemma4_model: the last layer must be a full-attention layer");
    }
    if (c.num_kv_shared_layers < 0 || c.num_kv_shared_layers >= n_layers) {
        throw std::invalid_argument("gemma4_model: num_kv_shared_layers must leave a layer keeping KV");
    }
    std::int64_t first_shared = n_layers - c.num_kv_shared_layers;
    for (std::int64_t layer = 0; layer < n_layers; layer++) {
        if (layer < first_shared) {
            kv_layer_of.push_back(kv_specs.size());
            reader_of.push_back(0);
            source_layer_of.push_back(layer);
            bool full = !sliding(layer);
            kv_specs.push_back(
                kv_layer{
                    .n_kv_heads = static_cast<std::size_t>(full ? c.global_n_kv_heads : c.n_kv_heads),
                    .head_dim = static_cast<std::size_t>(full ? c.global_head_dim : c.head_dim),
                    .n_q_heads = static_cast<std::size_t>(c.n_heads),
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
                std::format("gemma4_model: KV-shared layer {} has no earlier layer of its type", layer)
            );
        }
        kv_layer_of.push_back(kv_layer_of[source]);
        reader_of.push_back(kv_specs[kv_layer_of[source]].n_readers++);
        source_layer_of.push_back(source);
    }

    if (c.hidden_size_per_layer_input > 0) {
        embed_tokens_per_layer = text_weight(weights, "embed_tokens_per_layer.weight");
        per_layer_model_projection = text_weight(weights, "per_layer_model_projection.weight");
        per_layer_projection_norm = text_weight(weights, "per_layer_projection_norm.weight");
    }
    for (std::int64_t layer = 0; layer < n_layers; layer++) {
        const std::string p = "layers." + std::to_string(layer) + ".";
        layer_weights w{
            .input_layernorm = text_weight(weights, p + "input_layernorm.weight"),
            .post_attention_layernorm = text_weight(weights, p + "post_attention_layernorm.weight"),
            .pre_feedforward_layernorm = text_weight(weights, p + "pre_feedforward_layernorm.weight"),
            .post_feedforward_layernorm = text_weight(weights, p + "post_feedforward_layernorm.weight"),
            .q_proj = text_weight(weights, p + "self_attn.q_proj.weight"),
            .q_norm = text_weight(weights, p + "self_attn.q_norm.weight"),
            .o_proj = text_weight(weights, p + "self_attn.o_proj.weight"),
            .gate_proj = text_weight(weights, p + "mlp.gate_proj.weight"),
            .up_proj = text_weight(weights, p + "mlp.up_proj.weight"),
            .down_proj = text_weight(weights, p + "mlp.down_proj.weight"),
            .layer_scalar = text_weight(weights, p + "layer_scalar"),
        };
        if (layer < first_shared) {
            w.k_proj = text_weight(weights, p + "self_attn.k_proj.weight");
            w.k_norm = text_weight(weights, p + "self_attn.k_norm.weight");
            if (!kv_specs[kv_layer_of[layer]].values_only) {
                w.v_proj = text_weight(weights, p + "self_attn.v_proj.weight");
            }
        }
        if (c.hidden_size_per_layer_input > 0) {
            w.per_layer_input_gate = text_weight(weights, p + "per_layer_input_gate.weight");
            w.per_layer_projection = text_weight(weights, p + "per_layer_projection.weight");
            w.post_per_layer_input_norm = text_weight(weights, p + "post_per_layer_input_norm.weight");
        }
        if (c.enable_moe_block) {
            w.router_proj = text_weight(weights, p + "router.proj.weight");
            w.router_scale = text_weight(weights, p + "router.scale");
            w.router_per_expert_scale = text_weight(weights, p + "router.per_expert_scale");
            w.experts_gate_up_proj = text_weight(weights, p + "experts.gate_up_proj");
            w.experts_down_proj = text_weight(weights, p + "experts.down_proj");
            w.pre_feedforward_layernorm_2 = text_weight(weights, p + "pre_feedforward_layernorm_2.weight");
            w.post_feedforward_layernorm_1 = text_weight(weights, p + "post_feedforward_layernorm_1.weight");
            w.post_feedforward_layernorm_2 = text_weight(weights, p + "post_feedforward_layernorm_2.weight");
            check_size(w.router_proj, 1, c.hidden_size, p + "router.proj.weight");
            const std::int64_t experts = w.router_proj.size(0);
            if (c.top_k_experts < 1 || c.top_k_experts > experts) {
                throw std::invalid_argument("gemma4_model: top_k_experts must be between 1 and the experts");
            }
            check_size(w.router_per_expert_scale, 0, experts, p + "router.per_expert_scale");
            check_size(w.experts_gate_up_proj, 0, experts, p + "experts.gate_up_proj");
            check_size(w.experts_gate_up_proj, 2, c.hidden_size, p + "experts.gate_up_proj");
            check_size(w.experts_down_proj, 0, experts, p + "experts.down_proj");
            check_size(w.experts_down_proj, 1, c.hidden_size, p + "experts.down_proj");
            check_size(w.experts_down_proj, 2, w.experts_gate_up_proj.size(1) / 2, p + "experts.down_proj");
            check_size(w.experts_gate_up_proj, 1, 2 * w.experts_down_proj.size(2), p + "experts.gate_up_proj");
            if (!layers.empty()) {
                check_size(
                    w.experts_gate_up_proj,
                    0,
                    layers.front().experts_gate_up_proj.size(0),
                    p + "experts.gate_up_proj"
                );
                check_size(w.experts_down_proj, 2, layers.front().experts_down_proj.size(2), p + "experts.down_proj");
            }
        }
        const kv_layer& spec = kv_specs[kv_layer_of[layer]];
        const auto layer_head_dim = static_cast<std::int64_t>(spec.head_dim);
        const auto layer_kv_heads = static_cast<std::int64_t>(spec.n_kv_heads);
        check_size(w.q_proj, 0, c.n_heads * layer_head_dim, p + "self_attn.q_proj.weight");
        check_size(w.q_proj, 1, c.hidden_size, p + "self_attn.q_proj.weight");
        check_size(w.q_norm, 0, layer_head_dim, p + "self_attn.q_norm.weight");
        check_size(w.o_proj, 0, c.hidden_size, p + "self_attn.o_proj.weight");
        check_size(w.o_proj, 1, c.n_heads * layer_head_dim, p + "self_attn.o_proj.weight");
        if (w.k_proj.defined()) {
            check_size(w.k_proj, 0, layer_kv_heads * layer_head_dim, p + "self_attn.k_proj.weight");
            check_size(w.k_proj, 1, c.hidden_size, p + "self_attn.k_proj.weight");
            check_size(w.k_norm, 0, layer_head_dim, p + "self_attn.k_norm.weight");
        }
        if (w.v_proj.defined()) {
            check_size(w.v_proj, 0, layer_kv_heads * layer_head_dim, p + "self_attn.v_proj.weight");
            check_size(w.v_proj, 1, c.hidden_size, p + "self_attn.v_proj.weight");
        }
        layers.push_back(std::move(w));
    }
    sliding_ones = at::ones({c.head_dim}, embed_tokens.options());
    full_ones = at::ones({c.global_head_dim}, embed_tokens.options());
    hidden_ones = at::ones({c.hidden_size}, embed_tokens.options());
}

bool gemma4_model::sliding(std::int64_t layer) const {
    return config.layer_types[layer] == gemma4_layer_type::sliding_attention;
}

std::vector<kv_layer> gemma4_model::kv_layers() const {
    return kv_specs;
}

// [ntok, n_layers, hidden_size_per_layer_input].
at::Tensor gemma4_model::per_layer_inputs(const at::Tensor& tokens, const at::Tensor& embedded) const {
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

at::Tensor gemma4_model::attention(
    const at::Tensor& h,
    std::int64_t layer,
    const batch& g,
    std::span<const kv> lines,
    graph_segments& segments
) const {
    const layer_weights& w = layers[layer];
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
        k_norm_weight = layers[source_layer_of[layer]].k_norm;
    }
    std::optional<std::int64_t> window;
    if (spec.sliding_window) {
        window = static_cast<std::int64_t>(*spec.sliding_window);
    }
    at::Tensor o = at::empty_like(q);
    segments.run_between_segments(
        [&g, lines, kv_index, reader = reader_of[layer], rope, window, k_norm_weight](
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
                lines[kv_index],
                kv_index,
                reader,
                g,
                1.0,
                rope,
                window,
                k_norm_weight
            );
        },
        q,
        k,
        v,
        o
    );
    return at::linear(o.reshape({ntok, config.n_heads * head_dim}), w.o_proj);
}

at::Tensor gemma4_model::experts(const at::Tensor& x, std::int64_t layer, const at::Tensor& counters) const {
    const layer_weights& w = layers[layer];
    const double eps = config.rms_norm_eps;

    at::Tensor routed = rmsnorm_cuda(x, hidden_ones, eps) * w.router_scale *
        std::pow(static_cast<double>(config.hidden_size), -0.5);
    at::Tensor probabilities = at::softmax(at::linear(routed, w.router_proj), -1, at::kFloat);
    auto [top_weights, top_experts] = at::topk(probabilities, config.top_k_experts, -1);
    at::Tensor expert_scale = w.router_per_expert_scale.index_select(0, top_experts.flatten())
                                  .view(top_experts.sizes());
    top_weights = top_weights / top_weights.sum(-1, true) * expert_scale.to(at::kFloat);
    return routed_experts_cuda(
        rmsnorm_cuda(x, w.pre_feedforward_layernorm_2, eps),
        top_experts,
        top_weights,
        w.experts_gate_up_proj,
        w.experts_down_proj,
        expert_activation::gelu_tanh,
        counters
    );
}

at::Tensor gemma4_model::block(
    at::Tensor x,
    std::int64_t layer,
    const at::Tensor& per_layer_input,
    const batch& g,
    std::span<const kv> lines,
    graph_segments& segments
) const {
    const layer_weights& w = layers[layer];
    const double eps = config.rms_norm_eps;
    at::Tensor attended = attention(rmsnorm_cuda(x, w.input_layernorm, eps), layer, g, lines, segments);
    x = x + rmsnorm_cuda(attended, w.post_attention_layernorm, eps);

    at::Tensor h = rmsnorm_cuda(x, w.pre_feedforward_layernorm, eps);
    at::Tensor mlp = at::linear(gelu_tanh(at::linear(h, w.gate_proj)) * at::linear(h, w.up_proj), w.down_proj);
    if (config.enable_moe_block) {
        mlp = rmsnorm_cuda(mlp, w.post_feedforward_layernorm_1, eps) +
            rmsnorm_cuda(experts(x, layer, g.arrival_counters), w.post_feedforward_layernorm_2, eps);
    }
    x = x + rmsnorm_cuda(mlp, w.post_feedforward_layernorm, eps);

    if (per_layer_input.defined()) {
        at::Tensor gated = gelu_tanh(at::linear(x, w.per_layer_input_gate)) * per_layer_input;
        x = x + rmsnorm_cuda(at::linear(gated, w.per_layer_projection), w.post_per_layer_input_norm, eps);
    }
    return x * w.layer_scalar;
}

void gemma4_model::write_logits(const at::Tensor& x, at::Tensor logits) const {
    at::matmul_out(logits, rmsnorm_cuda(x, norm, config.rms_norm_eps), embed_tokens.t());
    if (config.final_logit_softcapping) {
        logits.div_(*config.final_logit_softcapping).tanh_().mul_(*config.final_logit_softcapping);
    }
}

model::output gemma4_model::forward(const batch& b, std::span<const kv> lines, graph_segments& segments) {
    check_batch(b);
    assert(b.per_query_state_request_count == 0);
    at::Tensor x = scaled(embed_tokens.index_select(0, b.tokens), std::sqrt(static_cast<double>(config.hidden_size)));
    at::Tensor per_layer;
    if (config.hidden_size_per_layer_input > 0) {
        per_layer = per_layer_inputs(b.tokens, x);
    }
    for (std::int64_t layer = 0; layer < n_layers; layer++) {
        x = block(x, layer, per_layer.defined() ? per_layer.select(1, layer) : at::Tensor(), b, lines, segments);
    }
    at::Tensor logits = at::empty({b.logit_tokens.size(0), embed_tokens.size(0)}, x.options());
    segments.run_between_segments(
        [this, &b](const at::Tensor& hidden_states, const at::Tensor& out) {
            write_logits(logit_inputs(hidden_states, b), out.narrow(0, 0, b.logit_tokens.size(0)));
        },
        x,
        logits
    );
    return output{.logits = logits};
}

std::int64_t gemma4_model::arrival_counter_count(std::int64_t, std::int64_t tokens) const {
    if (!config.enable_moe_block) {
        return 0;
    }
    const layer_weights& w = layers.front();
    return routed_experts_counters(
        tokens,
        config.top_k_experts,
        w.experts_gate_up_proj.size(0),
        w.experts_down_proj.size(2),
        config.hidden_size
    );
}

}  // namespace pulsarlm
