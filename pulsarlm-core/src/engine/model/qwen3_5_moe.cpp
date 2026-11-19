#include "pulsarlm/engine/model/qwen3_5_moe.hpp"

#include "pulsarlm/engine/model/paged_forward.hpp"
#include "pulsarlm/engine/model/partial_rope.hpp"
#include "pulsarlm/kernels/ops.hpp"
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
#include <vector>

namespace pulsarlm {

namespace {

const std::string prefix = "model.language_model.";

constexpr std::string_view model_name = "qwen3_5_moe_model";

at::Tensor text_weight(const c10::Dict<std::string, at::Tensor>& weights, const std::string& name) {
    return required_weight(weights, model_name, prefix + name);
}

// Hugging Face's Qwen3.5 RMSNorm scales by 1 + weight, in fp32.
at::Tensor zero_centered(const at::Tensor& weight) {
    return weight.to(at::kFloat) + 1.0;
}

void check_size(const at::Tensor& tensor, std::int64_t dim, std::int64_t size, const std::string& name) {
    if (tensor.dim() <= dim || tensor.size(dim) != size) {
        throw std::invalid_argument(
            std::format("qwen3_5_moe_model: {:?} must have {} in dim {}", prefix + name, size, dim)
        );
    }
}

}  // namespace

qwen3_5_moe_model::qwen3_5_moe_model(qwen3_5_moe_config config, const c10::Dict<std::string, at::Tensor>& weights)
: config(std::move(config)),
  n_layers(static_cast<std::int64_t>(this->config.layer_types.size())),
  embed_tokens(text_weight(weights, "embed_tokens.weight")),
  norm(zero_centered(text_weight(weights, "norm.weight"))) {
    const qwen3_5_moe_config& c = this->config;
    if (c.layer_types.empty()) {
        throw std::invalid_argument("qwen3_5_moe_model: layer_types must hold a layer");
    }
    const auto rotary_dim = static_cast<std::int64_t>(std::llround(c.partial_rotary_factor * c.head_dim));
    if (rotary_dim % 2 != 0 || rotary_dim > c.head_dim || c.head_dim % 2 != 0) {
        throw std::invalid_argument("qwen3_5_moe_model: the rotated dims must be an even part of an even head dim");
    }
    if (c.linear_n_value_heads % c.linear_n_key_heads != 0) {
        throw std::invalid_argument("qwen3_5_moe_model: the value heads must be a multiple of the key heads");
    }
    rope = rope_parameters{
        .theta = std::pow(c.rope_theta, static_cast<double>(c.head_dim) / static_cast<double>(rotary_dim)),
        .partial_rotary_factor = static_cast<double>(rotary_dim) / static_cast<double>(c.head_dim),
    };
    lm_head = weights.contains("lm_head.weight") ? weights.at("lm_head.weight") : embed_tokens;
    at::Tensor permutation = rope_permutation(c.head_dim, rotary_dim, embed_tokens.device());
    const std::int64_t conv_channels = 2 * c.linear_n_key_heads * c.linear_key_head_dim +
        c.linear_n_value_heads * c.linear_value_head_dim;

    for (std::int64_t layer = 0; layer < n_layers; layer++) {
        const std::string p = "layers." + std::to_string(layer) + ".";
        layer_weights w{
            .input_layernorm = zero_centered(text_weight(weights, p + "input_layernorm.weight")),
            .post_attention_layernorm = zero_centered(text_weight(weights, p + "post_attention_layernorm.weight")),
            .moe = moe_weights{
                .router = text_weight(weights, p + "mlp.gate.weight"),
                .experts_gate_up_proj = text_weight(weights, p + "mlp.experts.gate_up_proj"),
                .experts_down_proj = text_weight(weights, p + "mlp.experts.down_proj"),
                .shared_gate_proj = text_weight(weights, p + "mlp.shared_expert.gate_proj.weight"),
                .shared_up_proj = text_weight(weights, p + "mlp.shared_expert.up_proj.weight"),
                .shared_down_proj = text_weight(weights, p + "mlp.shared_expert.down_proj.weight"),
                .shared_expert_gate = text_weight(weights, p + "mlp.shared_expert_gate.weight"),
            },
        };
        if (c.num_experts_per_tok < 1 || w.moe.router.size(0) < c.num_experts_per_tok) {
            throw std::invalid_argument("qwen3_5_moe_model: num_experts_per_tok must be between 1 and the experts");
        }
        const std::int64_t experts = w.moe.router.size(0);
        check_size(w.moe.router, 1, c.hidden_size, p + "mlp.gate.weight");
        check_size(w.moe.experts_gate_up_proj, 0, experts, p + "mlp.experts.gate_up_proj");
        check_size(w.moe.experts_gate_up_proj, 2, c.hidden_size, p + "mlp.experts.gate_up_proj");
        check_size(w.moe.experts_down_proj, 0, experts, p + "mlp.experts.down_proj");
        check_size(w.moe.experts_down_proj, 1, c.hidden_size, p + "mlp.experts.down_proj");
        check_size(w.moe.experts_down_proj, 2, w.moe.experts_gate_up_proj.size(1) / 2, p + "mlp.experts.down_proj");
        check_size(w.moe.experts_gate_up_proj, 1, 2 * w.moe.experts_down_proj.size(2), p + "mlp.experts.gate_up_proj");
        check_size(w.moe.shared_gate_proj, 1, c.hidden_size, p + "mlp.shared_expert.gate_proj.weight");
        check_size(w.moe.shared_up_proj, 0, w.moe.shared_gate_proj.size(0), p + "mlp.shared_expert.up_proj.weight");
        check_size(w.moe.shared_up_proj, 1, c.hidden_size, p + "mlp.shared_expert.up_proj.weight");
        check_size(w.moe.shared_down_proj, 0, c.hidden_size, p + "mlp.shared_expert.down_proj.weight");
        check_size(w.moe.shared_down_proj, 1, w.moe.shared_gate_proj.size(0), p + "mlp.shared_expert.down_proj.weight");
        check_size(w.moe.shared_expert_gate, 0, 1, p + "mlp.shared_expert_gate.weight");
        check_size(w.moe.shared_expert_gate, 1, c.hidden_size, p + "mlp.shared_expert_gate.weight");
        if (!layers.empty()) {
            const moe_weights& first = layers.front().moe;
            check_size(
                w.moe.experts_gate_up_proj,
                0,
                first.experts_gate_up_proj.size(0),
                p + "mlp.experts.gate_up_proj"
            );
            check_size(w.moe.experts_down_proj, 2, first.experts_down_proj.size(2), p + "mlp.experts.down_proj");
        }
        if (full_attention(layer)) {
            const std::string a = p + "self_attn.";
            check_size(text_weight(weights, a + "q_proj.weight"), 0, c.n_heads * 2 * c.head_dim, a + "q_proj.weight");
            check_size(text_weight(weights, a + "q_proj.weight"), 1, c.hidden_size, a + "q_proj.weight");
            check_size(text_weight(weights, a + "k_proj.weight"), 0, c.n_kv_heads * c.head_dim, a + "k_proj.weight");
            check_size(text_weight(weights, a + "k_proj.weight"), 1, c.hidden_size, a + "k_proj.weight");
            check_size(text_weight(weights, a + "v_proj.weight"), 0, c.n_kv_heads * c.head_dim, a + "v_proj.weight");
            check_size(text_weight(weights, a + "v_proj.weight"), 1, c.hidden_size, a + "v_proj.weight");
            check_size(text_weight(weights, a + "o_proj.weight"), 0, c.hidden_size, a + "o_proj.weight");
            check_size(text_weight(weights, a + "o_proj.weight"), 1, c.n_heads * c.head_dim, a + "o_proj.weight");
            check_size(text_weight(weights, a + "q_norm.weight"), 0, c.head_dim, a + "q_norm.weight");
            check_size(text_weight(weights, a + "k_norm.weight"), 0, c.head_dim, a + "k_norm.weight");
            w.attention = attention_weights{
                .q_proj = permute_queries(text_weight(weights, p + "self_attn.q_proj.weight"), c.n_heads, permutation),
                .k_proj = permute_heads(text_weight(weights, p + "self_attn.k_proj.weight"), c.n_kv_heads, permutation),
                .v_proj = text_weight(weights, p + "self_attn.v_proj.weight"),
                .o_proj = text_weight(weights, p + "self_attn.o_proj.weight"),
                .q_norm = zero_centered(text_weight(weights, p + "self_attn.q_norm.weight"))
                              .index_select(0, permutation),
                .k_norm = zero_centered(text_weight(weights, p + "self_attn.k_norm.weight"))
                              .index_select(0, permutation),
            };
            w.kv_layer = kv_specs.size();
            kv_specs.push_back(
                kv_layer{
                    .n_kv_heads = static_cast<std::size_t>(c.n_kv_heads),
                    .head_dim = static_cast<std::size_t>(c.head_dim),
                    .n_q_heads = static_cast<std::size_t>(c.n_heads),
                }
            );
        } else {
            const std::string l = p + "linear_attn.";
            const std::int64_t value_dim = c.linear_n_value_heads * c.linear_value_head_dim;
            check_size(text_weight(weights, l + "in_proj_qkv.weight"), 0, conv_channels, l + "in_proj_qkv.weight");
            check_size(text_weight(weights, l + "in_proj_qkv.weight"), 1, c.hidden_size, l + "in_proj_qkv.weight");
            check_size(text_weight(weights, l + "in_proj_z.weight"), 0, value_dim, l + "in_proj_z.weight");
            check_size(text_weight(weights, l + "in_proj_b.weight"), 0, c.linear_n_value_heads, l + "in_proj_b.weight");
            check_size(text_weight(weights, l + "in_proj_a.weight"), 0, c.linear_n_value_heads, l + "in_proj_a.weight");
            check_size(text_weight(weights, l + "conv1d.weight"), 0, conv_channels, l + "conv1d.weight");
            check_size(text_weight(weights, l + "conv1d.weight"), 2, c.linear_conv_kernel, l + "conv1d.weight");
            check_size(text_weight(weights, l + "dt_bias"), 0, c.linear_n_value_heads, l + "dt_bias");
            check_size(text_weight(weights, l + "A_log"), 0, c.linear_n_value_heads, l + "A_log");
            check_size(text_weight(weights, l + "norm.weight"), 0, c.linear_value_head_dim, l + "norm.weight");
            check_size(text_weight(weights, l + "out_proj.weight"), 0, c.hidden_size, l + "out_proj.weight");
            check_size(text_weight(weights, l + "out_proj.weight"), 1, value_dim, l + "out_proj.weight");
            w.deltanet = deltanet_weights{
                .in_proj_qkv = text_weight(weights, p + "linear_attn.in_proj_qkv.weight"),
                .in_proj_z = text_weight(weights, p + "linear_attn.in_proj_z.weight"),
                .in_proj_b = text_weight(weights, p + "linear_attn.in_proj_b.weight"),
                .in_proj_a = text_weight(weights, p + "linear_attn.in_proj_a.weight"),
                .conv = text_weight(weights, p + "linear_attn.conv1d.weight"),
                .dt_bias = text_weight(weights, p + "linear_attn.dt_bias").to(at::kFloat),
                .neg_a = -text_weight(weights, p + "linear_attn.A_log").to(at::kFloat).exp(),
                .norm = text_weight(weights, p + "linear_attn.norm.weight"),
                .out_proj = text_weight(weights, p + "linear_attn.out_proj.weight"),
            };
            w.state_kv_layer = kv_specs.size();
            kv_specs.push_back(
                kv_layer{
                    .n_kv_heads = 0,
                    .head_dim = 0,
                    .state = recurrent_state{
                        .sizes = {c.linear_n_value_heads, c.linear_key_head_dim, c.linear_value_head_dim},
                        .dtype = at::kFloat,
                    },
                }
            );
            w.conv_kv_layer = kv_specs.size();
            kv_specs.push_back(
                kv_layer{
                    .n_kv_heads = 0,
                    .head_dim = 0,
                    .state = recurrent_state{
                        .sizes = {conv_channels, c.linear_conv_kernel - 1},
                        .dtype = embed_tokens.scalar_type(),
                        .input_window = true,
                    },
                }
            );
        }
        layers.push_back(std::move(w));
    }
}

bool qwen3_5_moe_model::full_attention(std::int64_t layer) const {
    return config.layer_types[layer] == qwen3_5_layer_type::full_attention;
}

std::vector<kv_layer> qwen3_5_moe_model::kv_layers() const {
    return kv_specs;
}

at::Tensor
qwen3_5_moe_model::attention(const at::Tensor& h, std::int64_t layer, const batch& g, std::span<const kv> lines) const {
    const attention_weights& w = *layers[layer].attention;
    const qwen3_5_moe_config& c = config;
    const std::int64_t ntok = h.size(0);
    const std::size_t kv_index = layers[layer].kv_layer;

    at::Tensor query_gate = at::linear(h, w.q_proj).view({ntok, c.n_heads, 2 * c.head_dim});
    at::Tensor q = rmsnorm_cuda(query_gate.narrow(-1, 0, c.head_dim), w.q_norm, c.rms_norm_eps);
    at::Tensor gate = query_gate.narrow(-1, c.head_dim, c.head_dim).reshape({ntok, c.n_heads * c.head_dim});
    q = rope_q(q.view({ntok, c.n_heads * c.head_dim}), g.pos, c.n_heads, c.head_dim, rope);
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
        lines[kv_index],
        kv_index,
        0,
        g,
        1.0 / std::sqrt(static_cast<double>(c.head_dim)),
        rope
    );
    return at::linear(o.reshape({ntok, c.n_heads * c.head_dim}) * at::sigmoid(gate), w.o_proj);
}

at::Tensor
qwen3_5_moe_model::deltanet(const at::Tensor& h, std::int64_t layer, const batch& g, std::span<const kv> lines) const {
    const deltanet_weights& w = *layers[layer].deltanet;
    const std::size_t state_layer = layers[layer].state_kv_layer;
    const std::size_t conv_layer = layers[layer].conv_kv_layer;
    at::Tensor gated = gated_deltanet(
        g,
        gated_deltanet_inputs{
            .qkv = at::linear(h, w.in_proj_qkv),
            .a = at::linear(h, w.in_proj_a),
            .b = at::linear(h, w.in_proj_b),
            .z = at::linear(h, w.in_proj_z),
        },
        gated_deltanet_weights{.conv = w.conv, .dt_bias = w.dt_bias, .neg_a = w.neg_a, .norm = w.norm},
        lines[state_layer],
        lines[conv_layer],
        state_layer,
        conv_layer,
        config.linear_n_key_heads,
        config.rms_norm_eps,
        gate_activation::silu
    );
    return at::linear(gated, w.out_proj);
}

at::Tensor qwen3_5_moe_model::moe(const at::Tensor& h, std::int64_t layer, const at::Tensor& counters) const {
    const moe_weights& m = layers[layer].moe;
    at::Tensor shared = at::linear(
        at::silu(at::linear(h, m.shared_gate_proj)) * at::linear(h, m.shared_up_proj),
        m.shared_down_proj
    );
    at::Tensor routed = routed_experts_cuda(
        h,
        at::linear(h, m.router),
        config.num_experts_per_tok,
        0,
        m.experts_gate_up_proj,
        m.experts_down_proj,
        expert_activation::silu,
        counters
    );
    return sigmoid_gated_add_cuda(routed, at::linear(h, m.shared_expert_gate), shared);
}

model::output qwen3_5_moe_model::forward(const batch& b, std::span<const kv> lines) {
    check_batch(b);
    assert(b.kind != batch_kind::verify);
    const double eps = config.rms_norm_eps;
    at::Tensor x = embed_tokens.index_select(0, b.tokens);
    for (std::int64_t layer = 0; layer < n_layers; layer++) {
        const layer_weights& w = layers[layer];
        at::Tensor h = rmsnorm_cuda(x, w.input_layernorm, eps);
        x = x + (full_attention(layer) ? attention(h, layer, b, lines) : deltanet(h, layer, b, lines));
        x = x + moe(rmsnorm_cuda(x, w.post_attention_layernorm, eps), layer, b.arrival_counters);
    }
    return output{.logits = at::matmul(rmsnorm_cuda(logit_inputs(x, b), norm, eps), lm_head.t())};
}

std::int64_t qwen3_5_moe_model::arrival_counter_count(std::int64_t sequences, std::int64_t tokens) const {
    const moe_weights& m = layers.front().moe;
    return std::max(
        gated_delta_rule_counters(sequences, config.linear_n_value_heads),
        routed_experts_counters(
            tokens,
            config.num_experts_per_tok,
            m.experts_gate_up_proj.size(0),
            m.experts_down_proj.size(2),
            m.experts_down_proj.size(1)
        )
    );
}

}  // namespace pulsarlm
