#include "pulsarlm/engine/model/qwen3_5_moe.hpp"

#include "pulsarlm/engine/model/paged_forward.hpp"
#include "pulsarlm/engine/model/partial_rope.hpp"
#include "pulsarlm/kernels/ops.hpp"

#include <ATen/ATen.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pulsarlm {

namespace {

const std::string prefix = "model.language_model.";

at::Tensor need(const c10::Dict<std::string, at::Tensor>& weights, const std::string& name) {
    if (!weights.contains(name)) {
        throw std::invalid_argument(std::format("qwen3_5_moe_model: missing weight {:?}", name));
    }
    return weights.at(name);
}

at::Tensor need_text(const c10::Dict<std::string, at::Tensor>& weights, const std::string& name) {
    return need(weights, prefix + name);
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
  embed_tokens(need_text(weights, "embed_tokens.weight")),
  norm(zero_centered(need_text(weights, "norm.weight"))) {
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
            .input_layernorm = zero_centered(need_text(weights, p + "input_layernorm.weight")),
            .post_attention_layernorm = zero_centered(need_text(weights, p + "post_attention_layernorm.weight")),
            .moe = moe_weights{
                .router = need_text(weights, p + "mlp.gate.weight"),
                .experts_gate_up_proj = need_text(weights, p + "mlp.experts.gate_up_proj"),
                .experts_down_proj = need_text(weights, p + "mlp.experts.down_proj"),
                .shared_gate_proj = need_text(weights, p + "mlp.shared_expert.gate_proj.weight"),
                .shared_up_proj = need_text(weights, p + "mlp.shared_expert.up_proj.weight"),
                .shared_down_proj = need_text(weights, p + "mlp.shared_expert.down_proj.weight"),
                .shared_expert_gate = need_text(weights, p + "mlp.shared_expert_gate.weight"),
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
        kv_layer_of.push_back(kv_specs.size());
        if (full_attention(layer)) {
            const std::string a = p + "self_attn.";
            check_size(need_text(weights, a + "q_proj.weight"), 0, c.n_heads * 2 * c.head_dim, a + "q_proj.weight");
            check_size(need_text(weights, a + "q_proj.weight"), 1, c.hidden_size, a + "q_proj.weight");
            check_size(need_text(weights, a + "k_proj.weight"), 0, c.n_kv_heads * c.head_dim, a + "k_proj.weight");
            check_size(need_text(weights, a + "k_proj.weight"), 1, c.hidden_size, a + "k_proj.weight");
            check_size(need_text(weights, a + "v_proj.weight"), 0, c.n_kv_heads * c.head_dim, a + "v_proj.weight");
            check_size(need_text(weights, a + "v_proj.weight"), 1, c.hidden_size, a + "v_proj.weight");
            check_size(need_text(weights, a + "o_proj.weight"), 0, c.hidden_size, a + "o_proj.weight");
            check_size(need_text(weights, a + "o_proj.weight"), 1, c.n_heads * c.head_dim, a + "o_proj.weight");
            check_size(need_text(weights, a + "q_norm.weight"), 0, c.head_dim, a + "q_norm.weight");
            check_size(need_text(weights, a + "k_norm.weight"), 0, c.head_dim, a + "k_norm.weight");
            w.attention = attention_weights{
                .q_proj = permute_queries(need_text(weights, p + "self_attn.q_proj.weight"), c.n_heads, permutation),
                .k_proj = permute_heads(need_text(weights, p + "self_attn.k_proj.weight"), c.n_kv_heads, permutation),
                .v_proj = need_text(weights, p + "self_attn.v_proj.weight"),
                .o_proj = need_text(weights, p + "self_attn.o_proj.weight"),
                .q_norm = zero_centered(need_text(weights, p + "self_attn.q_norm.weight")).index_select(0, permutation),
                .k_norm = zero_centered(need_text(weights, p + "self_attn.k_norm.weight")).index_select(0, permutation),
            };
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
            check_size(need_text(weights, l + "in_proj_qkv.weight"), 0, conv_channels, l + "in_proj_qkv.weight");
            check_size(need_text(weights, l + "in_proj_qkv.weight"), 1, c.hidden_size, l + "in_proj_qkv.weight");
            check_size(need_text(weights, l + "in_proj_z.weight"), 0, value_dim, l + "in_proj_z.weight");
            check_size(need_text(weights, l + "in_proj_b.weight"), 0, c.linear_n_value_heads, l + "in_proj_b.weight");
            check_size(need_text(weights, l + "in_proj_a.weight"), 0, c.linear_n_value_heads, l + "in_proj_a.weight");
            check_size(need_text(weights, l + "conv1d.weight"), 0, conv_channels, l + "conv1d.weight");
            check_size(need_text(weights, l + "conv1d.weight"), 2, c.linear_conv_kernel, l + "conv1d.weight");
            check_size(need_text(weights, l + "dt_bias"), 0, c.linear_n_value_heads, l + "dt_bias");
            check_size(need_text(weights, l + "A_log"), 0, c.linear_n_value_heads, l + "A_log");
            check_size(need_text(weights, l + "norm.weight"), 0, c.linear_value_head_dim, l + "norm.weight");
            check_size(need_text(weights, l + "out_proj.weight"), 0, c.hidden_size, l + "out_proj.weight");
            check_size(need_text(weights, l + "out_proj.weight"), 1, value_dim, l + "out_proj.weight");
            w.deltanet = deltanet_weights{
                .in_proj_qkv = need_text(weights, p + "linear_attn.in_proj_qkv.weight"),
                .in_proj_z = need_text(weights, p + "linear_attn.in_proj_z.weight"),
                .in_proj_b = need_text(weights, p + "linear_attn.in_proj_b.weight"),
                .in_proj_a = need_text(weights, p + "linear_attn.in_proj_a.weight"),
                .conv = need_text(weights, p + "linear_attn.conv1d.weight"),
                .dt_bias = need_text(weights, p + "linear_attn.dt_bias").to(at::kFloat),
                .neg_a = -need_text(weights, p + "linear_attn.A_log").to(at::kFloat).exp(),
                .norm = need_text(weights, p + "linear_attn.norm.weight"),
                .out_proj = need_text(weights, p + "linear_attn.out_proj.weight"),
            };
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

at::Tensor qwen3_5_moe_model::attention(
    const at::Tensor& h,
    std::int64_t layer,
    const group_batch& g,
    std::span<const kv> lines
) const {
    const attention_weights& w = *layers[layer].attention;
    const qwen3_5_moe_config& c = config;
    const std::int64_t ntok = h.size(0);
    const std::size_t kv_index = kv_layer_of[layer];

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

at::Tensor qwen3_5_moe_model::deltanet(
    const at::Tensor& h,
    std::int64_t layer,
    const group_batch& g,
    std::span<const kv> lines
) const {
    const deltanet_weights& w = *layers[layer].deltanet;
    const qwen3_5_moe_config& c = config;
    const std::int64_t ntok = h.size(0);
    const std::size_t state_layer = kv_layer_of[layer];
    const std::size_t conv_layer = state_layer + 1;
    std::optional<at::Tensor> cu_seqlens_q;
    if (g.is_prefill()) {
        cu_seqlens_q = g.cu_seqlens_q;
    }

    at::Tensor convolved = causal_conv_cuda(
        at::linear(h, w.in_proj_qkv),
        w.conv,
        lines[conv_layer].k,
        window_layer_rows(g, conv_layer),
        cu_seqlens_q,
        /*dilation=*/1
    );
    at::Tensor gated = gated_delta_rule_cuda(
        convolved,
        at::linear(h, w.in_proj_a),
        at::linear(h, w.in_proj_b),
        w.dt_bias,
        w.neg_a,
        lines[state_layer].k,
        state_layer_rows(g, state_layer),
        cu_seqlens_q,
        c.linear_n_key_heads,
        at::linear(h, w.in_proj_z),
        w.norm,
        c.rms_norm_eps,
        gate_activation::silu,
        g.arrival_counters
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

at::Tensor qwen3_5_moe_model::group_forward(const group_batch& g, std::span<const kv> lines) const {
    const double eps = config.rms_norm_eps;
    at::Tensor x = embed_tokens.index_select(0, g.tokens);
    for (std::int64_t layer = 0; layer < n_layers; layer++) {
        const layer_weights& w = layers[layer];
        at::Tensor h = rmsnorm_cuda(x, w.input_layernorm, eps);
        x = x + (full_attention(layer) ? attention(h, layer, g, lines) : deltanet(h, layer, g, lines));
        x = x + moe(rmsnorm_cuda(x, w.post_attention_layernorm, eps), layer, g.arrival_counters);
    }
    return at::matmul(rmsnorm_cuda(logit_inputs(x, g), norm, eps), lm_head.t());
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

step_logits qwen3_5_moe_model::forward(const step_batch& batch, std::span<const kv> lines) {
    return paged_forward(batch, [&](const group_batch& g) { return group_forward(g, lines); });
}

}  // namespace pulsarlm
