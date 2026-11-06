#include "pulsarlm/model/qwen3_5.hpp"

#include "pulsarlm/model/paged_forward.hpp"
#include "pulsarlm/model/partial_rope.hpp"
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
#include <vector>

namespace pulsarlm {

namespace {

const std::string prefix = "model.language_model.";

at::Tensor need(const c10::Dict<std::string, at::Tensor>& weights, const std::string& name) {
    if (!weights.contains(name)) {
        throw std::invalid_argument(std::format("Qwen3_5Model: missing weight {:?}", name));
    }
    return weights.at(name);
}

at::Tensor need_text(const c10::Dict<std::string, at::Tensor>& weights, const std::string& name) {
    return need(weights, prefix + name);
}

bool quantized(const c10::Dict<std::string, at::Tensor>& weights, const std::string& name) {
    return weights.contains(prefix + name + ".weight_scale_inv");
}

const fp8_block& block_of(const std::optional<fp8_block>& block, const std::string& name) {
    if (!block) {
        throw std::invalid_argument(std::format("Qwen3_5Model: {:?} has fp8 scales but no fp8_weight_block", name));
    }
    return *block;
}

// name is a linear's prefix, such as layers.0.mlp.gate_proj. An fp8 linear may be on the host; it is copied to device
// only for its repack.
Linear need_linear(
    const c10::Dict<std::string, at::Tensor>& weights,
    const std::string& name,
    const std::optional<fp8_block>& block,
    at::Device device
) {
    at::Tensor weight = need_text(weights, name + ".weight");
    if (!quantized(weights, name)) {
        return Linear::dense(std::move(weight));
    }
    const fp8_block& b = block_of(block, name);
    return Linear::fp8(weight.to(device), need_text(weights, name + ".weight_scale_inv").to(device), b.n, b.k);
}

// permute reorders a linear's output rows. An fp8 linear's scales expand to one per row first, since the permutation
// moves rows across the scale blocks.
template <class row_permutation>
Linear need_permuted_linear(
    const c10::Dict<std::string, at::Tensor>& weights,
    const std::string& name,
    const std::optional<fp8_block>& block,
    at::Device device,
    row_permutation permute
) {
    const at::Tensor weight = need_text(weights, name + ".weight").to(device);
    if (!quantized(weights, name)) {
        return Linear::dense(permute(weight));
    }
    const fp8_block& b = block_of(block, name);
    const at::Tensor row_scale = need_text(weights, name + ".weight_scale_inv")
                                     .to(device)
                                     .repeat_interleave(b.n, 0)
                                     .narrow(0, 0, weight.size(0));
    return Linear::fp8(permute(weight.view(at::kByte)).view(at::kFloat8_e4m3fn), permute(row_scale), 1, b.k);
}

// Hugging Face's Qwen3.5 RMSNorm scales by 1 + weight, in fp32.
at::Tensor zero_centered(const at::Tensor& weight) {
    return weight.to(at::kFloat) + 1.0;
}

}  // namespace

Qwen3_5Model::Qwen3_5Model(qwen3_5_config config, const c10::Dict<std::string, at::Tensor>& weights)
: config(std::move(config)),
  n_layers(static_cast<std::int64_t>(this->config.layer_types.size())),
  embed_tokens(need_text(weights, "embed_tokens.weight")),
  norm(zero_centered(need_text(weights, "norm.weight"))) {
    const qwen3_5_config& c = this->config;
    const auto rotary_dim = static_cast<std::int64_t>(std::llround(c.partial_rotary_factor * c.head_dim));
    if (rotary_dim % 2 != 0 || rotary_dim > c.head_dim || c.head_dim % 2 != 0) {
        throw std::invalid_argument("Qwen3_5Model: the rotated dims must be an even part of an even head dim");
    }
    if (c.linear_n_value_heads % c.linear_n_key_heads != 0) {
        throw std::invalid_argument("Qwen3_5Model: the value heads must be a multiple of the key heads");
    }
    rope = rope_parameters{
        .theta = std::pow(c.rope_theta, static_cast<double>(c.head_dim) / static_cast<double>(rotary_dim)),
        .partial_rotary_factor = static_cast<double>(rotary_dim) / static_cast<double>(c.head_dim),
    };
    lm_head = weights.contains("lm_head.weight") ? weights.at("lm_head.weight") : embed_tokens;
    const at::Device device = embed_tokens.device();
    at::Tensor permutation = rope_permutation(c.head_dim, rotary_dim, device);
    const std::int64_t conv_channels = 2 * c.linear_n_key_heads * c.linear_key_head_dim +
        c.linear_n_value_heads * c.linear_value_head_dim;

    for (std::int64_t layer = 0; layer < n_layers; layer++) {
        const std::string p = "layers." + std::to_string(layer) + ".";
        LayerWeights w{
            .input_layernorm = zero_centered(need_text(weights, p + "input_layernorm.weight")),
            .post_attention_layernorm = zero_centered(need_text(weights, p + "post_attention_layernorm.weight")),
            .gate_proj = need_linear(weights, p + "mlp.gate_proj", c.fp8_weight_block, device),
            .up_proj = need_linear(weights, p + "mlp.up_proj", c.fp8_weight_block, device),
            .down_proj = need_linear(weights, p + "mlp.down_proj", c.fp8_weight_block, device),
        };
        kv_layer_of.push_back(kv_specs.size());
        if (full_attention(layer)) {
            w.attention = AttentionWeights{
                .q_proj = need_permuted_linear(
                    weights,
                    p + "self_attn.q_proj",
                    c.fp8_weight_block,
                    device,
                    [&](const at::Tensor& rows) { return permute_queries(rows, c.n_heads, permutation); }
                ),
                .k_proj = need_permuted_linear(
                    weights,
                    p + "self_attn.k_proj",
                    c.fp8_weight_block,
                    device,
                    [&](const at::Tensor& rows) { return permute_heads(rows, c.n_kv_heads, permutation); }
                ),
                .v_proj = need_linear(weights, p + "self_attn.v_proj", c.fp8_weight_block, device),
                .o_proj = need_linear(weights, p + "self_attn.o_proj", c.fp8_weight_block, device),
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
            w.deltanet = DeltaNetWeights{
                .in_proj_qkv = need_linear(weights, p + "linear_attn.in_proj_qkv", c.fp8_weight_block, device),
                .in_proj_z = need_linear(weights, p + "linear_attn.in_proj_z", c.fp8_weight_block, device),
                .in_proj_b = need_linear(weights, p + "linear_attn.in_proj_b", c.fp8_weight_block, device),
                .in_proj_a = need_linear(weights, p + "linear_attn.in_proj_a", c.fp8_weight_block, device),
                .conv = need_text(weights, p + "linear_attn.conv1d.weight"),
                .dt_bias = need_text(weights, p + "linear_attn.dt_bias").to(at::kFloat),
                .neg_a = -need_text(weights, p + "linear_attn.A_log").to(at::kFloat).exp(),
                .norm = need_text(weights, p + "linear_attn.norm.weight"),
                .out_proj = need_linear(weights, p + "linear_attn.out_proj", c.fp8_weight_block, device),
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
                    },
                }
            );
        }
        layers.push_back(std::move(w));
    }
}

bool Qwen3_5Model::full_attention(std::int64_t layer) const {
    return config.layer_types[layer] == qwen3_5_layer_type::full_attention;
}

std::vector<kv_layer> Qwen3_5Model::kv_layers() const {
    return kv_specs;
}

at::Tensor
Qwen3_5Model::attention(const at::Tensor& h, std::int64_t layer, const GroupBatch& g, std::span<const kv> lines) const {
    const AttentionWeights& w = *layers[layer].attention;
    const qwen3_5_config& c = config;
    const std::int64_t ntok = h.size(0);
    const std::size_t kv_index = kv_layer_of[layer];

    at::Tensor query_gate = w.q_proj.apply(h).view({ntok, c.n_heads, 2 * c.head_dim});
    at::Tensor q = rmsnorm_cuda(query_gate.narrow(-1, 0, c.head_dim), w.q_norm, c.rms_norm_eps);
    at::Tensor gate = query_gate.narrow(-1, c.head_dim, c.head_dim).reshape({ntok, c.n_heads * c.head_dim});
    q = rope_q(q.view({ntok, c.n_heads * c.head_dim}), g.pos, c.n_heads, c.head_dim, rope);
    at::Tensor k = rmsnorm_cuda(w.k_proj.apply(h).view({ntok, c.n_kv_heads, c.head_dim}), w.k_norm, c.rms_norm_eps);
    at::Tensor v = w.v_proj.apply(h).view({ntok, c.n_kv_heads, c.head_dim}).contiguous();
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
    return w.o_proj.apply(o.reshape({ntok, c.n_heads * c.head_dim}) * at::sigmoid(gate));
}

at::Tensor
Qwen3_5Model::deltanet(const at::Tensor& h, std::int64_t layer, const GroupBatch& g, std::span<const kv> lines) const {
    const DeltaNetWeights& w = *layers[layer].deltanet;
    const qwen3_5_config& c = config;
    const std::int64_t ntok = h.size(0);
    const std::size_t state_layer = kv_layer_of[layer];
    const std::size_t conv_layer = state_layer + 1;
    std::optional<at::Tensor> cu_seqlens_q;
    if (g.is_prefill()) {
        cu_seqlens_q = g.cu_seqlens_q;
    }

    at::Tensor convolved = causal_conv_cuda(
        w.in_proj_qkv.apply(h),
        w.conv,
        lines[conv_layer].k,
        g.page_tables.select(0, static_cast<std::int64_t>(conv_layer)),
        cu_seqlens_q,
        g.page_end_tokens,
        /*dilation=*/1
    );
    at::Tensor core = gated_delta_rule_cuda(
        convolved,
        w.in_proj_a.apply(h),
        w.in_proj_b.apply(h),
        w.dt_bias,
        w.neg_a,
        lines[state_layer].k,
        g.page_tables.select(0, static_cast<std::int64_t>(state_layer)),
        cu_seqlens_q,
        g.page_end_tokens,
        c.linear_n_key_heads
    );
    at::Tensor z = w.in_proj_z.apply(h).view(core.sizes());
    at::Tensor gated = gated_rmsnorm_cuda(core, z, w.norm, c.rms_norm_eps, gate_activation::silu);
    return w.out_proj.apply(gated.view({ntok, c.linear_n_value_heads * c.linear_value_head_dim}));
}

at::Tensor Qwen3_5Model::group_forward(const GroupBatch& g, std::span<const kv> lines) const {
    const double eps = config.rms_norm_eps;
    at::Tensor x = embed_tokens.index_select(0, g.tokens);
    for (std::int64_t layer = 0; layer < n_layers; layer++) {
        const LayerWeights& w = layers[layer];
        at::Tensor h = rmsnorm_cuda(x, w.input_layernorm, eps);
        x = x + (full_attention(layer) ? attention(h, layer, g, lines) : deltanet(h, layer, g, lines));
        at::Tensor h2 = rmsnorm_cuda(x, w.post_attention_layernorm, eps);
        x = x + w.down_proj.apply(at::silu(w.gate_proj.apply(h2)) * w.up_proj.apply(h2));
    }
    return at::matmul(rmsnorm_cuda(logit_inputs(x, g), norm, eps), lm_head.t());
}

StepLogits Qwen3_5Model::forward(const StepBatch& batch, std::span<const kv> lines) {
    return paged_forward(batch, [&](const GroupBatch& g) { return group_forward(g, lines); });
}

}  // namespace pulsarlm
