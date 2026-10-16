#include "pulsarlm/model/qwen3_5.hpp"

#include "pulsarlm/model/paged_forward.hpp"
#include "pulsarlm/ops.hpp"

#include <ATen/ATen.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace pulsarlm {

namespace {

const std::string prefix = "model.language_model.";
constexpr std::int64_t delta_rule_chunk = 64;

at::Tensor need(const c10::Dict<std::string, at::Tensor>& weights, const std::string& name) {
    if (!weights.contains(name)) {
        throw std::invalid_argument(std::format("Qwen3_5Model: missing weight {:?}", name));
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

// Head dim i of the paged ops holds Hugging Face's head dim permutation[i]: the paged ops rotate pairs (j, j + half)
// for j below rotary_dim / 2, which hold Hugging Face's rotated pairs (j, j + rotary_dim / 2); the dims past rotary_dim
// fill the rest in order.
at::Tensor rope_permutation(std::int64_t head_dim, std::int64_t rotary_dim, at::Device device) {
    std::vector<std::int64_t> permutation(head_dim);
    std::int64_t half = head_dim / 2;
    std::int64_t next_unrotated = rotary_dim;
    for (std::int64_t i = 0; i < head_dim; i++) {
        std::int64_t pair = i % half;
        permutation[i] = pair < rotary_dim / 2 ? pair + (i >= half ? rotary_dim / 2 : 0) : next_unrotated++;
    }
    return at::tensor(permutation).to(device);
}

// weight [heads * head_dim, in] with each head's output dims permuted.
at::Tensor permute_heads(const at::Tensor& weight, std::int64_t heads, const at::Tensor& permutation) {
    std::int64_t head_dim = permutation.size(0);
    return weight.view({heads, head_dim, -1}).index_select(1, permutation).reshape(weight.sizes()).contiguous();
}

// q_proj [heads * 2 * head_dim, in] with the query half of each head permuted and the gate half kept.
at::Tensor permute_queries(const at::Tensor& weight, std::int64_t heads, const at::Tensor& permutation) {
    std::int64_t head_dim = permutation.size(0);
    at::Tensor halves = weight.view({heads, 2, head_dim, -1});
    at::Tensor queries = halves.select(1, 0).index_select(1, permutation);
    return at::stack({queries, halves.select(1, 1)}, 1).reshape(weight.sizes()).contiguous();
}

at::Tensor l2_normalized(const at::Tensor& x) {
    return x * at::rsqrt(x.pow(2).sum(-1, true) + 1e-6);
}

// The gated delta rule over one sequence, Hugging Face's chunked form. q and k are [tokens, heads, key dim], v
// [tokens, heads, value dim], decay (log) and beta [tokens, heads], state [heads, key dim, value dim], all fp32, q and
// k normalized and q scaled. Returns the outputs [tokens, heads, value dim] and the state after the last token.
std::pair<at::Tensor, at::Tensor> chunked_delta_rule(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& decay,
    const at::Tensor& beta,
    at::Tensor state
) {
    const std::int64_t n = q.size(0);
    const std::int64_t pad = (delta_rule_chunk - n % delta_rule_chunk) % delta_rule_chunk;
    const std::int64_t chunks = (n + pad) / delta_rule_chunk;
    auto by_chunk = [&](const at::Tensor& x) {
        at::Tensor heads_first = x.transpose(0, 1);
        at::Tensor padded = x.dim() == 3 ? at::constant_pad_nd(heads_first, {0, 0, 0, pad})
                                         : at::constant_pad_nd(heads_first, {0, pad});
        std::vector<std::int64_t> sizes{padded.size(0), chunks, delta_rule_chunk};
        if (x.dim() == 3) {
            sizes.push_back(x.size(2));
        }
        return padded.reshape(sizes);
    };
    at::Tensor query = by_chunk(q);
    at::Tensor key = by_chunk(k);
    at::Tensor value = by_chunk(v);
    at::Tensor gate = by_chunk(decay);
    at::Tensor rate = by_chunk(beta).unsqueeze(-1);
    at::Tensor k_beta = key * rate;
    at::Tensor v_beta = value * rate;

    at::Tensor upper = at::ones({delta_rule_chunk, delta_rule_chunk}, q.options().dtype(at::kBool)).triu(1);
    at::Tensor cum_decay = gate.cumsum(-1);
    at::Tensor pairwise = (cum_decay.unsqueeze(-1) - cum_decay.unsqueeze(-2))
                              .masked_fill(upper, -std::numeric_limits<float>::infinity())
                              .exp();
    at::Tensor ut_system = at::matmul(k_beta, key.transpose(-1, -2)) * pairwise;
    at::Tensor intra_chunk = at::matmul(query, key.transpose(-1, -2)) * pairwise;
    at::Tensor decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1);
    at::Tensor new_values = at::linalg_solve_triangular(ut_system, v_beta, false, true, true);
    at::Tensor k_cum_decay = at::linalg_solve_triangular(ut_system, decayed_k_beta, false, true, true);

    query = query * cum_decay.exp().unsqueeze(-1);
    key = key * (cum_decay.narrow(-1, delta_rule_chunk - 1, 1) - cum_decay).exp().unsqueeze(-1);
    at::Tensor chunk_decay = cum_decay.select(-1, delta_rule_chunk - 1).exp().unsqueeze(-1).unsqueeze(-1);
    at::Tensor out = at::empty_like(new_values);
    for (std::int64_t i = 0; i < chunks; i++) {
        at::Tensor v_new = new_values.select(1, i) - at::matmul(k_cum_decay.select(1, i), state);
        out.select(1, i).copy_(at::matmul(query.select(1, i), state) + at::matmul(intra_chunk.select(1, i), v_new));
        state = state * chunk_decay.select(1, i) + at::matmul(key.select(1, i).transpose(-1, -2), v_new);
    }
    at::Tensor outputs = out.reshape({out.size(0), -1, out.size(-1)}).narrow(1, 0, n).transpose(0, 1);
    return {outputs, state};
}

// One token of each sequence under the gated delta rule, Hugging Face's recurrent form. q and k are [seqs, heads, key
// dim], v [seqs, heads, value dim], decay and beta [seqs, heads], state [seqs, heads, key dim, value dim], fp32.
std::pair<at::Tensor, at::Tensor> delta_rule_step(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& decay,
    const at::Tensor& beta,
    at::Tensor state
) {
    state = state * decay.exp().unsqueeze(-1).unsqueeze(-1);
    at::Tensor remembered = (state * k.unsqueeze(-1)).sum(-2);
    at::Tensor delta = (v - remembered) * beta.unsqueeze(-1);
    state = state + k.unsqueeze(-1) * delta.unsqueeze(-2);
    return {(state * q.unsqueeze(-1)).sum(-2), state};
}

// Each channel of window [..., channels, kernel - 1 + n] convolved with its own kernel, for the last n inputs.
at::Tensor depthwise_conv(const at::Tensor& window, const at::Tensor& weight) {
    return at::conv1d(
        window,
        weight,
        std::nullopt,
        at::IntArrayRef{1},
        at::IntArrayRef{0},
        at::IntArrayRef{1},
        weight.size(0)
    );
}

// line_rows [seqs, 2] of a recurrent layer: each sequence's running line and boundary line.
at::Tensor line_column(const at::Tensor& line_rows, std::int64_t column) {
    return line_rows.select(1, column).to(at::kLong);
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
    at::Tensor permutation = rope_permutation(c.head_dim, rotary_dim, embed_tokens.device());
    const std::int64_t conv_channels = 2 * c.linear_n_key_heads * c.linear_key_head_dim +
        c.linear_n_value_heads * c.linear_value_head_dim;

    for (std::int64_t layer = 0; layer < n_layers; layer++) {
        const std::string p = "layers." + std::to_string(layer) + ".";
        LayerWeights w{
            .input_layernorm = zero_centered(need_text(weights, p + "input_layernorm.weight")),
            .post_attention_layernorm = zero_centered(need_text(weights, p + "post_attention_layernorm.weight")),
            .gate_proj = need_text(weights, p + "mlp.gate_proj.weight"),
            .up_proj = need_text(weights, p + "mlp.up_proj.weight"),
            .down_proj = need_text(weights, p + "mlp.down_proj.weight"),
        };
        kv_layer_of.push_back(kv_specs.size());
        if (full_attention(layer)) {
            w.attention = AttentionWeights{
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
            w.deltanet = DeltaNetWeights{
                .in_proj_qkv = need_text(weights, p + "linear_attn.in_proj_qkv.weight"),
                .in_proj_z = need_text(weights, p + "linear_attn.in_proj_z.weight"),
                .in_proj_b = need_text(weights, p + "linear_attn.in_proj_b.weight"),
                .in_proj_a = need_text(weights, p + "linear_attn.in_proj_a.weight"),
                .conv = need_text(weights, p + "linear_attn.conv1d.weight"),
                .dt_bias = need_text(weights, p + "linear_attn.dt_bias"),
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
                    },
                }
            );
        }
        layers.push_back(std::move(w));
    }
    for (std::int64_t layer = 3 * n_layers / 4; layer < n_layers; layer++) {
        if (full_attention(layer)) {
            relevance_layers.insert(layer);
        }
    }
}

bool Qwen3_5Model::full_attention(std::int64_t layer) const {
    return config.layer_types[layer] == qwen3_5_layer_type::full_attention;
}

std::vector<kv_layer> Qwen3_5Model::kv_layers() const {
    return kv_specs;
}

void Qwen3_5Model::set_relevance_layers(const std::vector<std::int64_t>& layers) {
    for (std::int64_t layer : layers) {
        if (layer < 0 || layer >= n_layers || !full_attention(layer)) {
            throw std::invalid_argument(std::format("Qwen3_5Model: layer {} is not a full-attention layer", layer));
        }
    }
    relevance_layers = std::unordered_set<std::int64_t>(layers.begin(), layers.end());
}

at::Tensor
Qwen3_5Model::attention(const at::Tensor& h, std::int64_t layer, const GroupBatch& g, std::span<const kv> lines) const {
    const AttentionWeights& w = *layers[layer].attention;
    const qwen3_5_config& c = config;
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
        g,
        relevance_layers.contains(layer),
        1.0 / std::sqrt(static_cast<double>(c.head_dim)),
        rope
    );
    return at::linear(o.reshape({ntok, c.n_heads * c.head_dim}) * at::sigmoid(gate), w.o_proj);
}

// A sequence's delta rule state and convolution inputs start from its running lines and end there; the state after
// its first page_end_tokens tokens goes to its boundary lines.
at::Tensor
Qwen3_5Model::deltanet(const at::Tensor& h, std::int64_t layer, const GroupBatch& g, std::span<const kv> lines) const {
    const DeltaNetWeights& w = *layers[layer].deltanet;
    const qwen3_5_config& c = config;
    const std::int64_t ntok = h.size(0);
    const std::int64_t key_dim = c.linear_n_key_heads * c.linear_key_head_dim;
    const std::int64_t value_heads = c.linear_n_value_heads;
    const std::int64_t value_dim = value_heads * c.linear_value_head_dim;
    const std::int64_t history = c.linear_conv_kernel - 1;
    const std::size_t state_layer = kv_layer_of[layer];
    const std::size_t conv_layer = state_layer + 1;
    const at::Tensor& states = lines[state_layer].k;
    const at::Tensor& conv_inputs = lines[conv_layer].k;

    at::Tensor mixed = at::linear(h, w.in_proj_qkv);
    at::Tensor z = at::linear(h, w.in_proj_z).view({ntok, value_heads, c.linear_value_head_dim});
    at::Tensor beta = at::sigmoid(at::linear(h, w.in_proj_b)).to(at::kFloat);
    at::Tensor decay = at::softplus(at::linear(h, w.in_proj_a).to(at::kFloat) + w.dt_bias.to(at::kFloat)) * w.neg_a;
    auto heads = [&](const at::Tensor& convolved) {
        const std::int64_t n = convolved.size(0);
        const std::int64_t repeats = value_heads / c.linear_n_key_heads;
        at::Tensor q = convolved.narrow(-1, 0, key_dim).view({n, c.linear_n_key_heads, c.linear_key_head_dim});
        at::Tensor k = convolved.narrow(-1, key_dim, key_dim).view({n, c.linear_n_key_heads, c.linear_key_head_dim});
        at::Tensor v = convolved.narrow(-1, 2 * key_dim, value_dim).view({n, value_heads, c.linear_value_head_dim});
        q = l2_normalized(q.repeat_interleave(repeats, 1).to(at::kFloat)) /
            std::sqrt(static_cast<double>(c.linear_key_head_dim));
        k = l2_normalized(k.repeat_interleave(repeats, 1).to(at::kFloat));
        return std::tuple{q, k, v.to(at::kFloat)};
    };

    at::Tensor state_rows = g.page_tables.select(0, static_cast<std::int64_t>(state_layer));
    at::Tensor conv_rows = g.page_tables.select(0, static_cast<std::int64_t>(conv_layer));
    at::Tensor core;
    if (!g.is_prefill()) {
        at::Tensor ends_page = (g.page_end_tokens > 0).view({-1, 1, 1});
        at::Tensor window = at::cat({conv_inputs.index_select(0, line_column(conv_rows, 0)), mixed.unsqueeze(-1)}, -1);
        at::Tensor convolved = at::silu(depthwise_conv(window, w.conv)).squeeze(-1);
        at::Tensor kept = window.narrow(-1, 1, history);
        at::Tensor boundary_conv = line_column(conv_rows, 1);
        conv_inputs.index_copy_(0, line_column(conv_rows, 0), kept);
        conv_inputs
            .index_copy_(0, boundary_conv, at::where(ends_page, kept, conv_inputs.index_select(0, boundary_conv)));
        auto [q, k, v] = heads(convolved);
        at::Tensor state;
        std::tie(
            core,
            state
        ) = delta_rule_step(q, k, v, decay, beta, states.index_select(0, line_column(state_rows, 0)));
        at::Tensor boundary_state = line_column(state_rows, 1);
        states.index_copy_(0, line_column(state_rows, 0), state);
        states.index_copy_(
            0,
            boundary_state,
            at::where(ends_page.unsqueeze(-1), state, states.index_select(0, boundary_state))
        );
    } else {
        at::Tensor starts = g.cu_seqlens_q.cpu();
        at::Tensor page_ends = g.page_end_tokens.cpu();
        at::Tensor host_state_rows = state_rows.cpu();
        at::Tensor host_conv_rows = conv_rows.cpu();
        std::vector<at::Tensor> outputs;
        for (std::int64_t s = 0; s + 1 < starts.size(0); s++) {
            const std::int64_t first = starts[s].item<std::int64_t>();
            const std::int64_t n = starts[s + 1].item<std::int64_t>() - first;
            const std::int64_t page_end = page_ends[s].item<std::int64_t>();
            at::Tensor running_conv = conv_inputs.select(0, host_conv_rows[s][0].item<std::int64_t>());
            at::Tensor running_state = states.select(0, host_state_rows[s][0].item<std::int64_t>());
            at::Tensor window = at::cat({running_conv, mixed.narrow(0, first, n).t()}, -1);
            at::Tensor convolved = at::silu(depthwise_conv(window.unsqueeze(0), w.conv)).squeeze(0).t();
            if (page_end > 0) {
                conv_inputs.select(0, host_conv_rows[s][1].item<std::int64_t>())
                    .copy_(window.narrow(-1, page_end, history));
            }
            running_conv.copy_(window.narrow(-1, n, history));

            auto [q, k, v] = heads(convolved);
            at::Tensor token_decay = decay.narrow(0, first, n);
            at::Tensor token_beta = beta.narrow(0, first, n);
            at::Tensor state = running_state.clone();
            std::vector<at::Tensor> parts;
            for (auto [part_first, part_end] : {std::pair{std::int64_t{0}, page_end}, std::pair{page_end, n}}) {
                if (part_end == part_first) {
                    continue;
                }
                const std::int64_t length = part_end - part_first;
                at::Tensor part;
                std::tie(part, state) = chunked_delta_rule(
                    q.narrow(0, part_first, length),
                    k.narrow(0, part_first, length),
                    v.narrow(0, part_first, length),
                    token_decay.narrow(0, part_first, length),
                    token_beta.narrow(0, part_first, length),
                    state
                );
                parts.push_back(part);
                if (part_end == page_end) {
                    states.select(0, host_state_rows[s][1].item<std::int64_t>()).copy_(state);
                }
            }
            running_state.copy_(state);
            outputs.push_back(at::cat(parts));
        }
        core = at::cat(outputs);
    }

    at::Tensor normed = core.to(h.scalar_type()).to(at::kFloat);
    normed = normed * at::rsqrt(normed.pow(2).mean(-1, true) + c.rms_norm_eps);
    at::Tensor gated = (w.norm * normed.to(h.scalar_type())).to(at::kFloat) * at::silu(z.to(at::kFloat));
    return at::linear(gated.to(h.scalar_type()).view({ntok, value_dim}), w.out_proj);
}

at::Tensor Qwen3_5Model::group_forward(const GroupBatch& g, std::span<const kv> lines) const {
    const double eps = config.rms_norm_eps;
    at::Tensor x = embed_tokens.index_select(0, g.tokens);
    for (std::int64_t layer = 0; layer < n_layers; layer++) {
        const LayerWeights& w = layers[layer];
        at::Tensor h = rmsnorm_cuda(x, w.input_layernorm, eps);
        x = x + (full_attention(layer) ? attention(h, layer, g, lines) : deltanet(h, layer, g, lines));
        at::Tensor h2 = rmsnorm_cuda(x, w.post_attention_layernorm, eps);
        x = x + at::linear(at::silu(at::linear(h2, w.gate_proj)) * at::linear(h2, w.up_proj), w.down_proj);
    }
    return at::matmul(rmsnorm_cuda(logit_inputs(x, g), norm, eps), lm_head.t());
}

StepLogits Qwen3_5Model::forward(const StepBatch& batch, std::span<const kv> lines) {
    return paged_forward(batch, [&](const GroupBatch& g) { return group_forward(g, lines); });
}

}  // namespace pulsarlm
