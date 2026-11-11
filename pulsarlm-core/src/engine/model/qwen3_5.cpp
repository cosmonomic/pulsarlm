#include "pulsarlm/engine/model/qwen3_5.hpp"

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
        throw std::invalid_argument(std::format("qwen3_5_model: missing weight {:?}", name));
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
        throw std::invalid_argument(std::format("qwen3_5_model: {:?} has fp8 scales but no fp8_weight_block", name));
    }
    return *block;
}

// name is a linear's prefix, such as layers.0.mlp.gate_proj. An fp8 linear may be on the host; it is copied to device
// only for its repack.
linear need_linear(
    const c10::Dict<std::string, at::Tensor>& weights,
    const std::string& name,
    const std::optional<fp8_block>& block,
    at::Device device
) {
    at::Tensor weight = need_text(weights, name + ".weight");
    if (!quantized(weights, name)) {
        return linear::dense(std::move(weight));
    }
    const fp8_block& b = block_of(block, name);
    return linear::fp8(weight.to(device), need_text(weights, name + ".weight_scale_inv").to(device), b.n, b.k);
}

// permute reorders a linear's output rows. An fp8 linear's scales expand to one per row first, since the permutation
// moves rows across the scale blocks.
template <class row_permutation>
linear need_permuted_linear(
    const c10::Dict<std::string, at::Tensor>& weights,
    const std::string& name,
    const std::optional<fp8_block>& block,
    at::Device device,
    row_permutation permute
) {
    const at::Tensor weight = need_text(weights, name + ".weight").to(device);
    if (!quantized(weights, name)) {
        return linear::dense(permute(weight));
    }
    const fp8_block& b = block_of(block, name);
    const at::Tensor row_scale = need_text(weights, name + ".weight_scale_inv")
                                     .to(device)
                                     .repeat_interleave(b.n, 0)
                                     .narrow(0, 0, weight.size(0));
    return linear::fp8(permute(weight.view(at::kByte)).view(at::kFloat8_e4m3fn), permute(row_scale), 1, b.k);
}

// Hugging Face's Qwen3.5 RMSNorm scales by 1 + weight, in fp32.
at::Tensor zero_centered(const at::Tensor& weight) {
    return weight.to(at::kFloat) + 1.0;
}

std::int64_t per_rank(std::int64_t total, std::int64_t ranks, const char* what) {
    if (total % ranks != 0) {
        throw std::invalid_argument(std::format("qwen3_5_model: {} {} do not split over {} ranks", total, what, ranks));
    }
    return total / ranks;
}

// A rank's rows of the vocabulary: all of it on one rank, else its share of the vocabulary padded to a multiple of 64.
std::int64_t vocab_rows(std::int64_t vocab_size, std::int64_t ranks) {
    if (ranks == 1) {
        return vocab_size;
    }
    return per_rank((vocab_size + 63) / 64 * 64, ranks, "padded vocabulary rows");
}

void check_size(const at::Tensor& tensor, std::int64_t dim, std::int64_t size, const std::string& name) {
    if (tensor.size(dim) != size) {
        throw std::invalid_argument(
            std::format(
                "qwen3_5_model: {:?} has {} in dim {}, this rank's share is {}",
                name,
                tensor.size(dim),
                dim,
                size
            )
        );
    }
}

}  // namespace

qwen3_5_model::qwen3_5_model(
    qwen3_5_config config,
    const c10::Dict<std::string, at::Tensor>& weights,
    communicator comm
)
: config(std::move(config)),
  comm(std::move(comm)),
  n_layers(static_cast<std::int64_t>(this->config.layer_types.size())),
  embed_tokens(need_text(weights, "embed_tokens.weight")),
  norm(zero_centered(need_text(weights, "norm.weight"))) {
    const qwen3_5_config& c = this->config;
    const auto rotary_dim = static_cast<std::int64_t>(std::llround(c.partial_rotary_factor * c.head_dim));
    if (rotary_dim % 2 != 0 || rotary_dim > c.head_dim || c.head_dim % 2 != 0) {
        throw std::invalid_argument("qwen3_5_model: the rotated dims must be an even part of an even head dim");
    }
    if (c.linear_n_value_heads % c.linear_n_key_heads != 0) {
        throw std::invalid_argument("qwen3_5_model: the value heads must be a multiple of the key heads");
    }
    const auto ranks = static_cast<std::int64_t>(this->comm.size());
    const auto rank = static_cast<std::int64_t>(this->comm.rank());
    n_heads = per_rank(c.n_heads, ranks, "query heads");
    linear_n_key_heads = per_rank(c.linear_n_key_heads, ranks, "linear attention key heads");
    linear_n_value_heads = per_rank(c.linear_n_value_heads, ranks, "linear attention value heads");
    std::int64_t first_kv_head = 0;
    if (ranks <= c.n_kv_heads) {
        n_kv_heads = per_rank(c.n_kv_heads, ranks, "KV heads");
        first_kv_head = rank * n_kv_heads;
    } else {
        if (ranks % c.n_kv_heads != 0) {
            throw std::invalid_argument(
                std::format("qwen3_5_model: {} ranks are not a multiple of the {} KV heads", ranks, c.n_kv_heads)
            );
        }
        n_kv_heads = 1;
        first_kv_head = rank / (ranks / c.n_kv_heads);
    }
    check_size(embed_tokens, 0, vocab_rows(c.vocab_size, ranks), "embed_tokens.weight");
    rope = rope_parameters{
        .theta = std::pow(c.rope_theta, static_cast<double>(c.head_dim) / static_cast<double>(rotary_dim)),
        .partial_rotary_factor = static_cast<double>(rotary_dim) / static_cast<double>(c.head_dim),
    };
    lm_head = weights.contains("lm_head.weight") ? weights.at("lm_head.weight") : embed_tokens;
    check_size(lm_head, 0, embed_tokens.size(0), "lm_head.weight");
    const at::Device device = embed_tokens.device();
    at::Tensor permutation = rope_permutation(c.head_dim, rotary_dim, device);
    const std::int64_t key_dim = linear_n_key_heads * c.linear_key_head_dim;
    const std::int64_t value_dim = linear_n_value_heads * c.linear_value_head_dim;
    const std::int64_t conv_channels = 2 * key_dim + value_dim;
    const std::int64_t full_key_dim = c.linear_n_key_heads * c.linear_key_head_dim;
    const auto row_range = [](std::int64_t first, std::int64_t count) {
        return std::pair{static_cast<std::size_t>(first), static_cast<std::size_t>(first + count)};
    };

    for (std::int64_t layer = 0; layer < n_layers; layer++) {
        const std::string p = "layers." + std::to_string(layer) + ".";
        layer_weights w{
            .input_layernorm = zero_centered(need_text(weights, p + "input_layernorm.weight")),
            .post_attention_layernorm = zero_centered(need_text(weights, p + "post_attention_layernorm.weight")),
            .gate_proj = need_linear(weights, p + "mlp.gate_proj", c.fp8_weight_block, device),
            .up_proj = need_linear(weights, p + "mlp.up_proj", c.fp8_weight_block, device),
            .down_proj = need_linear(weights, p + "mlp.down_proj", c.fp8_weight_block, device),
        };
        kv_layer_of.push_back(kv_shards.size());
        if (full_attention(layer)) {
            const std::string a = p + "self_attn.";
            check_size(need_text(weights, a + "q_proj.weight"), 0, n_heads * 2 * c.head_dim, a + "q_proj.weight");
            check_size(need_text(weights, a + "k_proj.weight"), 0, n_kv_heads * c.head_dim, a + "k_proj.weight");
            check_size(need_text(weights, a + "v_proj.weight"), 0, n_kv_heads * c.head_dim, a + "v_proj.weight");
            check_size(need_text(weights, a + "o_proj.weight"), 1, n_heads * c.head_dim, a + "o_proj.weight");
            w.attention = attention_weights{
                .q_proj = need_permuted_linear(
                    weights,
                    a + "q_proj",
                    c.fp8_weight_block,
                    device,
                    [&](const at::Tensor& rows) { return permute_queries(rows, n_heads, permutation); }
                ),
                .k_proj = need_permuted_linear(
                    weights,
                    a + "k_proj",
                    c.fp8_weight_block,
                    device,
                    [&](const at::Tensor& rows) { return permute_heads(rows, n_kv_heads, permutation); }
                ),
                .v_proj = need_linear(weights, a + "v_proj", c.fp8_weight_block, device),
                .o_proj = need_linear(weights, a + "o_proj", c.fp8_weight_block, device),
                .q_norm = zero_centered(need_text(weights, a + "q_norm.weight")).index_select(0, permutation),
                .k_norm = zero_centered(need_text(weights, a + "k_norm.weight")).index_select(0, permutation),
            };
            kv_shards.push_back(
                kv_layer_shard{
                    .full =
                        kv_layer{
                            .n_kv_heads = static_cast<std::size_t>(c.n_kv_heads),
                            .head_dim = static_cast<std::size_t>(c.head_dim),
                            .n_q_heads = static_cast<std::size_t>(c.n_heads),
                        },
                    .rows = {row_range(first_kv_head, n_kv_heads)},
                }
            );
        } else {
            const std::string l = p + "linear_attn.";
            check_size(need_text(weights, l + "in_proj_qkv.weight"), 0, conv_channels, l + "in_proj_qkv.weight");
            check_size(need_text(weights, l + "in_proj_z.weight"), 0, value_dim, l + "in_proj_z.weight");
            check_size(need_text(weights, l + "in_proj_b.weight"), 0, linear_n_value_heads, l + "in_proj_b.weight");
            check_size(need_text(weights, l + "in_proj_a.weight"), 0, linear_n_value_heads, l + "in_proj_a.weight");
            check_size(need_text(weights, l + "conv1d.weight"), 0, conv_channels, l + "conv1d.weight");
            check_size(need_text(weights, l + "dt_bias"), 0, linear_n_value_heads, l + "dt_bias");
            check_size(need_text(weights, l + "A_log"), 0, linear_n_value_heads, l + "A_log");
            check_size(need_text(weights, l + "out_proj.weight"), 1, value_dim, l + "out_proj.weight");
            w.deltanet = deltanet_weights{
                .in_proj_qkv = need_linear(weights, l + "in_proj_qkv", c.fp8_weight_block, device),
                .in_proj_z = need_linear(weights, l + "in_proj_z", c.fp8_weight_block, device),
                .in_proj_b = need_linear(weights, l + "in_proj_b", c.fp8_weight_block, device),
                .in_proj_a = need_linear(weights, l + "in_proj_a", c.fp8_weight_block, device),
                .conv = need_text(weights, l + "conv1d.weight"),
                .dt_bias = need_text(weights, l + "dt_bias").to(at::kFloat),
                .neg_a = -need_text(weights, l + "A_log").to(at::kFloat).exp(),
                .norm = need_text(weights, l + "norm.weight"),
                .out_proj = need_linear(weights, l + "out_proj", c.fp8_weight_block, device),
            };
            kv_shards.push_back(
                kv_layer_shard{
                    .full =
                        kv_layer{
                            .n_kv_heads = 0,
                            .head_dim = 0,
                            .state =
                                recurrent_state{
                                    .sizes = {c.linear_n_value_heads, c.linear_key_head_dim, c.linear_value_head_dim},
                                    .dtype = at::kFloat,
                                },
                        },
                    .rows = {row_range(rank * linear_n_value_heads, linear_n_value_heads)},
                }
            );
            kv_shards.push_back(
                kv_layer_shard{
                    .full =
                        kv_layer{
                            .n_kv_heads = 0,
                            .head_dim = 0,
                            .state =
                                recurrent_state{
                                    .sizes =
                                        {2 * full_key_dim + c.linear_n_value_heads * c.linear_value_head_dim,
                                         c.linear_conv_kernel - 1},
                                    .dtype = embed_tokens.scalar_type(),
                                },
                        },
                    .rows = {
                        row_range(rank * key_dim, key_dim),
                        row_range(full_key_dim + rank * key_dim, key_dim),
                        row_range(2 * full_key_dim + rank * value_dim, value_dim),
                    },
                }
            );
        }
        layers.push_back(std::move(w));
    }
}

bool qwen3_5_model::full_attention(std::int64_t layer) const {
    return config.layer_types[layer] == qwen3_5_layer_type::full_attention;
}

// The full layers, cut along the line's first dimension to this rank's rows; an attention layer's query heads are
// this rank's.
std::vector<kv_layer> qwen3_5_model::kv_layers() const {
    std::vector<kv_layer> layers;
    for (const kv_layer_shard& shard : kv_shards) {
        kv_layer layer = shard.full;
        std::size_t rows = 0;
        for (auto [first, last] : shard.rows) {
            rows += last - first;
        }
        if (layer.state) {
            layer.state->sizes.front() = static_cast<std::int64_t>(rows);
        } else {
            layer.n_kv_heads = rows;
            layer.n_q_heads = static_cast<std::size_t>(n_heads);
        }
        layers.push_back(std::move(layer));
    }
    return layers;
}

std::vector<kv_layer_shard> qwen3_5_model::kv_layer_shards() const {
    return kv_shards;
}

at::Tensor qwen3_5_model::attention(
    const at::Tensor& h,
    std::int64_t layer,
    const group_batch& g,
    std::span<const kv> lines
) const {
    const attention_weights& w = *layers[layer].attention;
    const qwen3_5_config& c = config;
    const std::int64_t ntok = h.size(0);
    const std::size_t kv_index = kv_layer_of[layer];

    at::Tensor query_gate = w.q_proj.apply(h).view({ntok, n_heads, 2 * c.head_dim});
    at::Tensor q = rmsnorm_cuda(query_gate.narrow(-1, 0, c.head_dim), w.q_norm, c.rms_norm_eps);
    at::Tensor gate = query_gate.narrow(-1, c.head_dim, c.head_dim).reshape({ntok, n_heads * c.head_dim});
    q = rope_q(q.view({ntok, n_heads * c.head_dim}), g.pos, n_heads, c.head_dim, rope);
    at::Tensor k = rmsnorm_cuda(w.k_proj.apply(h).view({ntok, n_kv_heads, c.head_dim}), w.k_norm, c.rms_norm_eps);
    at::Tensor v = w.v_proj.apply(h).view({ntok, n_kv_heads, c.head_dim}).contiguous();
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
    at::Tensor out = w.o_proj.apply(o.reshape({ntok, n_heads * c.head_dim}) * at::sigmoid(gate));
    comm.all_reduce(out);
    return out;
}

at::Tensor qwen3_5_model::deltanet(
    const at::Tensor& h,
    std::int64_t layer,
    const group_batch& g,
    std::span<const kv> lines
) const {
    const deltanet_weights& w = *layers[layer].deltanet;
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
        linear_n_key_heads
    );
    at::Tensor z = w.in_proj_z.apply(h).view(core.sizes());
    at::Tensor gated = gated_rmsnorm_cuda(core, z, w.norm, c.rms_norm_eps, gate_activation::silu);
    at::Tensor out = w.out_proj.apply(gated.view({ntok, linear_n_value_heads * c.linear_value_head_dim}));
    comm.all_reduce(out);
    return out;
}

// A token outside this rank's vocabulary rows looks up row 0 and contributes zeros to the sum over the ranks.
at::Tensor qwen3_5_model::embed(const at::Tensor& tokens) const {
    const std::int64_t rows = embed_tokens.size(0);
    const at::Tensor local = tokens - static_cast<std::int64_t>(comm.rank()) * rows;
    const at::Tensor outside = (local < 0).logical_or(local >= rows);
    at::Tensor x = embed_tokens.index_select(0, local.masked_fill(outside, 0)).masked_fill(outside.unsqueeze(1), 0.0);
    comm.all_reduce(x);
    return x;
}

at::Tensor qwen3_5_model::group_forward(const group_batch& g, std::span<const kv> lines) const {
    const double eps = config.rms_norm_eps;
    at::Tensor x = embed(g.tokens);
    for (std::int64_t layer = 0; layer < n_layers; layer++) {
        const layer_weights& w = layers[layer];
        at::Tensor h = rmsnorm_cuda(x, w.input_layernorm, eps);
        x = x + (full_attention(layer) ? attention(h, layer, g, lines) : deltanet(h, layer, g, lines));
        at::Tensor h2 = rmsnorm_cuda(x, w.post_attention_layernorm, eps);
        at::Tensor mlp = w.down_proj.apply(at::silu(w.gate_proj.apply(h2)) * w.up_proj.apply(h2));
        comm.all_reduce(mlp);
        x = x + mlp;
    }
    const at::Tensor rank_logits = at::matmul(rmsnorm_cuda(logit_inputs(x, g), norm, eps), lm_head.t()).contiguous();
    const std::int64_t rows = rank_logits.size(0);
    return comm.all_gather(rank_logits)
        .view({static_cast<std::int64_t>(comm.size()), rows, rank_logits.size(1)})
        .transpose(0, 1)
        .reshape({rows, -1})
        .narrow(1, 0, config.vocab_size);
}

step_logits qwen3_5_model::forward(const step_batch& batch, std::span<const kv> lines) {
    return paged_forward(batch, [&](const group_batch& g) { return group_forward(g, lines); });
}

}  // namespace pulsarlm
