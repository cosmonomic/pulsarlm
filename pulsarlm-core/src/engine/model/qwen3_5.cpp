#include "pulsarlm/engine/model/qwen3_5.hpp"

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

constexpr std::string_view model_name = "qwen3_5_model";

bool quantized(const c10::Dict<std::string, at::Tensor>& weights, const std::string& name) {
    return weights.contains(name + ".weight_scale_inv");
}

const fp8_block& block_of(const std::optional<fp8_block>& block, const std::string& name) {
    if (!block) {
        throw std::invalid_argument(std::format("qwen3_5_model: {:?} has fp8 scales but no fp8_weight_block", name));
    }
    return *block;
}

// name is a linear's full prefix, such as model.language_model.layers.0.mlp.gate_proj. An fp8 linear may be on the host;
// it is copied to device only for its repack.
linear need_linear(
    const c10::Dict<std::string, at::Tensor>& weights,
    const std::string& name,
    const std::optional<fp8_block>& block,
    at::Device device
) {
    at::Tensor weight = required_weight(weights, model_name, name + ".weight");
    if (!quantized(weights, name)) {
        return linear::dense(std::move(weight));
    }
    const fp8_block& b = block_of(block, name);
    return linear::fp8(
        weight.to(device),
        required_weight(weights, model_name, name + ".weight_scale_inv").to(device),
        b.n,
        b.k
    );
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
    const at::Tensor weight = required_weight(weights, model_name, name + ".weight").to(device);
    if (!quantized(weights, name)) {
        return linear::dense(permute(weight));
    }
    const fp8_block& b = block_of(block, name);
    const at::Tensor row_scale = required_weight(weights, model_name, name + ".weight_scale_inv")
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
  embed_tokens(required_weight(weights, model_name, prefix + "embed_tokens.weight")),
  norm(zero_centered(required_weight(weights, model_name, prefix + "norm.weight"))) {
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

    // p is the layer's full weight prefix. The MTP layer, a full-attention layer, keeps a hidden row per line and
    // holds the pages the other attention layers hold.
    auto load_layer = [&](const std::string& p, bool attends, bool drafts) {
        layer_weights w{
            .input_layernorm = zero_centered(required_weight(weights, model_name, p + "input_layernorm.weight")),
            .post_attention_layernorm = zero_centered(
                required_weight(weights, model_name, p + "post_attention_layernorm.weight")
            ),
            .gate_proj = need_linear(weights, p + "mlp.gate_proj", c.fp8_weight_block, device),
            .up_proj = need_linear(weights, p + "mlp.up_proj", c.fp8_weight_block, device),
            .down_proj = need_linear(weights, p + "mlp.down_proj", c.fp8_weight_block, device),
        };
        if (attends) {
            const std::string a = p + "self_attn.";
            check_size(
                required_weight(weights, model_name, a + "q_proj.weight"),
                0,
                n_heads * 2 * c.head_dim,
                a + "q_proj.weight"
            );
            check_size(
                required_weight(weights, model_name, a + "k_proj.weight"),
                0,
                n_kv_heads * c.head_dim,
                a + "k_proj.weight"
            );
            check_size(
                required_weight(weights, model_name, a + "v_proj.weight"),
                0,
                n_kv_heads * c.head_dim,
                a + "v_proj.weight"
            );
            check_size(
                required_weight(weights, model_name, a + "o_proj.weight"),
                1,
                n_heads * c.head_dim,
                a + "o_proj.weight"
            );
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
                .q_norm = zero_centered(required_weight(weights, model_name, a + "q_norm.weight"))
                              .index_select(0, permutation),
                .k_norm = zero_centered(required_weight(weights, model_name, a + "k_norm.weight"))
                              .index_select(0, permutation),
            };
            w.kv_layer = kv_shards.size();
            kv_shards.push_back(
                kv_layer_shard{
                    .full =
                        kv_layer{
                            .n_kv_heads = static_cast<std::size_t>(c.n_kv_heads),
                            .head_dim = static_cast<std::size_t>(c.head_dim),
                            .n_q_heads = static_cast<std::size_t>(c.n_heads),
                            .last_hidden_dim = drafts ? static_cast<std::size_t>(c.hidden_size) : 0,
                            .mirrors_held_pages = drafts,
                        },
                    .rows = {row_range(first_kv_head, n_kv_heads)},
                }
            );
        } else {
            const std::string l = p + "linear_attn.";
            check_size(
                required_weight(weights, model_name, l + "in_proj_qkv.weight"),
                0,
                conv_channels,
                l + "in_proj_qkv.weight"
            );
            check_size(
                required_weight(weights, model_name, l + "in_proj_z.weight"),
                0,
                value_dim,
                l + "in_proj_z.weight"
            );
            check_size(
                required_weight(weights, model_name, l + "in_proj_b.weight"),
                0,
                linear_n_value_heads,
                l + "in_proj_b.weight"
            );
            check_size(
                required_weight(weights, model_name, l + "in_proj_a.weight"),
                0,
                linear_n_value_heads,
                l + "in_proj_a.weight"
            );
            check_size(
                required_weight(weights, model_name, l + "conv1d.weight"),
                0,
                conv_channels,
                l + "conv1d.weight"
            );
            check_size(required_weight(weights, model_name, l + "dt_bias"), 0, linear_n_value_heads, l + "dt_bias");
            check_size(required_weight(weights, model_name, l + "A_log"), 0, linear_n_value_heads, l + "A_log");
            check_size(
                required_weight(weights, model_name, l + "out_proj.weight"),
                1,
                value_dim,
                l + "out_proj.weight"
            );
            w.deltanet = deltanet_weights{
                .in_proj_qkv = need_linear(weights, l + "in_proj_qkv", c.fp8_weight_block, device),
                .in_proj_z = need_linear(weights, l + "in_proj_z", c.fp8_weight_block, device),
                .in_proj_b = need_linear(weights, l + "in_proj_b", c.fp8_weight_block, device),
                .in_proj_a = need_linear(weights, l + "in_proj_a", c.fp8_weight_block, device),
                .conv = required_weight(weights, model_name, l + "conv1d.weight"),
                .dt_bias = required_weight(weights, model_name, l + "dt_bias").to(at::kFloat),
                .neg_a = -required_weight(weights, model_name, l + "A_log").to(at::kFloat).exp(),
                .norm = required_weight(weights, model_name, l + "norm.weight"),
                .out_proj = need_linear(weights, l + "out_proj", c.fp8_weight_block, device),
            };
            w.state_kv_layer = kv_shards.size();
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
            w.conv_kv_layer = kv_shards.size();
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
                                    .input_window = true,
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
        return w;
    };
    for (std::int64_t layer = 0; layer < n_layers; layer++) {
        layers.push_back(load_layer(prefix + "layers." + std::to_string(layer) + ".", full_attention(layer), false));
    }
    if (weights.contains("mtp.fc.weight")) {
        at::Tensor pre_fc_norm_embedding = zero_centered(
            required_weight(weights, model_name, "mtp.pre_fc_norm_embedding.weight")
        );
        at::Tensor pre_fc_norm_hidden = zero_centered(
            required_weight(weights, model_name, "mtp.pre_fc_norm_hidden.weight")
        );
        at::Tensor fc = required_weight(weights, model_name, "mtp.fc.weight");
        at::Tensor mtp_norm = zero_centered(required_weight(weights, model_name, "mtp.norm.weight"));
        check_size(fc, 1, 2 * c.hidden_size, "mtp.fc.weight");
        mtp = mtp_weights{
            .pre_fc_norm_embedding = std::move(pre_fc_norm_embedding),
            .pre_fc_norm_hidden = std::move(pre_fc_norm_hidden),
            .fc = std::move(fc),
            .norm = std::move(mtp_norm),
            .layer = load_layer("mtp.layers.0.", true, true),
        };
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
    const layer_weights& layer,
    const batch& g,
    std::span<const kv> lines
) const {
    const attention_weights& w = *layer.attention;
    const qwen3_5_config& c = config;
    const std::int64_t ntok = h.size(0);
    const std::size_t kv_index = layer.kv_layer;

    at::Tensor query_gate = w.q_proj.apply(h).view({ntok, n_heads, 2 * c.head_dim});
    at::Tensor q = rmsnorm_cuda(query_gate.narrow(-1, 0, c.head_dim), w.q_norm, c.rms_norm_eps);
    at::Tensor gate = query_gate.narrow(-1, c.head_dim, c.head_dim).reshape({ntok, n_heads * c.head_dim});
    q = rope_q(q.view({ntok, n_heads * c.head_dim}), g.pos, n_heads, c.head_dim, rope);
    at::Tensor k = rmsnorm_cuda(w.k_proj.apply(h).view({ntok, n_kv_heads, c.head_dim}), w.k_norm, c.rms_norm_eps);
    at::Tensor v = w.v_proj.apply(h).view({ntok, n_kv_heads, c.head_dim}).contiguous();
    const double scale = 1.0 / std::sqrt(static_cast<double>(c.head_dim));
    at::Tensor o = paged_attention(q, k, v, lines[kv_index], kv_index, 0, g, scale, rope);
    at::Tensor out = w.o_proj.apply(o.reshape({ntok, n_heads * c.head_dim}) * at::sigmoid(gate));
    comm.all_reduce(out);
    return out;
}

at::Tensor qwen3_5_model::deltanet(
    const at::Tensor& h,
    const layer_weights& layer,
    const batch& g,
    std::span<const kv> lines
) const {
    const deltanet_weights& w = *layer.deltanet;
    at::Tensor gated = gated_deltanet(
        g,
        gated_deltanet_inputs{
            .qkv = w.in_proj_qkv.apply(h),
            .a = w.in_proj_a.apply(h),
            .b = w.in_proj_b.apply(h),
            .z = w.in_proj_z.apply(h),
        },
        gated_deltanet_weights{.conv = w.conv, .dt_bias = w.dt_bias, .neg_a = w.neg_a, .norm = w.norm},
        lines[layer.state_kv_layer],
        lines[layer.conv_kv_layer],
        layer.state_kv_layer,
        layer.conv_kv_layer,
        linear_n_key_heads,
        config.rms_norm_eps,
        gate_activation::silu
    );
    at::Tensor out = w.out_proj.apply(gated);
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

at::Tensor qwen3_5_model::decoder_layer(
    const at::Tensor& x,
    const layer_weights& w,
    const batch& g,
    std::span<const kv> lines
) const {
    const double eps = config.rms_norm_eps;
    at::Tensor h = rmsnorm_cuda(x, w.input_layernorm, eps);
    at::Tensor mixed = x + (w.attention ? attention(h, w, g, lines) : deltanet(h, w, g, lines));
    at::Tensor h2 = rmsnorm_cuda(mixed, w.post_attention_layernorm, eps);
    at::Tensor mlp = w.down_proj.apply(at::silu(w.gate_proj.apply(h2)) * w.up_proj.apply(h2));
    comm.all_reduce(mlp);
    return mixed + mlp;
}

at::Tensor qwen3_5_model::logits(const at::Tensor& rows) const {
    const at::Tensor rank_logits = at::matmul(rows, lm_head.t()).contiguous();
    const std::int64_t row_count = rank_logits.size(0);
    return comm.all_gather(rank_logits)
        .view({static_cast<std::int64_t>(comm.size()), row_count, rank_logits.size(1)})
        .transpose(0, 1)
        .reshape({row_count, -1})
        .narrow(1, 0, config.vocab_size);
}

// Normed row by row, the logit rows' hidden rows equal the normed hidden rows' logit rows.
model::output qwen3_5_model::forward(const batch& b, std::span<const kv> lines) {
    check_batch(b);
    assert(mtp || b.kind != batch_kind::verify);
    at::Tensor x = embed(b.tokens);
    for (std::int64_t layer = 0; layer < n_layers; layer++) {
        x = decoder_layer(x, layers[layer], b, lines);
    }
    if (!mtp) {
        return output{.logits = logits(rmsnorm_cuda(logit_inputs(x, b), norm, config.rms_norm_eps))};
    }
    at::Tensor normed = rmsnorm_cuda(x, norm, config.rms_norm_eps);
    keep_last_hidden(b, normed, lines[mtp->layer.kv_layer], mtp->layer.kv_layer);
    return output{.logits = logits(logit_inputs(normed, b)), .hidden = normed};
}

std::int64_t qwen3_5_model::arrival_counter_count(std::int64_t sequences, std::int64_t) const {
    return gated_delta_rule_counters(sequences, linear_n_value_heads);
}

draft_head* qwen3_5_model::drafter() {
    return mtp ? this : nullptr;
}

std::int64_t qwen3_5_model::hidden_dim() const {
    return config.hidden_size;
}

// The token's embedding and the hidden row, each normed, go through fc into the MTP layer, whose output is normed.
model::output qwen3_5_model::draft(const batch& b, const at::Tensor& hidden, std::span<const kv> lines) {
    const mtp_weights& m = *mtp;
    const double eps = config.rms_norm_eps;
    at::Tensor embedded = rmsnorm_cuda(embed(b.tokens), m.pre_fc_norm_embedding, eps);
    at::Tensor previous = rmsnorm_cuda(hidden.contiguous(), m.pre_fc_norm_hidden, eps);
    at::Tensor x = at::linear(at::cat({embedded, previous}, -1), m.fc);
    x = decoder_layer(x, m.layer, b, lines);
    at::Tensor normed = rmsnorm_cuda(x, m.norm, eps);
    return output{.logits = logits(logit_inputs(normed, b)), .hidden = normed};
}

}  // namespace pulsarlm
