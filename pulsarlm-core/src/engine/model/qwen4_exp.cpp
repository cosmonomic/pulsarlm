#include "pulsarlm/engine/model/qwen4_exp.hpp"

#include "pulsarlm/engine/model/paged_forward.hpp"
#include "pulsarlm/engine/model/partial_rope.hpp"
#include "pulsarlm/kernels/ops.hpp"

#include <ATen/ATen.h>

#include <algorithm>
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

// The weight of the checkpoint's full name.
at::Tensor need_named(const c10::Dict<std::string, at::Tensor>& weights, const std::string& name) {
    if (!weights.contains(name)) {
        throw std::invalid_argument(std::format("qwen4_exp_model: missing weight {:?}", name));
    }
    return weights.at(name);
}

at::Tensor need(const c10::Dict<std::string, at::Tensor>& weights, const std::string& name) {
    return need_named(weights, prefix + name);
}

// Hugging Face's Qwen4-Exp RMSNorm scales by 1 + weight, in fp32.
at::Tensor zero_centered(const at::Tensor& weight) {
    return weight.to(at::kFloat) + 1.0;
}

// The shards of the table whose weights' full names start with p, concatenated in order.
at::Tensor ngram_table(const c10::Dict<std::string, at::Tensor>& weights, const std::string& p) {
    std::vector<at::Tensor> shards;
    while (weights.contains(std::format("{}shard_{}.weight", p, shards.size()))) {
        shards.push_back(weights.at(std::format("{}shard_{}.weight", p, shards.size())));
    }
    if (shards.empty()) {
        throw std::invalid_argument(std::format("qwen4_exp_model: missing weight {:?}", p + "shard_0.weight"));
    }
    return at::cat(shards);
}

std::int64_t split(std::int64_t count, std::int64_t parts, const char* what) {
    if (count % parts != 0) {
        throw std::invalid_argument(
            std::format("qwen4_exp_model: {} {} do not split across {} GPUs", count, what, parts)
        );
    }
    return count / parts;
}

}  // namespace

qwen4_exp_model::qwen4_exp_model(
    qwen4_exp_config config,
    const c10::Dict<std::string, at::Tensor>& weights,
    std::int64_t draft_tokens,
    communicator comm
)
: config(std::move(config)),
  draft_tokens(draft_tokens),
  comm(std::move(comm)),
  n_layers(static_cast<std::int64_t>(this->config.layer_types.size())),
  embed_tokens(need(weights, "embed_tokens.weight")) {
    const qwen4_exp_config& c = this->config;
    const auto gpus = static_cast<std::int64_t>(this->comm.size());
    const auto rank = static_cast<std::int64_t>(this->comm.rank());
    if (c.n_kv_heads % gpus != 0 && gpus % c.n_kv_heads != 0) {
        throw std::invalid_argument(
            std::format("qwen4_exp_model: {} KV heads do not split across or repeat over {} GPUs", c.n_kv_heads, gpus)
        );
    }
    heads = shard_heads{
        .n_heads = split(c.n_heads, gpus, "query heads"),
        .n_kv_heads = std::max<std::int64_t>(1, c.n_kv_heads / gpus),
        .linear_n_key_heads = split(c.linear_n_key_heads, gpus, "Gated DeltaNet key heads"),
        .linear_n_value_heads = split(c.linear_n_value_heads, gpus, "Gated DeltaNet value heads"),
    };
    first_vocab_row = rank * embed_tokens.size(0);
    if (embed_tokens.size(0) * gpus < c.vocab_size) {
        throw std::invalid_argument("qwen4_exp_model: the GPUs' embedding rows do not cover the vocabulary");
    }
    const auto rotary_dim = static_cast<std::int64_t>(std::llround(c.partial_rotary_factor * c.head_dim));
    if (rotary_dim % 2 != 0 || rotary_dim > c.head_dim || c.head_dim % 2 != 0) {
        throw std::invalid_argument("qwen4_exp_model: the rotated dims must be an even part of an even head dim");
    }
    if (c.linear_n_value_heads % c.linear_n_key_heads != 0) {
        throw std::invalid_argument("qwen4_exp_model: the value heads must be a multiple of the key heads");
    }
    if (c.hc_count < 2 || c.ngram_size < 2) {
        throw std::invalid_argument("qwen4_exp_model: hc_count and ngram_size must be at least 2");
    }
    if (c.indexer_budget % c.indexer_compress_ratio != 0 || rotary_dim > c.indexer_head_dim) {
        throw std::invalid_argument(
            "qwen4_exp_model: the indexer budget must be whole blocks and the rotated dims fit an index head"
        );
    }
    for (std::int64_t layer : c.ple_layers) {
        if (layer < 0 || layer >= n_layers || c.layer_types[layer] != qwen4_exp_layer_type::linear_attention) {
            throw std::invalid_argument("qwen4_exp_model: a PLE layer must be a linear attention layer");
        }
    }
    rope = rope_parameters{
        .theta = std::pow(c.rope_theta, static_cast<double>(c.head_dim) / static_cast<double>(rotary_dim)),
        .partial_rotary_factor = static_cast<double>(rotary_dim) / static_cast<double>(c.head_dim),
    };
    index_inv_freq = 1.0 /
        at::pow(c.rope_theta, at::arange(0, rotary_dim, 2, at::kLong).to(at::kFloat) / static_cast<double>(rotary_dim))
            .to(embed_tokens.device());
    lm_head = weights.contains("lm_head.weight") ? weights.at("lm_head.weight") : embed_tokens;
    at::Tensor permutation = rope_permutation(c.head_dim, rotary_dim, embed_tokens.device());
    const std::int64_t key_channels = heads.linear_n_key_heads * c.linear_key_head_dim;
    const std::int64_t value_channels = heads.linear_n_value_heads * c.linear_value_head_dim;
    const std::int64_t full_key_channels = c.linear_n_key_heads * c.linear_key_head_dim;
    const std::int64_t full_value_channels = c.linear_n_value_heads * c.linear_value_head_dim;
    const std::int64_t first_kv_head = gpus <= c.n_kv_heads ? rank * heads.n_kv_heads : rank / (gpus / c.n_kv_heads);
    auto rows = [](std::int64_t first, std::int64_t count) {
        return std::pair{static_cast<std::size_t>(first), static_cast<std::size_t>(first + count)};
    };
    auto hyper_connection = [&](const std::string& p) {
        return hyper_connection_weights{
            .hc_norm = zero_centered(need_named(weights, p + "hc_norm.weight")),
            .down_and_inject = at::cat(
                {need_named(weights, p + "input_mix_weight_down.weight"),
                 need_named(weights, p + "block_inject_weight.weight")}
            ),
            .input_mix_weight_up = need_named(weights, p + "input_mix_weight_up.weight"),
        };
    };
    auto mixer_named = [&](const std::string& p) {
        return mixer_weights{
            .hc_norm = zero_centered(need_named(weights, p + "hc_norm.weight")),
            .input_mix_weight_down = need_named(weights, p + "input_mix_weight_down.weight"),
            .input_mix_weight_up = need_named(weights, p + "input_mix_weight_up.weight"),
        };
    };
    hyper_connection_mixer = mixer_named(prefix + "hyper_connection_mixer.");
    auto recurrent = [](std::vector<std::int64_t> sizes, at::ScalarType dtype, std::int64_t slots = 1) {
        return kv_layer{
            .n_kv_heads = 0,
            .head_dim = 0,
            .state = recurrent_state{
                .sizes = std::move(sizes),
                .dtype = dtype,
                .slots = static_cast<std::size_t>(slots)
            }
        };
    };
    auto add_kv_layer = [&](kv_layer layer, kv_layer full, std::vector<std::pair<std::size_t, std::size_t>> held) {
        kv_specs.push_back(std::move(layer));
        kv_shards.push_back(kv_layer_shard{.full = std::move(full), .rows = std::move(held)});
    };
    auto whole = [&](std::vector<std::int64_t> sizes, at::ScalarType dtype) {
        kv_layer layer = recurrent(std::move(sizes), dtype);
        std::int64_t first_size = layer.state->sizes.front();
        add_kv_layer(layer, layer, {rows(0, first_size)});
    };

    // p is the layer's weight prefix; layer n_layers is the MTP layer, an indexed attention layer without PLE.
    auto load_layer = [&](const std::string& p, std::int64_t layer) {
        layer_weights w{
            .attn_hyper_connection = hyper_connection(p + "attn_hyper_connection."),
            .mlp_hyper_connection = hyper_connection(p + "mlp_hyper_connection."),
            .moe = moe_weights{
                .router_and_gate = at::cat(
                    {need_named(weights, p + "mlp.gate.weight"),
                     need_named(weights, p + "mlp.shared_expert_gate.weight")}
                ),
                .experts_gate_up_proj = need_named(weights, p + "mlp.experts.gate_up_proj"),
                .experts_down_proj = need_named(weights, p + "mlp.experts.down_proj"),
                .shared_gate_up_proj = at::cat(
                    {need_named(weights, p + "mlp.shared_expert.gate_proj.weight"),
                     need_named(weights, p + "mlp.shared_expert.up_proj.weight")}
                ),
                .shared_down_proj = need_named(weights, p + "mlp.shared_expert.down_proj.weight"),
            },
        };
        const std::int64_t experts = w.moe.router_and_gate.size(0) - 1;
        if (experts < c.num_experts_per_tok || c.num_experts_per_tok < 1) {
            throw std::invalid_argument("qwen4_exp_model: num_experts_per_tok must be between 1 and the experts");
        }
        const std::int64_t held_experts = w.moe.experts_gate_up_proj.size(0);
        if (held_experts == experts) {
            w.moe.first_expert = 0;
        } else if (held_experts * gpus == experts) {
            w.moe.first_expert = rank * held_experts;
        } else {
            throw std::invalid_argument(
                std::format("qwen4_exp_model: {} experts held of {} over {} GPUs", held_experts, experts, gpus)
            );
        }
        kv_layer_of.push_back(kv_specs.size());
        if (indexed_attention(layer)) {
            w.attention = attention_weights{
                .projections = at::cat(
                    {permute_queries(need_named(weights, p + "self_attn.q_proj.weight"), heads.n_heads, permutation),
                     permute_heads(need_named(weights, p + "self_attn.k_proj.weight"), heads.n_kv_heads, permutation),
                     need_named(weights, p + "self_attn.v_proj.weight"),
                     need_named(weights, p + "self_attn.indexer.index_qk_proj.weight")}
                ),
                .o_proj = need_named(weights, p + "self_attn.o_proj.weight"),
                .q_norm = zero_centered(need_named(weights, p + "self_attn.q_norm.weight"))
                              .index_select(0, permutation),
                .k_norm = zero_centered(need_named(weights, p + "self_attn.k_norm.weight"))
                              .index_select(0, permutation),
                .index_q_norm = zero_centered(need_named(weights, p + "self_attn.indexer.q_layernorm.weight")),
                .index_k_norm = zero_centered(need_named(weights, p + "self_attn.indexer.k_layernorm.weight")),
            };
            auto attention_layer = [&](std::int64_t n_kv_heads, std::int64_t n_q_heads) {
                return kv_layer{
                    .n_kv_heads = static_cast<std::size_t>(n_kv_heads),
                    .head_dim = static_cast<std::size_t>(c.head_dim),
                    .n_q_heads = static_cast<std::size_t>(n_q_heads),
                    .index_head_dim = static_cast<std::size_t>(c.indexer_head_dim),
                };
            };
            add_kv_layer(
                attention_layer(heads.n_kv_heads, heads.n_heads),
                attention_layer(c.n_kv_heads, c.n_heads),
                {rows(first_kv_head, heads.n_kv_heads)}
            );
            if (layer == n_layers) {
                for (kv_layer* drafting : {&kv_specs.back(), &kv_shards.back().full}) {
                    drafting->mirrors_held_pages = true;
                    drafting->last_hidden_dim = static_cast<std::size_t>(c.hc_count * c.hidden_size);
                }
            }
        } else {
            w.deltanet = deltanet_weights{
                .in_proj = at::cat(
                    {need_named(weights, p + "linear_attn.in_proj_qkv.weight"),
                     need_named(weights, p + "linear_attn.in_proj_a.weight"),
                     need_named(weights, p + "linear_attn.in_proj_b.weight"),
                     need_named(weights, p + "linear_attn.in_proj_z.weight")}
                ),
                .conv = need_named(weights, p + "linear_attn.conv1d.weight"),
                .dt_bias = need_named(weights, p + "linear_attn.dt_bias").to(at::kFloat),
                .neg_a = -need_named(weights, p + "linear_attn.A_log").to(at::kFloat).exp(),
                .norm = need_named(weights, p + "linear_attn.norm.weight"),
                .out_proj = need_named(weights, p + "linear_attn.out_proj.weight"),
            };
            add_kv_layer(
                recurrent(
                    {heads.linear_n_value_heads, c.linear_key_head_dim, c.linear_value_head_dim},
                    at::kFloat,
                    draft_tokens + 1
                ),
                recurrent(
                    {c.linear_n_value_heads, c.linear_key_head_dim, c.linear_value_head_dim},
                    at::kFloat,
                    draft_tokens + 1
                ),
                {rows(rank * heads.linear_n_value_heads, heads.linear_n_value_heads)}
            );
            add_kv_layer(
                recurrent(
                    {2 * key_channels + value_channels, c.linear_conv_kernel - 1 + draft_tokens},
                    embed_tokens.scalar_type()
                ),
                recurrent(
                    {2 * full_key_channels + full_value_channels, c.linear_conv_kernel - 1 + draft_tokens},
                    embed_tokens.scalar_type()
                ),
                {rows(rank * key_channels, key_channels),
                 rows(full_key_channels + rank * key_channels, key_channels),
                 rows(2 * full_key_channels + rank * value_channels, value_channels)}
            );
        }
        if (std::ranges::contains(c.ple_layers, layer)) {
            const std::string e = p + "ple.ple_embedding.";
            at::Tensor table = ngram_table(weights, e + "ngram_embedding.");
            w.ple = ple_weights{
                .key_proj = need_named(weights, p + "ple.key_proj.weight"),
                .value_proj = need_named(weights, p + "ple.value_proj.weight"),
                .norm_key = zero_centered(need_named(weights, p + "ple.norm_key.weight")),
                .norm_query = zero_centered(need_named(weights, p + "ple.norm_query.weight")),
                .norm_conv = zero_centered(need_named(weights, p + "ple.norm_conv.weight")),
                .conv = need_named(weights, p + "ple.conv1d.weight"),
                .layer_multipliers = need_named(weights, e + "layer_multipliers").to(at::kLong),
                .head_vocab_sizes = need_named(weights, e + "ngram_heads_vocab_sizes").to(at::kLong),
                .head_offsets = need_named(weights, e + "ngram_heads_offsets").to(at::kLong),
                .ngram_embedding = table,
                .first_row = rank * table.size(0),
            };
            if ((w.ple->head_offsets + w.ple->head_vocab_sizes).max().item<std::int64_t>() > table.size(0) * gpus) {
                throw std::invalid_argument("qwen4_exp_model: the GPUs' n-gram table rows do not cover its heads");
            }
            if (w.ple->layer_multipliers.size(0) != c.ngram_size ||
                w.ple->head_vocab_sizes.size(0) % (c.ngram_size - 1) != 0) {
                throw std::invalid_argument("qwen4_exp_model: the n-gram buffers must match ngram_size");
            }
            const std::int64_t history = (w.ple->conv.size(2) - 1) * c.ngram_size;
            whole({c.hc_count * c.hidden_size, history + draft_tokens}, embed_tokens.scalar_type());
            whole({c.ngram_size - 1 + draft_tokens}, at::kLong);
        }
        layers.push_back(std::move(w));
    };
    for (std::int64_t layer = 0; layer < n_layers; layer++) {
        load_layer(prefix + "layers." + std::to_string(layer) + ".", layer);
    }
    if (weights.contains("mtp.fc_embedding.weight")) {
        load_layer("mtp.layers.0.", n_layers);
        mtp = mtp_weights{
            .pre_fc_norm_embedding = zero_centered(need_named(weights, "mtp.pre_fc_norm_embedding.weight")),
            .pre_fc_norm_hidden = zero_centered(need_named(weights, "mtp.pre_fc_norm_hidden.weight")),
            .fc_embedding = need_named(weights, "mtp.fc_embedding.weight"),
            .fc_hidden = need_named(weights, "mtp.fc_hidden.weight"),
            .mixer = mixer_named("mtp.hyper_connection_mixer."),
        };
    }
}

std::vector<kv_layer> qwen4_exp_model::kv_layers() const {
    return kv_specs;
}

std::vector<kv_layer_shard> qwen4_exp_model::kv_layer_shards() const {
    return kv_shards;
}

at::Tensor qwen4_exp_model::lookup(const at::Tensor& table, std::int64_t first_row, const at::Tensor& ids) const {
    at::Tensor rows = ids - first_row;
    at::Tensor held = rows.ge(0).logical_and(rows.lt(table.size(0))).unsqueeze(-1);
    std::vector<std::int64_t> sizes = ids.sizes().vec();
    sizes.push_back(table.size(1));
    at::Tensor found = table.index_select(0, rows.clamp(0, table.size(0) - 1).flatten()).view(sizes);
    at::Tensor summed = at::where(held, found, at::zeros({}, found.options()));
    comm.all_reduce(summed);
    return summed;
}

at::Tensor qwen4_exp_model::logits(const at::Tensor& x) const {
    at::Tensor part = at::matmul(x, lm_head.t()).contiguous();
    const auto gpus = static_cast<std::int64_t>(comm.size());
    return comm.all_gather(part)
        .view({gpus, part.size(0), part.size(1)})
        .permute({1, 0, 2})
        .reshape({part.size(0), gpus * part.size(1)})
        .narrow(1, 0, config.vocab_size);
}

bool qwen4_exp_model::indexed_attention(std::int64_t layer) const {
    return layer == n_layers || config.layer_types[layer] == qwen4_exp_layer_type::indexed_attention;
}

// Hugging Face takes a query's top blocks with torch.topk, whose order among equal scores is unspecified; here the lower
// block wins.
at::Tensor qwen4_exp_model::selected_blocks(
    const attention_weights& w,
    const at::Tensor& queries,
    const kv& lines,
    const at::Tensor& page_tables,
    const std::optional<at::Tensor>& cu_seqlens_q,
    const at::Tensor& key_counts,
    const at::Tensor& page_positions
) const {
    const qwen4_exp_config& c = config;
    const std::int64_t ratio = c.indexer_compress_ratio;
    const std::int64_t ntok = queries.size(0);
    const std::int64_t page_size = lines.index_keys.size(1);
    TORCH_CHECK(page_size % ratio == 0, "qwen4_exp_model: a page must hold whole index blocks");
    const std::int64_t max_blocks = page_tables.size(1) * page_size / ratio;
    const std::int64_t words = max_blocks / 32 + 1;

    at::Tensor blocks = index_blocks_cuda(
        lines.index_keys,
        page_tables,
        page_positions,
        key_counts,
        w.index_k_norm,
        index_inv_freq,
        ratio,
        c.rms_norm_eps
    );

    const std::int64_t chunk_rows = std::max<std::int64_t>(1, (std::int64_t{1} << 26) / max_blocks);
    std::vector<at::Tensor> selected;
    for (std::int64_t first = 0; first < ntok; first += chunk_rows) {
        const std::int64_t rows = std::min(chunk_rows, ntok - first);
        at::Tensor scores = index_scores_cuda(
            queries.narrow(0, first, rows).contiguous(),
            blocks,
            key_counts,
            cu_seqlens_q,
            first,
            ratio,
            1.0 / std::sqrt(static_cast<double>(c.indexer_head_dim))
        );
        selected.push_back(
            select_blocks_cuda(scores, key_counts, cu_seqlens_q, first, c.indexer_budget / ratio, ratio, words)
        );
    }
    return selected.size() == 1 ? selected.front() : at::cat(selected);
}

at::Tensor qwen4_exp_model::stream_norm(const at::Tensor& x, const at::Tensor& weight) const {
    return stream_rmsnorm_cuda(x.contiguous(), weight, x.size(1) / config.hidden_size, config.rms_norm_eps);
}

at::Tensor qwen4_exp_model::mixed_streams(const at::Tensor& normed, const mixer_weights& w) const {
    const auto hc = static_cast<double>(config.hc_count);
    at::Tensor mix_weights = at::silu(at::linear(normed, w.input_mix_weight_down) / hc);
    mix_weights = at::sigmoid(at::linear(mix_weights, w.input_mix_weight_up));
    at::Tensor streams = normed.view({normed.size(0), config.hc_count, config.hidden_size});
    return (mix_weights.view(streams.sizes()) * streams).mean(-2);
}

qwen4_exp_model::mixed_input qwen4_exp_model::mix(const at::Tensor& x, const hyper_connection_weights& w) const {
    auto [normed, projected] = stream_rmsnorm_linear_cuda(
        x.contiguous(),
        w.hc_norm,
        x.size(1) / config.hidden_size,
        config.rms_norm_eps,
        w.down_and_inject
    );
    return mix_normed(normed, projected, w);
}

std::pair<at::Tensor, qwen4_exp_model::mixed_input> qwen4_exp_model::injected_mix(
    const at::Tensor& x,
    const at::Tensor& out,
    const at::Tensor& injection_weights,
    const hyper_connection_weights& w
) const {
    auto [summed, normed, projected] = injected_stream_rmsnorm_linear_cuda(
        x.contiguous(),
        out.contiguous(),
        injection_weights,
        w.hc_norm,
        x.size(1) / config.hidden_size,
        config.rms_norm_eps,
        w.down_and_inject
    );
    return {summed, mix_normed(normed, projected, w)};
}

qwen4_exp_model::mixed_input qwen4_exp_model::mix_normed(
    const at::Tensor& normed,
    const at::Tensor& projected,
    const hyper_connection_weights& w
) const {
    const std::int64_t rank = w.input_mix_weight_up.size(1);
    auto [input, injection_weights] = hyper_connection_mix_cuda(
        normed,
        projected.narrow(1, 0, rank),
        w.input_mix_weight_up,
        projected.narrow(1, rank, config.hc_count)
    );
    return mixed_input{.input = input, .injection_weights = injection_weights};
}

at::Tensor qwen4_exp_model::final_mix(const at::Tensor& x) const {
    return mixed_streams(stream_norm(x, hyper_connection_mixer.hc_norm), hyper_connection_mixer);
}

at::Tensor qwen4_exp_model::attention(
    const at::Tensor& h,
    std::int64_t layer,
    const group_batch& g,
    std::span<const kv> lines
) const {
    const attention_weights& w = *layers[layer].attention;
    const qwen4_exp_config& c = config;
    const std::int64_t ntok = h.size(0);
    const std::size_t kv_index = kv_layer_of[layer];

    const std::int64_t query_rows = heads.n_heads * 2 * c.head_dim;
    const std::int64_t key_rows = heads.n_kv_heads * c.head_dim;
    at::Tensor projected = small_batch_linear_cuda(h, w.projections);
    at::Tensor query_gate = projected.narrow(1, 0, query_rows).view({ntok, heads.n_heads, 2 * c.head_dim});
    at::Tensor q = rmsnorm_cuda(query_gate.narrow(-1, 0, c.head_dim), w.q_norm, c.rms_norm_eps);
    at::Tensor gate = query_gate.narrow(-1, c.head_dim, c.head_dim);
    q = rope_q(q.view({ntok, heads.n_heads * c.head_dim}), g.pos, heads.n_heads, c.head_dim, rope);
    at::Tensor k = rmsnorm_cuda(
        projected.narrow(1, query_rows, key_rows).view({ntok, heads.n_kv_heads, c.head_dim}),
        w.k_norm,
        c.rms_norm_eps
    );
    at::Tensor v = projected.narrow(1, query_rows + key_rows, key_rows)
                       .view({ntok, heads.n_kv_heads, c.head_dim})
                       .contiguous();

    const kv& line = lines[kv_index];
    const auto layer_row = static_cast<std::int64_t>(kv_index);
    at::Tensor page_tables = g.page_tables.select(0, layer_row);
    at::Tensor page_positions = g.page_positions.select(0, layer_row);
    at::Tensor key_counts = g.key_counts.select(0, layer_row);
    std::optional<at::Tensor> cu_seqlens_q;
    if (g.is_prefill()) {
        cu_seqlens_q = g.cu_seqlens_q;
    }
    write_kv_cuda(line.k, line.v, k, v, page_tables, cu_seqlens_q, key_counts);

    at::Tensor index_qk = projected.narrow(1, query_rows + 2 * key_rows, projected.size(1) - query_rows - 2 * key_rows);
    at::Tensor index_queries = index_queries_cuda(
        index_qk,
        w.index_q_norm,
        index_inv_freq,
        g.pos,
        c.indexer_n_heads,
        c.rms_norm_eps
    );
    at::Tensor index_keys = index_qk.narrow(1, c.indexer_n_heads * c.indexer_head_dim, c.indexer_head_dim)
                                .contiguous()
                                .view({ntok, 1, -1});
    at::Tensor index_pool = line.index_keys.unsqueeze(2);
    write_kv_cuda(index_pool, index_pool, index_keys, index_keys, page_tables, cu_seqlens_q, key_counts);
    at::Tensor
        selected = selected_blocks(w, index_queries, line, page_tables, cu_seqlens_q, key_counts, page_positions);

    std::optional<at::Tensor> mass_sum, weight_sum, cu_view_pages, query_weights;
    if (!g.page_mass_sum.empty() && g.page_mass_sum[kv_index].defined()) {
        mass_sum = g.page_mass_sum[kv_index];
        weight_sum = g.page_weight_sum[kv_index];
        cu_view_pages = g.cu_view_pages.select(0, layer_row);
        query_weights = g.query_weights;
    }
    at::Tensor o = attn_indexed_cuda(
        q,
        line.k,
        line.v,
        page_tables,
        cu_seqlens_q,
        key_counts,
        page_positions,
        rope,
        1.0 / std::sqrt(static_cast<double>(c.head_dim)),
        selected,
        c.indexer_compress_ratio,
        c.indexer_budget / c.indexer_compress_ratio + 1,
        mass_sum,
        weight_sum,
        cu_view_pages,
        query_weights
    );
    at::Tensor out = small_batch_linear_cuda(
        sigmoid_gated_cuda(o.view({ntok, heads.n_heads, c.head_dim}), gate).view({ntok, heads.n_heads * c.head_dim}),
        w.o_proj
    );
    comm.all_reduce(out);
    return out;
}

at::Tensor qwen4_exp_model::deltanet(
    const at::Tensor& h,
    std::int64_t layer,
    const group_batch& g,
    std::span<const kv> lines
) const {
    const deltanet_weights& w = *layers[layer].deltanet;
    const qwen4_exp_config& c = config;
    const std::int64_t ntok = h.size(0);
    const std::size_t state_layer = kv_layer_of[layer];
    const std::size_t conv_layer = state_layer + 1;
    std::optional<at::Tensor> cu_seqlens_q;
    if (g.is_prefill()) {
        cu_seqlens_q = g.cu_seqlens_q;
    }

    const std::int64_t channels = w.conv.size(0);
    const std::int64_t value_heads = heads.linear_n_value_heads;
    at::Tensor projected = small_batch_linear_cuda(h, w.in_proj);
    at::Tensor convolved = causal_conv_cuda(
        projected.narrow(1, 0, channels),
        w.conv,
        lines[conv_layer].k,
        recurrent_rows(g, conv_layer, 1),
        cu_seqlens_q,
        /*dilation=*/1
    );
    at::Tensor gated = gated_delta_rule_cuda(
        convolved,
        projected.narrow(1, channels, value_heads),
        projected.narrow(1, channels + value_heads, value_heads),
        w.dt_bias,
        w.neg_a,
        lines[state_layer].k,
        recurrent_rows(g, state_layer, draft_tokens + 1),
        cu_seqlens_q,
        heads.linear_n_key_heads,
        projected.narrow(1, channels + 2 * value_heads, value_heads * c.linear_value_head_dim),
        w.norm,
        c.rms_norm_eps,
        c.linear_output_gate,
        g.arrival_counters
    );
    at::Tensor out = small_batch_linear_cuda(gated, w.out_proj);
    comm.all_reduce(out);
    return out;
}

at::Tensor qwen4_exp_model::moe(const at::Tensor& h, std::int64_t layer, const at::Tensor& counters) const {
    const moe_weights& m = layers[layer].moe;
    at::Tensor logits = small_batch_linear_cuda(h, m.router_and_gate);
    const std::int64_t experts = m.router_and_gate.size(0) - 1;
    at::Tensor routed = routed_experts_cuda(
        h,
        logits.narrow(1, 0, experts),
        config.num_experts_per_tok,
        m.first_expert,
        m.experts_gate_up_proj,
        m.experts_down_proj,
        expert_activation::silu,
        counters
    );
    at::Tensor shared = small_batch_linear_cuda(
        small_batch_gated_linear_cuda(h, m.shared_gate_up_proj),
        m.shared_down_proj
    );
    at::Tensor out = sigmoid_gated_add_cuda(routed, logits.narrow(1, experts, 1), shared);
    comm.all_reduce(out);
    return out;
}

// Hugging Face shifts each sequence right by k with EOS in the places before the sequence's start and before a token
// following an EOS, so a token's k-th previous token is EOS when any of its previous k is EOS.
// A token line keeps the last window tokens before a sequence's next one, each XOR eos_token_id, the n-gram context and
// the tokens of up to draft_tokens tokens a step does not keep; after a step that did not keep its last rejected
// tokens, the valid ones end window - rejected entries in.
at::Tensor
qwen4_exp_model::ngram_ids(const ple_weights& w, std::size_t token_layer, const group_batch& g, const kv& lines) const {
    const std::int64_t ntok = g.tokens.size(0);
    const std::int64_t context = config.ngram_size - 1;
    const std::int64_t eos = config.eos_token_id;
    const std::int64_t window = lines.k.size(1);
    const auto longs = g.tokens.options().dtype(at::kLong);
    at::Tensor rows = g.page_tables.select(0, static_cast<std::int64_t>(token_layer)).to(at::kLong);
    at::Tensor slot = rows.select(1, 0);
    const std::int64_t num_seqs = rows.size(0);
    at::Tensor starts = g.is_prefill() ? g.cu_seqlens_q.to(at::kLong) : at::arange(num_seqs + 1, longs);
    at::Tensor counts = starts.narrow(0, 1, num_seqs) - starts.narrow(0, 0, num_seqs);
    at::Tensor valid = window - g.rejected_tokens.to(at::kLong);
    at::Tensor sequence_of = at::repeat_interleave(counts, std::optional<std::int64_t>(ntok));
    at::Tensor local = at::arange(ntok, longs) - starts.index_select(0, sequence_of);

    // Entry c of a sequence's inputs: its line's valid entries, then its tokens in the group, EOS before the first.
    at::Tensor tokens = g.tokens.to(at::kLong);
    at::Tensor inputs = at::cat({at::bitwise_xor(lines.k.index_select(0, slot), eos).flatten(), tokens});
    auto input_at = [&](const at::Tensor& sequence, const at::Tensor& c) {
        at::Tensor flat_sequence = sequence.flatten();
        at::Tensor flat_c = c.flatten();
        at::Tensor sequence_valid = valid.index_select(0, flat_sequence);
        at::Tensor in_line = flat_sequence * window + flat_c.clamp(0, window - 1);
        at::Tensor in_group = num_seqs * window + starts.index_select(0, flat_sequence) + flat_c - sequence_valid;
        at::Tensor read = inputs.index_select(0, at::where(flat_c < sequence_valid, in_line, in_group.clamp_min(0)));
        return at::where(flat_c < 0, eos, read).view(c.sizes());
    };

    std::vector<at::Tensor> shifted{tokens};
    at::Tensor after_eos = at::zeros({ntok}, longs.dtype(at::kBool));
    at::Tensor own = valid.index_select(0, sequence_of) + local;
    for (std::int64_t k = 1; k <= context; k++) {
        at::Tensor previous = input_at(sequence_of, own - k);
        after_eos = after_eos.logical_or(previous.eq(eos));
        shifted.push_back(at::where(after_eos, eos, previous));
    }
    const std::int64_t heads_per_ngram = w.head_vocab_sizes.size(0) / context;
    std::vector<at::Tensor> ids;
    for (std::int64_t n = 2; n <= config.ngram_size; n++) {
        at::Tensor mixed = shifted[0] * w.layer_multipliers[0];
        for (std::int64_t k = 1; k < n; k++) {
            mixed = at::bitwise_xor(mixed, shifted[k] * w.layer_multipliers[k]);
        }
        at::Tensor sizes = w.head_vocab_sizes.narrow(0, (n - 2) * heads_per_ngram, heads_per_ngram);
        at::Tensor offsets = w.head_offsets.narrow(0, (n - 2) * heads_per_ngram, heads_per_ngram);
        ids.push_back(at::remainder(mixed.unsqueeze(-1), sizes) + offsets);
    }

    // The window after the group's first count tokens of each sequence.
    at::Tensor sequences = at::arange(num_seqs, longs).unsqueeze(1).expand({num_seqs, window});
    at::Tensor entries = at::arange(window, longs).unsqueeze(0);
    auto window_after = [&](const at::Tensor& count) {
        return at::bitwise_xor(input_at(sequences, valid.unsqueeze(1) + count.unsqueeze(1) - window + entries), eos);
    };
    at::Tensor history = g.history_positions;
    at::Tensor page_ends = at::floor_divide(history + counts, g.page_size) * g.page_size - history;
    at::Tensor boundary = rows.gather(
                                  1,
                                  (1 + at::remainder(at::floor_divide(history + page_ends, g.page_size) - 1, 2))
                                      .unsqueeze(1)
    )
                              .squeeze(1);
    at::Tensor at_page_end = at::where(
        (page_ends > 0).unsqueeze(1),
        window_after(page_ends.clamp_min(0)),
        lines.k.index_select(0, boundary)
    );
    lines.k.index_copy_(0, boundary, at_page_end);
    lines.k.index_copy_(0, slot, window_after(counts));
    return at::cat(ids, -1);
}

at::Tensor
qwen4_exp_model::ple(const at::Tensor& x, std::int64_t layer, const group_batch& g, std::span<const kv> lines) const {
    const ple_weights& w = *layers[layer].ple;
    const qwen4_exp_config& c = config;
    const std::int64_t ntok = x.size(0);
    const std::size_t conv_layer = kv_layer_of[layer] + 2;
    const std::size_t token_layer = conv_layer + 1;
    std::optional<at::Tensor> cu_seqlens_q;
    if (g.is_prefill()) {
        cu_seqlens_q = g.cu_seqlens_q;
    }

    at::Tensor ids = ngram_ids(w, token_layer, g, lines[token_layer]);
    at::Tensor embeddings = lookup(w.ngram_embedding, w.first_row, ids).view({ntok, -1});
    at::Tensor keys = stream_norm(at::linear(embeddings, w.key_proj), w.norm_key).view({ntok, c.hc_count, -1});
    at::Tensor queries = stream_norm(x, w.norm_query).view({ntok, c.hc_count, -1});
    at::Tensor gate = (keys * queries).sum(-1, true) / std::sqrt(static_cast<double>(c.hidden_size));
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign();
    at::Tensor gated = (at::sigmoid(gate) * at::linear(embeddings, w.value_proj).unsqueeze(1)).flatten(1);
    at::Tensor convolved = causal_conv_cuda(
        stream_norm(gated, w.norm_conv),
        w.conv,
        lines[conv_layer].k,
        recurrent_rows(g, conv_layer, 1),
        cu_seqlens_q,
        c.ngram_size
    );
    return gated + convolved;
}

// A block's output joins the streams in the next mix's kernel, so x lags it until then.
std::pair<at::Tensor, at::Tensor>
qwen4_exp_model::group_forward(const group_batch& g, std::span<const kv> lines) const {
    at::Tensor x = lookup(embed_tokens, first_vocab_row, g.tokens).repeat({1, config.hc_count});
    // The last block's output and its injection weights, not yet added to x.
    std::optional<std::pair<at::Tensor, at::Tensor>> pending;
    auto inject_pending = [&] {
        if (pending) {
            x = hyper_connection_inject_cuda(x, pending->first, pending->second);
            pending.reset();
        }
    };
    auto mix_pending = [&](const hyper_connection_weights& w) {
        if (!pending) {
            return mix(x, w);
        }
        auto [summed, mixed] = injected_mix(x, pending->first, pending->second, w);
        x = summed;
        pending.reset();
        return mixed;
    };
    for (std::int64_t layer = 0; layer < n_layers; layer++) {
        const layer_weights& w = layers[layer];
        if (w.ple) {
            inject_pending();
            x = x + ple(x, layer, g, lines);
        }
        mixed_input mixed = mix_pending(w.attn_hyper_connection);
        at::Tensor out = indexed_attention(layer) ? attention(mixed.input, layer, g, lines)
                                                  : deltanet(mixed.input, layer, g, lines);
        pending.emplace(out, mixed.injection_weights);
        mixed = mix_pending(w.mlp_hyper_connection);
        pending.emplace(moe(mixed.input, layer, g.arrival_counters), mixed.injection_weights);
    }
    inject_pending();
    return {logits(final_mix(logit_inputs(x, g))), x};
}

std::int64_t qwen4_exp_model::arrival_counter_count(std::int64_t sequences, std::int64_t tokens) const {
    const moe_weights& m = layers.front().moe;
    return std::max(
        gated_delta_rule_counters(sequences, heads.linear_n_value_heads),
        routed_experts_counters(
            tokens,
            config.num_experts_per_tok,
            m.experts_gate_up_proj.size(0),
            m.experts_down_proj.size(2),
            m.experts_down_proj.size(1)
        )
    );
}

step_logits qwen4_exp_model::forward(const step_batch& batch, std::span<const kv> lines) {
    step_logits out;
    if (batch.prefill_group) {
        auto [logits, hidden] = group_forward(*batch.prefill_group, lines);
        out.prefill_logits = logits;
        if (mtp) {
            out.prefill_hidden = hidden;
        }
    }
    if (batch.decode_group) {
        auto [logits, hidden] = group_forward(*batch.decode_group, lines);
        out.decode_logits = logits;
        if (mtp) {
            out.decode_hidden = hidden;
        }
    }
    return out;
}

draft_head* qwen4_exp_model::drafter() {
    return mtp ? this : nullptr;
}

std::int64_t qwen4_exp_model::hidden_dim() const {
    return config.hc_count * config.hidden_size;
}

// The line of each sequence's view index key_count - 2, the position before its last row; a sequence whose last row is
// its first position keeps its line's row.
void qwen4_exp_model::write_last_hidden(const group_batch& g, const at::Tensor& hidden, const kv& line) const {
    const auto layer_row = static_cast<std::int64_t>(kv_layer_of[n_layers]);
    const std::int64_t page_size = line.k.size(1);
    at::Tensor key_counts = g.key_counts.select(0, layer_row).to(at::kLong);
    at::Tensor previous = (key_counts - 2).clamp_min(0);
    at::Tensor line_ids = g.page_tables.select(0, layer_row)
                              .to(at::kLong)
                              .gather(1, at::floor_divide(previous, page_size).unsqueeze(1));
    line_ids = line_ids.squeeze(1);
    at::Tensor last_rows = g.is_prefill() ? g.cu_seqlens_q.narrow(0, 1, key_counts.size(0)).to(at::kLong) - 1
                                          : at::arange(key_counts.size(0), key_counts.options());
    at::Tensor rows = at::where(
        (key_counts >= 2).unsqueeze(1),
        hidden.index_select(0, last_rows),
        line.last_hidden.index_select(0, line_ids)
    );
    line.last_hidden.index_copy_(0, line_ids, rows);
}

draft_output qwen4_exp_model::draft(const group_batch& g, const at::Tensor& hidden, std::span<const kv> lines) {
    const mtp_weights& m = *mtp;
    const qwen4_exp_config& c = config;
    const layer_weights& w = layers[n_layers];
    const std::int64_t ntok = g.tokens.size(0);
    write_last_hidden(g, hidden, lines[kv_layer_of[n_layers]]);
    at::Tensor embedded = at::linear(
        rmsnorm_cuda(lookup(embed_tokens, first_vocab_row, g.tokens), m.pre_fc_norm_embedding, c.rms_norm_eps),
        m.fc_embedding
    );
    at::Tensor streams = at::linear(
                             rmsnorm_cuda(hidden.contiguous(), m.pre_fc_norm_hidden, c.rms_norm_eps)
                                 .view({ntok, c.hc_count, c.hidden_size}),
                             m.fc_hidden
    )
                             .view({ntok, c.hc_count * c.hidden_size});
    at::Tensor whole = at::ones({ntok, c.hc_count}, embedded.options());
    auto [x, mixed] = injected_mix(streams, embedded, whole, w.attn_hyper_connection);
    at::Tensor attended = attention(mixed.input, n_layers, g, lines);
    auto [summed, mlp_mixed] = injected_mix(x, attended, mixed.injection_weights, w.mlp_hyper_connection);
    x = hyper_connection_inject_cuda(
        summed,
        moe(mlp_mixed.input, n_layers, g.arrival_counters),
        mlp_mixed.injection_weights
    );
    at::Tensor rows = logit_inputs(x, g);
    return draft_output{
        .logits = logits(mixed_streams(stream_norm(rows, m.mixer.hc_norm), m.mixer)),
        .hidden = x,
    };
}

}  // namespace pulsarlm
