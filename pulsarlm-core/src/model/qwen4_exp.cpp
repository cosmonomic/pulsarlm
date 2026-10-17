#include "pulsarlm/model/qwen4_exp.hpp"

#include "pulsarlm/model/paged_forward.hpp"
#include "pulsarlm/model/partial_rope.hpp"
#include "pulsarlm/ops.hpp"

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

at::Tensor need(const c10::Dict<std::string, at::Tensor>& weights, const std::string& name) {
    if (!weights.contains(prefix + name)) {
        throw std::invalid_argument(std::format("Qwen4ExpModel: missing weight {:?}", prefix + name));
    }
    return weights.at(prefix + name);
}

// Hugging Face's Qwen4-Exp RMSNorm scales by 1 + weight, in fp32.
at::Tensor zero_centered(const at::Tensor& weight) {
    return weight.to(at::kFloat) + 1.0;
}

// The table's shards, concatenated in order.
at::Tensor ngram_table(const c10::Dict<std::string, at::Tensor>& weights, const std::string& p) {
    std::vector<at::Tensor> shards;
    while (weights.contains(std::format("{}{}shard_{}.weight", prefix, p, shards.size()))) {
        shards.push_back(weights.at(std::format("{}{}shard_{}.weight", prefix, p, shards.size())));
    }
    if (shards.empty()) {
        throw std::invalid_argument(std::format("Qwen4ExpModel: missing weight {:?}", prefix + p + "shard_0.weight"));
    }
    return at::cat(shards);
}

}  // namespace

Qwen4ExpModel::Qwen4ExpModel(qwen4_exp_config config, const c10::Dict<std::string, at::Tensor>& weights)
: config(std::move(config)),
  n_layers(static_cast<std::int64_t>(this->config.layer_types.size())),
  embed_tokens(need(weights, "embed_tokens.weight")) {
    const qwen4_exp_config& c = this->config;
    const auto rotary_dim = static_cast<std::int64_t>(std::llround(c.partial_rotary_factor * c.head_dim));
    if (rotary_dim % 2 != 0 || rotary_dim > c.head_dim || c.head_dim % 2 != 0) {
        throw std::invalid_argument("Qwen4ExpModel: the rotated dims must be an even part of an even head dim");
    }
    if (c.linear_n_value_heads % c.linear_n_key_heads != 0) {
        throw std::invalid_argument("Qwen4ExpModel: the value heads must be a multiple of the key heads");
    }
    if (c.hc_count < 2 || c.ngram_size < 2) {
        throw std::invalid_argument("Qwen4ExpModel: hc_count and ngram_size must be at least 2");
    }
    if (c.indexer_budget % c.indexer_compress_ratio != 0 || rotary_dim > c.indexer_head_dim) {
        throw std::invalid_argument(
            "Qwen4ExpModel: the indexer budget must be whole blocks and the rotated dims fit an index head"
        );
    }
    for (std::int64_t layer : c.ple_layers) {
        if (layer < 0 || layer >= n_layers || c.layer_types[layer] != qwen4_exp_layer_type::linear_attention) {
            throw std::invalid_argument("Qwen4ExpModel: a PLE layer must be a linear attention layer");
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
    const std::int64_t conv_channels = 2 * c.linear_n_key_heads * c.linear_key_head_dim +
        c.linear_n_value_heads * c.linear_value_head_dim;
    auto mixer = [&](const std::string& p) {
        return MixerWeights{
            .hc_norm = zero_centered(need(weights, p + "hc_norm.weight")),
            .input_mix_weight_down = need(weights, p + "input_mix_weight_down.weight"),
            .input_mix_weight_up = need(weights, p + "input_mix_weight_up.weight"),
        };
    };
    auto hyper_connection = [&](const std::string& p) {
        return HyperConnectionWeights{
            .mixer = mixer(p),
            .block_inject_weight = need(weights, p + "block_inject_weight.weight"),
        };
    };
    hyper_connection_mixer = mixer("hyper_connection_mixer.");
    auto recurrent = [](std::vector<std::int64_t> sizes, at::ScalarType dtype) {
        return kv_layer{
            .n_kv_heads = 0,
            .head_dim = 0,
            .state = recurrent_state{.sizes = std::move(sizes), .dtype = dtype}
        };
    };

    for (std::int64_t layer = 0; layer < n_layers; layer++) {
        const std::string p = "layers." + std::to_string(layer) + ".";
        LayerWeights w{
            .attn_hyper_connection = hyper_connection(p + "attn_hyper_connection."),
            .mlp_hyper_connection = hyper_connection(p + "mlp_hyper_connection."),
            .moe = MoeWeights{
                .router = need(weights, p + "mlp.gate.weight"),
                .experts_gate_up_proj = need(weights, p + "mlp.experts.gate_up_proj"),
                .experts_down_proj = need(weights, p + "mlp.experts.down_proj"),
                .shared_gate_proj = need(weights, p + "mlp.shared_expert.gate_proj.weight"),
                .shared_up_proj = need(weights, p + "mlp.shared_expert.up_proj.weight"),
                .shared_down_proj = need(weights, p + "mlp.shared_expert.down_proj.weight"),
                .shared_expert_gate = need(weights, p + "mlp.shared_expert_gate.weight"),
            },
        };
        if (w.moe.router.size(0) < c.num_experts_per_tok || c.num_experts_per_tok < 1) {
            throw std::invalid_argument("Qwen4ExpModel: num_experts_per_tok must be between 1 and the experts");
        }
        kv_layer_of.push_back(kv_specs.size());
        if (indexed_attention(layer)) {
            w.attention = AttentionWeights{
                .q_proj = permute_queries(need(weights, p + "self_attn.q_proj.weight"), c.n_heads, permutation),
                .k_proj = permute_heads(need(weights, p + "self_attn.k_proj.weight"), c.n_kv_heads, permutation),
                .v_proj = need(weights, p + "self_attn.v_proj.weight"),
                .o_proj = need(weights, p + "self_attn.o_proj.weight"),
                .q_norm = zero_centered(need(weights, p + "self_attn.q_norm.weight")).index_select(0, permutation),
                .k_norm = zero_centered(need(weights, p + "self_attn.k_norm.weight")).index_select(0, permutation),
                .index_qk_proj = need(weights, p + "self_attn.indexer.index_qk_proj.weight"),
                .index_q_norm = zero_centered(need(weights, p + "self_attn.indexer.q_layernorm.weight")),
                .index_k_norm = zero_centered(need(weights, p + "self_attn.indexer.k_layernorm.weight")),
            };
            kv_specs.push_back(
                kv_layer{
                    .n_kv_heads = static_cast<std::size_t>(c.n_kv_heads),
                    .head_dim = static_cast<std::size_t>(c.head_dim),
                    .n_q_heads = static_cast<std::size_t>(c.n_heads),
                    .index_head_dim = static_cast<std::size_t>(c.indexer_head_dim),
                }
            );
        } else {
            w.deltanet = DeltaNetWeights{
                .in_proj_qkv = need(weights, p + "linear_attn.in_proj_qkv.weight"),
                .in_proj_z = need(weights, p + "linear_attn.in_proj_z.weight"),
                .in_proj_b = need(weights, p + "linear_attn.in_proj_b.weight"),
                .in_proj_a = need(weights, p + "linear_attn.in_proj_a.weight"),
                .conv = need(weights, p + "linear_attn.conv1d.weight"),
                .dt_bias = need(weights, p + "linear_attn.dt_bias").to(at::kFloat),
                .neg_a = -need(weights, p + "linear_attn.A_log").to(at::kFloat).exp(),
                .norm = need(weights, p + "linear_attn.norm.weight"),
                .out_proj = need(weights, p + "linear_attn.out_proj.weight"),
            };
            kv_specs.push_back(
                recurrent({c.linear_n_value_heads, c.linear_key_head_dim, c.linear_value_head_dim}, at::kFloat)
            );
            kv_specs.push_back(recurrent({conv_channels, c.linear_conv_kernel - 1}, embed_tokens.scalar_type()));
        }
        if (std::ranges::contains(c.ple_layers, layer)) {
            const std::string e = p + "ple.ple_embedding.";
            w.ple = PleWeights{
                .key_proj = need(weights, p + "ple.key_proj.weight"),
                .value_proj = need(weights, p + "ple.value_proj.weight"),
                .norm_key = zero_centered(need(weights, p + "ple.norm_key.weight")),
                .norm_query = zero_centered(need(weights, p + "ple.norm_query.weight")),
                .norm_conv = zero_centered(need(weights, p + "ple.norm_conv.weight")),
                .conv = need(weights, p + "ple.conv1d.weight"),
                .layer_multipliers = need(weights, e + "layer_multipliers").to(at::kLong),
                .head_vocab_sizes = need(weights, e + "ngram_heads_vocab_sizes").to(at::kLong),
                .head_offsets = need(weights, e + "ngram_heads_offsets").to(at::kLong),
                .ngram_embedding = ngram_table(weights, e + "ngram_embedding."),
            };
            if (w.ple->layer_multipliers.size(0) != c.ngram_size ||
                w.ple->head_vocab_sizes.size(0) % (c.ngram_size - 1) != 0) {
                throw std::invalid_argument("Qwen4ExpModel: the n-gram buffers must match ngram_size");
            }
            const std::int64_t history = (w.ple->conv.size(2) - 1) * c.ngram_size;
            kv_specs.push_back(recurrent({c.hc_count * c.hidden_size, history}, embed_tokens.scalar_type()));
            kv_specs.push_back(recurrent({c.ngram_size - 1}, at::kLong));
        }
        layers.push_back(std::move(w));
    }
}

bool Qwen4ExpModel::indexed_attention(std::int64_t layer) const {
    return config.layer_types[layer] == qwen4_exp_layer_type::indexed_attention;
}

at::Tensor Qwen4ExpModel::index_rope(const at::Tensor& x, const at::Tensor& positions) const {
    const std::int64_t rotary_dim = 2 * index_inv_freq.size(0);
    at::Tensor freqs = positions.to(at::kFloat).unsqueeze(-1) * index_inv_freq;
    at::Tensor angles = at::cat({freqs, freqs}, -1);
    std::vector<std::int64_t> sizes = angles.sizes().vec();
    while (static_cast<std::int64_t>(sizes.size()) < x.dim()) {
        sizes.insert(sizes.end() - 1, 1);
    }
    at::Tensor cos = angles.cos().to(x.scalar_type()).view(sizes);
    at::Tensor sin = angles.sin().to(x.scalar_type()).view(sizes);
    at::Tensor rotated = x.narrow(-1, 0, rotary_dim);
    const std::int64_t half = rotary_dim / 2;
    at::Tensor rotated_half = at::cat({-rotated.narrow(-1, half, half), rotated.narrow(-1, 0, half)}, -1);
    return at::cat({rotated * cos + rotated_half * sin, x.narrow(-1, rotary_dim, x.size(-1) - rotary_dim)}, -1);
}

// Hugging Face takes a query's top blocks with torch.topk, whose order among equal scores is unspecified; here the lower
// block wins.
at::Tensor Qwen4ExpModel::selected_blocks(
    const AttentionWeights& w,
    const at::Tensor& queries,
    const kv& lines,
    const at::Tensor& page_tables,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& key_counts,
    const at::Tensor& rope_layout
) const {
    const qwen4_exp_config& c = config;
    const std::int64_t ratio = c.indexer_compress_ratio;
    const std::int64_t ntok = queries.size(0);
    const std::int64_t num_seqs = page_tables.size(0);
    const std::int64_t page_size = lines.index_keys.size(1);
    TORCH_CHECK(page_size % ratio == 0, "Qwen4ExpModel: a page must hold whole index blocks");
    const std::int64_t max_blocks = page_tables.size(1) * page_size / ratio;
    const std::int64_t words = max_blocks / 32 + 1;
    const auto longs = queries.options().dtype(at::kLong);

    at::Tensor blocks = index_blocks_cuda(
        lines.index_keys,
        page_tables,
        rope_layout,
        w.index_k_norm,
        index_inv_freq,
        ratio,
        c.rms_norm_eps
    );

    at::Tensor starts = cu_seqlens_q.to(at::kLong);
    at::Tensor counts = starts.narrow(0, 1, num_seqs) - starts.narrow(0, 0, num_seqs);
    at::Tensor sequence_of = at::repeat_interleave(counts, std::optional<std::int64_t>(ntok));
    at::Tensor view_index = key_counts.to(at::kLong).index_select(0, sequence_of) -
        counts.index_select(0, sequence_of) + at::arange(ntok, longs) - starts.index_select(0, sequence_of);
    at::Tensor complete_blocks = at::floor_divide(view_index + 1, ratio);

    const std::int64_t chunk_rows = std::max<std::int64_t>(1, (std::int64_t{1} << 26) / max_blocks);
    std::vector<at::Tensor> selected;
    for (std::int64_t first = 0; first < ntok; first += chunk_rows) {
        const std::int64_t rows = std::min(chunk_rows, ntok - first);
        at::Tensor row_blocks = complete_blocks.narrow(0, first, rows).contiguous();
        at::Tensor scores = index_scores_cuda(
            queries.narrow(0, first, rows).contiguous(),
            blocks,
            sequence_of.narrow(0, first, rows).contiguous(),
            row_blocks,
            1.0 / std::sqrt(static_cast<double>(c.indexer_head_dim))
        );
        selected.push_back(select_blocks_cuda(
            scores,
            row_blocks,
            view_index.narrow(0, first, rows).contiguous(),
            c.indexer_budget / ratio,
            ratio,
            words
        ));
    }
    return at::cat(selected);
}

std::vector<kv_layer> Qwen4ExpModel::kv_layers() const {
    return kv_specs;
}

at::Tensor Qwen4ExpModel::stream_norm(const at::Tensor& x, const at::Tensor& weight) const {
    at::Tensor streams = x.to(at::kFloat).view({x.size(0), -1, config.hidden_size});
    at::Tensor normed = streams * at::rsqrt(streams.pow(2).mean(-1, true) + config.rms_norm_eps);
    return (normed.flatten(1) * weight).to(x.scalar_type());
}

at::Tensor Qwen4ExpModel::mixed_streams(const at::Tensor& normed, const MixerWeights& w) const {
    const auto hc = static_cast<double>(config.hc_count);
    at::Tensor mix_weights = at::silu(at::linear(normed, w.input_mix_weight_down) / hc);
    mix_weights = at::sigmoid(at::linear(mix_weights, w.input_mix_weight_up));
    at::Tensor streams = normed.view({normed.size(0), config.hc_count, config.hidden_size});
    return (mix_weights.view(streams.sizes()) * streams).mean(-2);
}

Qwen4ExpModel::MixedInput Qwen4ExpModel::mix(const at::Tensor& x, const HyperConnectionWeights& w) const {
    at::Tensor normed = stream_norm(x, w.mixer.hc_norm);
    return MixedInput{
        .input = mixed_streams(normed, w.mixer),
        .injection_weights = 2 *
            at::sigmoid(at::linear(normed, w.block_inject_weight) / static_cast<double>(config.hc_count)),
    };
}

at::Tensor Qwen4ExpModel::final_mix(const at::Tensor& x) const {
    return mixed_streams(stream_norm(x, hyper_connection_mixer.hc_norm), hyper_connection_mixer);
}

at::Tensor Qwen4ExpModel::attention(
    const at::Tensor& h,
    std::int64_t layer,
    const GroupBatch& g,
    std::span<const kv> lines
) const {
    const AttentionWeights& w = *layers[layer].attention;
    const qwen4_exp_config& c = config;
    const std::int64_t ntok = h.size(0);
    const std::size_t kv_index = kv_layer_of[layer];

    at::Tensor query_gate = at::linear(h, w.q_proj).view({ntok, c.n_heads, 2 * c.head_dim});
    at::Tensor q = rmsnorm_cuda(query_gate.narrow(-1, 0, c.head_dim), w.q_norm, c.rms_norm_eps);
    at::Tensor gate = query_gate.narrow(-1, c.head_dim, c.head_dim).reshape({ntok, c.n_heads * c.head_dim});
    q = rope_q(
        q.view({ntok, c.n_heads * c.head_dim}),
        g.pos.select(0, static_cast<std::int64_t>(kv_index)),
        c.n_heads,
        c.head_dim,
        rope
    );
    at::Tensor k = rmsnorm_cuda(
        at::linear(h, w.k_proj).view({ntok, c.n_kv_heads, c.head_dim}),
        w.k_norm,
        c.rms_norm_eps
    );
    at::Tensor v = at::linear(h, w.v_proj).view({ntok, c.n_kv_heads, c.head_dim}).contiguous();

    const kv& line = lines[kv_index];
    const auto layer_row = static_cast<std::int64_t>(kv_index);
    at::Tensor page_tables = g.page_tables.select(0, layer_row);
    at::Tensor key_counts = g.key_counts.select(0, layer_row);
    std::optional<at::Tensor> cu_seqlens_write;
    at::Tensor cu_seqlens_q;
    if (g.is_prefill()) {
        cu_seqlens_write = g.cu_seqlens_q;
        cu_seqlens_q = g.cu_seqlens_q;
    } else {
        cu_seqlens_q = at::arange(page_tables.size(0) + 1, page_tables.options());
    }
    write_kv_cuda(line.k, line.v, k, v, page_tables, cu_seqlens_write, key_counts);

    at::Tensor index_qk = at::linear(h, w.index_qk_proj);
    at::Tensor index_queries = rmsnorm_cuda(
        index_qk.narrow(1, 0, c.indexer_n_heads * c.indexer_head_dim)
            .view({ntok, c.indexer_n_heads, c.indexer_head_dim}),
        w.index_q_norm,
        c.rms_norm_eps
    );
    index_queries = index_rope(index_queries, g.pos.select(0, layer_row));
    at::Tensor index_keys = index_qk.narrow(1, c.indexer_n_heads * c.indexer_head_dim, c.indexer_head_dim)
                                .contiguous()
                                .view({ntok, 1, -1});
    at::Tensor index_pool = line.index_keys.unsqueeze(2);
    write_kv_cuda(index_pool, index_pool, index_keys, index_keys, page_tables, cu_seqlens_write, key_counts);
    at::Tensor selected = selected_blocks(w, index_queries, line, page_tables, cu_seqlens_q, key_counts, g.rope_layout);

    std::optional<at::Tensor> mass, cu_view_pages, mass_decay;
    at::Tensor page_mass = g.page_mass.empty() ? at::Tensor() : g.page_mass[kv_index];
    if (page_mass.defined()) {
        mass = at::zeros({page_mass.size(0), line.k.size(1), c.n_heads}, q.options().dtype(at::kFloat));
        cu_view_pages = g.cu_view_pages.select(0, layer_row);
        mass_decay = g.attention_mass_decay;
    }
    at::Tensor o = attn_indexed_cuda(
        q,
        line.k,
        line.v,
        page_tables,
        cu_seqlens_q,
        key_counts,
        g.rope_layout,
        rope,
        1.0 / std::sqrt(static_cast<double>(c.head_dim)),
        selected,
        c.indexer_compress_ratio,
        mass,
        cu_view_pages,
        mass_decay,
        g.mass_length_gain
    );
    if (mass) {
        at::maximum_out(page_mass, page_mass, mass->mean(-1).amax(-1));
    }
    return at::linear(o.reshape({ntok, c.n_heads * c.head_dim}) * at::sigmoid(gate), w.o_proj);
}

at::Tensor
Qwen4ExpModel::deltanet(const at::Tensor& h, std::int64_t layer, const GroupBatch& g, std::span<const kv> lines) const {
    const DeltaNetWeights& w = *layers[layer].deltanet;
    const qwen4_exp_config& c = config;
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
        g.page_tables.select(0, static_cast<std::int64_t>(conv_layer)),
        cu_seqlens_q,
        g.page_end_tokens,
        /*dilation=*/1
    );
    at::Tensor core = gated_delta_rule_cuda(
        convolved,
        at::linear(h, w.in_proj_a),
        at::linear(h, w.in_proj_b),
        w.dt_bias,
        w.neg_a,
        lines[state_layer].k,
        g.page_tables.select(0, static_cast<std::int64_t>(state_layer)),
        cu_seqlens_q,
        g.page_end_tokens,
        c.linear_n_key_heads
    );
    at::Tensor z = at::linear(h, w.in_proj_z).view(core.sizes());
    at::Tensor gated = gated_rmsnorm_cuda(core, z, w.norm, c.rms_norm_eps, c.linear_output_gate);
    return at::linear(gated.view({ntok, c.linear_n_value_heads * c.linear_value_head_dim}), w.out_proj);
}

at::Tensor Qwen4ExpModel::moe(const at::Tensor& h, std::int64_t layer) const {
    const MoeWeights& m = layers[layer].moe;
    at::Tensor probabilities = at::softmax(at::linear(h, m.router), -1, at::kFloat);
    auto [top_weights, top_experts] = at::topk(probabilities, config.num_experts_per_tok, -1);
    top_weights = top_weights / top_weights.sum(-1, true);
    at::Tensor routed = routed_experts(
        h,
        top_experts,
        top_weights,
        m.experts_gate_up_proj,
        m.experts_down_proj,
        [](const at::Tensor& x) { return at::silu(x); }
    );
    at::Tensor shared = at::linear(
        at::silu(at::linear(h, m.shared_gate_proj)) * at::linear(h, m.shared_up_proj),
        m.shared_down_proj
    );
    return routed + at::sigmoid(at::linear(h, m.shared_expert_gate)) * shared;
}

// Hugging Face shifts each sequence right by k with EOS in the places before the sequence's start and before a token
// following an EOS, so a token's k-th previous token is EOS when any of its previous k is EOS.
at::Tensor
Qwen4ExpModel::ngram_ids(const PleWeights& w, std::size_t token_layer, const GroupBatch& g, const kv& lines) const {
    const std::int64_t ntok = g.tokens.size(0);
    const std::int64_t context = config.ngram_size - 1;
    const std::int64_t eos = config.eos_token_id;
    const auto longs = g.tokens.options().dtype(at::kLong);
    at::Tensor rows = g.page_tables.select(0, static_cast<std::int64_t>(token_layer)).to(at::kLong);
    at::Tensor running = rows.select(1, 0);
    at::Tensor boundary = rows.select(1, 1);
    const std::int64_t num_seqs = rows.size(0);
    at::Tensor starts = g.is_prefill() ? g.cu_seqlens_q.to(at::kLong) : at::arange(num_seqs + 1, longs);
    at::Tensor counts = starts.narrow(0, 1, num_seqs) - starts.narrow(0, 0, num_seqs);
    at::Tensor sequence_of = at::repeat_interleave(counts, std::optional<std::int64_t>(ntok));
    at::Tensor local = at::arange(ntok, longs) - starts.index_select(0, sequence_of);

    // Each sequence's history is its last context tokens, then its tokens in the group.
    at::Tensor tokens = g.tokens.to(at::kLong);
    at::Tensor history = at::cat({at::bitwise_xor(lines.k.index_select(0, running), eos).flatten(), tokens});
    auto history_at = [&](const at::Tensor& sequence, const at::Tensor& index) {
        at::Tensor before = sequence * context + index;
        at::Tensor after = num_seqs * context + starts.index_select(0, sequence.flatten()).view(sequence.sizes()) +
            index - context;
        return history.index_select(0, at::where(index < context, before, after).flatten()).view(index.sizes());
    };

    std::vector<at::Tensor> shifted{tokens};
    at::Tensor after_eos = at::zeros({ntok}, longs.dtype(at::kBool));
    for (std::int64_t k = 1; k <= context; k++) {
        at::Tensor previous = history_at(sequence_of, local + context - k);
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

    at::Tensor sequences = at::arange(num_seqs, longs).unsqueeze(1).expand({num_seqs, context});
    at::Tensor last = at::arange(context, longs).unsqueeze(0);
    at::Tensor ends = history_at(sequences, counts.unsqueeze(1) + last);
    at::Tensor page_ends = g.page_end_tokens.to(at::kLong).unsqueeze(1);
    at::Tensor at_page_end = at::where(
        page_ends > 0,
        at::bitwise_xor(history_at(sequences, page_ends + last), eos),
        lines.k.index_select(0, boundary)
    );
    lines.k.index_copy_(0, running, at::bitwise_xor(ends, eos));
    lines.k.index_copy_(0, boundary, at_page_end);
    return at::cat(ids, -1);
}

at::Tensor
Qwen4ExpModel::ple(const at::Tensor& x, std::int64_t layer, const GroupBatch& g, std::span<const kv> lines) const {
    const PleWeights& w = *layers[layer].ple;
    const qwen4_exp_config& c = config;
    const std::int64_t ntok = x.size(0);
    const std::size_t conv_layer = kv_layer_of[layer] + 2;
    const std::size_t token_layer = conv_layer + 1;
    std::optional<at::Tensor> cu_seqlens_q;
    if (g.is_prefill()) {
        cu_seqlens_q = g.cu_seqlens_q;
    }

    at::Tensor ids = ngram_ids(w, token_layer, g, lines[token_layer]);
    at::Tensor embeddings = w.ngram_embedding.index_select(0, ids.flatten()).view({ntok, -1});
    at::Tensor keys = stream_norm(at::linear(embeddings, w.key_proj), w.norm_key).view({ntok, c.hc_count, -1});
    at::Tensor queries = stream_norm(x, w.norm_query).view({ntok, c.hc_count, -1});
    at::Tensor gate = (keys * queries).sum(-1, true) / std::sqrt(static_cast<double>(c.hidden_size));
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign();
    at::Tensor gated = (at::sigmoid(gate) * at::linear(embeddings, w.value_proj).unsqueeze(1)).flatten(1);
    at::Tensor convolved = causal_conv_cuda(
        stream_norm(gated, w.norm_conv),
        w.conv,
        lines[conv_layer].k,
        g.page_tables.select(0, static_cast<std::int64_t>(conv_layer)),
        cu_seqlens_q,
        g.page_end_tokens,
        c.ngram_size
    );
    return gated + convolved;
}

at::Tensor Qwen4ExpModel::group_forward(const GroupBatch& g, std::span<const kv> lines) const {
    auto inject = [](const at::Tensor& x, const at::Tensor& out, const MixedInput& mixed) {
        return x + (out.unsqueeze(1) * mixed.injection_weights.unsqueeze(-1)).flatten(1);
    };
    at::Tensor x = embed_tokens.index_select(0, g.tokens).repeat({1, config.hc_count});
    for (std::int64_t layer = 0; layer < n_layers; layer++) {
        const LayerWeights& w = layers[layer];
        if (w.ple) {
            x = x + ple(x, layer, g, lines);
        }
        MixedInput mixed = mix(x, w.attn_hyper_connection);
        x = inject(
            x,
            indexed_attention(layer) ? attention(mixed.input, layer, g, lines) : deltanet(mixed.input, layer, g, lines),
            mixed
        );
        mixed = mix(x, w.mlp_hyper_connection);
        x = inject(x, moe(mixed.input, layer), mixed);
    }
    return at::matmul(final_mix(logit_inputs(x, g)), lm_head.t());
}

StepLogits Qwen4ExpModel::forward(const StepBatch& batch, std::span<const kv> lines) {
    return paged_forward(batch, [&](const GroupBatch& g) { return group_forward(g, lines); });
}

}  // namespace pulsarlm
