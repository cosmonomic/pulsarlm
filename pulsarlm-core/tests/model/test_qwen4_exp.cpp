#include "pulsarlm/model/qwen4_exp.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAFunctions.h>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <format>
#include <string>
#include <vector>

// Qwen4ExpModel's paged forward against a dense ATen Qwen4-Exp written from Hugging Face's code, with the delta rule
// run token by token, the experts one by one, QSA's blocks selected query by query and the n-gram ids hashed on the
// host: chunked prefill then decode, checking the logits and the lines of every recurrent KV layer at the end and at the
// newest page end.

namespace {

using pulsarlm::qwen4_exp_layer_type;

constexpr std::int64_t hidden = 64;
constexpr std::int64_t vocab = 300;
constexpr std::int64_t n_heads = 4;
constexpr std::int64_t n_kv_heads = 2;
constexpr std::int64_t head_dim = 64;
constexpr double rope_theta = 1e6;
constexpr double partial_rotary_factor = 0.25;
constexpr std::int64_t rotary_dim = 16;
constexpr std::int64_t key_heads = 2;
constexpr std::int64_t value_heads = 4;
constexpr std::int64_t key_head_dim = 32;
constexpr std::int64_t value_head_dim = 32;
constexpr std::int64_t conv_kernel = 4;
constexpr std::int64_t channels = 2 * key_heads * key_head_dim + value_heads * value_head_dim;
constexpr std::int64_t n_experts = 8;
constexpr std::int64_t moe_intermediate = 32;
constexpr std::int64_t shared_intermediate = 48;
constexpr std::int64_t hc = 4;
constexpr std::int64_t hc_lowrank = 16;
constexpr std::int64_t ple_layer = 1;
constexpr std::int64_t ngram_size = 3;
constexpr std::int64_t ple_conv_kernel = 4;
constexpr std::int64_t ple_history = (ple_conv_kernel - 1) * ngram_size;
constexpr std::int64_t eos = 7;
const std::vector<std::int64_t> head_vocab_sizes{101, 103, 107, 109};
const std::vector<std::int64_t> layer_multipliers{1000003, 999983, 12345679};
constexpr std::int64_t ngram_rows = 430;
constexpr std::int64_t ngram_dim = 16;
constexpr std::int64_t ngram_shards = 2;
constexpr std::int64_t indexer_heads = 4;
constexpr std::int64_t indexer_dim = 32;
constexpr std::int64_t compress_ratio = 4;
constexpr double rms_eps = 1e-6;
constexpr std::int64_t page_size = 16;
const std::vector<qwen4_exp_layer_type> layer_types{
    qwen4_exp_layer_type::linear_attention,
    qwen4_exp_layer_type::linear_attention,
    qwen4_exp_layer_type::indexed_attention,
    qwen4_exp_layer_type::linear_attention,
};
const std::vector<std::int32_t> view_lines{3, 0, 5, 1};
constexpr std::int32_t running_line = 2;
constexpr std::int32_t boundary_line = 4;
constexpr std::int64_t n_lines = 6;
constexpr std::int64_t n_tokens = 49;
const std::vector<std::int64_t> eos_positions{5, 20, 21, 40};

at::TensorOptions cuda(at::ScalarType dtype) {
    return at::device(at::kCUDA).dtype(dtype);
}

at::Tensor int32_cuda(const std::vector<std::int32_t>& values) {
    return at::tensor(values, at::dtype(at::kInt)).to(at::kCUDA);
}

bool indexed(std::size_t layer) {
    return layer_types[layer] == qwen4_exp_layer_type::indexed_attention;
}

std::vector<std::int64_t> head_offsets() {
    std::vector<std::int64_t> offsets{0};
    for (std::size_t head = 0; head + 1 < head_vocab_sizes.size(); head++) {
        offsets.push_back(offsets.back() + head_vocab_sizes[head]);
    }
    return offsets;
}

struct weights {
    c10::Dict<std::string, at::Tensor> named;
    at::Tensor ngram_embedding;

    explicit weights(at::ScalarType dtype) {
        at::manual_seed(13);
        auto add = [&](const std::string& name, at::Tensor t) { named.insert(name, t.to(cuda(dtype))); };
        auto projection = [](std::int64_t out, std::int64_t in) { return at::randn({out, in}) / std::sqrt(in); };
        auto norm = [](std::int64_t size) { return 0.1 * at::randn({size}); };
        auto hyper_connection = [&](const std::string& p, bool injects) {
            add(p + "hc_norm.weight", norm(hc * hidden));
            add(p + "input_mix_weight_down.weight", projection(hc_lowrank, hc * hidden));
            add(p + "input_mix_weight_up.weight", projection(hc * hidden, hc_lowrank));
            if (injects) {
                add(p + "block_inject_weight.weight", projection(hc, hc * hidden));
            }
        };
        const std::string p = "model.language_model.";
        add(p + "embed_tokens.weight", at::randn({vocab, hidden}));
        add("lm_head.weight", projection(vocab, hidden));
        hyper_connection(p + "hyper_connection_mixer.", false);
        for (std::size_t layer = 0; layer < layer_types.size(); layer++) {
            std::string l = p + "layers." + std::to_string(layer) + ".";
            hyper_connection(l + "attn_hyper_connection.", true);
            hyper_connection(l + "mlp_hyper_connection.", true);
            add(l + "mlp.gate.weight", projection(n_experts, hidden));
            add(l + "mlp.experts.gate_up_proj",
                at::randn({n_experts, 2 * moe_intermediate, hidden}) / std::sqrt(hidden));
            add(l + "mlp.experts.down_proj",
                at::randn({n_experts, hidden, moe_intermediate}) / std::sqrt(moe_intermediate));
            add(l + "mlp.shared_expert.gate_proj.weight", projection(shared_intermediate, hidden));
            add(l + "mlp.shared_expert.up_proj.weight", projection(shared_intermediate, hidden));
            add(l + "mlp.shared_expert.down_proj.weight", projection(hidden, shared_intermediate));
            add(l + "mlp.shared_expert_gate.weight", projection(1, hidden));
            if (indexed(layer)) {
                add(l + "self_attn.indexer.index_qk_proj.weight",
                    projection((indexer_heads + 1) * indexer_dim, hidden));
                add(l + "self_attn.indexer.q_layernorm.weight", norm(indexer_dim));
                add(l + "self_attn.indexer.k_layernorm.weight", norm(indexer_dim));
                add(l + "self_attn.q_proj.weight", projection(2 * n_heads * head_dim, hidden));
                add(l + "self_attn.k_proj.weight", projection(n_kv_heads * head_dim, hidden));
                add(l + "self_attn.v_proj.weight", projection(n_kv_heads * head_dim, hidden));
                add(l + "self_attn.o_proj.weight", projection(hidden, n_heads * head_dim));
                add(l + "self_attn.q_norm.weight", norm(head_dim));
                add(l + "self_attn.k_norm.weight", norm(head_dim));
            } else {
                add(l + "linear_attn.in_proj_qkv.weight", projection(channels, hidden));
                add(l + "linear_attn.in_proj_z.weight", projection(value_heads * value_head_dim, hidden));
                add(l + "linear_attn.in_proj_b.weight", projection(value_heads, hidden));
                add(l + "linear_attn.in_proj_a.weight", projection(value_heads, hidden));
                add(l + "linear_attn.conv1d.weight", at::randn({channels, 1, conv_kernel}) / 2.0);
                add(l + "linear_attn.dt_bias", at::randn({value_heads}));
                add(l + "linear_attn.A_log", at::log(at::rand({value_heads}) * 15.0 + 0.5));
                add(l + "linear_attn.norm.weight", 1.0 + 0.1 * at::randn({value_head_dim}));
                add(l + "linear_attn.out_proj.weight", projection(hidden, value_heads * value_head_dim));
            }
            if (static_cast<std::int64_t>(layer) == ple_layer) {
                const std::int64_t embedding_dim = ngram_dim * static_cast<std::int64_t>(head_vocab_sizes.size());
                add(l + "ple.key_proj.weight", projection(hc * hidden, embedding_dim));
                add(l + "ple.value_proj.weight", projection(hidden, embedding_dim));
                add(l + "ple.norm_key.weight", norm(hc * hidden));
                add(l + "ple.norm_query.weight", norm(hc * hidden));
                add(l + "ple.norm_conv.weight", norm(hc * hidden));
                add(l + "ple.conv1d.weight", at::randn({hc * hidden, 1, ple_conv_kernel}) / 2.0);
                const std::string e = l + "ple.ple_embedding.";
                named.insert(e + "layer_multipliers", at::tensor(layer_multipliers).to(at::kCUDA));
                named.insert(e + "ngram_heads_vocab_sizes", at::tensor(head_vocab_sizes).to(at::kCUDA));
                named.insert(e + "ngram_heads_offsets", at::tensor(head_offsets()).to(at::kCUDA));
                ngram_embedding = at::randn({ngram_rows, ngram_dim}).to(cuda(dtype));
                std::vector<at::Tensor> shards = ngram_embedding.chunk(ngram_shards);
                for (std::size_t shard = 0; shard < shards.size(); shard++) {
                    named.insert(
                        std::format("{}ngram_embedding.shard_{}.weight", e, shard),
                        shards[shard].contiguous()
                    );
                }
            }
        }
    }

    at::Tensor operator()(const std::string& name) const {
        return named.at(name).to(at::kFloat);
    }

    at::Tensor layer(std::size_t layer, const std::string& name) const {
        return (*this)("model.language_model.layers." + std::to_string(layer) + "." + name);
    }
};

at::Tensor rms_norm(const at::Tensor& x, const at::Tensor& weight) {
    return x * at::rsqrt(x.pow(2).mean(-1, true) + rms_eps) * (1.0 + weight);
}

// An RMSNorm of each stream of hidden dims, by its part of weight [hc * hidden].
at::Tensor stream_norm(const at::Tensor& x, const at::Tensor& weight) {
    at::Tensor streams = x.view({x.size(0), hc, hidden});
    return (streams * at::rsqrt(streams.pow(2).mean(-1, true) + rms_eps)).flatten(1) * (1.0 + weight);
}

at::Tensor rotate_half(const at::Tensor& x) {
    std::int64_t half = x.size(-1) / 2;
    return at::cat({-x.narrow(-1, half, half), x.narrow(-1, 0, half)}, -1);
}

// Hugging Face's partial RoPE: the first rotary_dim dims rotate among themselves, angles in fp64. x is [tokens,
// heads, head_dim].
at::Tensor rope(const at::Tensor& x) {
    auto exponents = at::arange(0, rotary_dim, 2, cuda(at::kDouble)) / static_cast<double>(rotary_dim);
    auto angles = at::outer(at::arange(x.size(0), cuda(at::kDouble)), at::pow(rope_theta, -exponents));
    angles = at::cat({angles, angles}, -1).unsqueeze(1);
    at::Tensor rotated = x.narrow(-1, 0, rotary_dim);
    rotated = rotated * angles.cos().to(at::kFloat) + rotate_half(rotated) * angles.sin().to(at::kFloat);
    return at::cat({rotated, x.narrow(-1, rotary_dim, head_dim - rotary_dim)}, -1);
}

at::Tensor l2_normalized(const at::Tensor& x) {
    return x * at::rsqrt(x.pow(2).sum(-1, true) + 1e-6);
}

struct dense_output {
    at::Tensor logits;  // [tokens, vocab]
    std::vector<at::Tensor>
        states;  // per layer, S after each token [tokens, value_heads, key_head_dim, value_head_dim]
    std::vector<at::Tensor> conv_inputs;  // per layer, [tokens, channels]
    at::Tensor ple_conv_inputs;  // [tokens, hc * hidden]
};

struct mixed_input {
    at::Tensor input;
    at::Tensor injection_weights;
};

mixed_input mix(const weights& w, const std::string& p, const at::Tensor& x) {
    at::Tensor normed = stream_norm(x, w(p + "hc_norm.weight"));
    at::Tensor mix_weights = at::silu(at::linear(normed, w(p + "input_mix_weight_down.weight")) / hc);
    mix_weights = at::sigmoid(at::linear(mix_weights, w(p + "input_mix_weight_up.weight"))).view({-1, hc, hidden});
    mixed_input out{.input = (mix_weights * normed.view({-1, hc, hidden})).mean(1)};
    if (w.named.contains(p + "block_inject_weight.weight")) {
        out.injection_weights = 2 * at::sigmoid(at::linear(normed, w(p + "block_inject_weight.weight")) / hc);
    }
    return out;
}

at::Tensor inject(const at::Tensor& x, const at::Tensor& out, const mixed_input& mixed) {
    return x + (out.unsqueeze(1) * mixed.injection_weights.unsqueeze(-1)).flatten(1);
}

at::Tensor deltanet(const weights& w, std::size_t layer, const at::Tensor& h, dense_output& out) {
    const std::int64_t n = h.size(0);
    auto mixed = at::linear(h, w.layer(layer, "linear_attn.in_proj_qkv.weight"));
    out.conv_inputs[layer] = mixed;
    auto padded = at::cat({at::zeros({channels, conv_kernel - 1}, cuda(at::kFloat)), mixed.t()}, 1);
    auto convolved = at::silu(
                         at::conv1d(
                             padded.unsqueeze(0),
                             w.layer(layer, "linear_attn.conv1d.weight"),
                             std::nullopt,
                             at::IntArrayRef{1},
                             at::IntArrayRef{0},
                             at::IntArrayRef{1},
                             channels
                         )
    )
                         .squeeze(0)
                         .t();
    std::int64_t key_dim = key_heads * key_head_dim;
    auto q = convolved.narrow(1, 0, key_dim).view({n, key_heads, key_head_dim});
    auto k = convolved.narrow(1, key_dim, key_dim).view({n, key_heads, key_head_dim});
    auto v = convolved.narrow(1, 2 * key_dim, value_heads * value_head_dim).view({n, value_heads, value_head_dim});
    q = l2_normalized(q.repeat_interleave(value_heads / key_heads, 1)) / std::sqrt(static_cast<double>(key_head_dim));
    k = l2_normalized(k.repeat_interleave(value_heads / key_heads, 1));
    auto beta = at::sigmoid(at::linear(h, w.layer(layer, "linear_attn.in_proj_b.weight")));
    auto decay = -w.layer(layer, "linear_attn.A_log").exp() *
        at::softplus(
            at::linear(h, w.layer(layer, "linear_attn.in_proj_a.weight")) + w.layer(layer, "linear_attn.dt_bias")
        );

    auto state = at::zeros({value_heads, key_head_dim, value_head_dim}, cuda(at::kFloat));
    std::vector<at::Tensor> outputs;
    std::vector<at::Tensor> states;
    for (std::int64_t t = 0; t < n; t++) {
        state = state * decay[t].exp().view({value_heads, 1, 1});
        auto remembered = (state * k[t].unsqueeze(-1)).sum(-2);
        auto delta = (v[t] - remembered) * beta[t].unsqueeze(-1);
        state = state + k[t].unsqueeze(-1) * delta.unsqueeze(-2);
        outputs.push_back((state * q[t].unsqueeze(-1)).sum(-2));
        states.push_back(state);
    }
    out.states[layer] = at::stack(states);
    auto core = at::stack(outputs);
    auto normed = core * at::rsqrt(core.pow(2).mean(-1, true) + rms_eps) * w.layer(layer, "linear_attn.norm.weight");
    auto z = at::linear(h, w.layer(layer, "linear_attn.in_proj_z.weight")).view({n, value_heads, value_head_dim});
    auto gated = (normed * at::sigmoid(z)).reshape({n, value_heads * value_head_dim});
    return at::linear(gated, w.layer(layer, "linear_attn.out_proj.weight"));
}

// Hugging Face's partial RoPE in the indexer: the first rotary_dim dims rotate among themselves, its frequencies and
// angles in fp32. x is [..., indexer_dim], positions x's leading dims or all but its last two.
at::Tensor index_rope(const at::Tensor& x, const at::Tensor& positions) {
    auto inv_freq = 1.0 / at::pow(rope_theta, at::arange(0, rotary_dim, 2, at::kLong).to(at::kFloat) / rotary_dim);
    auto freqs = positions.to(at::kFloat).unsqueeze(-1) * inv_freq.to(at::kCUDA);
    auto angles = at::cat({freqs, freqs}, -1);
    if (x.dim() > angles.dim()) {
        angles = angles.unsqueeze(-2);
    }
    at::Tensor rotated = x.narrow(-1, 0, rotary_dim);
    rotated = rotated * angles.cos() + rotate_half(rotated) * angles.sin();
    return at::cat({rotated, x.narrow(-1, rotary_dim, x.size(-1) - rotary_dim)}, -1);
}

// Hugging Face's QSA indexer, query by query: [n, n] whether query i attends key j. The lower block wins among equal
// scores, where torch.topk leaves the order unspecified.
at::Tensor qsa_selection(const weights& w, std::size_t layer, const at::Tensor& h, std::int64_t budget) {
    const std::int64_t n = h.size(0);
    auto qk = at::linear(h, w.layer(layer, "self_attn.indexer.index_qk_proj.weight"));
    auto q = rms_norm(
        qk.narrow(1, 0, indexer_heads * indexer_dim).view({n, indexer_heads, indexer_dim}),
        w.layer(layer, "self_attn.indexer.q_layernorm.weight")
    );
    q = index_rope(q, at::arange(n, cuda(at::kLong)));
    auto raw_keys = qk.narrow(1, indexer_heads * indexer_dim, indexer_dim);
    auto attends = at::zeros({n, n}, cuda(at::kBool));
    for (std::int64_t i = 0; i < n; i++) {
        const std::int64_t complete = (i + 1) / compress_ratio;
        if (complete > 0) {
            auto pooled = raw_keys.narrow(0, 0, complete * compress_ratio)
                              .view({complete, compress_ratio, indexer_dim})
                              .mean(1);
            pooled = rms_norm(pooled, w.layer(layer, "self_attn.indexer.k_layernorm.weight"));
            auto keys = index_rope(pooled, at::arange(complete, cuda(at::kLong)) * compress_ratio);
            auto scores = at::relu(at::matmul(q[i], keys.t())).sum(0) / std::sqrt(static_cast<double>(indexer_dim));
            auto [sorted, order] = at::sort(scores, /*stable=*/true, /*dim=*/0, /*descending=*/true);
            auto top = order.narrow(0, 0, std::min(budget / compress_ratio, complete)).cpu();
            for (std::int64_t t = 0; t < top.size(0); t++) {
                attends[i].narrow(0, top[t].item<std::int64_t>() * compress_ratio, compress_ratio).fill_(true);
            }
        }
        attends[i].narrow(0, complete * compress_ratio, i + 1 - complete * compress_ratio).fill_(true);
    }
    return attends;
}

at::Tensor attention(const weights& w, std::size_t layer, const at::Tensor& h, std::int64_t budget) {
    const std::int64_t n = h.size(0);
    auto query_gate = at::linear(h, w.layer(layer, "self_attn.q_proj.weight")).view({n, n_heads, 2 * head_dim});
    auto q = rms_norm(query_gate.narrow(-1, 0, head_dim), w.layer(layer, "self_attn.q_norm.weight"));
    auto gate = query_gate.narrow(-1, head_dim, head_dim).reshape({n, n_heads * head_dim});
    auto k = rms_norm(
        at::linear(h, w.layer(layer, "self_attn.k_proj.weight")).view({n, n_kv_heads, head_dim}),
        w.layer(layer, "self_attn.k_norm.weight")
    );
    auto v = at::linear(h, w.layer(layer, "self_attn.v_proj.weight")).view({n, n_kv_heads, head_dim});
    auto group = n_heads / n_kv_heads;
    auto q_heads = rope(q).transpose(0, 1);
    auto k_heads = rope(k).repeat_interleave(group, 1).transpose(0, 1);
    auto v_heads = v.repeat_interleave(group, 1).transpose(0, 1);
    auto attends = qsa_selection(w, layer, h, budget);
    auto scores = at::matmul(q_heads, k_heads.transpose(1, 2)) / std::sqrt(static_cast<double>(head_dim));
    auto probabilities = at::softmax(scores.masked_fill(attends.logical_not(), -INFINITY), -1);
    auto o = at::matmul(probabilities, v_heads).transpose(0, 1).reshape({n, n_heads * head_dim});
    return at::linear(o * at::sigmoid(gate), w.layer(layer, "self_attn.o_proj.weight"));
}

// Hugging Face's qwen4_exp block, each expert over the tokens that chose it.
at::Tensor sparse_moe(const weights& w, std::size_t layer, const at::Tensor& h, std::int64_t top_k) {
    auto probabilities = at::softmax(at::linear(h, w.layer(layer, "mlp.gate.weight")), -1);
    auto [top_weights, top_experts] = at::topk(probabilities, top_k, -1);
    top_weights = top_weights / top_weights.sum(-1, true);
    auto gate_up = w.layer(layer, "mlp.experts.gate_up_proj");
    auto down = w.layer(layer, "mlp.experts.down_proj");
    auto out = at::zeros_like(h);
    for (std::int64_t expert = 0; expert < n_experts; expert++) {
        auto chosen = top_experts.eq(expert);
        auto tokens_of = chosen.any(-1).nonzero().squeeze(-1);
        if (tokens_of.numel() == 0) {
            continue;
        }
        auto weight = (top_weights * chosen).sum(-1).index_select(0, tokens_of).unsqueeze(-1);
        auto halves = at::linear(h.index_select(0, tokens_of), gate_up[expert]).chunk(2, -1);
        out.index_add_(0, tokens_of, at::linear(at::silu(halves[0]) * halves[1], down[expert]) * weight);
    }
    auto shared = at::linear(
        at::silu(at::linear(h, w.layer(layer, "mlp.shared_expert.gate_proj.weight"))) *
            at::linear(h, w.layer(layer, "mlp.shared_expert.up_proj.weight")),
        w.layer(layer, "mlp.shared_expert.down_proj.weight")
    );
    return out + at::sigmoid(at::linear(h, w.layer(layer, "mlp.shared_expert_gate.weight"))) * shared;
}

// Hugging Face's _shift_right_ignore_eos over the history of ngram_size - 1 EOS then the tokens: the token shift
// places back within its segment, a segment starting after each EOS, else EOS.
std::vector<std::int64_t> ngram_ids(const std::vector<std::int64_t>& tokens) {
    const std::int64_t context = ngram_size - 1;
    std::vector<std::int64_t> history(context, eos);
    history.insert(history.end(), tokens.begin(), tokens.end());
    auto shifted = [&](std::int64_t position, std::int64_t shift) {
        std::int64_t previous_eos = -1;
        for (std::int64_t i = 0; i < position; i++) {
            if (history[i] == eos) {
                previous_eos = i;
            }
        }
        return position - (previous_eos + 1) >= shift && position - shift >= 0 ? history[position - shift] : eos;
    };
    const std::int64_t heads_per_ngram = static_cast<std::int64_t>(head_vocab_sizes.size()) / context;
    std::vector<std::int64_t> offsets = head_offsets();
    std::vector<std::int64_t> ids;
    for (std::int64_t position = context; position < static_cast<std::int64_t>(history.size()); position++) {
        for (std::int64_t n = 2; n <= ngram_size; n++) {
            std::int64_t mixed = shifted(position, 0) * layer_multipliers[0];
            for (std::int64_t k = 1; k < n; k++) {
                mixed ^= shifted(position, k) * layer_multipliers[k];
            }
            for (std::int64_t head = (n - 2) * heads_per_ngram; head < (n - 1) * heads_per_ngram; head++) {
                ids.push_back(mixed % head_vocab_sizes[head] + offsets[head]);
            }
        }
    }
    return ids;
}

at::Tensor ple(const weights& w, const at::Tensor& x, const std::vector<std::int64_t>& tokens, dense_output& out) {
    const std::int64_t n = x.size(0);
    auto ids = at::tensor(ngram_ids(tokens)).to(at::kCUDA);
    auto embeddings = w.ngram_embedding.to(at::kFloat).index_select(0, ids).view({n, -1});
    auto keys = stream_norm(
                    at::linear(embeddings, w.layer(ple_layer, "ple.key_proj.weight")),
                    w.layer(ple_layer, "ple.norm_key.weight")
    )
                    .view({n, hc, hidden});
    auto queries = stream_norm(x, w.layer(ple_layer, "ple.norm_query.weight")).view({n, hc, hidden});
    auto gate = (keys * queries).sum(-1, true) / std::sqrt(static_cast<double>(hidden));
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign();
    auto value = at::linear(embeddings, w.layer(ple_layer, "ple.value_proj.weight"));
    auto gated = (at::sigmoid(gate) * value.unsqueeze(1)).flatten(1);
    auto normed = stream_norm(gated, w.layer(ple_layer, "ple.norm_conv.weight"));
    out.ple_conv_inputs = normed;
    auto padded = at::cat({at::zeros({hc * hidden, ple_history}, cuda(at::kFloat)), normed.t()}, 1);
    auto convolved = at::silu(
                         at::conv1d(
                             padded.unsqueeze(0),
                             w.layer(ple_layer, "ple.conv1d.weight"),
                             std::nullopt,
                             at::IntArrayRef{1},
                             at::IntArrayRef{0},
                             at::IntArrayRef{ngram_size},
                             hc * hidden
                         )
    )
                         .squeeze(0)
                         .t();
    return gated + convolved;
}

dense_output
dense_forward(const weights& w, const std::vector<std::int64_t>& tokens, std::int64_t top_k, std::int64_t budget) {
    dense_output out{
        .states = std::vector<at::Tensor>(layer_types.size()),
        .conv_inputs = std::vector<at::Tensor>(layer_types.size()),
    };
    auto ids = at::tensor(tokens).to(at::kCUDA);
    auto x = w("model.language_model.embed_tokens.weight").index_select(0, ids).repeat({1, hc});
    for (std::size_t layer = 0; layer < layer_types.size(); layer++) {
        const std::string l = "model.language_model.layers." + std::to_string(layer) + ".";
        if (static_cast<std::int64_t>(layer) == ple_layer) {
            x = x + ple(w, x, tokens, out);
        }
        mixed_input mixed = mix(w, l + "attn_hyper_connection.", x);
        x = inject(
            x,
            indexed(layer) ? attention(w, layer, mixed.input, budget) : deltanet(w, layer, mixed.input, out),
            mixed
        );
        mixed = mix(w, l + "mlp_hyper_connection.", x);
        x = inject(x, sparse_moe(w, layer, mixed.input, top_k), mixed);
    }
    out.logits = at::linear(mix(w, "model.language_model.hyper_connection_mixer.", x).input, w("lm_head.weight"));
    return out;
}

pulsarlm::qwen4_exp_config model_config(std::int64_t experts_per_token, std::int64_t budget) {
    return {
        .vocab_size = vocab,
        .hidden_size = hidden,
        .layer_types = layer_types,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .head_dim = head_dim,
        .rope_theta = rope_theta,
        .partial_rotary_factor = partial_rotary_factor,
        .linear_n_key_heads = key_heads,
        .linear_n_value_heads = value_heads,
        .linear_key_head_dim = key_head_dim,
        .linear_value_head_dim = value_head_dim,
        .linear_conv_kernel = conv_kernel,
        .linear_output_gate = pulsarlm::gate_activation::sigmoid,
        .rms_norm_eps = rms_eps,
        .num_experts_per_tok = experts_per_token,
        .hc_count = hc,
        .ple_layers = {ple_layer},
        .ngram_size = ngram_size,
        .eos_token_id = eos,
        .indexer_n_heads = indexer_heads,
        .indexer_head_dim = indexer_dim,
        .indexer_budget = budget,
        .indexer_compress_ratio = compress_ratio,
    };
}

// A step of the single sequence forwarding the tokens at positions [first, last).
pulsarlm::GroupBatch step(
    const at::Tensor& tokens,
    std::int64_t first,
    std::int64_t last,
    bool prefill,
    const std::vector<pulsarlm::kv_layer>& specs
) {
    std::vector<std::int32_t> rows;
    for (const pulsarlm::kv_layer& spec : specs) {
        if (spec.state) {
            rows.insert(rows.end(), {running_line, boundary_line});
            rows.resize(rows.size() + view_lines.size() - 2, 0);
        } else {
            rows.insert(rows.end(), view_lines.begin(), view_lines.end());
        }
    }
    std::int64_t page_end = last / page_size * page_size;
    auto n_kv_layers = static_cast<std::int64_t>(specs.size());
    auto key_count = int32_cuda({static_cast<std::int32_t>(last)}).expand({n_kv_layers, 1}).contiguous();
    pulsarlm::GroupBatch g{
        .tokens = tokens.narrow(0, first, last - first),
        .pos = at::arange(first, last, cuda(at::kLong)),
        .page_tables = int32_cuda(rows).view({n_kv_layers, 1, -1}),
        .page_end_tokens = int32_cuda({static_cast<std::int32_t>(page_end > first ? page_end - first : 0)}),
        .page_positions = (at::arange(static_cast<std::int64_t>(rows.size()) / n_kv_layers, cuda(at::kInt)) * page_size)
                              .expand({n_kv_layers, 1, -1})
                              .contiguous(),
    };
    g.key_counts = key_count;
    if (prefill) {
        g.cu_seqlens_q = int32_cuda({0, static_cast<std::int32_t>(last - first)});
    }
    return g;
}

double max_relative_error(const at::Tensor& got, const at::Tensor& expected) {
    return ((got.to(at::kFloat) - expected).abs().max() / expected.abs().max()).item<double>();
}

void check_against_dense(at::ScalarType dtype, double tolerance, std::int64_t experts_per_token, std::int64_t budget) {
    weights w(dtype);
    pulsarlm::Qwen4ExpModel model(model_config(experts_per_token, budget), w.named);
    std::vector<pulsarlm::kv_layer> specs = model.kv_layers();
    REQUIRE(specs.size() == 9);
    std::vector<pulsarlm::kv> kv;
    for (const pulsarlm::kv_layer& spec : specs) {
        if (spec.state) {
            std::vector<std::int64_t> sizes{n_lines};
            sizes.insert(sizes.end(), spec.state->sizes.begin(), spec.state->sizes.end());
            at::Tensor lines = at::zeros(sizes, cuda(spec.state->dtype));
            kv.push_back(pulsarlm::kv{.k = lines, .v = lines});
        } else {
            kv.push_back(
                pulsarlm::kv{
                    .k = at::zeros({n_lines, page_size, n_kv_heads, head_dim}, cuda(dtype)),
                    .v = at::zeros({n_lines, page_size, n_kv_heads, head_dim}, cuda(dtype)),
                    .index_keys = at::zeros({n_lines, page_size, indexer_dim}, cuda(dtype)),
                }
            );
        }
    }

    auto tokens = at::randint(vocab, {n_tokens}, cuda(at::kLong));
    for (std::int64_t position : eos_positions) {
        tokens[position] = eos;
    }
    at::Tensor host_tokens = tokens.cpu();
    std::vector<std::int64_t> token_ids(
        host_tokens.data_ptr<std::int64_t>(),
        host_tokens.data_ptr<std::int64_t>() + n_tokens
    );
    dense_output dense = dense_forward(w, token_ids, experts_per_token, budget);

    struct chunk {
        std::int64_t first;
        std::int64_t last;
        bool prefill;
    };
    for (chunk c :
         {chunk{0, 21, true}, chunk{21, 37, true}, chunk{37, 47, true}, chunk{47, 48, false}, chunk{48, 49, false}}) {
        INFO("tokens [" << c.first << ", " << c.last << ")" << (c.prefill ? " prefill" : " decode"));
        pulsarlm::StepBatch batch;
        (c.prefill ? batch.prefill : batch.decode) = step(tokens, c.first, c.last, c.prefill, specs);
        pulsarlm::StepLogits logits = model.forward(batch, kv);
        auto expected_logits = dense.logits.narrow(0, c.first, c.last - c.first);
        CHECK(max_relative_error(*(c.prefill ? logits.prefill : logits.decode), expected_logits) < tolerance);
    }

    std::size_t kv_index = 0;
    for (std::size_t layer = 0; layer < layer_types.size(); layer++) {
        INFO("layer " << layer);
        if (indexed(layer)) {
            kv_index++;
            continue;
        }
        const at::Tensor& states = kv[kv_index].k;
        const at::Tensor& conv = kv[kv_index + 1].k;
        CHECK(max_relative_error(states[running_line], dense.states[layer][n_tokens - 1]) < tolerance);
        CHECK(max_relative_error(states[boundary_line], dense.states[layer][47]) < tolerance);
        auto inputs = dense.conv_inputs[layer];
        CHECK(max_relative_error(conv[running_line], inputs.narrow(0, n_tokens - 3, 3).t()) < tolerance);
        CHECK(max_relative_error(conv[boundary_line], inputs.narrow(0, 45, 3).t()) < tolerance);
        kv_index += 2;
        if (static_cast<std::int64_t>(layer) != ple_layer) {
            continue;
        }
        const at::Tensor& ple_conv = kv[kv_index].k;
        const at::Tensor& ple_tokens = kv[kv_index + 1].k;
        CHECK(
            max_relative_error(
                ple_conv[running_line],
                dense.ple_conv_inputs.narrow(0, n_tokens - ple_history, ple_history).t()
            ) < tolerance
        );
        CHECK(
            max_relative_error(
                ple_conv[boundary_line],
                dense.ple_conv_inputs.narrow(0, 48 - ple_history, ple_history).t()
            ) < tolerance
        );
        CHECK(at::equal(ple_tokens[running_line], at::bitwise_xor(tokens.narrow(0, n_tokens - 2, 2), eos)));
        CHECK(at::equal(ple_tokens[boundary_line], at::bitwise_xor(tokens.narrow(0, 46, 2), eos)));
        kv_index += 2;
    }
}

}  // namespace

TEST_CASE("Qwen4-Exp kv_layers add a PLE layer's convolution and token lines", "[model][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    weights w(at::kFloat);
    pulsarlm::Qwen4ExpModel model(model_config(2, 8), w.named);
    std::vector<pulsarlm::kv_layer> specs = model.kv_layers();
    REQUIRE(specs.size() == 9);
    CHECK(specs[4].state->sizes == std::vector<std::int64_t>{hc * hidden, ple_history});
    CHECK(specs[5].state->sizes == std::vector<std::int64_t>{ngram_size - 1});
    CHECK(specs[5].state->dtype == at::kLong);
    CHECK(!specs[6].state);
    CHECK(specs[6].index_head_dim == indexer_dim);
}

// A budget of 2 blocks, so that QSA selects from the 12 complete blocks the last queries see.
TEST_CASE("Qwen4-Exp matches a dense forward in fp32", "[model][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    check_against_dense(at::kFloat, 1e-4, 2, 8);
}

// A token's top experts and top blocks change with bf16's rounding of its router logits and block scores, so in bf16
// every token takes every expert and every block.
TEST_CASE("Qwen4-Exp matches a dense forward in bf16", "[model][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    check_against_dense(at::kBFloat16, 5e-2, n_experts, n_tokens * compress_ratio);
}
