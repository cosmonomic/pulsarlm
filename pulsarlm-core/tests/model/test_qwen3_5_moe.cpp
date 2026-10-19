#include "pulsarlm/model/qwen3_5_moe.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAFunctions.h>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

// Qwen3_5MoeModel's paged forward against a dense ATen Qwen3.5-MoE written from Hugging Face's code, with the delta rule
// run token by token and the experts one by one: chunked prefill then decode, checking the logits, the states the
// recurrent lines hold at the end and at the newest page end, and the K the attention lines hold.

namespace {

using pulsarlm::qwen3_5_layer_type;

constexpr std::int64_t hidden = 64;
constexpr std::int64_t shared_intermediate = 128;
constexpr std::int64_t n_experts = 8;
constexpr std::int64_t moe_intermediate = 32;
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
constexpr double rms_eps = 1e-6;
constexpr std::int64_t page_size = 16;
const std::vector<qwen3_5_layer_type> layer_types{
    qwen3_5_layer_type::linear_attention,
    qwen3_5_layer_type::linear_attention,
    qwen3_5_layer_type::full_attention,
    qwen3_5_layer_type::linear_attention,
};
const std::vector<std::int32_t> view_lines{3, 0, 5, 1};
constexpr std::int32_t running_line = 2;
constexpr std::int32_t boundary_line = 4;
constexpr std::int64_t n_lines = 6;
constexpr std::int64_t n_tokens = 49;

at::TensorOptions cuda(at::ScalarType dtype) {
    return at::device(at::kCUDA).dtype(dtype);
}

at::Tensor int32_cuda(const std::vector<std::int32_t>& values) {
    return at::tensor(values, at::dtype(at::kInt)).to(at::kCUDA);
}

bool full(std::size_t layer) {
    return layer_types[layer] == qwen3_5_layer_type::full_attention;
}

struct weights {
    c10::Dict<std::string, at::Tensor> named;

    explicit weights(at::ScalarType dtype) {
        at::manual_seed(13);
        auto add = [&](const std::string& name, at::Tensor t) { named.insert(name, t.to(cuda(dtype))); };
        auto projection = [](std::int64_t out, std::int64_t in) { return at::randn({out, in}) / std::sqrt(in); };
        auto norm = [](std::int64_t size) { return 0.1 * at::randn({size}); };
        const std::string p = "model.language_model.";
        add(p + "embed_tokens.weight", at::randn({vocab, hidden}));
        add(p + "norm.weight", norm(hidden));
        add("lm_head.weight", projection(vocab, hidden));
        for (std::size_t layer = 0; layer < layer_types.size(); layer++) {
            std::string l = p + "layers." + std::to_string(layer) + ".";
            add(l + "input_layernorm.weight", norm(hidden));
            add(l + "post_attention_layernorm.weight", norm(hidden));
            add(l + "mlp.shared_expert.gate_proj.weight", projection(shared_intermediate, hidden));
            add(l + "mlp.shared_expert.up_proj.weight", projection(shared_intermediate, hidden));
            add(l + "mlp.shared_expert.down_proj.weight", projection(hidden, shared_intermediate));
            add(l + "mlp.gate.weight", projection(n_experts, hidden));
            add(l + "mlp.shared_expert_gate.weight", projection(1, hidden));
            add(l + "mlp.experts.gate_up_proj",
                at::randn({n_experts, 2 * moe_intermediate, hidden}) / std::sqrt(hidden));
            add(l + "mlp.experts.down_proj",
                at::randn({n_experts, hidden, moe_intermediate}) / std::sqrt(moe_intermediate));
            if (full(layer)) {
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
    std::vector<at::Tensor> k;  // per layer, unrotated [tokens, n_kv_heads, head_dim]; undefined for linear layers
    std::vector<at::Tensor>
        states;  // per layer, S after each token [tokens, value_heads, key_head_dim, value_head_dim]
    std::vector<at::Tensor> conv_inputs;  // per layer, [tokens, channels]
};

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
    auto gated = (normed * at::silu(z)).reshape({n, value_heads * value_head_dim});
    return at::linear(gated, w.layer(layer, "linear_attn.out_proj.weight"));
}

at::Tensor attention(const weights& w, std::size_t layer, const at::Tensor& h, dense_output& out) {
    const std::int64_t n = h.size(0);
    auto query_gate = at::linear(h, w.layer(layer, "self_attn.q_proj.weight")).view({n, n_heads, 2 * head_dim});
    auto q = rms_norm(query_gate.narrow(-1, 0, head_dim), w.layer(layer, "self_attn.q_norm.weight"));
    auto gate = query_gate.narrow(-1, head_dim, head_dim).reshape({n, n_heads * head_dim});
    auto k = rms_norm(
        at::linear(h, w.layer(layer, "self_attn.k_proj.weight")).view({n, n_kv_heads, head_dim}),
        w.layer(layer, "self_attn.k_norm.weight")
    );
    auto v = at::linear(h, w.layer(layer, "self_attn.v_proj.weight")).view({n, n_kv_heads, head_dim});
    out.k[layer] = k;
    auto group = n_heads / n_kv_heads;
    auto q_heads = rope(q).transpose(0, 1);
    auto k_heads = rope(k).repeat_interleave(group, 1).transpose(0, 1);
    auto v_heads = v.repeat_interleave(group, 1).transpose(0, 1);
    auto causal = at::ones({n, n}, cuda(at::kBool)).tril();
    auto scores = at::matmul(q_heads, k_heads.transpose(1, 2)) / std::sqrt(static_cast<double>(head_dim));
    auto probabilities = at::softmax(scores.masked_fill(causal.logical_not(), -INFINITY), -1);
    auto o = at::matmul(probabilities, v_heads).transpose(0, 1).reshape({n, n_heads * head_dim});
    return at::linear(o * at::sigmoid(gate), w.layer(layer, "self_attn.o_proj.weight"));
}

at::Tensor swiglu(const weights& w, std::size_t layer, const std::string& mlp, const at::Tensor& h) {
    auto gated = at::silu(at::linear(h, w.layer(layer, mlp + "gate_proj.weight"))) *
        at::linear(h, w.layer(layer, mlp + "up_proj.weight"));
    return at::linear(gated, w.layer(layer, mlp + "down_proj.weight"));
}

// Hugging Face's qwen3_5_moe block, each expert over the tokens that chose it.
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
    auto shared = at::sigmoid(at::linear(h, w.layer(layer, "mlp.shared_expert_gate.weight"))) *
        swiglu(w, layer, "mlp.shared_expert.", h);
    return out + shared;
}

dense_output dense_forward(const weights& w, const at::Tensor& tokens, std::int64_t top_k) {
    dense_output out{
        .k = std::vector<at::Tensor>(layer_types.size()),
        .states = std::vector<at::Tensor>(layer_types.size()),
        .conv_inputs = std::vector<at::Tensor>(layer_types.size()),
    };
    auto x = w("model.language_model.embed_tokens.weight").index_select(0, tokens);
    for (std::size_t layer = 0; layer < layer_types.size(); layer++) {
        auto h = rms_norm(x, w.layer(layer, "input_layernorm.weight"));
        x = x + (full(layer) ? attention(w, layer, h, out) : deltanet(w, layer, h, out));
        auto h2 = rms_norm(x, w.layer(layer, "post_attention_layernorm.weight"));
        x = x + sparse_moe(w, layer, h2, top_k);
    }
    out.logits = at::linear(rms_norm(x, w("model.language_model.norm.weight")), w("lm_head.weight"));
    return out;
}

// Pulsar's head dim i holds Hugging Face's head dim permutation[i].
std::vector<std::int64_t> rope_permutation() {
    std::vector<std::int64_t> permutation;
    for (std::int64_t i = 0; i < rotary_dim / 2; i++) {
        permutation.push_back(i);
    }
    for (std::int64_t i = rotary_dim; i < rotary_dim + head_dim / 2 - rotary_dim / 2; i++) {
        permutation.push_back(i);
    }
    for (std::int64_t i = rotary_dim / 2; i < rotary_dim; i++) {
        permutation.push_back(i);
    }
    for (std::int64_t i = rotary_dim + head_dim / 2 - rotary_dim / 2; i < head_dim; i++) {
        permutation.push_back(i);
    }
    return permutation;
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

void check_against_dense(at::ScalarType dtype, double tolerance, std::int64_t top_k) {
    weights w(dtype);
    pulsarlm::Qwen3_5MoeModel model(
        pulsarlm::qwen3_5_moe_config{
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
            .rms_norm_eps = rms_eps,
            .num_experts_per_tok = top_k,
        },
        w.named
    );
    std::vector<pulsarlm::kv_layer> specs = model.kv_layers();
    REQUIRE(specs.size() == 7);
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
                }
            );
        }
    }

    auto tokens = at::randint(vocab, {n_tokens}, cuda(at::kLong));
    dense_output dense = dense_forward(w, tokens, top_k);

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

    auto permutation = at::tensor(rope_permutation()).to(at::kCUDA);
    auto lines = at::tensor(std::vector<std::int64_t>(view_lines.begin(), view_lines.end())).to(at::kCUDA);
    std::size_t kv_index = 0;
    for (std::size_t layer = 0; layer < layer_types.size(); layer++) {
        INFO("layer " << layer);
        if (full(layer)) {
            auto stored = kv[kv_index].k.index_select(0, lines).view({-1, n_kv_heads, head_dim});
            auto expected = dense.k[layer].index_select(-1, permutation);
            CHECK(max_relative_error(stored.narrow(0, 0, n_tokens), expected) < tolerance);
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
    }
}

}  // namespace

TEST_CASE("Qwen3.5-MoE matches a dense forward", "[model][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    SECTION("top 2 experts in fp32") {
        check_against_dense(at::kFloat, 1e-4, 2);
    }
    // A token's top experts change with bf16's rounding of its router logits, so in bf16 every token takes every
    // expert.
    SECTION("every expert in bf16") {
        check_against_dense(at::kBFloat16, 5e-2, n_experts);
    }
}
