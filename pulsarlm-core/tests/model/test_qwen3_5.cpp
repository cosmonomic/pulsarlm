#include "pulsarlm/engine/communicator.hpp"
#include "pulsarlm/engine/model/qwen3_5.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// qwen3_5_model's paged forward against a dense ATen Qwen3.5 written from Hugging Face's code, with the delta rule run
// token by token: chunked prefill then decode, checking the logits, the states the recurrent lines hold at the end and
// at the newest page end, and the K the attention lines hold. Split over ranks on one device, it matches the model on
// one GPU: the same logits on every rank, and each rank's lines the rows of one GPU's lines it holds.

namespace {

using pulsarlm::qwen3_5_layer_type;

constexpr std::int64_t hidden = 64;
constexpr std::int64_t intermediate = 128;
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
    c10::Dict<std::string, at::Tensor> named_tensors;

    explicit weights(at::ScalarType dtype) {
        at::manual_seed(13);
        auto add = [&](const std::string& name, at::Tensor t) { named_tensors.insert(name, t.to(cuda(dtype))); };
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
            add(l + "mlp.gate_proj.weight", projection(intermediate, hidden));
            add(l + "mlp.up_proj.weight", projection(intermediate, hidden));
            add(l + "mlp.down_proj.weight", projection(hidden, intermediate));
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
        return named_tensors.at(name).to(at::kFloat);
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

at::Tensor
attention(const weights& w, std::size_t layer, const at::Tensor& h, dense_output& out, std::int64_t kv_heads) {
    const std::int64_t n = h.size(0);
    auto query_gate = at::linear(h, w.layer(layer, "self_attn.q_proj.weight")).view({n, n_heads, 2 * head_dim});
    auto q = rms_norm(query_gate.narrow(-1, 0, head_dim), w.layer(layer, "self_attn.q_norm.weight"));
    auto gate = query_gate.narrow(-1, head_dim, head_dim).reshape({n, n_heads * head_dim});
    auto k = rms_norm(
        at::linear(h, w.layer(layer, "self_attn.k_proj.weight")).view({n, kv_heads, head_dim}),
        w.layer(layer, "self_attn.k_norm.weight")
    );
    auto v = at::linear(h, w.layer(layer, "self_attn.v_proj.weight")).view({n, kv_heads, head_dim});
    out.k[layer] = k;
    auto group = n_heads / kv_heads;
    auto q_heads = rope(q).transpose(0, 1);
    auto k_heads = rope(k).repeat_interleave(group, 1).transpose(0, 1);
    auto v_heads = v.repeat_interleave(group, 1).transpose(0, 1);
    auto causal = at::ones({n, n}, cuda(at::kBool)).tril();
    auto scores = at::matmul(q_heads, k_heads.transpose(1, 2)) / std::sqrt(static_cast<double>(head_dim));
    auto probabilities = at::softmax(scores.masked_fill(causal.logical_not(), -INFINITY), -1);
    auto o = at::matmul(probabilities, v_heads).transpose(0, 1).reshape({n, n_heads * head_dim});
    return at::linear(o * at::sigmoid(gate), w.layer(layer, "self_attn.o_proj.weight"));
}

dense_output dense_forward(const weights& w, const at::Tensor& tokens, std::int64_t kv_heads = n_kv_heads) {
    dense_output out{
        .k = std::vector<at::Tensor>(layer_types.size()),
        .states = std::vector<at::Tensor>(layer_types.size()),
        .conv_inputs = std::vector<at::Tensor>(layer_types.size()),
    };
    auto x = w("model.language_model.embed_tokens.weight").index_select(0, tokens);
    for (std::size_t layer = 0; layer < layer_types.size(); layer++) {
        auto h = rms_norm(x, w.layer(layer, "input_layernorm.weight"));
        x = x + (full(layer) ? attention(w, layer, h, out, kv_heads) : deltanet(w, layer, h, out));
        auto h2 = rms_norm(x, w.layer(layer, "post_attention_layernorm.weight"));
        auto mlp = at::silu(at::linear(h2, w.layer(layer, "mlp.gate_proj.weight"))) *
            at::linear(h2, w.layer(layer, "mlp.up_proj.weight"));
        x = x + at::linear(mlp, w.layer(layer, "mlp.down_proj.weight"));
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
pulsarlm::group_batch step(
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
    pulsarlm::group_batch g{
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

struct fp8_linear {
    at::Tensor weight;
    at::Tensor scale;
};

// One scale per block, its largest magnitude mapped to e4m3's largest finite value 448.
fp8_linear quantized(const at::Tensor& weight, pulsarlm::fp8_block block) {
    const std::int64_t n = weight.size(0), k = weight.size(1);
    const std::int64_t rows = (n + block.n - 1) / block.n;
    at::Tensor padded = at::zeros({rows * block.n, k}, cuda(at::kFloat));
    padded.narrow(0, 0, n).copy_(weight);
    const at::Tensor scale = padded.view({rows, block.n, k / block.k, block.k}).abs().amax({1, 3}).clamp_min(1e-12) /
        448.0;
    const at::Tensor expanded = scale.repeat_interleave(block.n, 0).narrow(0, 0, n).repeat_interleave(block.k, 1);
    return {.weight = (weight.to(at::kFloat) / expanded).to(at::kFloat8_e4m3fn), .scale = scale};
}

// in_proj_a and in_proj_b stay dense, as in Qwen's fp8 checkpoints.
bool quantized_name(const std::string& name) {
    return name.ends_with("proj.weight") && !name.ends_with("in_proj_a.weight") && !name.ends_with("in_proj_b.weight");
}

// The model's weights: w's, with the linears quantized to fp8 and w's own replaced by their dequantized values, which
// the dense forward then reads.
c10::Dict<std::string, at::Tensor> fp8_model_weights(weights& w, pulsarlm::fp8_block block, at::ScalarType dtype) {
    c10::Dict<std::string, at::Tensor> model = w.named_tensors.copy();
    for (const auto& entry : w.named_tensors.copy()) {
        if (!quantized_name(entry.key())) {
            continue;
        }
        const fp8_linear q = quantized(entry.value(), block);
        const at::Tensor expanded = q.scale.repeat_interleave(block.n, 0)
                                        .narrow(0, 0, q.weight.size(0))
                                        .repeat_interleave(block.k, 1);
        w.named_tensors.insert_or_assign(entry.key(), (q.weight.to(at::kFloat) * expanded).to(dtype));
        model.insert_or_assign(entry.key(), q.weight);
        model.insert(entry.key() + "_scale_inv", q.scale);
    }
    return model;
}

pulsarlm::qwen3_5_config model_config(std::optional<pulsarlm::fp8_block> block, std::int64_t kv_heads = n_kv_heads) {
    return pulsarlm::qwen3_5_config{
        .vocab_size = vocab,
        .hidden_size = hidden,
        .layer_types = layer_types,
        .n_heads = n_heads,
        .n_kv_heads = kv_heads,
        .head_dim = head_dim,
        .rope_theta = rope_theta,
        .partial_rotary_factor = partial_rotary_factor,
        .linear_n_key_heads = key_heads,
        .linear_n_value_heads = value_heads,
        .linear_key_head_dim = key_head_dim,
        .linear_value_head_dim = value_head_dim,
        .linear_conv_kernel = conv_kernel,
        .rms_norm_eps = rms_eps,
        .fp8_weight_block = block,
    };
}

// n_lines zeroed lines of each KV layer.
std::vector<pulsarlm::kv> make_lines(const std::vector<pulsarlm::kv_layer>& specs, at::ScalarType dtype) {
    std::vector<pulsarlm::kv> kv;
    for (const pulsarlm::kv_layer& spec : specs) {
        if (spec.state) {
            std::vector<std::int64_t> sizes{n_lines};
            sizes.insert(sizes.end(), spec.state->sizes.begin(), spec.state->sizes.end());
            at::Tensor lines = at::zeros(sizes, cuda(spec.state->dtype));
            kv.push_back(pulsarlm::kv{.k = lines, .v = lines});
        } else {
            const std::vector<std::int64_t> sizes{
                n_lines,
                page_size,
                static_cast<std::int64_t>(spec.n_kv_heads),
                static_cast<std::int64_t>(spec.head_dim),
            };
            kv.push_back(pulsarlm::kv{.k = at::zeros(sizes, cuda(dtype)), .v = at::zeros(sizes, cuda(dtype))});
        }
    }
    return kv;
}

double max_relative_error(const at::Tensor& got, const at::Tensor& expected) {
    return ((got.to(at::kFloat) - expected).abs().max() / expected.abs().max()).item<double>();
}

void check_against_dense(at::ScalarType dtype, double tolerance, std::optional<pulsarlm::fp8_block> block = {}) {
    weights w(dtype);
    const c10::Dict<std::string, at::Tensor> model_weights = block ? fp8_model_weights(w, *block, dtype)
                                                                   : w.named_tensors;
    pulsarlm::qwen3_5_model model(model_config(block), model_weights);
    std::vector<pulsarlm::kv_layer> specs = model.kv_layers();
    REQUIRE(specs.size() == 7);
    std::vector<pulsarlm::kv> kv = make_lines(specs, dtype);

    auto tokens = at::randint(vocab, {n_tokens}, cuda(at::kLong));
    dense_output dense = dense_forward(w, tokens);

    struct chunk {
        std::int64_t first;
        std::int64_t last;
        bool prefill;
    };
    for (chunk c :
         {chunk{0, 21, true}, chunk{21, 37, true}, chunk{37, 47, true}, chunk{47, 48, false}, chunk{48, 49, false}}) {
        INFO("tokens [" << c.first << ", " << c.last << ")" << (c.prefill ? " prefill" : " decode"));
        pulsarlm::step_batch batch;
        (c.prefill ? batch.prefill_group : batch.decode_group) = step(tokens, c.first, c.last, c.prefill, specs);
        pulsarlm::step_logits logits = model.forward(batch, kv);
        auto expected_logits = dense.logits.narrow(0, c.first, c.last - c.first);
        CHECK(
            max_relative_error(*(c.prefill ? logits.prefill_logits : logits.decode_logits), expected_logits) < tolerance
        );
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

using cut_ranges = std::vector<std::pair<std::int64_t, std::int64_t>>;

// The [first, last) ranges of dim along which a weight is split over the ranks, in weight rows or columns.
struct weight_cut {
    std::int64_t dim;
    cut_ranges ranges;
};

// Rank's cut of a weight split over ranks as vLLM v0.31.0 splits Qwen3.5; nullopt for a weight every rank holds whole.
std::optional<weight_cut>
cut_of(const std::string& name, std::int64_t rank, std::int64_t ranks, std::int64_t kv_heads) {
    const std::int64_t heads = n_heads / ranks;
    const std::int64_t key_dim = key_heads * key_head_dim;
    const std::int64_t rank_key_dim = key_dim / ranks;
    const std::int64_t rank_value_dim = value_heads * value_head_dim / ranks;
    const std::int64_t rank_value_heads = value_heads / ranks;
    const auto range = [](std::int64_t first, std::int64_t count) { return std::pair{first, first + count}; };
    if (name.ends_with("q_proj.weight")) {
        return weight_cut{0, {range(rank * heads * 2 * head_dim, heads * 2 * head_dim)}};
    }
    if (name.ends_with("k_proj.weight") || name.ends_with("v_proj.weight")) {
        if (ranks <= kv_heads) {
            const std::int64_t rank_kv_heads = kv_heads / ranks;
            return weight_cut{0, {range(rank * rank_kv_heads * head_dim, rank_kv_heads * head_dim)}};
        }
        return weight_cut{0, {range(rank / (ranks / kv_heads) * head_dim, head_dim)}};
    }
    if (name.ends_with("o_proj.weight")) {
        return weight_cut{1, {range(rank * heads * head_dim, heads * head_dim)}};
    }
    if (name.ends_with("gate_proj.weight") || name.ends_with("up_proj.weight")) {
        return weight_cut{0, {range(rank * intermediate / ranks, intermediate / ranks)}};
    }
    if (name.ends_with("down_proj.weight")) {
        return weight_cut{1, {range(rank * intermediate / ranks, intermediate / ranks)}};
    }
    if (name.ends_with("in_proj_qkv.weight") || name.ends_with("conv1d.weight")) {
        return weight_cut{
            0,
            {range(rank * rank_key_dim, rank_key_dim),
             range(key_dim + rank * rank_key_dim, rank_key_dim),
             range(2 * key_dim + rank * rank_value_dim, rank_value_dim)},
        };
    }
    if (name.ends_with("in_proj_z.weight")) {
        return weight_cut{0, {range(rank * rank_value_dim, rank_value_dim)}};
    }
    if (name.ends_with("in_proj_b.weight") || name.ends_with("in_proj_a.weight") || name.ends_with("dt_bias") ||
        name.ends_with("A_log")) {
        return weight_cut{0, {range(rank * rank_value_heads, rank_value_heads)}};
    }
    if (name.ends_with("out_proj.weight")) {
        return weight_cut{1, {range(rank * rank_value_dim, rank_value_dim)}};
    }
    return std::nullopt;
}

// The ranges of t along dim, each divided by unit, concatenated; an fp8 tensor is cut as bytes.
at::Tensor cut(const at::Tensor& t, std::int64_t dim, const cut_ranges& ranges, std::int64_t unit) {
    const bool fp8 = t.scalar_type() == at::kFloat8_e4m3fn;
    const at::Tensor source = fp8 ? t.view(at::kByte) : t;
    std::vector<at::Tensor> parts;
    for (auto [first, last] : ranges) {
        parts.push_back(source.narrow(dim, first / unit, (last - first) / unit));
    }
    const at::Tensor joined = at::cat(parts, dim).contiguous();
    return fp8 ? joined.view(at::kFloat8_e4m3fn) : joined;
}

// The embedding or head padded with zero rows to a multiple of 64, then rank's rows.
at::Tensor vocab_rows(const at::Tensor& t, std::int64_t rank, std::int64_t ranks) {
    const std::int64_t padded = (vocab + 63) / 64 * 64;
    const at::Tensor rows = at::cat({t, at::zeros({padded - vocab, t.size(1)}, t.options())}, 0);
    return rows.narrow(0, rank * padded / ranks, padded / ranks).contiguous();
}

// Rank's slices of a model's weights; an fp8 scale is cut along its weight's ranges in blocks.
c10::Dict<std::string, at::Tensor> rank_weights(
    const c10::Dict<std::string, at::Tensor>& full_weights,
    std::int64_t rank,
    std::int64_t ranks,
    std::int64_t kv_heads,
    std::optional<pulsarlm::fp8_block> block
) {
    c10::Dict<std::string, at::Tensor> sliced;
    const std::string scale_suffix = "_scale_inv";
    for (const auto& entry : full_weights) {
        const std::string& name = entry.key();
        if (name.ends_with("embed_tokens.weight") || name == "lm_head.weight") {
            sliced.insert(name, vocab_rows(entry.value(), rank, ranks));
            continue;
        }
        const bool scale = name.ends_with(scale_suffix);
        const std::string weight_name = scale ? name.substr(0, name.size() - scale_suffix.size()) : name;
        const std::optional<weight_cut> c = cut_of(weight_name, rank, ranks, kv_heads);
        if (!c) {
            sliced.insert(name, entry.value());
            continue;
        }
        const std::int64_t unit = scale ? (c->dim == 0 ? block->n : block->k) : 1;
        sliced.insert(name, cut(entry.value(), c->dim, c->ranges, unit));
    }
    return sliced;
}

// Each rank's forward on its own thread and stream, as the collectives need every rank to run.
std::vector<pulsarlm::step_logits> forward_on_ranks(
    std::vector<pulsarlm::qwen3_5_model>& models,
    const pulsarlm::step_batch& batch,
    const std::vector<std::vector<pulsarlm::kv>>& lines
) {
    c10::cuda::device_synchronize();
    std::vector<pulsarlm::step_logits> logits(models.size());
    std::vector<std::thread> threads;
    for (std::size_t rank = 0; rank < models.size(); rank++) {
        threads.emplace_back([&, rank] {
            c10::cuda::CUDAGuard device(0);
            c10::cuda::CUDAStreamGuard stream(c10::cuda::getStreamFromPool());
            logits[rank] = models[rank].forward(batch, lines[rank]);
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    c10::cuda::device_synchronize();
    return logits;
}

// The rows of one GPU's line tensor a rank holds: along KV heads, dim 2 of [lines, page_size, KV heads, head_dim], or
// along a state's first size, dim 1 of [lines, state sizes...].
at::Tensor shard_rows(const at::Tensor& lines, const pulsarlm::kv_layer_shard& shard) {
    const std::int64_t dim = shard.full.state ? 1 : 2;
    std::vector<at::Tensor> parts;
    for (auto [first, last] : shard.rows) {
        parts.push_back(lines.narrow(dim, static_cast<std::int64_t>(first), static_cast<std::int64_t>(last - first)));
    }
    return at::cat(parts, dim);
}

// The weights with the attention layers' KV heads cut to the first.
c10::Dict<std::string, at::Tensor>
one_kv_head(const c10::Dict<std::string, at::Tensor>& model_weights, std::optional<pulsarlm::fp8_block> block) {
    c10::Dict<std::string, at::Tensor> cut_weights;
    for (const auto& entry : model_weights) {
        const std::string& name = entry.key();
        const bool weight = name.ends_with("k_proj.weight") || name.ends_with("v_proj.weight");
        const bool scale = name.ends_with("k_proj.weight_scale_inv") || name.ends_with("v_proj.weight_scale_inv");
        if (weight || scale) {
            cut_weights.insert(name, cut(entry.value(), 0, {{0, head_dim}}, scale ? block->n : 1));
        } else {
            cut_weights.insert(name, entry.value());
        }
    }
    return cut_weights;
}

bool same_layer(const pulsarlm::kv_layer& a, const pulsarlm::kv_layer& b) {
    return a.n_kv_heads == b.n_kv_heads && a.head_dim == b.head_dim && a.n_q_heads == b.n_q_heads &&
        a.n_readers == b.n_readers && a.values_only == b.values_only && a.sliding_window == b.sliding_window &&
        a.index_head_dim == b.index_head_dim && a.state == b.state;
}

// With kv_heads below n_kv_heads, the attention layers keep only their first KV head, in the model and the dense
// forward alike.
void check_split_over_ranks(
    at::ScalarType dtype,
    double tolerance,
    std::int64_t ranks,
    std::int64_t kv_heads,
    std::optional<pulsarlm::fp8_block> block = {}
) {
    weights w(dtype);
    const c10::Dict<std::string, at::Tensor> all_kv_heads = block ? fp8_model_weights(w, *block, dtype)
                                                                  : w.named_tensors;
    const c10::Dict<std::string, at::Tensor> model_weights = kv_heads == n_kv_heads ? all_kv_heads
                                                                                    : one_kv_head(all_kv_heads, block);
    if (kv_heads != n_kv_heads) {
        w.named_tensors = one_kv_head(w.named_tensors, std::nullopt);
    }
    pulsarlm::qwen3_5_model single(model_config(block, kv_heads), model_weights);
    const std::vector<pulsarlm::kv_layer> specs = single.kv_layers();
    const std::vector<pulsarlm::kv> single_lines = make_lines(specs, dtype);

    std::vector<pulsarlm::communicator> comms = pulsarlm::communicator::make_on_one_device(
        at::Device(at::kCUDA, 0),
        static_cast<std::size_t>(ranks)
    );
    std::vector<pulsarlm::qwen3_5_model> models;
    std::vector<std::vector<pulsarlm::kv>> lines;
    for (std::int64_t rank = 0; rank < ranks; rank++) {
        models.emplace_back(
            model_config(block, kv_heads),
            rank_weights(model_weights, rank, ranks, kv_heads, block),
            comms[rank]
        );
        lines.push_back(make_lines(models.back().kv_layers(), dtype));
    }

    auto tokens = at::randint(vocab, {n_tokens}, cuda(at::kLong));
    dense_output dense = dense_forward(w, tokens, kv_heads);

    struct chunk {
        std::int64_t first;
        std::int64_t last;
        bool prefill;
    };
    for (chunk c :
         {chunk{0, 21, true}, chunk{21, 37, true}, chunk{37, 47, true}, chunk{47, 48, false}, chunk{48, 49, false}}) {
        INFO("tokens [" << c.first << ", " << c.last << ")" << (c.prefill ? " prefill" : " decode"));
        pulsarlm::step_batch batch;
        (c.prefill ? batch.prefill_group : batch.decode_group) = step(tokens, c.first, c.last, c.prefill, specs);
        const auto group_logits = [&](const pulsarlm::step_logits& l) {
            return *(c.prefill ? l.prefill_logits : l.decode_logits);
        };
        single.forward(batch, single_lines);
        std::vector<pulsarlm::step_logits> logits = forward_on_ranks(models, batch, lines);
        const at::Tensor first_rank = group_logits(logits.front());
        CHECK(max_relative_error(first_rank, dense.logits.narrow(0, c.first, c.last - c.first)) < tolerance);
        for (std::int64_t rank = 1; rank < ranks; rank++) {
            CHECK(at::equal(group_logits(logits[rank]), first_rank));
        }
    }

    for (std::int64_t rank = 0; rank < ranks; rank++) {
        const std::vector<pulsarlm::kv_layer_shard> shards = models[rank].kv_layer_shards();
        REQUIRE(shards.size() == specs.size());
        for (std::size_t layer = 0; layer < shards.size(); layer++) {
            INFO("rank " << rank << ", KV layer " << layer);
            REQUIRE(same_layer(shards[layer].full, specs[layer]));
            const pulsarlm::kv& held = lines[rank][layer];
            CHECK(
                max_relative_error(held.k, shard_rows(single_lines[layer].k, shards[layer]).to(at::kFloat)) < tolerance
            );
            CHECK(
                max_relative_error(held.v, shard_rows(single_lines[layer].v, shards[layer]).to(at::kFloat)) < tolerance
            );
        }
    }
}

}  // namespace

TEST_CASE("Qwen3.5 matches a dense forward in fp32", "[model][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    check_against_dense(at::kFloat, 1e-4);
}

TEST_CASE("Qwen3.5 matches a dense forward in bf16", "[model][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    check_against_dense(at::kBFloat16, 5e-2);
}

TEST_CASE("Qwen3.5 with fp8 linears matches a dense forward of their dequantized weights", "[model][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    check_against_dense(at::kBFloat16, 5e-2, pulsarlm::fp8_block{.n = 32, .k = 32});
}

TEST_CASE("Qwen3.5 split over ranks matches one GPU in fp32", "[model][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const std::int64_t kv_heads = GENERATE(n_kv_heads, std::int64_t{1});
    INFO(kv_heads << " KV heads over 2 ranks");
    check_split_over_ranks(at::kFloat, 1e-4, 2, kv_heads);
}

TEST_CASE("Qwen3.5 with fp8 linears split over ranks matches one GPU", "[model][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const std::int64_t kv_heads = GENERATE(n_kv_heads, std::int64_t{1});
    INFO(kv_heads << " KV heads over 2 ranks");
    check_split_over_ranks(at::kHalf, 5e-2, 2, kv_heads, pulsarlm::fp8_block{.n = 32, .k = 32});
}
