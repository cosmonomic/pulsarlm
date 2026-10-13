#include "pulsar/model/qwen3.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAFunctions.h>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

// Qwen3Model's paged forward against a dense ATen Qwen3: chunked prefill then decode over pages scattered across the
// lines, checking the logits, the unrotated K the lines hold, and the per-page attention mass.

namespace {

constexpr std::int64_t n_layers = 2;
constexpr std::int64_t n_heads = 4;
constexpr std::int64_t n_kv_heads = 2;
constexpr std::int64_t head_dim = 64;
constexpr std::int64_t hidden = 128;
constexpr std::int64_t intermediate = 256;
constexpr std::int64_t vocab = 300;
constexpr double rope_theta = 1e6;
constexpr double rms_eps = 1e-6;
constexpr std::int64_t page_size = 16;
constexpr double mass_decay = 0.25;
const std::vector<std::int64_t> relevance_layers{0, 1};
const std::vector<std::int32_t> view_lines{3, 0, 5};
constexpr std::int64_t n_lines = 6;

at::TensorOptions cuda(at::ScalarType dtype) {
    return at::device(at::kCUDA).dtype(dtype);
}

at::Tensor int32_cuda(const std::vector<std::int32_t>& values) {
    return at::tensor(values, at::dtype(at::kInt)).to(at::kCUDA);
}

struct weights {
    c10::Dict<std::string, at::Tensor> named;

    explicit weights(at::ScalarType dtype) {
        at::manual_seed(11);
        auto add = [&](const std::string& name, at::Tensor t) { named.insert(name, t.to(cuda(dtype))); };
        auto projection = [](std::int64_t out, std::int64_t in) { return at::randn({out, in}) / std::sqrt(in); };
        auto norm = [](std::int64_t size) { return 1.0 + 0.1 * at::randn({size}); };
        add("model.embed_tokens.weight", at::randn({vocab, hidden}));
        add("model.norm.weight", norm(hidden));
        for (std::int64_t layer = 0; layer < n_layers; layer++) {
            std::string p = "model.layers." + std::to_string(layer) + ".";
            add(p + "input_layernorm.weight", norm(hidden));
            add(p + "post_attention_layernorm.weight", norm(hidden));
            add(p + "self_attn.q_proj.weight", projection(n_heads * head_dim, hidden));
            add(p + "self_attn.k_proj.weight", projection(n_kv_heads * head_dim, hidden));
            add(p + "self_attn.v_proj.weight", projection(n_kv_heads * head_dim, hidden));
            add(p + "self_attn.o_proj.weight", projection(hidden, n_heads * head_dim));
            add(p + "self_attn.q_norm.weight", norm(head_dim));
            add(p + "self_attn.k_norm.weight", norm(head_dim));
            add(p + "mlp.gate_proj.weight", projection(intermediate, hidden));
            add(p + "mlp.up_proj.weight", projection(intermediate, hidden));
            add(p + "mlp.down_proj.weight", projection(hidden, intermediate));
        }
    }

    at::Tensor operator()(const std::string& name) const {
        return named.at(name).to(at::kFloat);
    }

    at::Tensor layer(std::int64_t layer, const std::string& name) const {
        return (*this)("model.layers." + std::to_string(layer) + "." + name);
    }
};

at::Tensor rms_norm(const at::Tensor& x, const at::Tensor& weight) {
    return x * at::rsqrt(x.pow(2).mean(-1, true) + rms_eps) * weight;
}

// Hugging Face's rotate_half RoPE, angles in fp64. x is [tokens, heads, head_dim].
at::Tensor rope(const at::Tensor& x) {
    auto exponents = at::arange(0, head_dim, 2, cuda(at::kDouble)) / static_cast<double>(head_dim);
    auto inverse_frequency = at::pow(rope_theta, -exponents);
    auto positions = at::arange(x.size(0), cuda(at::kDouble));
    auto angles = at::outer(positions, inverse_frequency);
    angles = at::cat({angles, angles}, -1).unsqueeze(1);
    auto cos = angles.cos().to(at::kFloat);
    auto sin = angles.sin().to(at::kFloat);
    auto half = head_dim / 2;
    auto rotated = at::cat({-x.narrow(-1, half, half), x.narrow(-1, 0, half)}, -1);
    return x * cos + rotated * sin;
}

struct dense_output {
    at::Tensor logits;  // [tokens, vocab]
    std::vector<at::Tensor> unrotated_k;  // per layer [tokens, n_kv_heads, head_dim]
    std::vector<at::Tensor> attention;  // per layer [n_heads, tokens, tokens], query by key
};

dense_output dense_forward(const weights& w, const at::Tensor& tokens) {
    const std::int64_t n = tokens.size(0);
    dense_output out;
    auto x = w("model.embed_tokens.weight").index_select(0, tokens);
    auto causal = at::ones({n, n}, cuda(at::kBool)).tril();
    for (std::int64_t layer = 0; layer < n_layers; layer++) {
        auto h = rms_norm(x, w.layer(layer, "input_layernorm.weight"));
        auto q = at::linear(h, w.layer(layer, "self_attn.q_proj.weight")).view({n, n_heads, head_dim});
        auto k = at::linear(h, w.layer(layer, "self_attn.k_proj.weight")).view({n, n_kv_heads, head_dim});
        auto v = at::linear(h, w.layer(layer, "self_attn.v_proj.weight")).view({n, n_kv_heads, head_dim});
        q = rms_norm(q, w.layer(layer, "self_attn.q_norm.weight"));
        k = rms_norm(k, w.layer(layer, "self_attn.k_norm.weight"));
        out.unrotated_k.push_back(k);
        auto group = n_heads / n_kv_heads;
        auto q_heads = rope(q).transpose(0, 1);
        auto k_heads = rope(k).repeat_interleave(group, 1).transpose(0, 1);
        auto v_heads = v.repeat_interleave(group, 1).transpose(0, 1);
        auto scores = at::matmul(q_heads, k_heads.transpose(1, 2)) / std::sqrt(static_cast<double>(head_dim));
        auto probabilities = at::softmax(scores.masked_fill(causal.logical_not(), -INFINITY), -1);
        out.attention.push_back(probabilities);
        auto o = at::matmul(probabilities, v_heads).transpose(0, 1).reshape({n, n_heads * head_dim});
        x = x + at::linear(o, w.layer(layer, "self_attn.o_proj.weight"));
        auto h2 = rms_norm(x, w.layer(layer, "post_attention_layernorm.weight"));
        auto gate = at::linear(h2, w.layer(layer, "mlp.gate_proj.weight"));
        auto up = at::linear(h2, w.layer(layer, "mlp.up_proj.weight"));
        x = x + at::linear(at::silu(gate) * up, w.layer(layer, "mlp.down_proj.weight"));
    }
    out.logits = at::matmul(rms_norm(x, w("model.norm.weight")), w("model.embed_tokens.weight").t());
    return out;
}

// The page mass of a step forwarding the queries at positions [first, last): each query's attention weights, graded
// by the decay over the step's later queries and stated against its own key count, summed over queries, averaged
// over heads, maxed over each page's slots and over the relevance layers.
at::Tensor dense_page_mass(const dense_output& dense, std::int64_t first, std::int64_t last) {
    std::int64_t view_pages = static_cast<std::int64_t>(view_lines.size());
    auto page_mass = at::zeros({view_pages}, cuda(at::kFloat));
    for (std::int64_t layer : relevance_layers) {
        auto slot_mass = at::zeros({view_pages * page_size, n_heads}, cuda(at::kFloat));
        for (std::int64_t p = first; p < last; p++) {
            double graded = mass_decay * std::pow(1.0 - mass_decay, static_cast<double>(last - 1 - p)) * (p + 1);
            auto weights = dense.attention[layer].select(1, p).narrow(1, 0, last).t();
            slot_mass.narrow(0, 0, last).add_(weights * graded);
        }
        auto layer_mass = slot_mass.mean(-1).view({view_pages, page_size}).amax(-1);
        page_mass = at::maximum(page_mass, layer_mass);
    }
    return page_mass;
}

// A step of the single sequence forwarding the tokens at positions [first, last).
pulsar::GroupBatch step(const at::Tensor& tokens, std::int64_t first, std::int64_t last, bool prefill) {
    std::vector<std::int32_t> slots;
    for (std::int64_t p = first; p < last; p++) {
        slots.push_back(view_lines[p / page_size] * page_size + static_cast<std::int32_t>(p % page_size));
    }
    auto key_count = int32_cuda({static_cast<std::int32_t>(last)});
    pulsar::GroupBatch g{
        .tokens = tokens.narrow(0, first, last - first),
        .pos = at::arange(first, last, cuda(at::kLong)),
        .slot_mapping = int32_cuda(slots),
        .page_tables = int32_cuda(view_lines).view({1, -1}),
        .rope_layout = int32_cuda({0, 0, -1}).view({1, 3}),
        .cu_view_pages = int32_cuda({0, static_cast<std::int32_t>(view_lines.size())}),
        .page_mass = at::zeros({static_cast<std::int64_t>(view_lines.size())}, cuda(at::kFloat)),
        .attention_mass_decay = at::full({1}, mass_decay, cuda(at::kFloat)),
    };
    if (prefill) {
        g.cu_seqlens_q = int32_cuda({0, static_cast<std::int32_t>(last - first)});
        g.seqlens_k = key_count;
    } else {
        g.context_lens = key_count;
    }
    return g;
}

double max_relative_error(const at::Tensor& got, const at::Tensor& expected) {
    return ((got.to(at::kFloat) - expected).abs().max() / expected.abs().max()).item<double>();
}

void check_against_dense(at::ScalarType dtype, double tolerance) {
    weights w(dtype);
    pulsar::Qwen3Model
        model(n_layers, n_heads, n_kv_heads, head_dim, hidden, intermediate, vocab, rope_theta, rms_eps, w.named);
    model.set_relevance_layers(relevance_layers);
    pulsar::kv_lines kv{
        .k = at::zeros({n_layers, n_lines, page_size, n_kv_heads, head_dim}, cuda(dtype)),
        .v = at::zeros({n_layers, n_lines, page_size, n_kv_heads, head_dim}, cuda(dtype)),
    };

    const std::int64_t n_tokens = 40;
    auto tokens = at::randint(vocab, {n_tokens}, cuda(at::kLong));
    dense_output dense = dense_forward(w, tokens);

    struct chunk {
        std::int64_t first;
        std::int64_t last;
        bool prefill;
    };
    for (chunk c :
         {chunk{0, 21, true}, chunk{21, 37, true}, chunk{37, 38, false}, chunk{38, 39, false}, chunk{39, 40, false}}) {
        INFO("tokens [" << c.first << ", " << c.last << ")" << (c.prefill ? " prefill" : " decode"));
        pulsar::StepBatch batch;
        (c.prefill ? batch.prefill : batch.decode) = step(tokens, c.first, c.last, c.prefill);
        pulsar::StepLogits logits = model.forward(batch, kv);
        const pulsar::GroupBatch& g = c.prefill ? *batch.prefill : *batch.decode;

        auto expected_logits = dense.logits.narrow(0, c.first, c.last - c.first);
        CHECK(max_relative_error(*(c.prefill ? logits.prefill : logits.decode), expected_logits) < tolerance);
        CHECK(max_relative_error(g.page_mass, dense_page_mass(dense, c.first, c.last)) < tolerance);
    }

    auto lines = at::tensor(std::vector<std::int64_t>(view_lines.begin(), view_lines.end())).to(at::kCUDA);
    for (std::int64_t layer = 0; layer < n_layers; layer++) {
        auto stored = kv.k.select(0, layer).index_select(0, lines).view({-1, n_kv_heads, head_dim});
        CHECK(max_relative_error(stored.narrow(0, 0, n_tokens), dense.unrotated_k[layer]) < tolerance);
    }
}

}  // namespace

TEST_CASE("Qwen3 matches a dense forward in fp32", "[model][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    check_against_dense(at::kFloat, 1e-4);
}

TEST_CASE("Qwen3 matches a dense forward in bf16", "[model][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    check_against_dense(at::kBFloat16, 1e-2);
}
