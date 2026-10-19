#include "pulsarlm/model/gemma4.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAFunctions.h>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// Gemma4Model's paged forward against a dense ATen Gemma 4 written from Hugging Face's modeling code: chunked prefill
// then decode over pages scattered across the lines, checking the logits and the K and V the lines hold.

namespace {

using pulsarlm::gemma4_config;
using pulsarlm::gemma4_layer_type;

constexpr auto sliding = gemma4_layer_type::sliding_attention;
constexpr auto full = gemma4_layer_type::full_attention;
constexpr std::int64_t hidden = 128;
constexpr std::int64_t intermediate = 256;
constexpr std::int64_t vocab = 300;
constexpr std::int64_t page_size = 16;
const std::vector<std::int32_t> view_lines{3, 0, 5};
constexpr std::int64_t n_lines = 6;

at::TensorOptions cuda(at::ScalarType dtype) {
    return at::device(at::kCUDA).dtype(dtype);
}

at::Tensor int32_cuda(const std::vector<std::int32_t>& values) {
    return at::tensor(values, at::dtype(at::kInt)).to(at::kCUDA);
}

gemma4_config base_config() {
    return gemma4_config{
        .vocab_size = vocab,
        .hidden_size = hidden,
        .n_heads = 4,
        .n_kv_heads = 2,
        .head_dim = 64,
        .global_n_kv_heads = 2,
        .global_head_dim = 128,
        .attention_k_eq_v = false,
        .layer_types = {},
        .sliding_window = 6,
        .sliding_rope = {.theta = 1e4},
        .full_rope = {.theta = 1e6, .partial_rotary_factor = 0.25},
        .rms_norm_eps = 1e-6,
        .final_logit_softcapping = 30.0,
        .num_kv_shared_layers = 0,
        .hidden_size_per_layer_input = 0,
    };
}

// Per-layer inputs, and a sliding and a full layer reading earlier layers' KV through double-wide MLPs.
gemma4_config shared_kv_config() {
    gemma4_config c = base_config();
    c.layer_types = {sliding, sliding, full, sliding, full};
    c.num_kv_shared_layers = 2;
    c.hidden_size_per_layer_input = 16;
    return c;
}

// Full layers keeping one KV head as both K and V, sliding ones two.
gemma4_config k_eq_v_config() {
    gemma4_config c = base_config();
    c.layer_types = {sliding, sliding, full, sliding, sliding, full};
    c.attention_k_eq_v = true;
    c.global_n_kv_heads = 1;
    c.sliding_window = 20;
    return c;
}

constexpr std::int64_t n_experts = 8;
constexpr std::int64_t moe_intermediate = 32;

// Experts beside every MLP, on the K=V layout.
gemma4_config moe_config() {
    gemma4_config c = k_eq_v_config();
    c.layer_types = {sliding, sliding, full};
    c.global_n_kv_heads = 2;
    c.enable_moe_block = true;
    c.top_k_experts = 2;
    return c;
}

std::int64_t n_layers(const gemma4_config& c) {
    return static_cast<std::int64_t>(c.layer_types.size());
}

bool is_sliding(const gemma4_config& c, std::int64_t layer) {
    return c.layer_types[layer] == sliding;
}

bool keeps_kv(const gemma4_config& c, std::int64_t layer) {
    return layer < n_layers(c) - c.num_kv_shared_layers;
}

std::int64_t head_dim(const gemma4_config& c, std::int64_t layer) {
    return is_sliding(c, layer) ? c.head_dim : c.global_head_dim;
}

std::int64_t kv_heads(const gemma4_config& c, std::int64_t layer) {
    return is_sliding(c, layer) ? c.n_kv_heads : c.global_n_kv_heads;
}

bool k_eq_v(const gemma4_config& c, std::int64_t layer) {
    return c.attention_k_eq_v && !is_sliding(c, layer);
}

struct weights {
    c10::Dict<std::string, at::Tensor> named;

    weights(const gemma4_config& c, at::ScalarType dtype) {
        at::manual_seed(11);
        auto add = [&](const std::string& name, at::Tensor t) {
            named.insert("model.language_model." + name, t.to(cuda(dtype)));
        };
        auto projection = [](std::int64_t out, std::int64_t in) { return at::randn({out, in}) / std::sqrt(in); };
        auto norm = [](std::int64_t size) { return 1.0 + 0.1 * at::randn({size}); };
        add("embed_tokens.weight", at::randn({vocab, hidden}) / std::sqrt(hidden));
        add("norm.weight", norm(hidden));
        const std::int64_t width = c.hidden_size_per_layer_input;
        if (width > 0) {
            add("embed_tokens_per_layer.weight", at::randn({vocab, n_layers(c) * width}) / std::sqrt(width));
            add("per_layer_model_projection.weight", projection(n_layers(c) * width, hidden));
            add("per_layer_projection_norm.weight", norm(width));
        }
        for (std::int64_t layer = 0; layer < n_layers(c); layer++) {
            std::string p = "layers." + std::to_string(layer) + ".";
            std::int64_t dim = head_dim(c, layer);
            std::int64_t mlp = keeps_kv(c, layer) ? intermediate : 2 * intermediate;
            add(p + "input_layernorm.weight", norm(hidden));
            add(p + "post_attention_layernorm.weight", norm(hidden));
            add(p + "pre_feedforward_layernorm.weight", norm(hidden));
            add(p + "post_feedforward_layernorm.weight", norm(hidden));
            add(p + "self_attn.q_proj.weight", projection(c.n_heads * dim, hidden));
            // At attention scale 1, unit-norm q and k give scores of deviation sqrt(dim), and bf16's rounding of a
            // key then moves the softmax past any tolerance; quarter-scale norms keep it comparable.
            add(p + "self_attn.q_norm.weight", 0.25 * norm(dim));
            add(p + "self_attn.k_proj.weight", projection(kv_heads(c, layer) * dim, hidden));
            add(p + "self_attn.k_norm.weight", 0.25 * norm(dim));
            if (!k_eq_v(c, layer)) {
                add(p + "self_attn.v_proj.weight", projection(kv_heads(c, layer) * dim, hidden));
            }
            add(p + "self_attn.o_proj.weight", projection(hidden, c.n_heads * dim));
            add(p + "mlp.gate_proj.weight", projection(mlp, hidden));
            add(p + "mlp.up_proj.weight", projection(mlp, hidden));
            add(p + "mlp.down_proj.weight", projection(hidden, mlp));
            add(p + "layer_scalar", at::full({1}, 0.5 + 0.1 * static_cast<double>(layer)));
            if (width > 0) {
                add(p + "per_layer_input_gate.weight", projection(width, hidden));
                add(p + "per_layer_projection.weight", projection(hidden, width));
                add(p + "post_per_layer_input_norm.weight", norm(hidden));
            }
            if (c.enable_moe_block) {
                add(p + "router.proj.weight", projection(n_experts, hidden));
                add(p + "router.scale", norm(hidden));
                add(p + "router.per_expert_scale", norm(n_experts));
                add(p + "experts.gate_up_proj",
                    at::randn({n_experts, 2 * moe_intermediate, hidden}) / std::sqrt(hidden));
                add(p + "experts.down_proj",
                    at::randn({n_experts, hidden, moe_intermediate}) / std::sqrt(moe_intermediate));
                add(p + "pre_feedforward_layernorm_2.weight", norm(hidden));
                add(p + "post_feedforward_layernorm_1.weight", norm(hidden));
                add(p + "post_feedforward_layernorm_2.weight", norm(hidden));
            }
        }
    }

    at::Tensor operator()(const std::string& name) const {
        return named.at("model.language_model." + name).to(at::kFloat);
    }

    at::Tensor layer(std::int64_t layer, const std::string& name) const {
        return (*this)("layers." + std::to_string(layer) + "." + name);
    }
};

at::Tensor rms_norm(const at::Tensor& x, double eps, const std::optional<at::Tensor>& weight = std::nullopt) {
    at::Tensor normed = x * at::pow(x.pow(2).mean(-1, true) + eps, -0.5);
    return weight ? normed * *weight : normed;
}

// Hugging Face's rotate_half RoPE with "proportional" partial rotation, angles in fp64. x is [tokens, heads, dim].
at::Tensor rope(const at::Tensor& x, const pulsarlm::rope_parameters& parameters) {
    const std::int64_t dim = x.size(-1), half = dim / 2;
    const auto rotated_pairs = static_cast<std::int64_t>(std::floor(parameters.partial_rotary_factor * dim / 2));
    auto pairs = at::arange(half, cuda(at::kDouble));
    auto inverse_frequency = at::pow(parameters.theta, -2.0 * pairs / static_cast<double>(dim));
    inverse_frequency = at::where(pairs < rotated_pairs, inverse_frequency, at::zeros_like(inverse_frequency));
    auto angles = at::outer(at::arange(x.size(0), cuda(at::kDouble)), inverse_frequency);
    angles = at::cat({angles, angles}, -1).unsqueeze(1);
    auto rotated = at::cat({-x.narrow(-1, half, half), x.narrow(-1, 0, half)}, -1);
    return x * angles.cos().to(at::kFloat) + rotated * angles.sin().to(at::kFloat);
}

struct dense_output {
    at::Tensor logits;  // [tokens, vocab]
    // Per layer keeping KV, [tokens, kv heads, head dim]: K unrotated and V, or V alone where K=V.
    std::vector<at::Tensor> k;
    std::vector<at::Tensor> v;
};

// The dtype's rounding of scale.
double rounded(double scale, at::ScalarType dtype) {
    return at::scalar_tensor(scale, at::dtype(dtype)).item<double>();
}

dense_output dense_forward(const gemma4_config& c, const weights& w, const at::Tensor& tokens, at::ScalarType dtype) {
    const std::int64_t n = tokens.size(0);
    const double eps = c.rms_norm_eps;
    dense_output out;
    auto x = w("embed_tokens.weight").index_select(0, tokens) * rounded(std::sqrt(hidden), dtype);
    at::Tensor per_layer;
    const std::int64_t width = c.hidden_size_per_layer_input;
    if (width > 0) {
        auto by_token = (w("embed_tokens_per_layer.weight").index_select(0, tokens) *
                         rounded(std::sqrt(static_cast<double>(width)), dtype))
                            .view({n, n_layers(c), width});
        auto projected = (at::linear(x, w("per_layer_model_projection.weight")) * std::pow(hidden, -0.5))
                             .view({n, n_layers(c), width});
        projected = rms_norm(projected, eps, w("per_layer_projection_norm.weight"));
        per_layer = (projected + by_token) * std::pow(2.0, -0.5);
    }
    auto positions = at::arange(n, cuda(at::kLong));
    auto distance = positions.unsqueeze(1) - positions.unsqueeze(0);
    std::vector<std::int64_t> kv_of;
    for (std::int64_t layer = 0; layer < n_layers(c); layer++) {
        std::int64_t source = layer;
        if (!keeps_kv(c, layer)) {
            source = n_layers(c) - c.num_kv_shared_layers - 1;
            while (is_sliding(c, source) != is_sliding(c, layer)) {
                source--;
            }
        }
        kv_of.push_back(source);
        const std::int64_t dim = head_dim(c, layer), heads = kv_heads(c, layer);
        const pulsarlm::rope_parameters& parameters = is_sliding(c, layer) ? c.sliding_rope : c.full_rope;

        auto h = rms_norm(x, eps, w.layer(layer, "input_layernorm.weight"));
        auto q = rms_norm(
            at::linear(h, w.layer(layer, "self_attn.q_proj.weight")).view({n, c.n_heads, dim}),
            eps,
            w.layer(layer, "self_attn.q_norm.weight")
        );
        if (keeps_kv(c, layer)) {
            auto raw_k = at::linear(h, w.layer(layer, "self_attn.k_proj.weight")).view({n, heads, dim});
            auto raw_v = k_eq_v(c, layer)
                ? raw_k
                : at::linear(h, w.layer(layer, "self_attn.v_proj.weight")).view({n, heads, dim});
            out.k.push_back(rms_norm(raw_k, eps, w.layer(layer, "self_attn.k_norm.weight")));
            out.v.push_back(rms_norm(raw_v, eps));
        }
        auto group = c.n_heads / heads;
        auto q_heads = rope(q, parameters).transpose(0, 1);
        auto k_heads = rope(out.k[source], parameters).repeat_interleave(group, 1).transpose(0, 1);
        auto v_heads = out.v[source].repeat_interleave(group, 1).transpose(0, 1);
        auto attended = distance >= 0;
        if (is_sliding(c, layer)) {
            attended = attended.logical_and(distance < c.sliding_window);
        }
        auto scores = at::matmul(q_heads, k_heads.transpose(1, 2));
        auto probabilities = at::softmax(scores.masked_fill(attended.logical_not(), -INFINITY), -1);
        auto o = at::matmul(probabilities, v_heads).transpose(0, 1).reshape({n, c.n_heads * dim});
        x = x +
            rms_norm(
                at::linear(o, w.layer(layer, "self_attn.o_proj.weight")),
                eps,
                w.layer(layer, "post_attention_layernorm.weight")
            );

        auto h2 = rms_norm(x, eps, w.layer(layer, "pre_feedforward_layernorm.weight"));
        auto gate = at::gelu(at::linear(h2, w.layer(layer, "mlp.gate_proj.weight")), "tanh");
        auto mlp = at::linear(
            gate * at::linear(h2, w.layer(layer, "mlp.up_proj.weight")),
            w.layer(layer, "mlp.down_proj.weight")
        );
        if (c.enable_moe_block) {
            auto routed = rms_norm(x, eps) * w.layer(layer, "router.scale") * std::pow(hidden, -0.5);
            auto probabilities = at::softmax(at::linear(routed, w.layer(layer, "router.proj.weight")), -1);
            auto [top_weights, top_experts] = at::topk(probabilities, c.top_k_experts, -1);
            top_weights = top_weights / top_weights.sum(-1, true) *
                w.layer(layer, "router.per_expert_scale")
                    .index_select(0, top_experts.flatten())
                    .view(top_experts.sizes());
            auto expert_input = rms_norm(x, eps, w.layer(layer, "pre_feedforward_layernorm_2.weight"));
            auto gate_up = w.layer(layer, "experts.gate_up_proj");
            auto down = w.layer(layer, "experts.down_proj");
            auto experts_out = at::zeros_like(x);
            for (std::int64_t expert = 0; expert < n_experts; expert++) {
                auto chosen = top_experts.eq(expert);
                auto tokens_of = chosen.any(-1).nonzero().flatten();
                if (tokens_of.numel() == 0) {
                    continue;
                }
                auto weight = (top_weights * chosen).sum(-1).index_select(0, tokens_of).unsqueeze(1);
                auto halves = at::linear(expert_input.index_select(0, tokens_of), gate_up[expert]).chunk(2, -1);
                auto out = at::linear(at::gelu(halves[0], "tanh") * halves[1], down[expert]);
                experts_out.index_add_(0, tokens_of, out * weight);
            }
            mlp = rms_norm(mlp, eps, w.layer(layer, "post_feedforward_layernorm_1.weight")) +
                rms_norm(experts_out, eps, w.layer(layer, "post_feedforward_layernorm_2.weight"));
        }
        x = x + rms_norm(mlp, eps, w.layer(layer, "post_feedforward_layernorm.weight"));
        if (width > 0) {
            auto gated = at::gelu(at::linear(x, w.layer(layer, "per_layer_input_gate.weight")), "tanh") *
                per_layer.select(1, layer);
            x = x +
                rms_norm(
                    at::linear(gated, w.layer(layer, "per_layer_projection.weight")),
                    eps,
                    w.layer(layer, "post_per_layer_input_norm.weight")
                );
        }
        x = x * w.layer(layer, "layer_scalar");
    }
    auto logits = at::matmul(rms_norm(x, eps, w("norm.weight")), w("embed_tokens.weight").t());
    out.logits = at::tanh(logits / *c.final_logit_softcapping) * *c.final_logit_softcapping;
    return out;
}

// A step of the single sequence forwarding the tokens at positions [first, last), every KV layer over the same lines.
pulsarlm::GroupBatch
step(const at::Tensor& tokens, std::int64_t n_kv_layers, std::int64_t first, std::int64_t last, bool prefill) {
    auto key_count = int32_cuda({static_cast<std::int32_t>(last)}).expand({n_kv_layers, 1}).contiguous();
    pulsarlm::GroupBatch g{
        .tokens = tokens.narrow(0, first, last - first),
        .pos = at::arange(first, last, cuda(at::kLong)),
        .page_tables = int32_cuda(view_lines).view({1, 1, -1}).expand({n_kv_layers, 1, -1}).contiguous(),
        .page_positions = (at::arange(static_cast<std::int64_t>(view_lines.size()), cuda(at::kInt)) * page_size)
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

void check_against_dense(const gemma4_config& c, at::ScalarType dtype, double tolerance) {
    weights w(c, dtype);
    pulsarlm::Gemma4Model model(c, w.named);
    std::vector<pulsarlm::kv_layer> layers = model.kv_layers();
    REQUIRE(static_cast<std::int64_t>(layers.size()) == n_layers(c) - c.num_kv_shared_layers);
    std::vector<pulsarlm::kv> lines;
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        auto shape = std::vector<std::int64_t>{
            n_lines,
            page_size,
            static_cast<std::int64_t>(layers[layer].n_kv_heads),
            static_cast<std::int64_t>(layers[layer].head_dim)
        };
        at::Tensor v = at::zeros(shape, cuda(dtype));
        lines.push_back(pulsarlm::kv{.k = layers[layer].values_only ? v : at::zeros(shape, cuda(dtype)), .v = v});
    }

    const std::int64_t n_tokens = 40;
    auto tokens = at::randint(vocab, {n_tokens}, cuda(at::kLong));
    dense_output dense = dense_forward(c, w, tokens, dtype);

    struct chunk {
        std::int64_t first;
        std::int64_t last;
        bool prefill;
    };
    for (chunk ch :
         {chunk{0, 21, true}, chunk{21, 37, true}, chunk{37, 38, false}, chunk{38, 39, false}, chunk{39, 40, false}}) {
        INFO("tokens [" << ch.first << ", " << ch.last << ")" << (ch.prefill ? " prefill" : " decode"));
        pulsarlm::StepBatch batch;
        (ch.prefill
             ? batch.prefill
             : batch.decode) = step(tokens, static_cast<std::int64_t>(layers.size()), ch.first, ch.last, ch.prefill);
        pulsarlm::StepLogits logits = model.forward(batch, lines);
        auto expected = dense.logits.narrow(0, ch.first, ch.last - ch.first);
        CHECK(max_relative_error(*(ch.prefill ? logits.prefill : logits.decode), expected) < tolerance);
    }

    auto view = at::tensor(std::vector<std::int64_t>(view_lines.begin(), view_lines.end())).to(at::kCUDA);
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        INFO("KV layer " << layer);
        auto stored = [&](const at::Tensor& pool) {
            return pool.index_select(0, view).flatten(0, 1).narrow(0, 0, n_tokens);
        };
        CHECK(max_relative_error(stored(lines[layer].v), dense.v[layer]) < tolerance);
        if (!layers[layer].values_only) {
            CHECK(max_relative_error(stored(lines[layer].k), dense.k[layer]) < tolerance);
        }
    }
}

}  // namespace

TEST_CASE("Gemma 4 kv_layers name the layers keeping KV, with their windows and K=V", "[model][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    gemma4_config shared = shared_kv_config();
    std::vector<pulsarlm::kv_layer> layers = pulsarlm::Gemma4Model(shared, weights(shared, at::kFloat).named)
                                                 .kv_layers();
    REQUIRE(layers.size() == 3);
    CHECK((layers[0].sliding_window == std::optional<std::size_t>(6) && layers[0].head_dim == 64));
    CHECK((!layers[2].sliding_window && layers[2].head_dim == 128 && !layers[2].values_only));

    gemma4_config unified = k_eq_v_config();
    layers = pulsarlm::Gemma4Model(unified, weights(unified, at::kFloat).named).kv_layers();
    REQUIRE(layers.size() == 6);
    CHECK((layers[5].values_only && layers[5].n_kv_heads == 1 && !layers[0].values_only));
}

TEST_CASE("Gemma 4 with per-layer inputs and shared KV matches a dense forward", "[model][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    SECTION("fp32") {
        check_against_dense(shared_kv_config(), at::kFloat, 1e-4);
    }
    SECTION("bf16") {
        check_against_dense(shared_kv_config(), at::kBFloat16, 5e-2);
    }
}

TEST_CASE("Gemma 4 with K=V full layers matches a dense forward", "[model][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    SECTION("fp32") {
        check_against_dense(k_eq_v_config(), at::kFloat, 1e-4);
    }
    SECTION("bf16") {
        check_against_dense(k_eq_v_config(), at::kBFloat16, 5e-2);
    }
}

TEST_CASE("Gemma 4 with experts beside every MLP matches a dense forward", "[model][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    SECTION("fp32") {
        check_against_dense(moe_config(), at::kFloat, 1e-4);
    }
    // A token's top experts change with bf16's rounding of its router logits, so in bf16 every token takes every
    // expert.
    SECTION("bf16") {
        gemma4_config c = moe_config();
        c.top_k_experts = n_experts;
        check_against_dense(c, at::kBFloat16, 5e-2);
    }
}
