#include "pulsarlm/engine/model/qwen3.hpp"

#include "pulsarlm/engine/request_table.hpp"
#include "pulsarlm/engine/step_graphs.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

// qwen3_model's paged forward against a dense ATen Qwen3: chunked prefill then decode over pages scattered across the
// lines, checking the logits, the unrotated K the lines hold, and the per-page attention mass. The decode steps forward
// through the model or through step_graphs, which gathers them from a request table.

namespace {

// The model's forward of b, its calls between segments run at once.
pulsarlm::model::output
forward_eagerly(pulsarlm::model& forwarded, const pulsarlm::model::batch& b, std::span<const pulsarlm::kv> lines) {
    pulsarlm::graph_segments segments(b.tokens.size(0));
    return forwarded.forward(b, lines, segments);
}

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
const std::vector<std::int32_t> view_lines{3, 0, 5};
constexpr std::int64_t n_lines = 6;

at::TensorOptions cuda(at::ScalarType dtype) {
    return at::device(at::kCUDA).dtype(dtype);
}

at::Tensor int32_cuda(const std::vector<std::int32_t>& values) {
    return at::tensor(values, at::dtype(at::kInt)).to(at::kCUDA);
}

struct weights {
    c10::Dict<std::string, at::Tensor> named_tensors;

    explicit weights(at::ScalarType dtype) {
        at::manual_seed(11);
        auto add = [&](const std::string& name, at::Tensor t) { named_tensors.insert(name, t.to(cuda(dtype))); };
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
        return named_tensors.at(name).to(at::kFloat);
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
    std::vector<at::Tensor> scores;  // per layer [n_heads, tokens, tokens], query by key, -inf for unattended keys
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
        auto attended_scores = scores.masked_fill(causal.logical_not(), -INFINITY);
        out.scores.push_back(attended_scores);
        auto probabilities = at::softmax(attended_scores, -1);
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

// The query weight of the query at position p of a step forwarding the queries at positions [first, last).
double query_weight(std::int64_t p, std::int64_t last) {
    return 0.25 * std::pow(0.75, static_cast<double>(last - 1 - p));
}

// Layer layer's page mass_sum [view pages, n_heads] and weight_sum [view pages] of a step forwarding the queries at
// positions [first, last): per page and head, the sum over the queries attending the page of their query weight times
// their max scaled score over its keys, and per page the sum of those weights.
struct dense_mass {
    at::Tensor mass_sum;
    at::Tensor weight_sum;
};

dense_mass dense_page_mass(const dense_output& dense, std::int64_t layer, std::int64_t first, std::int64_t last) {
    std::int64_t view_pages = static_cast<std::int64_t>(view_lines.size());
    dense_mass mass{
        .mass_sum = at::zeros({view_pages, n_heads}, cuda(at::kFloat)),
        .weight_sum = at::zeros({view_pages}, cuda(at::kFloat)),
    };
    for (std::int64_t p = first; p < last; p++) {
        std::int64_t attended_pages = p / page_size + 1;
        auto scores = at::full({n_heads, view_pages * page_size}, -INFINITY, cuda(at::kFloat));
        scores.narrow(1, 0, p + 1).copy_(dense.scores[layer].select(1, p).narrow(1, 0, p + 1));
        auto page_max = scores.view({n_heads, view_pages, page_size}).amax(-1).t().narrow(0, 0, attended_pages);
        mass.mass_sum.narrow(0, 0, attended_pages).add_(page_max * query_weight(p, last));
        mass.weight_sum.narrow(0, 0, attended_pages).add_(query_weight(p, last));
    }
    return mass;
}

// t for every layer: [n_layers, t's sizes...].
at::Tensor every_layer(const at::Tensor& t) {
    std::vector<std::int64_t> sizes{n_layers};
    sizes.insert(sizes.end(), t.sizes().begin(), t.sizes().end());
    return t.unsqueeze(0).expand(sizes).contiguous();
}

// A step of the single sequence forwarding the tokens at positions [first, last).
pulsarlm::model::batch step(const at::Tensor& tokens, std::int64_t first, std::int64_t last, bool prefill) {
    auto key_count = every_layer(int32_cuda({static_cast<std::int32_t>(last)}));
    pulsarlm::model::batch g{
        .single_query_request_count = prefill ? 0 : 1,
        .tokens = tokens.narrow(0, first, last - first),
        .pos = at::arange(first, last, cuda(at::kLong)),
        .page_tables = every_layer(int32_cuda(view_lines).view({1, -1})),
        .history_positions = at::full({1}, first, cuda(at::kLong)),
        .rejected_tokens = int32_cuda({0}),
        .page_size = page_size,
        .page_positions = every_layer(
            (at::arange(static_cast<std::int64_t>(view_lines.size()), cuda(at::kInt)) * page_size).view({1, -1})
        ),
        .cu_view_pages = every_layer(int32_cuda({0, static_cast<std::int32_t>(view_lines.size())})),
    };
    std::vector<float> weights;
    for (std::int64_t p = first; p < last; p++) {
        weights.push_back(static_cast<float>(query_weight(p, last)));
    }
    g.query_weights = at::tensor(weights).to(at::kCUDA);
    for (std::int64_t layer = 0; layer < n_layers; layer++) {
        g.page_mass_sum.push_back(at::zeros({static_cast<std::int64_t>(view_lines.size()), n_heads}, cuda(at::kFloat)));
        g.page_weight_sum.push_back(at::zeros({static_cast<std::int64_t>(view_lines.size())}, cuda(at::kFloat)));
    }
    g.key_counts = key_count;
    g.cu_seqlens_q = int32_cuda({0, static_cast<std::int32_t>(last - first)});
    g.cu_seqlens_q_from_second_run = prefill ? g.cu_seqlens_q : int32_cuda({0});
    g.cu_seqlens_q_from_third_run = g.cu_seqlens_q_from_second_run;
    g.logit_tokens = at::arange(last - first, cuda(at::kLong));
    return g;
}

// The request table of one request at slot 0, its view's pages those of step, its decode queries weighing as
// query_weight weighs a step's last query.
pulsarlm::request_table make_table() {
    const auto pages = static_cast<std::int64_t>(view_lines.size());
    return pulsarlm::request_table{
        .page_tables = every_layer(int32_cuda(view_lines)).unsqueeze(0).contiguous(),
        .page_positions = every_layer(at::arange(pages, cuda(at::kInt)) * page_size).unsqueeze(0).contiguous(),
        .history_positions = at::zeros({1}, cuda(at::kLong)),
        .rope_positions = at::zeros({1}, cuda(at::kLong)),
        .key_counts = at::zeros({1, n_layers}, cuda(at::kInt)),
        .recurrent_layers = at::zeros({n_layers}, cuda(at::kBool)),
        .page_size = page_size,
        .rejected_tokens = at::zeros({1}, cuda(at::kInt)),
        .tokens = at::zeros({1, 1}, cuda(at::kLong)),
        .cycle_end_positions = at::zeros({1}, cuda(at::kLong)),
        .mass_start_positions = at::zeros({1}, cuda(at::kLong)),
        .cycle_tokens = at::zeros({1}, cuda(at::kLong)),
        .mass_tokens = at::zeros({1}, cuda(at::kLong)),
        .attention_mass_decays = at::full({1}, 0.25, cuda(at::kDouble)),
        .remaining_tokens = at::full({1}, std::numeric_limits<std::int64_t>::max(), cuda(at::kLong)),
        .seeds = at::zeros({1}, cuda(at::kLong)),
        .draw_counts = at::zeros({1}, cuda(at::kLong)),
        .temperatures = at::zeros({1}, cuda(at::kFloat)),
        .top_k = at::zeros({1}, cuda(at::kLong)),
        .top_p = at::zeros({1}, cuda(at::kFloat)),
    };
}

// Sets table's request to the one whose next token is tokens[position].
void set_next_token(const pulsarlm::request_table& table, const at::Tensor& tokens, std::int64_t position) {
    table.history_positions.fill_(position);
    table.rope_positions.fill_(position);
    table.key_counts.fill_(position);
    table.tokens.copy_(tokens.narrow(0, position, 1).view({1, 1}));
}

double max_relative_error(const at::Tensor& got, const at::Tensor& expected) {
    return ((got.to(at::kFloat) - expected).abs().max() / expected.abs().max()).item<double>();
}

void check_against_dense(at::ScalarType dtype, double tolerance, bool graphs) {
    // A CUDA graph is captured on a stream other than the default one.
    c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromPool());
    weights w(dtype);
    pulsarlm::qwen3_model model(
        pulsarlm::dense_config{
            .vocab_size = vocab,
            .n_layers = n_layers,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .rope = {.theta = rope_theta},
            .rms_norm_eps = rms_eps,
        },
        w.named_tensors
    );
    std::vector<pulsarlm::kv> kv;
    for (std::int64_t layer = 0; layer < n_layers; layer++) {
        kv.push_back(
            pulsarlm::kv{
                .k = at::zeros({n_lines, page_size, n_kv_heads, head_dim}, cuda(dtype)),
                .v = at::zeros({n_lines, page_size, n_kv_heads, head_dim}, cuda(dtype)),
            }
        );
    }

    const pulsarlm::request_table table = make_table();
    pulsarlm::step_graphs decode(
        model,
        kv,
        view_lines.size(),
        page_size,
        /*state_slots=*/1,
        std::vector<std::size_t>(n_layers, static_cast<std::size_t>(n_heads)),
        at::empty({0}, cuda(at::kLong)),
        table,
        /*bucket_token_counts=*/{}
    );
    const at::Tensor slot = at::zeros({1}, cuda(at::kLong));

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
        pulsarlm::model::batch batch = step(tokens, c.first, c.last, c.prefill);
        const pulsarlm::model::batch& g = batch;
        auto expected_logits = dense.logits.narrow(0, c.first, c.last - c.first);
        if (graphs && !c.prefill) {
            set_next_token(table, tokens, c.first);
            const at::Tensor sampled = decode
                                           .decode_step(
                                               slot,
                                               batch.cu_view_pages,
                                               batch.page_mass_sum,
                                               batch.page_weight_sum,
                                               /*filters=*/false
                                           )
                                           .samples;
            // The table's temperature 0 takes the arg max, whose logit lies within the tolerance of the dense one's.
            const at::Tensor expected_row = expected_logits.select(0, 0).to(at::kFloat);
            const double best = expected_row.max().item<double>();
            const double taken = expected_row[sampled.item<std::int64_t>()].item<double>();
            CHECK(best - taken <= tolerance * expected_row.abs().max().item<double>());
        } else {
            const pulsarlm::model::output output = forward_eagerly(model, batch, kv);
            CHECK(max_relative_error(output.logits, expected_logits) < tolerance);
        }
        for (std::int64_t layer = 0; layer < n_layers; layer++) {
            dense_mass expected = dense_page_mass(dense, layer, c.first, c.last);
            CHECK(max_relative_error(g.page_mass_sum[layer], expected.mass_sum) < tolerance);
            CHECK(max_relative_error(g.page_weight_sum[layer], expected.weight_sum) < 1e-5);
        }
    }

    auto lines = at::tensor(std::vector<std::int64_t>(view_lines.begin(), view_lines.end())).to(at::kCUDA);
    for (std::int64_t layer = 0; layer < n_layers; layer++) {
        auto stored = kv[layer].k.index_select(0, lines).view({-1, n_kv_heads, head_dim});
        CHECK(max_relative_error(stored.narrow(0, 0, n_tokens), dense.unrotated_k[layer]) < tolerance);
    }
}

}  // namespace

TEST_CASE("Qwen3 matches a dense forward in fp32", "[model][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    check_against_dense(at::kFloat, 1e-4, false);
}

TEST_CASE("Qwen3 matches a dense forward in bf16", "[model][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    check_against_dense(at::kBFloat16, 1e-2, false);
}

TEST_CASE("Qwen3's decode steps replayed from CUDA graphs match a dense forward", "[model][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    check_against_dense(at::kFloat, 1e-4, true);
    check_against_dense(at::kBFloat16, 1e-2, true);
}
