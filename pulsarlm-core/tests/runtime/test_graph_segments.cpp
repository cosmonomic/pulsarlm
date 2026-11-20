#include "pulsarlm/engine/communicator.hpp"
#include "pulsarlm/engine/graph_segments.hpp"
#include "pulsarlm/engine/model/interface.hpp"
#include "pulsarlm/engine/model/qwen3_5.hpp"
#include "pulsarlm/engine/request_table.hpp"
#include "pulsarlm/engine/step_graphs.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#undef CHECK
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <vector>

// A step's model computation replayed from step_graphs' segment graphs against the eager forward of the same padded
// step, on Qwen3.5 with its MTP layer and on a model that counts its calls between segments.

namespace {

// Graph replays of the ranks of a communicator on one device queue their spinning p2p kernels without the
// communicator's wait for every rank, so the process loads its CUDA modules before it runs any
// (communicator::make_on_one_device).
const bool loads_modules_eagerly = setenv("CUDA_MODULE_LOADING", "EAGER", 1) == 0;

using pulsarlm::qwen3_5_layer_type;

constexpr std::int64_t hidden = 64;
constexpr std::int64_t intermediate = 128;
constexpr std::int64_t vocab = 300;
constexpr std::int64_t n_heads = 4;
constexpr std::int64_t n_kv_heads = 2;
constexpr std::int64_t head_dim = 64;
constexpr std::int64_t key_heads = 2;
constexpr std::int64_t value_heads = 4;
constexpr std::int64_t key_head_dim = 32;
constexpr std::int64_t value_head_dim = 32;
constexpr std::int64_t conv_kernel = 4;
constexpr std::int64_t page_size = 16;
// A verify request's query tokens: its pending token and one draft.
constexpr std::int64_t state_slots = 2;
constexpr std::int64_t request_count = 4;
// Each request's view pages per attention layer, and its lines per recurrent layer: its state slots, then its two
// boundary lines.
constexpr std::int64_t view_pages = 8;
constexpr std::int64_t recurrent_lines = state_slots + 2;
constexpr std::int64_t n_lines = request_count * view_pages;
const std::vector<std::int64_t> buckets{1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64};
const std::vector<qwen3_5_layer_type> layer_types{
    qwen3_5_layer_type::linear_attention,
    qwen3_5_layer_type::full_attention,
    qwen3_5_layer_type::linear_attention,
    qwen3_5_layer_type::full_attention,
};

at::TensorOptions cuda(at::ScalarType dtype) {
    return at::device(at::kCUDA).dtype(dtype);
}

// Random weights of the shapes rank's shard of ranks takes, the vocabulary padded to 320 rows over more than one rank.
c10::Dict<std::string, at::Tensor> qwen3_5_weights(std::int64_t rank, std::int64_t ranks) {
    at::manual_seed(21 + rank);
    const std::int64_t rank_heads = n_heads / ranks;
    const std::int64_t rank_kv_heads = n_kv_heads / ranks;
    const std::int64_t rank_key_heads = key_heads / ranks;
    const std::int64_t rank_value_heads = value_heads / ranks;
    const std::int64_t rank_channels = 2 * rank_key_heads * key_head_dim + rank_value_heads * value_head_dim;
    const std::int64_t vocab_entries = ranks == 1 ? vocab : 320 / ranks;
    c10::Dict<std::string, at::Tensor> named;
    auto add = [&](const std::string& name, const at::Tensor& t) { named.insert(name, t.to(cuda(at::kFloat))); };
    auto projection = [](std::int64_t out, std::int64_t in) { return at::randn({out, in}) / std::sqrt(in); };
    auto norm = [](std::int64_t size) { return 0.1 * at::randn({size}); };
    auto mlp = [&](const std::string& l) {
        add(l + "input_layernorm.weight", norm(hidden));
        add(l + "post_attention_layernorm.weight", norm(hidden));
        add(l + "mlp.gate_proj.weight", projection(intermediate / ranks, hidden));
        add(l + "mlp.up_proj.weight", projection(intermediate / ranks, hidden));
        add(l + "mlp.down_proj.weight", projection(hidden, intermediate / ranks));
    };
    auto attention_layer = [&](const std::string& l) {
        mlp(l);
        add(l + "self_attn.q_proj.weight", projection(2 * rank_heads * head_dim, hidden));
        add(l + "self_attn.k_proj.weight", projection(rank_kv_heads * head_dim, hidden));
        add(l + "self_attn.v_proj.weight", projection(rank_kv_heads * head_dim, hidden));
        add(l + "self_attn.o_proj.weight", projection(hidden, rank_heads * head_dim));
        add(l + "self_attn.q_norm.weight", norm(head_dim));
        add(l + "self_attn.k_norm.weight", norm(head_dim));
    };
    const std::string p = "model.language_model.";
    add(p + "embed_tokens.weight", at::randn({vocab_entries, hidden}));
    add(p + "norm.weight", norm(hidden));
    add("lm_head.weight", projection(vocab_entries, hidden));
    for (std::size_t layer = 0; layer < layer_types.size(); layer++) {
        const std::string l = p + "layers." + std::to_string(layer) + ".";
        if (layer_types[layer] == qwen3_5_layer_type::full_attention) {
            attention_layer(l);
            continue;
        }
        mlp(l);
        add(l + "linear_attn.in_proj_qkv.weight", projection(rank_channels, hidden));
        add(l + "linear_attn.in_proj_z.weight", projection(rank_value_heads * value_head_dim, hidden));
        add(l + "linear_attn.in_proj_b.weight", projection(rank_value_heads, hidden));
        add(l + "linear_attn.in_proj_a.weight", projection(rank_value_heads, hidden));
        add(l + "linear_attn.conv1d.weight", at::randn({rank_channels, 1, conv_kernel}) / 2.0);
        add(l + "linear_attn.dt_bias", at::randn({rank_value_heads}));
        add(l + "linear_attn.A_log", at::log(at::rand({rank_value_heads}) * 15.0 + 0.5));
        add(l + "linear_attn.norm.weight", 1.0 + 0.1 * at::randn({value_head_dim}));
        add(l + "linear_attn.out_proj.weight", projection(hidden, rank_value_heads * value_head_dim));
    }
    add("mtp.pre_fc_norm_embedding.weight", norm(hidden));
    add("mtp.pre_fc_norm_hidden.weight", norm(hidden));
    add("mtp.fc.weight", projection(hidden, 2 * hidden));
    add("mtp.norm.weight", norm(hidden));
    attention_layer("mtp.layers.0.");
    return named;
}

pulsarlm::qwen3_5_config qwen3_5_settings() {
    return pulsarlm::qwen3_5_config{
        .vocab_size = vocab,
        .hidden_size = hidden,
        .layer_types = layer_types,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .head_dim = head_dim,
        .rope_theta = 1e6,
        .partial_rotary_factor = 0.25,
        .linear_n_key_heads = key_heads,
        .linear_n_value_heads = value_heads,
        .linear_key_head_dim = key_head_dim,
        .linear_value_head_dim = value_head_dim,
        .linear_conv_kernel = conv_kernel,
        .rms_norm_eps = 1e-6,
    };
}

// n_lines lines of random contents for each KV layer, an input window widened by the drafts as the engine lays it out.
std::vector<pulsarlm::kv> make_lines(std::span<const pulsarlm::kv_layer> layers) {
    at::manual_seed(22);
    std::vector<pulsarlm::kv> lines;
    for (const pulsarlm::kv_layer& layer : layers) {
        if (layer.state) {
            std::vector<std::int64_t> sizes{n_lines};
            sizes.insert(sizes.end(), layer.state->sizes.begin(), layer.state->sizes.end());
            if (layer.state->input_window) {
                sizes.back() += state_slots - 1;
            }
            at::Tensor states = 0.1 * at::randn(sizes, cuda(layer.state->dtype));
            lines.push_back(pulsarlm::kv{.k = states, .v = states});
            continue;
        }
        const std::vector<std::int64_t> sizes{
            n_lines,
            page_size,
            static_cast<std::int64_t>(layer.n_kv_heads),
            static_cast<std::int64_t>(layer.head_dim),
        };
        lines.push_back(
            pulsarlm::kv{
                .k = at::randn(sizes, cuda(at::kFloat)),
                .v = at::randn(sizes, cuda(at::kFloat)),
                .last_hidden = layer.last_hidden_dim > 0
                    ? at::randn({n_lines, static_cast<std::int64_t>(layer.last_hidden_dim)}, cuda(at::kFloat))
                    : at::Tensor(),
            }
        );
    }
    return lines;
}

std::vector<pulsarlm::kv> clone_lines(std::span<const pulsarlm::kv> lines) {
    std::vector<pulsarlm::kv> clones;
    for (const pulsarlm::kv& line : lines) {
        at::Tensor k = line.k.clone();
        clones.push_back(
            pulsarlm::kv{
                .k = k,
                .v = line.v.is_same(line.k) ? k : line.v.clone(),
                .index_keys = line.index_keys.defined() ? line.index_keys.clone() : at::Tensor(),
                .last_hidden = line.last_hidden.defined() ? line.last_hidden.clone() : at::Tensor(),
            }
        );
    }
    return clones;
}

bool equal_bytes(const at::Tensor& got, const at::Tensor& expected) {
    return got.sizes() == expected.sizes() &&
        at::equal(got.contiguous().view(-1).view(at::kByte), expected.contiguous().view(-1).view(at::kByte));
}

bool equal_lines(std::span<const pulsarlm::kv> got, std::span<const pulsarlm::kv> expected) {
    for (std::size_t layer = 0; layer < got.size(); layer++) {
        if (!equal_bytes(got[layer].k, expected[layer].k) || !equal_bytes(got[layer].v, expected[layer].v)) {
            return false;
        }
        if (got[layer].last_hidden.defined() && !equal_bytes(got[layer].last_hidden, expected[layer].last_hidden)) {
            return false;
        }
    }
    return true;
}

// slots zeroed request table entries on the current device for a model of layers whose steps keep slot_count state
// slots.
pulsarlm::request_table
make_table(std::span<const pulsarlm::kv_layer> layers, std::int64_t slots, std::int64_t slot_count) {
    const auto n_layers = static_cast<std::int64_t>(layers.size());
    std::vector<std::uint8_t> recurrent;
    for (const pulsarlm::kv_layer& layer : layers) {
        recurrent.push_back(layer.state ? 1 : 0);
    }
    return pulsarlm::request_table{
        .page_tables = at::zeros({slots, n_layers, view_pages}, cuda(at::kInt)),
        .page_positions = at::zeros({slots, n_layers, view_pages}, cuda(at::kInt)),
        .history_positions = at::zeros({slots}, cuda(at::kLong)),
        .rope_positions = at::zeros({slots}, cuda(at::kLong)),
        .key_counts = at::zeros({slots, n_layers}, cuda(at::kInt)),
        .recurrent_layers = at::tensor(recurrent, at::dtype(at::kByte)).to(at::kCUDA).to(at::kBool),
        .page_size = page_size,
        .rejected_tokens = at::zeros({slots}, cuda(at::kInt)),
        .tokens = at::zeros({slots, slot_count}, cuda(at::kLong)),
        .cycle_end_positions = at::zeros({slots}, cuda(at::kLong)),
        .mass_start_positions = at::zeros({slots}, cuda(at::kLong)),
        .attention_mass_decays = at::zeros({slots}, cuda(at::kDouble)),
        .seeds = at::zeros({slots}, cuda(at::kLong)),
        .draw_counts = at::zeros({slots}, cuda(at::kLong)),
        .temperatures = at::zeros({slots}, cuda(at::kFloat)),
        .top_k = at::zeros({slots}, cuda(at::kLong)),
        .top_p = at::zeros({slots}, cuda(at::kFloat)),
    };
}

enum class run_kind { decode, verify, prefill };

// A request's part of a step: decode its pending token, verify its pending token and draft, or prefill a chunk of
// prefill_token_count tokens.
struct step_part {
    std::int64_t request;
    run_kind kind;
    std::int64_t prefill_token_count = 0;
};

std::int64_t count_query_tokens(const step_part& part) {
    switch (part.kind) {
    case run_kind::decode:
        return 1;
    case run_kind::verify:
        return state_slots;
    case run_kind::prefill:
        return part.prefill_token_count;
    }
    return 0;
}

// The step of parts, given in run order, of requests at history_positions, its tokens and positions padded with zeros
// to the smallest of buckets holding them. Request r's view pages in an attention layer are the lines [r * view_pages,
// (r + 1) * view_pages), and its lines in a recurrent layer [r * recurrent_lines, (r + 1) * recurrent_lines).
pulsarlm::model::batch make_step(
    std::span<const step_part> parts,
    std::span<const std::int64_t> history_positions,
    std::span<const pulsarlm::kv_layer> layers,
    const pulsarlm::model& stepped,
    std::int64_t seed
) {
    at::manual_seed(seed);
    const auto n_layers = static_cast<std::int64_t>(layers.size());
    const auto requests = static_cast<std::int64_t>(parts.size());
    std::vector<std::int32_t> query_starts{0};
    std::vector<std::int64_t> positions;
    std::vector<std::int64_t> histories;
    std::vector<std::int32_t> kept_states;
    std::vector<std::int64_t> logit_tokens;
    std::int64_t single_queries = 0;
    std::int64_t verifies = 0;
    for (const step_part& part : parts) {
        const std::int64_t count = count_query_tokens(part);
        const std::int64_t history = history_positions[static_cast<std::size_t>(part.request)];
        histories.push_back(history);
        for (std::int64_t i = 0; i < count; i++) {
            positions.push_back(history + i);
        }
        const std::int64_t first = query_starts.back();
        for (std::int64_t token = part.kind == run_kind::verify ? first : first + count - 1; token < first + count;
             token++) {
            logit_tokens.push_back(token);
        }
        query_starts.push_back(static_cast<std::int32_t>(first + count));
        single_queries += part.kind == run_kind::decode ? 1 : 0;
        verifies += part.kind == run_kind::verify ? 1 : 0;
        if (part.kind != run_kind::prefill) {
            kept_states.push_back(static_cast<std::int32_t>(count));
        }
    }
    const std::int64_t token_count = query_starts.back();
    // A mass view per request, but per query token of a verify request, over the pages its keys fill.
    std::vector<std::int32_t> cu_view_pages{0};
    for (std::size_t s = 0; s < parts.size(); s++) {
        const std::int64_t count = count_query_tokens(parts[s]);
        const auto pages = static_cast<std::int32_t>((histories[s] + count + page_size - 1) / page_size);
        for (std::int64_t view = 0; view < (parts[s].kind == run_kind::verify ? count : 1); view++) {
            cu_view_pages.push_back(cu_view_pages.back() + pages);
        }
    }
    const std::int64_t padded_count = *std::ranges::lower_bound(buckets, token_count);
    std::vector<std::int32_t> tables(static_cast<std::size_t>(n_layers * requests * view_pages), 0);
    std::vector<std::int32_t> table_positions(tables.size(), 0);
    std::vector<std::int32_t> key_counts;
    for (std::int64_t layer = 0; layer < n_layers; layer++) {
        const bool recurrent = layers[static_cast<std::size_t>(layer)].state.has_value();
        for (std::int64_t s = 0; s < requests; s++) {
            const step_part& part = parts[static_cast<std::size_t>(s)];
            const std::int64_t history = histories[static_cast<std::size_t>(s)];
            const std::int64_t count = count_query_tokens(part);
            for (std::int64_t column = 0; column < view_pages; column++) {
                const auto cell = static_cast<std::size_t>((layer * requests + s) * view_pages + column);
                if (recurrent) {
                    tables[cell] = column < recurrent_lines
                        ? static_cast<std::int32_t>(part.request * recurrent_lines + column)
                        : 0;
                } else {
                    tables[cell] = static_cast<std::int32_t>(part.request * view_pages + column);
                    table_positions[cell] = static_cast<std::int32_t>(column * page_size);
                }
            }
            key_counts.push_back(static_cast<std::int32_t>((recurrent ? history % page_size : history) + count));
        }
    }
    auto ints = [](const std::vector<std::int32_t>& values) {
        return at::tensor(values, at::dtype(at::kInt)).to(at::kCUDA);
    };
    auto longs = [](const std::vector<std::int64_t>& values) {
        return at::tensor(values, at::dtype(at::kLong)).to(at::kCUDA);
    };
    at::Tensor tokens = at::zeros({padded_count}, cuda(at::kLong));
    tokens.narrow(0, 0, token_count).copy_(at::randint(vocab, {token_count}, cuda(at::kLong)));
    at::Tensor pos = at::zeros({padded_count}, cuda(at::kLong));
    pos.narrow(0, 0, token_count).copy_(longs(positions));
    const at::Tensor cu_seqlens_q = ints(query_starts);
    auto starts_from = [&](std::int64_t first) {
        return cu_seqlens_q.narrow(0, first, requests - first + 1) - query_starts[static_cast<std::size_t>(first)];
    };
    return pulsarlm::model::batch{
        .single_query_request_count = single_queries,
        .per_query_state_request_count = verifies,
        .tokens = tokens,
        .pos = pos,
        .page_tables = ints(tables).view({n_layers, requests, view_pages}),
        .history_positions = longs(histories),
        .rejected_tokens = at::zeros({requests}, cuda(at::kInt)),
        .page_size = page_size,
        .state_slots = state_slots,
        .cu_seqlens_q = cu_seqlens_q,
        .cu_seqlens_q_from_second_run = starts_from(single_queries),
        .cu_seqlens_q_from_third_run = starts_from(single_queries + verifies),
        .kept_state_counts = ints(kept_states),
        .key_counts = ints(key_counts).view({n_layers, requests}),
        .page_positions = ints(table_positions).view({n_layers, requests, view_pages}),
        .cu_view_pages = ints(cu_view_pages).view({1, -1}).expand({n_layers, -1}).contiguous(),
        .query_weights = 0.25 + at::rand({token_count}, cuda(at::kFloat)),
        .logit_tokens = longs(logit_tokens),
        .arrival_counters = at::zeros({stepped.arrival_counter_count(requests, padded_count)}, cuda(at::kInt)),
    };
}

// b with zeroed page mass of its own in each KV layer of layers that measures mass.
pulsarlm::model::batch with_page_mass(pulsarlm::model::batch b, std::span<const pulsarlm::kv_layer> layers) {
    const std::int64_t view_page_count = b.cu_view_pages[0][-1].item<std::int64_t>();
    b.page_mass_sum.clear();
    b.page_weight_sum.clear();
    for (const pulsarlm::kv_layer& layer : layers) {
        if (layer.state || layer.sliding_window || layer.mirrors_held_pages) {
            b.page_mass_sum.emplace_back();
            b.page_weight_sum.emplace_back();
            continue;
        }
        const auto heads = static_cast<std::int64_t>(layer.n_readers * layer.n_q_heads);
        b.page_mass_sum.push_back(at::zeros({view_page_count, heads}, cuda(at::kFloat)));
        b.page_weight_sum.push_back(at::zeros({view_page_count}, cuda(at::kFloat)));
    }
    return b;
}

std::string describe(std::span<const step_part> parts) {
    std::string described;
    for (const step_part& part : parts) {
        described += part.kind == run_kind::decode ? " decode "
            : part.kind == run_kind::verify        ? " verify "
                                                   : " prefill ";
        described += std::to_string(part.request);
        if (part.kind == run_kind::prefill) {
            described += " of " + std::to_string(part.prefill_token_count);
        }
    }
    return described;
}

// Multiplies the one-hot rows of its tokens by 2 per layer, each layer's product then a call between segments that
// counts its runs, and takes its logits from the rows at the logit tokens in a last call.
class counting_model final : public pulsarlm::model {
  public:
    static constexpr std::int64_t layer_count = 3;
    std::shared_ptr<std::vector<std::size_t>> call_counts = std::make_shared<std::vector<std::size_t>>(layer_count + 1);
    std::size_t forward_count = 0;

    std::vector<pulsarlm::kv_layer> kv_layers() const override {
        return {pulsarlm::kv_layer{.n_kv_heads = 1, .head_dim = 8, .n_q_heads = 1}};
    }

    output forward(const batch& b, std::span<const pulsarlm::kv>, pulsarlm::graph_segments& segments) override {
        forward_count++;
        at::Tensor x = at::one_hot(b.tokens % vocab, vocab).to(at::kFloat);
        for (std::int64_t layer = 0; layer < layer_count; layer++) {
            x = x * 2.0;
            segments.run_between_segments(
                [counts = call_counts, layer](const at::Tensor&) { (*counts)[static_cast<std::size_t>(layer)]++; },
                x
            );
        }
        at::Tensor logits = at::empty({b.logit_tokens.size(0), vocab}, x.options());
        segments.run_between_segments(
            [counts = call_counts, &b](const at::Tensor& states, const at::Tensor& out) {
                (*counts)[layer_count]++;
                out.narrow(0, 0, b.logit_tokens.size(0)).copy_(states.index_select(0, b.logit_tokens));
            },
            x,
            logits
        );
        return output{.logits = logits};
    }
};

// Every combination of runs; 32 and 3 tokens fill their buckets, 33 is one token above 32.
std::vector<std::vector<step_part>> make_steps() {
    using enum run_kind;
    return {
        {{0, prefill, 20}, {1, prefill, 9}, {2, prefill, 3}},
        {{0, decode}, {1, verify}, {2, prefill, 20}, {3, prefill, 10}},
        {{0, decode}, {1, decode}},
        {{0, verify}, {1, verify}, {2, verify}},
        {{2, decode}, {3, prefill, 4}},
        {{0, decode}, {3, decode}, {1, verify}, {2, prefill, 13}},
        {{3, verify}, {0, prefill, 23}},
        {{1, decode}, {2, prefill, 12}},
        {{0, decode}, {2, verify}},
    };
}

// One rank's model, its lines replayed from its step graphs and a copy that its eager forwards step.
struct rank_steps {
    std::unique_ptr<pulsarlm::qwen3_5_model> model;
    std::vector<pulsarlm::kv_layer> layers;
    std::vector<pulsarlm::kv> graph_lines;
    std::vector<pulsarlm::kv> eager_lines;
    std::unique_ptr<pulsarlm::step_graphs> graphs;
    pulsarlm::model::output eager;
    pulsarlm::model::output replayed;
};

// Runs body(rank) for every rank at once, each on its own thread and stream, after the device's work so far. Catch2's
// assertions are not safe to make from these threads.
template <typename body_type> void on_every_rank(std::size_t ranks, body_type body) {
    c10::cuda::device_synchronize();
    std::vector<std::thread> threads;
    for (std::size_t rank = 0; rank < ranks; rank++) {
        threads.emplace_back([&body, rank] {
            c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromPool());
            body(rank);
            c10::cuda::getCurrentCUDAStream().synchronize();
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
}

// Steps the requests through make_steps on ranks ranks of Qwen3.5 with its MTP layer, each step once eagerly and once
// replayed from segment graphs captured at startup, on copies of the same lines.
void check_replayed_steps(std::int64_t ranks) {
    std::vector<pulsarlm::communicator> comms = ranks == 1
        ? std::vector<pulsarlm::communicator>{pulsarlm::communicator{}}
        : pulsarlm::communicator::make_on_one_device(at::Device(at::kCUDA, 0), static_cast<std::size_t>(ranks));
    std::vector<rank_steps> shards(static_cast<std::size_t>(ranks));
    for (std::int64_t rank = 0; rank < ranks; rank++) {
        rank_steps& shard = shards[static_cast<std::size_t>(rank)];
        shard.model = std::make_unique<pulsarlm::qwen3_5_model>(
            qwen3_5_settings(),
            qwen3_5_weights(rank, ranks),
            comms[static_cast<std::size_t>(rank)]
        );
        shard.layers = shard.model->kv_layers();
        shard.graph_lines = make_lines(shard.layers);
        shard.graphs = std::make_unique<pulsarlm::step_graphs>(
            *shard.model,
            shard.graph_lines,
            view_pages,
            page_size,
            state_slots,
            std::vector<std::size_t>(shard.layers.size(), 0),
            at::zeros({1}, cuda(at::kLong)),
            make_table(shard.layers, request_count, state_slots),
            buckets
        );
    }
    on_every_rank(shards.size(), [&](std::size_t rank) { shards[rank].graphs->capture_startup(); });
    for (rank_steps& shard : shards) {
        for (std::int64_t bucket : buckets) {
            REQUIRE(shard.graphs->has_segments(bucket));
        }
        shard.eager_lines = clone_lines(shard.graph_lines);
    }
    std::vector<std::int64_t> history_positions(request_count, 0);
    const std::vector<std::vector<step_part>> steps = make_steps();
    for (std::size_t index = 0; index < steps.size(); index++) {
        const std::vector<step_part>& parts = steps[index];
        INFO(ranks << " ranks, step" << describe(parts));
        const pulsarlm::model::batch
            b = make_step(parts, history_positions, shards.front().layers, *shards.front().model, 100 + index);
        std::int64_t token_count = 0;
        for (const step_part& part : parts) {
            token_count += count_query_tokens(part);
        }
        // Each rank's forward takes arrival counters of its own, as the executor's shards do.
        std::vector<pulsarlm::model::batch> rank_batches;
        std::vector<pulsarlm::model::batch> replayed_batches;
        for (const rank_steps& shard : shards) {
            rank_batches.push_back(with_page_mass(b, shard.layers));
            rank_batches.back().arrival_counters = at::zeros_like(b.arrival_counters);
            replayed_batches.push_back(with_page_mass(b, shard.layers));
        }
        on_every_rank(shards.size(), [&](std::size_t rank) {
            pulsarlm::graph_segments eager_segments(token_count);
            shards[rank].eager = shards[rank]
                                     .model->forward(rank_batches[rank], shards[rank].eager_lines, eager_segments);
        });
        on_every_rank(shards.size(), [&](std::size_t rank) {
            shards[rank].replayed = shards[rank].graphs->replay_segments(replayed_batches[rank], token_count);
        });
        for (std::size_t rank = 0; rank < shards.size(); rank++) {
            INFO("rank " << rank);
            const rank_steps& shard = shards[rank];
            REQUIRE(equal_bytes(shard.replayed.logits, shard.eager.logits));
            REQUIRE(equal_bytes(shard.replayed.logits, shards.front().eager.logits));
            REQUIRE(equal_bytes(
                shard.replayed.hidden.narrow(0, 0, token_count),
                shard.eager.hidden.narrow(0, 0, token_count)
            ));
            REQUIRE(equal_lines(shard.graph_lines, shard.eager_lines));
            for (std::size_t layer = 0; layer < shard.layers.size(); layer++) {
                INFO("KV layer " << layer);
                const at::Tensor& mass_sum = rank_batches[rank].page_mass_sum[layer];
                if (mass_sum.defined()) {
                    REQUIRE(mass_sum.abs().sum().item<double>() > 0.0);
                    // The attention kernels add a page's mass by atomic adds, whose order varies from run to run.
                    REQUIRE(at::allclose(replayed_batches[rank].page_mass_sum[layer], mass_sum, 1e-5, 1e-6));
                    REQUIRE(
                        at::allclose(
                            replayed_batches[rank].page_weight_sum[layer],
                            rank_batches[rank].page_weight_sum[layer],
                            1e-5,
                            1e-6
                        )
                    );
                }
            }
        }
        for (const step_part& part : parts) {
            history_positions[static_cast<std::size_t>(part.request)] += count_query_tokens(part);
        }
    }
}

}  // namespace

TEST_CASE("mixed steps replayed from segment graphs equal the eager forward of the padded step", "[runtime][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    check_replayed_steps(1);
}

TEST_CASE(
    "mixed steps over two ranks on one device replayed from segment graphs equal the eager forward",
    "[runtime][cuda]"
) {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    check_replayed_steps(2);
}

TEST_CASE("a replayed step runs only the calls between segments, each once, and no forward", "[runtime][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    c10::cuda::CUDAStreamGuard stream_guard(c10::cuda::getStreamFromPool());
    counting_model model;
    const std::vector<pulsarlm::kv_layer> layers = model.kv_layers();
    const std::vector<pulsarlm::kv> lines{
        pulsarlm::kv{
            .k = at::zeros({n_lines, page_size, 1, 8}, cuda(at::kFloat)),
            .v = at::zeros({n_lines, page_size, 1, 8}, cuda(at::kFloat))
        },
    };
    pulsarlm::step_graphs graphs(
        model,
        lines,
        view_pages,
        page_size,
        /*state_slots=*/1,
        std::vector<std::size_t>(layers.size(), 0),
        at::zeros({1}, cuda(at::kLong)),
        make_table(layers, request_count, 1),
        buckets
    );
    graphs.capture_startup();
    const std::size_t startup_forwards = model.forward_count;
    std::ranges::fill(*model.call_counts, 0);
    std::vector<std::int64_t> history_positions(request_count, 0);
    const std::vector<step_part> parts{
        {.request = 0, .kind = run_kind::decode},
        {.request = 1, .kind = run_kind::prefill, .prefill_token_count = 20},
    };
    const pulsarlm::model::batch b = make_step(parts, history_positions, layers, model, 7);
    for (std::size_t replay = 1; replay <= 3; replay++) {
        const pulsarlm::model::output replayed = graphs.replay_segments(b, 21);
        const at::Tensor expected = at::one_hot(b.tokens.index_select(0, b.logit_tokens) % vocab, vocab)
                                        .to(at::kFloat) *
            8.0;
        CHECK(at::equal(replayed.logits, expected));
        CHECK(model.forward_count == startup_forwards);
        for (std::size_t call = 0; call < model.call_counts->size(); call++) {
            INFO("call " << call);
            CHECK((*model.call_counts)[call] == replay);
        }
    }
}
