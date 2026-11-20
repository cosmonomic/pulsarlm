#include "pulsarlm/engine/communicator.hpp"
#include "pulsarlm/digest.hpp"
#include "pulsarlm/engine/engine.hpp"
#include "pulsarlm/engine/event_loop.hpp"
#include "pulsarlm/engine/kv_cache.hpp"
#include "pulsarlm/engine/model/qwen3_5.hpp"
#include "pulsarlm/engine/request_callbacks.hpp"
#include "pulsarlm/engine/scheduler.hpp"
#include "pulsarlm/engine/step_graphs.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>

#undef CHECK
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// A Qwen3.5 engine split over two ranks on one device serves the requests an engine of the whole model on that device
// serves alike: the same greedy tokens, the same states with their mass joined over the ranks' query heads, the same
// pages evicted, the same tokens again when a fresh engine resumes from a state, its pages fetched from the store into
// both ranks, and the same pages in the store.

namespace {

using pulsarlm::digest;
using pulsarlm::page;
using pulsarlm::sharded_page;
using pulsarlm::task;

using token_list = std::vector<std::int64_t>;

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
constexpr std::int64_t channels = 2 * key_heads * key_head_dim + value_heads * value_head_dim;
constexpr std::size_t page_size = 16;
constexpr std::int64_t ranks = 2;
constexpr std::size_t prompt_tokens = 100;
constexpr std::size_t decode_tokens = 64;
constexpr std::size_t resume_tokens = 20;
constexpr std::size_t resume_decode_tokens = 32;
constexpr double tolerance = 1e-4;
constexpr auto end_timeout = std::chrono::seconds(120);
const std::vector<pulsarlm::qwen3_5_layer_type> layer_types{
    pulsarlm::qwen3_5_layer_type::linear_attention,
    pulsarlm::qwen3_5_layer_type::linear_attention,
    pulsarlm::qwen3_5_layer_type::full_attention,
    pulsarlm::qwen3_5_layer_type::linear_attention,
    pulsarlm::qwen3_5_layer_type::full_attention,
};
const pulsarlm::eviction_policy eviction{
    .n_sink = page_size,
    .n_working = 2 * page_size,
    .cycle_tokens = page_size,
    .eviction_threshold = 0.5,
};

pulsarlm::qwen3_5_config model_config(std::int64_t kv_heads) {
    return pulsarlm::qwen3_5_config{
        .vocab_size = vocab,
        .hidden_size = hidden,
        .layer_types = layer_types,
        .n_heads = n_heads,
        .n_kv_heads = kv_heads,
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

// The whole model's weights in fp32 on cuda:0, with kv_heads KV heads in each attention layer.
c10::Dict<std::string, at::Tensor> full_weights(std::int64_t kv_heads) {
    at::manual_seed(13);
    c10::Dict<std::string, at::Tensor> named;
    auto add = [&](const std::string& name, const at::Tensor& t) {
        named.insert(name, t.to(at::TensorOptions().device(at::kCUDA).dtype(at::kFloat)));
    };
    auto projection = [](std::int64_t out, std::int64_t in) { return at::randn({out, in}) / std::sqrt(in); };
    auto norm = [](std::int64_t size) { return 0.1 * at::randn({size}); };
    const std::string p = "model.language_model.";
    add(p + "embed_tokens.weight", at::randn({vocab, hidden}));
    add(p + "norm.weight", norm(hidden));
    add("lm_head.weight", projection(vocab, hidden));
    for (std::size_t layer = 0; layer < layer_types.size(); layer++) {
        const std::string l = p + "layers." + std::to_string(layer) + ".";
        add(l + "input_layernorm.weight", norm(hidden));
        add(l + "post_attention_layernorm.weight", norm(hidden));
        add(l + "mlp.gate_proj.weight", projection(intermediate, hidden));
        add(l + "mlp.up_proj.weight", projection(intermediate, hidden));
        add(l + "mlp.down_proj.weight", projection(hidden, intermediate));
        if (layer_types[layer] == pulsarlm::qwen3_5_layer_type::full_attention) {
            add(l + "self_attn.q_proj.weight", projection(2 * n_heads * head_dim, hidden));
            add(l + "self_attn.k_proj.weight", projection(kv_heads * head_dim, hidden));
            add(l + "self_attn.v_proj.weight", projection(kv_heads * head_dim, hidden));
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
    return named;
}

// full_weights with the MTP layer's mtp.* weights, and tokens 0 and 1 taking opposite rows of lm_head far longer than
// the others', so the model and its MTP layer each predict one of them and the drafts are sometimes right.
c10::Dict<std::string, at::Tensor> drafting_weights(std::int64_t kv_heads) {
    c10::Dict<std::string, at::Tensor> named = full_weights(kv_heads);
    at::manual_seed(17);
    auto add = [&](const std::string& name, const at::Tensor& t) {
        named.insert_or_assign(name, t.to(at::TensorOptions().device(at::kCUDA).dtype(at::kFloat)));
    };
    auto projection = [](std::int64_t out, std::int64_t in) { return at::randn({out, in}) / std::sqrt(in); };
    auto norm = [](std::int64_t size) { return 0.1 * at::randn({size}); };
    at::Tensor lm_head = named.at("lm_head.weight").cpu().clone();
    at::Tensor direction = at::randn({hidden});
    lm_head[0] = 50.0 * direction / direction.norm();
    lm_head[1] = -lm_head[0];
    add("lm_head.weight", lm_head);
    add("mtp.pre_fc_norm_embedding.weight", norm(hidden));
    add("mtp.pre_fc_norm_hidden.weight", norm(hidden));
    add("mtp.fc.weight", projection(hidden, 2 * hidden));
    add("mtp.norm.weight", norm(hidden));
    const std::string m = "mtp.layers.0.";
    add(m + "input_layernorm.weight", norm(hidden));
    add(m + "post_attention_layernorm.weight", norm(hidden));
    add(m + "mlp.gate_proj.weight", projection(intermediate, hidden));
    add(m + "mlp.up_proj.weight", projection(intermediate, hidden));
    add(m + "mlp.down_proj.weight", projection(hidden, intermediate));
    add(m + "self_attn.q_proj.weight", projection(2 * n_heads * head_dim, hidden));
    add(m + "self_attn.k_proj.weight", projection(kv_heads * head_dim, hidden));
    add(m + "self_attn.v_proj.weight", projection(kv_heads * head_dim, hidden));
    add(m + "self_attn.o_proj.weight", projection(hidden, n_heads * head_dim));
    add(m + "self_attn.q_norm.weight", norm(head_dim));
    add(m + "self_attn.k_norm.weight", norm(head_dim));
    return named;
}

// weights without mtp.*, as the loader leaves them out of a model that drafts nothing.
c10::Dict<std::string, at::Tensor> without_drafts(const c10::Dict<std::string, at::Tensor>& weights) {
    c10::Dict<std::string, at::Tensor> kept;
    for (const auto& entry : weights) {
        if (!entry.key().starts_with("mtp.")) {
            kept.insert(entry.key(), entry.value());
        }
    }
    return kept;
}

using cut_ranges = std::vector<std::pair<std::int64_t, std::int64_t>>;

struct weight_cut {
    std::int64_t dim;
    cut_ranges ranges;
};

// Rank's cut of a weight split over the ranks as vLLM v0.31.0 splits Qwen3.5; nullopt for a weight every rank holds
// whole.
std::optional<weight_cut> cut_of(const std::string& name, std::int64_t rank, std::int64_t kv_heads) {
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

at::Tensor cut(const at::Tensor& t, const weight_cut& c) {
    std::vector<at::Tensor> parts;
    for (auto [first, last] : c.ranges) {
        parts.push_back(t.narrow(c.dim, first, last - first));
    }
    return at::cat(parts, c.dim).contiguous();
}

// The embedding or head padded with zero rows to a multiple of 64, then rank's rows.
at::Tensor vocab_rows(const at::Tensor& t, std::int64_t rank) {
    const std::int64_t padded = (vocab + 63) / 64 * 64;
    const at::Tensor rows = at::cat({t, at::zeros({padded - vocab, t.size(1)}, t.options())}, 0);
    return rows.narrow(0, rank * padded / ranks, padded / ranks).contiguous();
}

c10::Dict<std::string, at::Tensor>
rank_weights(const c10::Dict<std::string, at::Tensor>& weights, std::int64_t rank, std::int64_t kv_heads) {
    c10::Dict<std::string, at::Tensor> sliced;
    for (const auto& entry : weights) {
        const std::string& name = entry.key();
        if (name.ends_with("embed_tokens.weight") || name == "lm_head.weight") {
            sliced.insert(name, vocab_rows(entry.value(), rank));
        } else if (std::optional<weight_cut> c = cut_of(name, rank, kv_heads)) {
            sliced.insert(name, cut(entry.value(), *c));
        } else {
            sliced.insert(name, entry.value());
        }
    }
    return sliced;
}

struct store_contents {
    std::unordered_map<digest, page, pulsarlm::digest_hash> pages;
    std::size_t fetched_pages = 0;
    std::vector<std::shared_ptr<const pulsarlm::state>> states;
    pulsarlm::state::lease_id_type next_lease_id = 1;
};

std::size_t page_bytes_of(const sharded_page& p) {
    std::size_t bytes = 0;
    for (const pulsarlm::page_shard& shard : p.shards) {
        bytes = std::max(bytes, shard.host_offset + static_cast<std::size_t>(shard.bytes.numel()));
    }
    return bytes;
}

// Pageable copies of sources, each shard into its bytes of the page on streams[shard.cache_shard].
std::vector<page> copy_pages_to_host(std::span<const sharded_page> sources, std::span<const cudaStream_t> streams) {
    std::vector<page> pages;
    pages.reserve(sources.size());
    for (const sharded_page& source : sources) {
        at::Tensor bytes = at::empty({static_cast<std::int64_t>(page_bytes_of(source))}, at::kByte);
        for (const pulsarlm::page_shard& shard : source.shards) {
            c10::cuda::CUDAStreamGuard guard(
                c10::cuda::getStreamFromExternal(streams[shard.cache_shard], shard.bytes.device().index())
            );
            bytes.narrow(0, static_cast<std::int64_t>(shard.host_offset), shard.bytes.numel()).copy_(shard.bytes);
        }
        pages.push_back(page{.bytes = std::move(bytes)});
    }
    return pages;
}

// Copies each shard's bytes of pages[i] into the shards of targets[i] on streams[shard.cache_shard].
void copy_pages_from_host(
    std::span<const sharded_page> targets,
    std::span<const page> pages,
    std::span<const cudaStream_t> streams
) {
    for (std::size_t i = 0; i < targets.size(); i++) {
        for (const pulsarlm::page_shard& shard : targets[i].shards) {
            c10::cuda::CUDAStreamGuard guard(
                c10::cuda::getStreamFromExternal(streams[shard.cache_shard], shard.bytes.device().index())
            );
            shard.bytes.copy_(
                pages[i].bytes.narrow(0, static_cast<std::int64_t>(shard.host_offset), shard.bytes.numel())
            );
        }
    }
}

// Keeps every page it is written, copied to the host, and every state; engines built on one contents share them. A
// lease never drops a page.
struct memory_store final : pulsarlm::page_store {
    std::shared_ptr<store_contents> contents;

    task<std::vector<bool>> read_pages(
        std::span<const digest> hashes,
        std::vector<sharded_page>&& lines,
        std::span<const cudaStream_t> streams
    ) override {
        std::shared_ptr<store_contents> shared = contents;
        std::vector<digest> requested(hashes.begin(), hashes.end());
        std::vector<sharded_page> targets = std::move(lines);
        std::vector<bool> found(requested.size());
        std::vector<sharded_page> found_targets;
        std::vector<page> found_pages;
        for (std::size_t i = 0; i < requested.size(); i++) {
            if (auto it = shared->pages.find(requested[i]); it != shared->pages.end()) {
                found[i] = true;
                found_targets.push_back(targets[i]);
                found_pages.push_back(it->second);
            }
        }
        shared->fetched_pages += found_pages.size();
        copy_pages_from_host(found_targets, found_pages, streams);
        co_return found;
    }

    task<std::expected<void, pulsarlm::page_error>> write_pages(
        std::vector<digest>&& keys,
        std::vector<sharded_page>&& pages,
        std::vector<std::pair<pulsarlm::state::lease_id_type, page_index_type>>&&,
        std::span<const cudaStream_t> streams
    ) override {
        std::shared_ptr<store_contents> shared = contents;
        std::vector<digest> put_keys = std::move(keys);
        std::vector<sharded_page> sources = std::move(pages);
        std::vector<page> copies = copy_pages_to_host(sources, streams);
        for (std::size_t i = 0; i < put_keys.size(); i++) {
            shared->pages.insert_or_assign(put_keys[i], std::move(copies[i]));
        }
        co_return {};
    }

    // The stored state reaching the most tokens of a proper prefix of tokens.
    task<std::expected<
        std::pair<std::shared_ptr<const pulsarlm::state>, pulsarlm::state::lease_id_type>,
        pulsarlm::page_error>>
    read_state(std::span<const std::int64_t> tokens, std::span<const digest> page_digests) override {
        std::shared_ptr<store_contents> shared = contents;
        std::shared_ptr<const pulsarlm::state> found = std::make_shared<const pulsarlm::state>();
        std::size_t found_tokens = 0;
        for (const std::shared_ptr<const pulsarlm::state>& s : shared->states) {
            std::size_t pages = 0;
            if (s->head) {
                auto head = std::ranges::find(page_digests, *s->head);
                if (head == page_digests.end()) {
                    continue;
                }
                pages = static_cast<std::size_t>(head - page_digests.begin()) + 1;
            }
            std::size_t start = pages * page_size;
            std::size_t covered = start + s->tail.size();
            if (covered < tokens.size() && covered > found_tokens &&
                std::ranges::equal(s->tail, tokens.subspan(start, s->tail.size()))) {
                found = s;
                found_tokens = covered;
            }
        }
        co_return std::pair(found, shared->next_lease_id++);
    }

    task<void> write_state(pulsarlm::state::lease_id_type, std::shared_ptr<const pulsarlm::state>&& s) override {
        contents->states.push_back(std::move(s));
        co_return;
    }

    void release(pulsarlm::state::lease_id_type) noexcept override {}
};

// Filled on the loop thread, read on the test thread.
struct request_results {
    std::mutex mutex;
    std::condition_variable end_condition;
    token_list tokens;
    std::optional<std::expected<pulsarlm::state, pulsarlm::request_error>> end;
};

class recording_callbacks final : public pulsarlm::request_callbacks {
    std::shared_ptr<request_results> recorded_results;

  public:
    explicit recording_callbacks(std::shared_ptr<request_results> recorded_results)
    : recorded_results(std::move(recorded_results)) {}

    void on_prefill(const pulsarlm::prefill_statistics&) noexcept override {}

    void on_decode(std::int64_t token, const pulsarlm::decode_statistics&) noexcept override {
        std::lock_guard lock(recorded_results->mutex);
        recorded_results->tokens.push_back(token);
    }

    void on_cycle(pulsarlm::cycle_end&&) noexcept override {}

    void on_end(
        std::expected<std::shared_ptr<const pulsarlm::state>, pulsarlm::request_error>&& result,
        const pulsarlm::request_statistics&
    ) noexcept override {
        {
            std::lock_guard lock(recorded_results->mutex);
            recorded_results->end.emplace(
                std::move(result).transform([](std::shared_ptr<const pulsarlm::state>&& ended) { return *ended; })
            );
        }
        recorded_results->end_condition.notify_all();
    }

    bool cancelled() const noexcept override {
        return false;
    }
};

// With decode_graphs, decoding requests replay graphs over views of up to 1024 tokens. A step drafts draft_tokens
// tokens.
pulsarlm::engine_config
engine_config(std::size_t shard_count, bool decode_graphs = false, std::size_t draft_tokens = 0) {
    return pulsarlm::engine_config{
        .model_name = "qwen3_5-test",
        .model_hash = "1",
        .page_size = page_size,
        .kv_dtype = at::kFloat,
        .kv_cache_pool_bytes = std::size_t{64} << 20,
        .devices = std::vector<at::Device>(shard_count, at::Device(at::kCUDA, 0)),
        .stop_ids = {},
        .prefill_tokens_per_step = 32,
        .decode_graphs = decode_graphs,
        .max_context_window = 1024,
        .draft_tokens = draft_tokens,
    };
}

// An engine on cuda:0 of the whole model, or of its ranks' shards, over a fresh store on contents.
std::unique_ptr<pulsarlm::engine> make_engine(
    const c10::Dict<std::string, at::Tensor>& weights,
    std::int64_t kv_heads,
    std::size_t shard_count,
    const std::shared_ptr<store_contents>& contents,
    bool decode_graphs = false,
    std::size_t draft_tokens = 0
) {
    std::vector<std::unique_ptr<pulsarlm::model>> shards;
    if (shard_count == 1) {
        shards.push_back(std::make_unique<pulsarlm::qwen3_5_model>(model_config(kv_heads), weights));
    } else {
        std::vector<pulsarlm::communicator> comms = pulsarlm::communicator::make_on_one_device(
            at::Device(at::kCUDA, 0),
            shard_count
        );
        for (std::int64_t rank = 0; rank < static_cast<std::int64_t>(shard_count); rank++) {
            shards.push_back(
                std::make_unique<pulsarlm::qwen3_5_model>(
                    model_config(kv_heads),
                    rank_weights(weights, rank, kv_heads),
                    comms[rank]
                )
            );
        }
    }
    auto store = std::make_unique<memory_store>();
    store->contents = contents;
    return std::make_unique<pulsarlm::engine>(
        std::move(shards),
        vocab,
        engine_config(shard_count, decode_graphs, draft_tokens),
        std::move(store)
    );
}

struct served {
    token_list tokens;
    pulsarlm::state end_state;
};

served serve(pulsarlm::engine& e, pulsarlm::request&& r) {
    auto recorded_results = std::make_shared<request_results>();
    e.submit(std::move(r), std::make_unique<recording_callbacks>(recorded_results));
    std::unique_lock lock(recorded_results->mutex);
    REQUIRE(recorded_results->end_condition.wait_for(lock, end_timeout, [&] {
        return recorded_results->end.has_value();
    }));
    REQUIRE(recorded_results->end->has_value());
    REQUIRE(e.exception() == nullptr);
    return served{.tokens = recorded_results->tokens, .end_state = std::move(**recorded_results->end)};
}

token_list random_tokens(std::size_t count, std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::uniform_int_distribution<std::int64_t> token(0, vocab - 1);
    token_list tokens(count);
    std::ranges::generate(tokens, [&] { return token(generator); });
    return tokens;
}

double relative_error(std::span<const float> got, std::span<const float> expected) {
    REQUIRE(got.size() == expected.size());
    double largest = 0.0;
    double difference = 0.0;
    for (std::size_t i = 0; i < got.size(); i++) {
        largest = std::max(largest, std::abs(static_cast<double>(expected[i])));
        difference = std::max(difference, std::abs(static_cast<double>(got[i]) - expected[i]));
    }
    return largest == 0.0 ? difference : difference / largest;
}

bool moments_close(const pulsarlm::moments& got, const pulsarlm::moments& expected) {
    const auto close = [](double a, double b) { return std::abs(a - b) <= tolerance * std::max(1.0, std::abs(b)); };
    return got.count == expected.count && close(got.mean, expected.mean) && close(got.m2, expected.m2);
}

// The same pages in every layer, in view order, with their sums and moments close.
void check_same_state(const pulsarlm::state& got, const pulsarlm::state& expected) {
    REQUIRE(got.head == expected.head);
    REQUIRE(got.tail == expected.tail);
    REQUIRE(got.layers.size() == expected.layers.size());
    for (std::size_t layer = 0; layer < got.layers.size(); layer++) {
        INFO("KV layer " << layer);
        const pulsarlm::layer_sequence& g = got.layers[layer];
        const pulsarlm::layer_sequence& e = expected.layers[layer];
        REQUIRE(g.pages.size() == e.pages.size());
        for (std::size_t i = 0; i < g.pages.size(); i++) {
            INFO("page " << i);
            REQUIRE(g.pages[i].hash == e.pages[i].hash);
            REQUIRE(g.pages[i].index == e.pages[i].index);
            CHECK(relative_error(g.pages[i].mass_sum, e.pages[i].mass_sum) < tolerance);
            CHECK(
                std::abs(g.pages[i].weight_sum - e.pages[i].weight_sum) <=
                tolerance * std::max(1.0f, std::abs(e.pages[i].weight_sum))
            );
        }
        REQUIRE(g.density_moments.size() == e.density_moments.size());
        for (std::size_t head = 0; head < g.density_moments.size(); head++) {
            CHECK(moments_close(g.density_moments[head], e.density_moments[head]));
        }
        CHECK(moments_close(g.score_moments, e.score_moments));
    }
}

// The largest relative error over the pages of the store, read as fp32, and the count of pages equal byte for byte.
struct store_comparison {
    double largest_error = 0.0;
    std::size_t identical_pages = 0;
};

store_comparison compare_stores(const store_contents& got, const store_contents& expected) {
    REQUIRE(got.pages.size() == expected.pages.size());
    store_comparison comparison;
    for (const auto& [key, expected_page] : expected.pages) {
        auto it = got.pages.find(key);
        REQUIRE(it != got.pages.end());
        const at::Tensor& g = it->second.bytes;
        const at::Tensor& e = expected_page.bytes;
        REQUIRE(g.numel() == e.numel());
        comparison.identical_pages += at::equal(g, e) ? 1 : 0;
        const at::Tensor g_floats = g.view(at::kFloat);
        const at::Tensor e_floats = e.view(at::kFloat);
        comparison.largest_error = std::max(
            comparison.largest_error,
            relative_error(
                std::span(g_floats.const_data_ptr<float>(), static_cast<std::size_t>(g_floats.numel())),
                std::span(e_floats.const_data_ptr<float>(), static_cast<std::size_t>(e_floats.numel()))
            )
        );
    }
    return comparison;
}

// The full pages of the history a state reached, less those of its first full-attention layer.
std::size_t evicted_pages(const pulsarlm::state& s, std::size_t history_tokens) {
    const std::size_t full_attention_kv_layer = 4;
    return history_tokens / page_size - s.layers[full_attention_kv_layer].pages.size();
}

}  // namespace

TEST_CASE("an engine split over two ranks serves as the engine of the whole model", "[runtime][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const std::int64_t kv_heads = GENERATE(n_kv_heads, std::int64_t{1});
    INFO(kv_heads << " KV heads over " << ranks << " ranks");
    const c10::Dict<std::string, at::Tensor> weights = full_weights(kv_heads);
    const token_list prompt = random_tokens(prompt_tokens, 1);
    const token_list continuation = random_tokens(resume_tokens, 2);

    auto whole_contents = std::make_shared<store_contents>();
    auto split_contents = std::make_shared<store_contents>();
    served whole;
    served split;
    {
        std::unique_ptr<pulsarlm::engine> whole_engine = make_engine(weights, kv_heads, 1, whole_contents);
        std::unique_ptr<pulsarlm::engine> split_engine = make_engine(weights, kv_heads, ranks, split_contents);
        const auto first_request = [&] {
            return pulsarlm::request{.tokens = prompt, .max_tokens = decode_tokens, .eviction = eviction};
        };
        whole = serve(*whole_engine, first_request());
        split = serve(*split_engine, first_request());
    }
    REQUIRE(whole.tokens.size() == decode_tokens);
    CHECK(split.tokens == whole.tokens);
    REQUIRE(evicted_pages(whole.end_state, prompt_tokens + decode_tokens - 1) > 0);
    check_same_state(split.end_state, whole.end_state);

    served whole_resumed;
    served split_resumed;
    {
        std::unique_ptr<pulsarlm::engine> whole_engine = make_engine(weights, kv_heads, 1, whole_contents);
        std::unique_ptr<pulsarlm::engine> split_engine = make_engine(weights, kv_heads, ranks, split_contents);
        const auto resumed_request = [&](const served& first) {
            token_list history = prompt;
            history.insert(history.end(), first.tokens.begin(), first.tokens.end() - 1);
            history.insert(history.end(), continuation.begin(), continuation.end());
            return pulsarlm::request{.tokens = history, .max_tokens = resume_decode_tokens, .eviction = eviction};
        };
        whole_resumed = serve(*whole_engine, resumed_request(whole));
        split_resumed = serve(*split_engine, resumed_request(split));
    }
    REQUIRE(split_contents->fetched_pages > 0);
    CHECK(split_contents->fetched_pages == whole_contents->fetched_pages);
    REQUIRE(whole_resumed.tokens.size() == resume_decode_tokens);
    CHECK(split_resumed.tokens == whole_resumed.tokens);
    check_same_state(split_resumed.end_state, whole_resumed.end_state);

    const store_comparison stores = compare_stores(*split_contents, *whole_contents);
    INFO(stores.identical_pages << " of " << whole_contents->pages.size() << " pages equal byte for byte");
    CHECK(stores.largest_error < tolerance);
}

pulsarlm::request sampled_request() {
    return pulsarlm::request{
        .tokens = random_tokens(prompt_tokens, 3),
        .sampling = pulsarlm::sampling_params{.temperature = 0.8, .seed = 7},
        .max_tokens = decode_tokens,
        .eviction = eviction,
    };
}

// Most decode steps complete no page, so each is launched before the previous one is written back and every rank reads
// its next tokens from its own request table, which its own samples advanced: a rank sampling differently would
// forward another token and change the logits of every rank.
TEST_CASE("an engine split over two ranks samples as the whole model while its steps overlap", "[runtime][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const c10::Dict<std::string, at::Tensor> weights = full_weights(n_kv_heads);
    served whole;
    served split;
    {
        auto whole_contents = std::make_shared<store_contents>();
        auto split_contents = std::make_shared<store_contents>();
        std::unique_ptr<pulsarlm::engine> whole_engine = make_engine(weights, n_kv_heads, 1, whole_contents);
        std::unique_ptr<pulsarlm::engine> split_engine = make_engine(weights, n_kv_heads, ranks, split_contents);
        whole = serve(*whole_engine, sampled_request());
        split = serve(*split_engine, sampled_request());
    }
    REQUIRE(whole.tokens.size() == decode_tokens);
    CHECK(split.tokens == whole.tokens);
    check_same_state(split.end_state, whole.end_state);
}

// The ranks of a communicator on one device cannot capture graphs at once, so the graphs run on one rank: each replay
// copies the tokens gathered from the request table that the previous step advanced.
TEST_CASE("decode graphs sample as the eager forward while steps overlap", "[runtime][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const c10::Dict<std::string, at::Tensor> weights = full_weights(n_kv_heads);
    served eager;
    served replayed;
    {
        auto eager_contents = std::make_shared<store_contents>();
        auto replayed_contents = std::make_shared<store_contents>();
        std::unique_ptr<pulsarlm::engine> eager_engine = make_engine(weights, n_kv_heads, 1, eager_contents);
        std::unique_ptr<pulsarlm::engine>
            replayed_engine = make_engine(weights, n_kv_heads, 1, replayed_contents, /*decode_graphs=*/true);
        eager = serve(*eager_engine, sampled_request());
        replayed = serve(*replayed_engine, sampled_request());
    }
    REQUIRE(eager.tokens.size() == decode_tokens);
    CHECK(replayed.tokens == eager.tokens);
    check_same_state(replayed.end_state, eager.end_state);
}

// Counts the decode groups it forwards of more sequences than step_graphs captures at startup.
class wide_group_counting_model final : public pulsarlm::qwen3_5_model {
  public:
    std::shared_ptr<std::atomic<std::size_t>> wide_decode_group_count = std::make_shared<std::atomic<std::size_t>>(0);

    using pulsarlm::qwen3_5_model::qwen3_5_model;

    output forward(const batch& b, std::span<const pulsarlm::kv> lines) override {
        if (b.kind == batch_kind::decode && b.page_tables.size(1) > pulsarlm::step_graphs::startup_sequences) {
            (*wide_decode_group_count)++;
        }
        return pulsarlm::qwen3_5_model::forward(b, lines);
    }
};

// A graph of more sequences than the startup graphs is captured in the step that first needs it and replayed after:
// with graphs the model forwards such a group only at a capture.
TEST_CASE("decode graphs captured during a step sample as the eager forward", "[runtime][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const std::size_t sequence_count = 12;
    const c10::Dict<std::string, at::Tensor> weights = full_weights(n_kv_heads);
    struct run {
        std::vector<served> all;
        std::size_t wide_decode_group_count;
    };
    const auto serve_all = [&](bool decode_graphs) {
        auto model = std::make_unique<wide_group_counting_model>(model_config(n_kv_heads), weights);
        std::shared_ptr<std::atomic<std::size_t>> wide_decode_group_count = model->wide_decode_group_count;
        auto store = std::make_unique<memory_store>();
        store->contents = std::make_shared<store_contents>();
        pulsarlm::engine e(std::move(model), vocab, engine_config(1, decode_graphs), std::move(store));
        std::vector<std::shared_ptr<request_results>> recorded_results;
        for (std::size_t s = 0; s < sequence_count; s++) {
            recorded_results.push_back(std::make_shared<request_results>());
            e.submit(
                pulsarlm::request{
                    .tokens = random_tokens(prompt_tokens, 10 + static_cast<std::uint32_t>(s)),
                    .sampling = pulsarlm::sampling_params{.temperature = 0.8, .seed = 30 + s},
                    .max_tokens = decode_tokens,
                    .eviction = eviction,
                },
                std::make_unique<recording_callbacks>(recorded_results.back())
            );
        }
        run finished;
        for (const std::shared_ptr<request_results>& results : recorded_results) {
            std::unique_lock lock(results->mutex);
            REQUIRE(results->end_condition.wait_for(lock, end_timeout, [&] { return results->end.has_value(); }));
            REQUIRE(e.exception() == nullptr);
            REQUIRE(results->end->has_value());
            finished.all.push_back(served{.tokens = results->tokens, .end_state = std::move(**results->end)});
        }
        finished.wide_decode_group_count = *wide_decode_group_count;
        return finished;
    };
    const run eager = serve_all(false);
    const run replayed = serve_all(true);

    for (std::size_t s = 0; s < sequence_count; s++) {
        INFO("request " << s);
        REQUIRE(eager.all[s].tokens.size() == decode_tokens);
        CHECK(replayed.all[s].tokens == eager.all[s].tokens);
        check_same_state(replayed.all[s].end_state, eager.all[s].end_state);
    }
    CHECK(replayed.wide_decode_group_count > 0);
    CHECK(replayed.wide_decode_group_count < eager.wide_decode_group_count);
}

// Counts the decode groups it forwards by their number of requests, below 64.
class decode_group_counting_model final : public pulsarlm::qwen3_5_model {
  public:
    using counts_type = std::array<std::atomic<std::size_t>, 64>;
    std::shared_ptr<counts_type> decode_group_counts = std::make_shared<counts_type>();

    using pulsarlm::qwen3_5_model::qwen3_5_model;

    output forward(const batch& b, std::span<const pulsarlm::kv> lines) override {
        if (b.kind == batch_kind::decode) {
            (*decode_group_counts)[static_cast<std::size_t>(b.page_tables.size(1))]++;
        }
        return pulsarlm::qwen3_5_model::forward(b, lines);
    }
};

struct counted_engine {
    std::unique_ptr<pulsarlm::engine> engine;
    std::shared_ptr<decode_group_counting_model::counts_type> decode_group_counts;
};

counted_engine make_counted_engine(const c10::Dict<std::string, at::Tensor>& weights, bool decode_graphs) {
    auto model = std::make_unique<decode_group_counting_model>(model_config(n_kv_heads), weights);
    std::shared_ptr<decode_group_counting_model::counts_type> counts = model->decode_group_counts;
    auto store = std::make_unique<memory_store>();
    store->contents = std::make_shared<store_contents>();
    return counted_engine{
        .engine = std::make_unique<pulsarlm::engine>(
            std::move(model),
            vocab,
            engine_config(1, decode_graphs),
            std::move(store)
        ),
        .decode_group_counts = std::move(counts),
    };
}

std::size_t total_count(const decode_group_counting_model::counts_type& counts) {
    std::size_t total = 0;
    for (const std::atomic<std::size_t>& count : counts) {
        total += count;
    }
    return total;
}

// Submits every request at once and waits for each to end.
std::vector<served> serve_together(pulsarlm::engine& e, std::vector<pulsarlm::request>&& requests) {
    std::vector<std::shared_ptr<request_results>> recorded_results;
    for (pulsarlm::request& r : requests) {
        recorded_results.push_back(std::make_shared<request_results>());
        e.submit(std::move(r), std::make_unique<recording_callbacks>(recorded_results.back()));
    }
    std::vector<served> all;
    for (const std::shared_ptr<request_results>& results : recorded_results) {
        std::unique_lock lock(results->mutex);
        REQUIRE(results->end_condition.wait_for(lock, end_timeout, [&] { return results->end.has_value(); }));
        REQUIRE(e.exception() == nullptr);
        REQUIRE(results->end->has_value());
        all.push_back(served{.tokens = results->tokens, .end_state = std::move(**results->end)});
    }
    return all;
}

// request_count requests of prompt_tokens tokens that measure no mass, sampling at temperature 0.8, with top_k 20 when
// filters.
std::vector<pulsarlm::request> decoding_requests(std::size_t request_count, bool filters, std::uint32_t seed) {
    std::vector<pulsarlm::request> requests;
    for (std::size_t s = 0; s < request_count; s++) {
        requests.push_back(
            pulsarlm::request{
                .tokens = random_tokens(prompt_tokens, seed + static_cast<std::uint32_t>(s)),
                .sampling =
                    pulsarlm::sampling_params{
                        .temperature = 0.8,
                        .top_k = filters ? std::size_t{20} : std::size_t{0},
                        .seed = seed + s,
                    },
                .max_tokens = decode_tokens,
            }
        );
    }
    return requests;
}

// A step of decode entries alone that all sample replays a whole-step graph per number of requests and per sampling
// kind, captured at startup for the unfiltered steps of up to step_graphs::startup_sequences requests: with graphs the
// model forwards a decode group of 3 requests only to capture the filtered step of 3, and one of 12 only to capture the
// unfiltered step of 12 and then the filtered one. The requests measure no mass, and the steps overlap while no
// request ends a cycle.
TEST_CASE("decode steps replayed from whole-step graphs sample as the eager forward", "[runtime][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const c10::Dict<std::string, at::Tensor> weights = full_weights(n_kv_heads);
    counted_engine eager = make_counted_engine(weights, false);
    counted_engine replayed = make_counted_engine(weights, true);
    struct phase {
        std::size_t request_count;
        bool filters;
        std::uint32_t seed;
    };
    for (const phase p : {phase{3, true, 80}, phase{12, false, 40}, phase{12, true, 60}}) {
        INFO(p.request_count << (p.filters ? " filtered" : " unfiltered") << " requests");
        const std::size_t captures_before = (*replayed.decode_group_counts)[p.request_count];
        const std::vector<served> replayed_runs = serve_together(
            *replayed.engine,
            decoding_requests(p.request_count, p.filters, p.seed)
        );
        const std::vector<served> eager_runs = serve_together(
            *eager.engine,
            decoding_requests(p.request_count, p.filters, p.seed)
        );
        for (std::size_t s = 0; s < p.request_count; s++) {
            INFO("request " << s);
            REQUIRE(eager_runs[s].tokens.size() == decode_tokens);
            CHECK(replayed_runs[s].tokens == eager_runs[s].tokens);
            check_same_state(replayed_runs[s].end_state, eager_runs[s].end_state);
        }
        CHECK((*replayed.decode_group_counts)[p.request_count] - captures_before == 1);
    }
}

// While one request prefills a long prompt over many steps, another decodes beside it, its decode group replayed from
// a startup graph while its previous step is still launched, as no request ends a cycle: with graphs the model forwards
// no decode group after the startup.
TEST_CASE(
    "decode groups replayed beside a prefill sample as the eager forward while steps overlap",
    "[runtime][cuda]"
) {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const c10::Dict<std::string, at::Tensor> weights = full_weights(n_kv_heads);
    const auto requests = [] {
        std::vector<pulsarlm::request> both;
        for (const auto& [length, seed] : {std::pair<std::size_t, std::uint32_t>{32, 90}, {400, 91}}) {
            both.push_back(
                pulsarlm::request{
                    .tokens = random_tokens(length, seed),
                    .sampling = pulsarlm::sampling_params{.temperature = 0.8, .seed = seed},
                    .max_tokens = decode_tokens,
                }
            );
        }
        return both;
    };
    counted_engine eager = make_counted_engine(weights, false);
    counted_engine replayed = make_counted_engine(weights, true);
    const std::size_t eager_before = total_count(*eager.decode_group_counts);
    const std::size_t replayed_before = total_count(*replayed.decode_group_counts);
    const std::vector<served> eager_runs = serve_together(*eager.engine, requests());
    const std::vector<served> replayed_runs = serve_together(*replayed.engine, requests());
    for (std::size_t s = 0; s < eager_runs.size(); s++) {
        INFO("request " << s);
        REQUIRE(eager_runs[s].tokens.size() == decode_tokens);
        CHECK(replayed_runs[s].tokens == eager_runs[s].tokens);
        check_same_state(replayed_runs[s].end_state, eager_runs[s].end_state);
    }
    CHECK(total_count(*eager.decode_group_counts) > eager_before);
    CHECK(total_count(*replayed.decode_group_counts) == replayed_before);
}

// The drafting engine's layers are the other's then the MTP layer's, and its root digest differs with its layout, so
// pages compare by index and mass.
void check_same_history(const pulsarlm::state& got, const pulsarlm::state& expected) {
    REQUIRE(got.head.has_value() == expected.head.has_value());
    REQUIRE(got.tail == expected.tail);
    REQUIRE(got.layers.size() == expected.layers.size() + 1);
    for (std::size_t layer = 0; layer < expected.layers.size(); layer++) {
        INFO("KV layer " << layer);
        const pulsarlm::layer_sequence& g = got.layers[layer];
        const pulsarlm::layer_sequence& e = expected.layers[layer];
        REQUIRE(g.pages.size() == e.pages.size());
        for (std::size_t i = 0; i < g.pages.size(); i++) {
            INFO("page " << i);
            REQUIRE(g.pages[i].index == e.pages[i].index);
            CHECK(relative_error(g.pages[i].mass_sum, e.pages[i].mass_sum) < tolerance);
        }
    }
}

// Two ranks on one device capture no graphs, as their captures would merge.
TEST_CASE("Qwen3.5 drafting with its MTP layer emits the tokens it emits without drafts", "[runtime][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const std::size_t shard_count = GENERATE(1, 2);
    const bool decode_graphs = shard_count == 1 ? GENERATE(false, true) : false;
    const std::size_t draft_tokens = GENERATE(1, 2);
    const double temperature = GENERATE(0.0, 0.8);
    INFO(
        shard_count << " shards, " << draft_tokens << " drafts at temperature " << temperature
                    << (decode_graphs ? " with" : " without") << " graphs"
    );
    const c10::Dict<std::string, at::Tensor> weights = drafting_weights(n_kv_heads);
    const auto request = [&] {
        return pulsarlm::request{
            .tokens = random_tokens(prompt_tokens, 4),
            .sampling = pulsarlm::sampling_params{.temperature = temperature, .seed = 7},
            .max_tokens = decode_tokens,
            .eviction = eviction,
        };
    };
    std::unique_ptr<pulsarlm::engine> plain = make_engine(
        without_drafts(weights),
        n_kv_heads,
        shard_count,
        std::make_shared<store_contents>(),
        decode_graphs
    );
    std::unique_ptr<pulsarlm::engine> drafting = make_engine(
        weights,
        n_kv_heads,
        shard_count,
        std::make_shared<store_contents>(),
        decode_graphs,
        draft_tokens
    );
    served expected = serve(*plain, request());
    served got = serve(*drafting, request());

    REQUIRE(expected.tokens.size() == decode_tokens);
    CHECK(got.tokens == expected.tokens);
    check_same_history(got.end_state, expected.end_state);
    const pulsarlm::engine_stat drafted = drafting->stat();
    CHECK(drafted.accepted_draft_count > 0);
    CHECK(drafted.accepted_draft_count < drafted.drafted_token_count);
}
