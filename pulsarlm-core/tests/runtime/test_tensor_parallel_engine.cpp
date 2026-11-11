#include "pulsarlm/engine/communicator.hpp"
#include "pulsarlm/digest.hpp"
#include "pulsarlm/engine/engine.hpp"
#include "pulsarlm/engine/event_loop.hpp"
#include "pulsarlm/engine/kv_cache.hpp"
#include "pulsarlm/engine/model/qwen3_5.hpp"
#include "pulsarlm/engine/request_callbacks.hpp"
#include "pulsarlm/engine/scheduler.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>

#undef CHECK
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
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
            shard.bytes.copy_(pages[i].bytes.narrow(0, static_cast<std::int64_t>(shard.host_offset), shard.bytes.numel()));
        }
    }
}

// Keeps every page it is put, copied to the host; engines built on one contents share its pages.
struct memory_store final : pulsarlm::page_store {
    std::shared_ptr<store_contents> contents;

    task<std::vector<bool>> fetch(
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

    task<std::expected<void, pulsarlm::page_error>>
    put(std::vector<digest>&& keys,
        std::vector<sharded_page>&& pages,
        std::vector<pulsarlm::page_owner>&&,
        std::span<const cudaStream_t> streams) override {
        std::shared_ptr<store_contents> shared = contents;
        std::vector<digest> put_keys = std::move(keys);
        std::vector<sharded_page> sources = std::move(pages);
        std::vector<page> copies = copy_pages_to_host(sources, streams);
        for (std::size_t i = 0; i < put_keys.size(); i++) {
            shared->pages.insert_or_assign(put_keys[i], std::move(copies[i]));
        }
        co_return {};
    }
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

    void on_token(std::int64_t token) noexcept override {
        std::lock_guard lock(recorded_results->mutex);
        recorded_results->tokens.push_back(token);
    }

    void on_cycle_end(pulsarlm::cycle_end&&) noexcept override {}

    void on_end(std::expected<pulsarlm::state, pulsarlm::request_error>&& result) noexcept override {
        {
            std::lock_guard lock(recorded_results->mutex);
            recorded_results->end.emplace(std::move(result));
        }
        recorded_results->end_condition.notify_all();
    }

    bool cancelled() const noexcept override {
        return false;
    }
};

pulsarlm::engine_config engine_config(std::size_t shard_count) {
    return pulsarlm::engine_config{
        .model_name = "qwen3_5-test",
        .page_size = page_size,
        .kv_dtype = at::kFloat,
        .kv_cache_pool_bytes = std::size_t{64} << 20,
        .devices = std::vector<at::Device>(shard_count, at::Device(at::kCUDA, 0)),
        .stop_ids = {},
        .vocab_size = vocab,
        .prefill_tokens_per_step = 32,
    };
}

// An engine on cuda:0 of the whole model, or of its ranks' shards, over a fresh store on contents.
std::unique_ptr<pulsarlm::engine> make_engine(
    const c10::Dict<std::string, at::Tensor>& weights,
    std::int64_t kv_heads,
    std::size_t shard_count,
    const std::shared_ptr<store_contents>& contents
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
    return std::make_unique<pulsarlm::engine>(std::move(shards), engine_config(shard_count), std::move(store));
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
            return pulsarlm::request{.tokens = prompt, .max_tokens = decode_tokens, .eviction = eviction, .id = 1};
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
        const auto resumed_request = [&](const pulsarlm::state& from) {
            return pulsarlm::request{
                .initial_state = from,
                .tokens = continuation,
                .max_tokens = resume_decode_tokens,
                .eviction = eviction,
                .id = 2,
            };
        };
        whole_resumed = serve(*whole_engine, resumed_request(whole.end_state));
        split_resumed = serve(*split_engine, resumed_request(split.end_state));
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
