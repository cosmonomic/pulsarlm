#include "pulsarlm/digest.hpp"
#include "pulsarlm/engine/communicator.hpp"
#include "pulsarlm/engine/engine.hpp"
#include "pulsarlm/engine/model/qwen4_exp.hpp"
#include "pulsarlm/engine/request_callbacks.hpp"

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
#include <format>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// A Qwen4-Exp engine drafting with its MTP layer emits the tokens the engine emits without drafts, greedy or sampled,
// and reaches the same states, the mass of the queries it keeps measured as the step-by-step run measures it, while its
// verify steps accept some drafts and reject others.

namespace {

using pulsarlm::digest;
using pulsarlm::page;
using pulsarlm::qwen4_exp_layer_type;
using pulsarlm::sharded_page;
using pulsarlm::task;

using token_list = std::vector<std::int64_t>;

constexpr std::int64_t hidden = 64;
constexpr std::int64_t vocab = 300;
constexpr std::int64_t n_heads = 4;
constexpr std::int64_t n_kv_heads = 2;
constexpr std::int64_t head_dim = 64;
constexpr double rope_theta = 1e6;
constexpr double partial_rotary_factor = 0.25;
constexpr std::int64_t rotary_dim = 16;
constexpr std::int64_t key_head_dim = 32;
constexpr std::int64_t value_head_dim = 32;
constexpr std::int64_t conv_kernel = 4;
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
constexpr std::int64_t ngram_dim = 16;
constexpr std::int64_t ngram_shards = 2;
constexpr std::int64_t indexer_heads = 4;
constexpr std::int64_t indexer_dim = 32;
constexpr std::int64_t compress_ratio = 4;
constexpr double rms_eps = 1e-6;
constexpr std::size_t page_size = 16;
const std::vector<qwen4_exp_layer_type> layer_types{
    qwen4_exp_layer_type::linear_attention,
    qwen4_exp_layer_type::linear_attention,
    qwen4_exp_layer_type::indexed_attention,
    qwen4_exp_layer_type::linear_attention,
};
constexpr std::size_t prompt_tokens = 100;
constexpr std::size_t decode_tokens = 64;
constexpr double tolerance = 1e-4;
constexpr auto end_timeout = std::chrono::seconds(120);
const pulsarlm::eviction_policy eviction{
    .n_sink = page_size,
    .n_working = 2 * page_size,
    .cycle_tokens = 2 * page_size,
    .eviction_threshold = 0.5,
};

// The sizes a split across GPUs divides.
struct split_dims {
    std::int64_t key_heads;
    std::int64_t value_heads;
    std::int64_t ngram_rows;

    constexpr std::int64_t key_channels() const {
        return key_heads * key_head_dim;
    }

    constexpr std::int64_t value_channels() const {
        return value_heads * value_head_dim;
    }

    constexpr std::int64_t channels() const {
        return 2 * key_channels() + value_channels();
    }
};

// Splits 2 ways.
constexpr split_dims base_dims{.key_heads = 2, .value_heads = 4, .ngram_rows = 430};

at::TensorOptions cuda(at::ScalarType dtype) {
    return at::device(at::kCUDA).dtype(dtype);
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
    split_dims dims;
    at::ScalarType dtype;
    c10::Dict<std::string, at::Tensor> named_tensors;
    at::Tensor ngram_embedding;

    // With drafts, the MTP layer's mtp.* weights too.
    weights(at::ScalarType dtype, split_dims dims, bool drafts)
    : dims(dims),
      dtype(dtype) {
        at::manual_seed(13);
        auto add = [&](const std::string& name, at::Tensor t) { named_tensors.insert(name, t.to(cuda(dtype))); };
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
        auto decoder_layer = [&](const std::string& l, bool indexed_layer) {
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
            if (indexed_layer) {
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
            }
        };
        const std::string p = "model.language_model.";
        add(p + "embed_tokens.weight", at::randn({vocab, hidden}));
        // Tokens 0 and 1 take opposite rows far longer than the others', so the target and the MTP layer each predict
        // one of them and the drafts are sometimes right.
        at::Tensor lm_head = projection(vocab, hidden);
        at::Tensor direction = at::randn({hidden});
        lm_head[0] = 50.0 * direction / direction.norm();
        lm_head[1] = -lm_head[0];
        add("lm_head.weight", lm_head);
        hyper_connection(p + "hyper_connection_mixer.", false);
        for (std::size_t layer = 0; layer < layer_types.size(); layer++) {
            std::string l = p + "layers." + std::to_string(layer) + ".";
            decoder_layer(l, indexed(layer));
            if (!indexed(layer)) {
                add(l + "linear_attn.in_proj_qkv.weight", projection(dims.channels(), hidden));
                add(l + "linear_attn.in_proj_z.weight", projection(dims.value_channels(), hidden));
                add(l + "linear_attn.in_proj_b.weight", projection(dims.value_heads, hidden));
                add(l + "linear_attn.in_proj_a.weight", projection(dims.value_heads, hidden));
                add(l + "linear_attn.conv1d.weight", at::randn({dims.channels(), 1, conv_kernel}) / 2.0);
                add(l + "linear_attn.dt_bias", at::randn({dims.value_heads}));
                add(l + "linear_attn.A_log", at::log(at::rand({dims.value_heads}) * 15.0 + 0.5));
                add(l + "linear_attn.norm.weight", 1.0 + 0.1 * at::randn({value_head_dim}));
                add(l + "linear_attn.out_proj.weight", projection(hidden, dims.value_channels()));
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
                named_tensors.insert(e + "layer_multipliers", at::tensor(layer_multipliers).to(at::kCUDA));
                named_tensors.insert(e + "ngram_heads_vocab_sizes", at::tensor(head_vocab_sizes).to(at::kCUDA));
                named_tensors.insert(e + "ngram_heads_offsets", at::tensor(head_offsets()).to(at::kCUDA));
                ngram_embedding = at::randn({dims.ngram_rows, ngram_dim}).to(cuda(dtype));
                std::vector<at::Tensor> shards = ngram_embedding.chunk(ngram_shards);
                for (std::size_t shard = 0; shard < shards.size(); shard++) {
                    named_tensors.insert(
                        std::format("{}ngram_embedding.shard_{}.weight", e, shard),
                        shards[shard].contiguous()
                    );
                }
            }
        }
        if (drafts) {
            add("mtp.pre_fc_norm_embedding.weight", norm(hidden));
            add("mtp.pre_fc_norm_hidden.weight", norm(hc * hidden));
            add("mtp.fc_embedding.weight", projection(hidden, hidden));
            add("mtp.fc_hidden.weight", projection(hidden, hidden));
            hyper_connection("mtp.hyper_connection_mixer.", false);
            decoder_layer("mtp.layers.0.", true);
        }
    }
};

pulsarlm::qwen4_exp_config model_config(const split_dims& dims, std::int64_t experts_per_token, std::int64_t budget) {
    return {
        .vocab_size = vocab,
        .hidden_size = hidden,
        .layer_types = layer_types,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .head_dim = head_dim,
        .rope_theta = rope_theta,
        .partial_rotary_factor = partial_rotary_factor,
        .linear_n_key_heads = dims.key_heads,
        .linear_n_value_heads = dims.value_heads,
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

    void on_token(std::int64_t token) noexcept override {
        std::lock_guard lock(recorded_results->mutex);
        recorded_results->tokens.push_back(token);
    }

    void on_cycle_end(pulsarlm::cycle_end&&) noexcept override {}

    void on_end(std::expected<pulsarlm::request_result, pulsarlm::request_error>&& result) noexcept override {
        {
            std::lock_guard lock(recorded_results->mutex);
            recorded_results->end.emplace(std::move(result).transform([](pulsarlm::request_result&& ended) {
                return *ended.state;
            }));
        }
        recorded_results->end_condition.notify_all();
    }

    bool cancelled() const noexcept override {
        return false;
    }
};

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
    if (e.exception()) {
        try {
            std::rethrow_exception(e.exception());
        } catch (const std::exception& error) {
            FAIL("the engine's loop exited: " << error.what());
        }
    }
    REQUIRE(recorded_results->end->has_value());
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

// The same pages in every layer, in view order, with their sums and moments close. Drafting lays the recurrent layers
// out for the drafts, which changes the root digest, so pages are compared by index.
void check_same_state(const pulsarlm::state& got, const pulsarlm::state& expected) {
    REQUIRE(got.head.has_value() == expected.head.has_value());
    REQUIRE(got.tail == expected.tail);
    REQUIRE(got.layers.size() == expected.layers.size());
    for (std::size_t layer = 0; layer < got.layers.size(); layer++) {
        INFO("KV layer " << layer);
        const pulsarlm::layer_sequence& g = got.layers[layer];
        const pulsarlm::layer_sequence& e = expected.layers[layer];
        REQUIRE(g.pages.size() == e.pages.size());
        for (std::size_t i = 0; i < g.pages.size(); i++) {
            INFO("page " << i);
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

at::Tensor part(
    const at::Tensor& t,
    std::int64_t dim,
    std::int64_t first,
    std::int64_t count,
    std::size_t rank,
    std::size_t gpus
) {
    const std::int64_t share = count / static_cast<std::int64_t>(gpus);
    return t.narrow(dim, first + static_cast<std::int64_t>(rank) * share, share);
}

// The vocabulary padded to a multiple of 64, as vLLM pads it before splitting it.
constexpr std::int64_t padded_vocab = (vocab + 63) / 64 * 64;

// Of a whole-model weight, the part GPU rank of gpus holds, as vLLM v0.31.0 splits Qwen4-Exp, whole experts with
// expert_parallel.
at::Tensor rank_part(
    const split_dims& dims,
    const std::string& name,
    const at::Tensor& t,
    std::size_t rank,
    std::size_t gpus,
    bool expert_parallel
) {
    const std::int64_t key_dim = dims.key_channels();
    const std::int64_t value_dim = dims.value_channels();
    auto ends = [&](const char* suffix) { return name.ends_with(suffix); };
    if (ends("embed_tokens.weight") || ends("lm_head.weight")) {
        at::Tensor padded = at::cat({t, at::zeros({padded_vocab - vocab, t.size(1)}, t.options())});
        return part(padded, 0, 0, padded_vocab, rank, gpus);
    }
    if (ends("self_attn.q_proj.weight")) {
        return part(t, 0, 0, t.size(0), rank, gpus);
    }
    if (ends("self_attn.k_proj.weight") || ends("self_attn.v_proj.weight")) {
        if (static_cast<std::int64_t>(gpus) <= n_kv_heads) {
            return part(t, 0, 0, t.size(0), rank, gpus);
        }
        const auto head = static_cast<std::int64_t>(rank) / (static_cast<std::int64_t>(gpus) / n_kv_heads);
        return t.narrow(0, head * head_dim, head_dim);
    }
    if (ends("self_attn.o_proj.weight") || ends("linear_attn.out_proj.weight") ||
        ends("mlp.shared_expert.down_proj.weight")) {
        return part(t, 1, 0, t.size(1), rank, gpus);
    }
    if (ends("linear_attn.in_proj_qkv.weight") || ends("linear_attn.conv1d.weight")) {
        return at::cat({
            part(t, 0, 0, key_dim, rank, gpus),
            part(t, 0, key_dim, key_dim, rank, gpus),
            part(t, 0, 2 * key_dim, value_dim, rank, gpus),
        });
    }
    if (ends("linear_attn.in_proj_z.weight") || ends("linear_attn.in_proj_b.weight") ||
        ends("linear_attn.in_proj_a.weight") || ends("linear_attn.dt_bias") || ends("linear_attn.A_log") ||
        ends("mlp.shared_expert.gate_proj.weight") || ends("mlp.shared_expert.up_proj.weight")) {
        return part(t, 0, 0, t.size(0), rank, gpus);
    }
    if (expert_parallel && (ends("mlp.experts.gate_up_proj") || ends("mlp.experts.down_proj"))) {
        return part(t, 0, 0, t.size(0), rank, gpus);
    }
    if (ends("mlp.experts.gate_up_proj")) {
        const std::int64_t intermediate = t.size(1) / 2;
        return at::cat(
            {part(t, 1, 0, intermediate, rank, gpus), part(t, 1, intermediate, intermediate, rank, gpus)},
            1
        );
    }
    if (ends("mlp.experts.down_proj")) {
        return part(t, 2, 0, t.size(2), rank, gpus);
    }
    return t;
}

// GPU rank's weights: each weight's part, and its rows of the n-gram table as two shards.
c10::Dict<std::string, at::Tensor>
rank_weights(const weights& w, std::size_t rank, std::size_t gpus, bool expert_parallel) {
    c10::Dict<std::string, at::Tensor> parts;
    for (const auto& entry : w.named_tensors) {
        if (entry.key().find("ngram_embedding.shard_") == std::string::npos) {
            parts.insert(
                entry.key(),
                rank_part(w.dims, entry.key(), entry.value(), rank, gpus, expert_parallel).contiguous()
            );
        }
    }
    const std::string table = std::format(
        "model.language_model.layers.{}.ple.ple_embedding.ngram_embedding.",
        ple_layer
    );
    std::vector<at::Tensor> shards = part(w.ngram_embedding, 0, 0, w.dims.ngram_rows, rank, gpus).chunk(ngram_shards);
    for (std::size_t shard = 0; shard < shards.size(); shard++) {
        parts.insert(std::format("{}shard_{}.weight", table, shard), shards[shard].contiguous());
    }
    return parts;
}

// Counts the verify groups it forwards and their rows.
class counting_model final : public pulsarlm::qwen4_exp_model {
  public:
    std::shared_ptr<std::size_t> verify_steps = std::make_shared<std::size_t>(0);
    std::shared_ptr<std::size_t> verify_rows = std::make_shared<std::size_t>(0);

    using pulsarlm::qwen4_exp_model::qwen4_exp_model;

    pulsarlm::step_logits forward(const pulsarlm::step_batch& batch, std::span<const pulsarlm::kv> lines) override {
        if (batch.verify_group) {
            (*verify_steps)++;
            *verify_rows += static_cast<std::size_t>(batch.verify_group->tokens.size(0));
        }
        return pulsarlm::qwen4_exp_model::forward(batch, lines);
    }
};

// With decode_graphs, decode and verify groups replay graphs over views of up to 1024 tokens.
pulsarlm::engine_config engine_config(
    std::size_t draft_tokens,
    std::vector<std::int64_t> stop_ids,
    bool decode_graphs,
    std::size_t shard_count,
    at::ScalarType kv_dtype
) {
    return pulsarlm::engine_config{
        .model_name = "qwen4_exp-test",
        .model_hash = "1",
        .page_size = page_size,
        .kv_dtype = kv_dtype,
        .kv_cache_pool_bytes = std::size_t{64} << 20,
        .devices = std::vector<at::Device>(shard_count, at::Device(at::kCUDA, 0)),
        .stop_ids = std::move(stop_ids),
        .prefill_tokens_per_step = 32,
        .decode_graphs = decode_graphs,
        .max_context_window = decode_graphs ? std::optional<std::size_t>(1024) : std::nullopt,
        .draft_tokens = draft_tokens,
    };
}

struct drafting_engine {
    std::unique_ptr<pulsarlm::engine> engine;
    std::shared_ptr<std::size_t> verify_steps;
    std::shared_ptr<std::size_t> verify_rows;
};

// Over a store on contents, a fresh one without; split over shard_count ranks on one device, rank 0's counts kept.
drafting_engine make_engine(
    const weights& w,
    std::size_t draft_tokens,
    std::vector<std::int64_t> stop_ids,
    bool decode_graphs,
    std::shared_ptr<store_contents> contents = std::make_shared<store_contents>(),
    std::size_t shard_count = 1
) {
    std::vector<std::unique_ptr<pulsarlm::model>> shards;
    drafting_engine made;
    if (shard_count == 1) {
        auto model = std::make_unique<counting_model>(model_config(w.dims, 2, 8), w.named_tensors);
        made.verify_steps = model->verify_steps;
        made.verify_rows = model->verify_rows;
        shards.push_back(std::move(model));
    } else {
        std::vector<pulsarlm::communicator> comms = pulsarlm::communicator::make_on_one_device(
            at::Device(at::kCUDA, 0),
            shard_count
        );
        for (std::size_t rank = 0; rank < shard_count; rank++) {
            auto model = std::make_unique<counting_model>(
                model_config(w.dims, 2, 8),
                rank_weights(w, rank, shard_count, false),
                comms[rank]
            );
            if (rank == 0) {
                made.verify_steps = model->verify_steps;
                made.verify_rows = model->verify_rows;
            }
            shards.push_back(std::move(model));
        }
    }
    auto store = std::make_unique<memory_store>();
    store->contents = std::move(contents);
    made.engine = std::make_unique<pulsarlm::engine>(
        std::move(shards),
        vocab,
        engine_config(draft_tokens, std::move(stop_ids), decode_graphs, shard_count, w.dtype),
        std::move(store)
    );
    return made;
}

}  // namespace

TEST_CASE("Qwen4-Exp drafting emits the tokens and reaches the state it does without drafts", "[runtime][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const std::size_t draft_tokens = GENERATE(1, 2);
    const double temperature = GENERATE(0.0, 0.8);
    const bool decode_graphs = GENERATE(false, true);
    INFO(
        draft_tokens << " drafts at temperature " << temperature << (decode_graphs ? " with" : " without") << " graphs"
    );
    const weights w(at::kFloat, base_dims, /*drafts=*/true);
    const auto request = [&] {
        return pulsarlm::request{
            .tokens = random_tokens(prompt_tokens, 1),
            .sampling = pulsarlm::sampling_params{.temperature = temperature, .seed = 7},
            .max_tokens = decode_tokens,
            .eviction = eviction,
        };
    };
    drafting_engine plain = make_engine(w, 0, {}, decode_graphs);
    drafting_engine drafting = make_engine(w, draft_tokens, {}, decode_graphs);
    served expected = serve(*plain.engine, request());
    served got = serve(*drafting.engine, request());

    REQUIRE(expected.tokens.size() == decode_tokens);
    CHECK(got.tokens == expected.tokens);
    check_same_state(got.end_state, expected.end_state);
    CHECK(*plain.verify_steps == 0);
    // Fewer verify steps than tokens accept drafts, and more rows than tokens reject them. A graph replays the forward
    // it captured without calling it, so the counts are a run's without graphs.
    if (!decode_graphs) {
        CHECK(*drafting.verify_steps < decode_tokens - 1);
        CHECK(*drafting.verify_rows > decode_tokens - 1);
    }
}

TEST_CASE("Qwen4-Exp drafting stops at the stop token it stops at without drafts", "[runtime][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const std::size_t draft_tokens = GENERATE(1, 2);
    const weights w(at::kFloat, base_dims, /*drafts=*/true);
    const auto request = [&] {
        return pulsarlm::request{.tokens = random_tokens(prompt_tokens, 2), .max_tokens = decode_tokens};
    };
    // The predicted tokens are mostly 0 and 1: 1 stops the request a few tokens in.
    drafting_engine plain = make_engine(w, 0, {1}, false);
    drafting_engine drafting = make_engine(w, draft_tokens, {1}, false);
    served expected = serve(*plain.engine, request());
    served got = serve(*drafting.engine, request());

    REQUIRE(!expected.tokens.empty());
    REQUIRE(expected.tokens.back() == 1);
    CHECK(got.tokens == expected.tokens);
    check_same_state(got.end_state, expected.end_state);
}

// A fresh engine resumes the first request's end state from the store, its first draft rows reading the streams the
// stored page keeps.
TEST_CASE("Qwen4-Exp drafting resumes a stored state as it does without drafts", "[runtime][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const std::size_t draft_tokens = GENERATE(1, 2);
    const weights w(at::kFloat, base_dims, /*drafts=*/true);
    const token_list prompt = random_tokens(prompt_tokens, 3);
    const token_list continuation = random_tokens(20, 4);
    const auto resumed_request = [&](const served& first) {
        token_list history = prompt;
        history.insert(history.end(), first.tokens.begin(), first.tokens.end() - 1);
        history.insert(history.end(), continuation.begin(), continuation.end());
        return pulsarlm::request{.tokens = history, .max_tokens = 32, .eviction = eviction};
    };
    auto plain_contents = std::make_shared<store_contents>();
    auto drafting_contents = std::make_shared<store_contents>();
    served plain_first = serve(
        *make_engine(w, 0, {}, false, plain_contents).engine,
        pulsarlm::request{.tokens = prompt, .max_tokens = decode_tokens, .eviction = eviction}
    );
    served drafting_first = serve(
        *make_engine(w, draft_tokens, {}, false, drafting_contents).engine,
        pulsarlm::request{.tokens = prompt, .max_tokens = decode_tokens, .eviction = eviction}
    );
    REQUIRE(drafting_first.tokens == plain_first.tokens);

    drafting_engine drafting = make_engine(w, draft_tokens, {}, false, drafting_contents);
    served expected = serve(*make_engine(w, 0, {}, false, plain_contents).engine, resumed_request(plain_first));
    served got = serve(*drafting.engine, resumed_request(drafting_first));

    REQUIRE(plain_contents->fetched_pages > 0);
    CHECK(drafting_contents->fetched_pages == plain_contents->fetched_pages);
    CHECK(got.tokens == expected.tokens);
    check_same_state(got.end_state, expected.end_state);
    CHECK(*drafting.verify_steps > 0);
}

// Graphs are not captured: two ranks' captures on one device would merge.
TEST_CASE("Qwen4-Exp split over two ranks drafts the tokens it emits without drafts", "[runtime][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const double temperature = GENERATE(0.0, 0.8);
    const bool decode_graphs = false;
    INFO("temperature " << temperature);
    const weights w(at::kFloat, base_dims, /*drafts=*/true);
    const auto request = [&] {
        return pulsarlm::request{
            .tokens = random_tokens(prompt_tokens, 5),
            .sampling = pulsarlm::sampling_params{.temperature = temperature, .seed = 7},
            .max_tokens = decode_tokens,
            .eviction = eviction,
        };
    };
    served expected = serve(*make_engine(w, 0, {}, decode_graphs).engine, request());
    drafting_engine drafting = make_engine(w, 1, {}, decode_graphs, std::make_shared<store_contents>(), 2);
    served got = serve(*drafting.engine, request());

    REQUIRE(expected.tokens.size() == decode_tokens);
    CHECK(got.tokens == expected.tokens);
    check_same_state(got.end_state, expected.end_state);
    CHECK(*drafting.verify_steps < decode_tokens - 1);
}

// Each token a drafting bf16 engine emits greedily is the one an engine without drafts emits after the same tokens,
// but at near-ties of bf16's rounding.
TEST_CASE("Qwen4-Exp drafting in bf16 emits the greedy token after each prefix", "[runtime][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const weights w(at::kBFloat16, base_dims, /*drafts=*/true);
    const token_list prompt = random_tokens(prompt_tokens, 6);
    drafting_engine drafting = make_engine(w, 1, {}, true);
    served got = serve(*drafting.engine, pulsarlm::request{.tokens = prompt, .max_tokens = decode_tokens});
    drafting_engine plain = make_engine(w, 0, {}, true);
    std::size_t disagreements = 0;
    for (std::size_t i = 0; i < got.tokens.size(); i++) {
        token_list prefix = prompt;
        prefix.insert(prefix.end(), got.tokens.begin(), got.tokens.begin() + static_cast<std::ptrdiff_t>(i));
        served next = serve(*plain.engine, pulsarlm::request{.tokens = prefix, .max_tokens = 1});
        if (next.tokens.front() != got.tokens[i]) {
            disagreements++;
            UNSCOPED_INFO("position " << i << ": drafting " << got.tokens[i] << ", plain " << next.tokens.front());
        }
    }
    served whole = serve(*plain.engine, pulsarlm::request{.tokens = prompt, .max_tokens = decode_tokens});
    std::size_t same_prefix = 0;
    while (same_prefix < whole.tokens.size() && whole.tokens[same_prefix] == got.tokens[same_prefix]) {
        same_prefix++;
    }
    INFO("a run without drafts agrees for " << same_prefix << " of " << got.tokens.size() << " tokens");
    CHECK(disagreements == 0);
}
