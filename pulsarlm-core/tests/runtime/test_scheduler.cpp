#include "pulsarlm/digest.hpp"
#include "pulsarlm/engine/event_loop.hpp"
#include "pulsarlm/engine/kv_cache.hpp"
#include "pulsarlm/engine/model/interface.hpp"
#include "pulsarlm/engine/model/paged_forward.hpp"
#include "pulsarlm/engine/request_callbacks.hpp"
#include "pulsarlm/engine/scheduler.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#undef CHECK
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using pulsarlm::digest;
using pulsarlm::digest_hash;
using pulsarlm::event_loop;
using pulsarlm::page;
using pulsarlm::request;
using pulsarlm::request_error;
using pulsarlm::result_writer;
using pulsarlm::sharded_page;
using pulsarlm::task;

using token_list = std::vector<std::int64_t>;

using state_pointer = std::shared_ptr<const pulsarlm::state>;
using lease_id_type = pulsarlm::state::lease_id_type;

struct written_state {
    lease_id_type lease_id;
    state_pointer s;
};

struct state_read {
    token_list tokens;
    std::vector<digest> page_digests;
};

struct store_contents {
    std::unordered_map<digest, page, digest_hash> pages;
    std::vector<std::vector<digest>> fetches;
    std::size_t puts = 0;
    std::vector<std::pair<lease_id_type, digest>> owners;
    std::vector<state_pointer> states;
    std::vector<state_read> state_reads;
    std::vector<lease_id_type> leases;
    std::vector<written_state> written_states;
    std::vector<lease_id_type> releases;
    lease_id_type next_lease_id = 100;
    bool lookup_fails = false;
    // The put, counting from 1, that fails with oom.
    std::optional<std::size_t> failing_put;
};

struct request_results {
    token_list tokens;
    std::vector<pulsarlm::cycle_end> cycle_ends;
    std::optional<std::expected<pulsarlm::state, request_error>> end;
    state_pointer end_state;
    std::size_t total_tokens = 0;
    std::size_t cached_tokens = 0;
    std::size_t device_cached_tokens = 0;
    std::size_t peak_device_memory = 0;
    bool cancelled = false;
    std::shared_ptr<const store_contents> observed_store;
    std::vector<std::size_t> writes_at_cycle_ends;
    std::optional<std::size_t> writes_at_end;
    std::optional<std::size_t> releases_at_end;
};

class recording_callbacks final : public pulsarlm::request_callbacks {
    std::shared_ptr<request_results> recorded_results;

    std::size_t observed_writes() const noexcept {
        return recorded_results->observed_store ? recorded_results->observed_store->written_states.size() : 0;
    }

  public:
    explicit recording_callbacks(std::shared_ptr<request_results> recorded_results)
    : recorded_results(std::move(recorded_results)) {}

    void on_token(std::int64_t token) noexcept override {
        recorded_results->tokens.push_back(token);
    }

    void on_cycle_end(pulsarlm::cycle_end&& ended) noexcept override {
        recorded_results->writes_at_cycle_ends.push_back(observed_writes());
        recorded_results->cycle_ends.push_back(std::move(ended));
    }

    void on_end(std::expected<pulsarlm::request_result, request_error>&& result) noexcept override {
        recorded_results->writes_at_end = observed_writes();
        if (recorded_results->observed_store) {
            recorded_results->releases_at_end = recorded_results->observed_store->releases.size();
        }
        if (result) {
            recorded_results->total_tokens = result->total_tokens;
            recorded_results->cached_tokens = result->cached_tokens;
            recorded_results->device_cached_tokens = result->device_cached_tokens;
            recorded_results->peak_device_memory = result->peak_device_memory;
            recorded_results->end_state = result->state;
        }
        recorded_results->end.emplace(std::move(result).transform([](pulsarlm::request_result&& ended) {
            return *ended.state;
        }));
    }

    bool cancelled() const noexcept override {
        return recorded_results->cancelled;
    }
};

result_writer make_writer(const std::shared_ptr<request_results>& recorded_results) {
    return result_writer(std::make_unique<recording_callbacks>(recorded_results));
}

constexpr std::size_t page_size = 4;
constexpr std::int64_t vocab = 1000;
// sum_model predicts tokens below vocab only.

const std::vector<pulsarlm::kv_layer> two_full_layers{
    pulsarlm::kv_layer{.n_kv_heads = 1, .head_dim = 2, .n_q_heads = 1},
    pulsarlm::kv_layer{.n_kv_heads = 1, .head_dim = 2, .n_q_heads = 1},
};

const std::vector<pulsarlm::kv_layer> recurrent_and_full{
    pulsarlm::kv_layer{
        .n_kv_heads = 1,
        .head_dim = 2,
        .state = pulsarlm::recurrent_state{.sizes = {1}, .dtype = at::kFloat},
    },
    pulsarlm::kv_layer{.n_kv_heads = 1, .head_dim = 2, .n_q_heads = 1},
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

bool tail_continues(const pulsarlm::state& s, const token_list& tokens, std::size_t start) {
    return start + s.tail.size() < tokens.size() &&
        std::equal(s.tail.begin(), s.tail.end(), tokens.begin() + static_cast<std::ptrdiff_t>(start));
}

// Keeps every page it is written, copied to pageable host memory, and every state, one per head and tail. A lease never
// drops a page.
struct memory_store final : pulsarlm::page_store {
    std::shared_ptr<store_contents> contents = std::make_shared<store_contents>();

    // The state holding head and tail replaces the one holding the same.
    void keep_state(state_pointer s) const {
        std::erase_if(contents->states, [&](const state_pointer& stored_state) {
            return stored_state->head == s->head && stored_state->tail == s->tail;
        });
        contents->states.push_back(std::move(s));
    }

    task<std::vector<bool>> read_pages(
        std::span<const digest> hashes,
        std::vector<sharded_page>&& lines,
        std::span<const cudaStream_t> streams
    ) override {
        std::shared_ptr<store_contents> shared = contents;
        std::vector<digest> keys(hashes.begin(), hashes.end());
        std::vector<sharded_page> targets = std::move(lines);
        shared->fetches.push_back(keys);
        std::vector<bool> page_found(keys.size());
        std::vector<sharded_page> hit_targets;
        std::vector<page> hit_pages;
        for (std::size_t i = 0; i < keys.size(); i++) {
            if (auto it = shared->pages.find(keys[i]); it != shared->pages.end()) {
                page_found[i] = true;
                hit_targets.push_back(targets[i]);
                hit_pages.push_back(it->second);
            }
        }
        copy_pages_from_host(hit_targets, hit_pages, streams);
        co_return page_found;
    }

    task<std::expected<void, pulsarlm::page_error>> write_pages(
        std::vector<digest>&& keys,
        std::vector<sharded_page>&& pages,
        std::vector<std::pair<lease_id_type, page_index_type>>&& owners,
        std::span<const cudaStream_t> streams
    ) override {
        std::shared_ptr<store_contents> shared = contents;
        std::vector<digest> put_keys = std::move(keys);
        std::vector<sharded_page> sources = std::move(pages);
        shared->puts++;
        if (shared->failing_put == shared->puts) {
            co_return std::unexpected(pulsarlm::page_error::oom);
        }
        for (const auto& [lease_id, index] : owners) {
            shared->owners.emplace_back(lease_id, put_keys[index]);
        }
        std::vector<page> copies = copy_pages_to_host(sources, streams);
        for (std::size_t i = 0; i < put_keys.size(); i++) {
            shared->pages.insert_or_assign(put_keys[i], std::move(copies[i]));
        }
        co_return {};
    }

    task<std::expected<std::pair<state_pointer, lease_id_type>, pulsarlm::page_error>>
    read_state(std::span<const std::int64_t> tokens, std::span<const digest> page_digests) override {
        std::shared_ptr<store_contents> shared = contents;
        token_list prompt(tokens.begin(), tokens.end());
        shared->state_reads.push_back(
            state_read{
                .tokens = prompt,
                .page_digests = std::vector<digest>(page_digests.begin(), page_digests.end()),
            }
        );
        if (shared->lookup_fails) {
            co_return std::unexpected(pulsarlm::page_error::miss);
        }
        lease_id_type lease_id = shared->next_lease_id++;
        shared->leases.push_back(lease_id);
        for (std::size_t pages = page_digests.size() + 1; pages > 0; pages--) {
            std::optional<digest> head = pages > 1 ? std::optional(page_digests[pages - 2]) : std::optional<digest>();
            state_pointer longest;
            for (const state_pointer& s : shared->states) {
                if (s->head == head && tail_continues(*s, prompt, (pages - 1) * page_size) &&
                    (!longest || s->tail.size() > longest->tail.size())) {
                    longest = s;
                }
            }
            if (longest) {
                co_return std::pair(longest, lease_id);
            }
        }
        co_return std::pair(std::make_shared<const pulsarlm::state>(), lease_id);
    }

    // A state names only pages the store has.
    task<void> write_state(lease_id_type lease_id, state_pointer&& s) override {
        for (std::size_t layer = 0; layer < s->layers.size(); layer++) {
            for (const pulsarlm::active_page& named : s->layers[layer].pages) {
                CHECK(contents->pages.contains(pulsarlm::store_key(named.hash, layer)));
            }
        }
        contents->written_states.push_back(written_state{.lease_id = lease_id, .s = s});
        keep_state(std::move(s));
        co_return;
    }

    void release(lease_id_type lease_id) noexcept override {
        contents->releases.push_back(lease_id);
    }
};

// Writes each token's id as its K and its RoPE position as its V in every attention KV layer, and requires each query's
// position to be the one its view index takes under every layer's page positions. The token after a query is
// the sum of the K of every key the query attends in the first full-attention layer, mod vocab; with a recurrent layer,
// whose state is the sum of every token forwarded, evicted or not, it is that sum mod vocab instead. In a full-attention
// layer, every query token attends every view page holding a key, with mass 1 in its one query head when the page
// holds a key important accepts for that layer, else 0. While the first full-attention layer has evicted nothing, every key a sliding layer reads must equal that layer's key at the same
// distance from the query.
class sum_model final : public pulsarlm::model {
  public:
    std::vector<pulsarlm::kv_layer> layers = two_full_layers;
    std::size_t forward_count = 0;
    std::vector<token_list> prefill_chunks;
    std::vector<std::int64_t> decode_sizes;
    // Each decode step's page_positions, on the host.
    std::vector<at::Tensor> decode_page_positions;
    std::function<void()> on_forward;
    std::function<bool(std::size_t, std::int64_t)> is_important = [](std::size_t, std::int64_t) { return false; };

    std::vector<pulsarlm::kv_layer> kv_layers() const override {
        return layers;
    }

    output forward(const batch& b, std::span<const pulsarlm::kv> lines) override {
        forward_count++;
        if (on_forward) {
            on_forward();
        }
        if (b.kind == batch_kind::prefill) {
            at::Tensor starts = b.cu_seqlens_q.cpu().to(at::kLong);
            prefill_chunks.push_back(token_list(starts.size(0) - 1));
            for (std::size_t s = 0; s < prefill_chunks.back().size(); s++) {
                prefill_chunks.back()[s] = starts[s + 1].item<std::int64_t>() - starts[s].item<std::int64_t>();
            }
        }
        if (b.kind == batch_kind::decode) {
            decode_sizes.push_back(b.tokens.size(0));
            decode_page_positions.push_back(b.page_positions.cpu());
        }
        return output{.logits = predict(b, lines)};
    }

  private:
    at::Tensor predict(const batch& g, std::span<const pulsarlm::kv> lines) const {
        std::int64_t n_tokens = g.tokens.size(0);
        at::Tensor positions = g.pos.cpu();
        at::Tensor all_tables = g.page_tables.cpu();
        at::Tensor all_page_positions = g.page_positions.cpu();
        at::Tensor key_counts = g.key_counts.cpu();
        at::Tensor query_starts = g.kind == batch_kind::decode ? at::arange(all_tables.size(1) + 1, at::kInt)
                                                               : g.cu_seqlens_q.cpu();
        auto position = positions.accessor<std::int64_t, 1>();
        auto page_position = all_page_positions.accessor<std::int32_t, 3>();
        auto key_count = key_counts.accessor<std::int32_t, 2>();
        auto query_start = query_starts.accessor<std::int32_t, 1>();
        std::size_t n_seqs = static_cast<std::size_t>(all_tables.size(1));

        std::size_t view = 0;
        while (layers[view].state || layers[view].sliding_window) {
            view++;
        }
        std::optional<std::vector<double>> state_sums;
        std::vector<at::Tensor> keys(lines.size());
        std::vector<at::Tensor> values(lines.size());
        std::vector<at::Tensor> tables;
        for (std::size_t layer = 0; layer < lines.size(); layer++) {
            tables.push_back(all_tables[static_cast<std::int64_t>(layer)]);
            if (layers[layer].state) {
                state_sums = run_state(g, lines[layer], tables.back());
                continue;
            }
            auto table = tables.back().accessor<std::int32_t, 2>();
            std::vector<std::int64_t> line_ids;
            std::vector<std::int64_t> offsets;
            for (std::size_t s = 0; s < n_seqs; s++) {
                for (auto row = static_cast<std::size_t>(query_start[s]);
                     row < static_cast<std::size_t>(query_start[s + 1]);
                     row++) {
                    std::size_t index = static_cast<std::size_t>(key_count[layer][s]) - (query_start[s + 1] - row);
                    REQUIRE(
                        position[row] ==
                        page_position[layer][s][index / page_size] + static_cast<std::int64_t>(index % page_size)
                    );
                    line_ids.push_back(table[s][index / page_size]);
                    offsets.push_back(static_cast<std::int64_t>(index % page_size));
                }
            }
            c10::List<std::optional<at::Tensor>> written{
                at::tensor(line_ids).to(g.tokens.device()),
                at::tensor(offsets).to(g.tokens.device()),
            };
            const pulsarlm::kv& layer_lines = lines[layer];
            layer_lines.k.select(2, 0).index_put_(
                written,
                g.tokens.to(at::kFloat).view({n_tokens, 1}).expand({n_tokens, 2})
            );
            layer_lines.v.select(2, 0).index_put_(
                written,
                g.pos.to(at::kFloat).view({n_tokens, 1}).expand({n_tokens, 2})
            );
            keys[layer] = layer_lines.k.select(2, 0).select(2, 0).cpu();
            values[layer] = layer_lines.v.select(2, 0).select(2, 0).cpu();
        }

        at::Tensor logits = at::zeros({n_tokens, vocab});
        auto logit = logits.accessor<float, 2>();
        auto key = keys[view].accessor<float, 2>();
        auto value = values[view].accessor<float, 2>();
        auto table = tables[view].accessor<std::int32_t, 2>();
        for (std::size_t s = 0; s < n_seqs; s++) {
            auto end_row = static_cast<std::size_t>(query_start[s + 1]);
            bool evicted = false;
            for (std::size_t j = 0; j < static_cast<std::size_t>(key_count[view][s]); j++) {
                auto key_position = page_position[view][s][j / page_size] + static_cast<std::int64_t>(j % page_size);
                evicted = evicted || key_position != static_cast<std::int64_t>(j);
            }
            for (auto row = static_cast<std::size_t>(query_start[s]); row < end_row; row++) {
                std::size_t view_index = static_cast<std::size_t>(key_count[view][s]) - (end_row - row);
                std::int64_t sum = 0;
                for (std::size_t j = 0; j <= view_index; j++) {
                    sum += static_cast<std::int64_t>(key[table[s][j / page_size]][j % page_size]);
                }
                if (state_sums) {
                    sum = static_cast<std::int64_t>((*state_sums)[row]);
                }
                logit[row][sum % vocab] = 1.0f;
                for (std::size_t layer = 0; layer < lines.size() && !evicted; layer++) {
                    std::optional<std::size_t> window = layers[layer].sliding_window;
                    if (!window) {
                        continue;
                    }
                    auto window_key = keys[layer].accessor<float, 2>();
                    auto window_table = tables[layer].accessor<std::int32_t, 2>();
                    std::size_t query = static_cast<std::size_t>(key_count[layer][s]) - (end_row - row);
                    std::size_t first = query + 1 > *window ? query + 1 - *window : 0;
                    for (std::size_t j = first; j <= query; j++) {
                        std::size_t full = view_index - (query - j);
                        REQUIRE(
                            window_key[window_table[s][j / page_size]][j % page_size] ==
                            key[table[s][full / page_size]][full % page_size]
                        );
                    }
                }
            }
        }
        if (!g.page_mass_sum.empty()) {
            at::Tensor cu_view_pages = g.cu_view_pages.cpu();
            auto first_page = cu_view_pages.accessor<std::int32_t, 2>();
            at::Tensor query_weights = g.query_weights.cpu();
            auto query_weight = query_weights.accessor<float, 1>();
            for (std::size_t layer = 0; layer < lines.size(); layer++) {
                const at::Tensor& layer_mass_sum = g.page_mass_sum[layer];
                REQUIRE(
                    layer_mass_sum.defined() ==
                    (!layers[layer].state && !layers[layer].sliding_window && !layers[layer].mirrors_held_pages)
                );
                if (!layer_mass_sum.defined()) {
                    continue;
                }
                auto layer_key = keys[layer].accessor<float, 2>();
                auto layer_table = tables[layer].accessor<std::int32_t, 2>();
                at::Tensor page_mass_sum = at::zeros(layer_mass_sum.sizes());
                at::Tensor page_weight_sum = at::zeros(g.page_weight_sum[layer].sizes());
                auto mass_sum = page_mass_sum.accessor<float, 2>();
                auto weight_sum = page_weight_sum.accessor<float, 1>();
                for (std::size_t s = 0; s < n_seqs; s++) {
                    float sequence_weight = 0.0f;
                    for (auto row = static_cast<std::size_t>(query_start[s]);
                         row < static_cast<std::size_t>(query_start[s + 1]);
                         row++) {
                        sequence_weight += query_weight[row];
                    }
                    for (std::size_t j = 0; j < static_cast<std::size_t>(key_count[layer][s]); j++) {
                        std::size_t row = first_page[layer][s] + j / page_size;
                        auto token = static_cast<std::int64_t>(layer_key[layer_table[s][j / page_size]][j % page_size]);
                        weight_sum[row] = sequence_weight;
                        if (is_important(layer, token)) {
                            mass_sum[row][0] = sequence_weight;
                        }
                    }
                }
                layer_mass_sum.copy_(page_mass_sum);
                g.page_weight_sum[layer].copy_(page_weight_sum);
            }
        }
        return pulsarlm::logit_inputs(logits.to(g.tokens.device()), g);
    }

    // Each row's sum of every token forwarded through it. Starts from the slot of each sequence's state_rows, writes
    // the sum after its last token into its last slot, and the sum at its newest page end into that page's boundary
    // line.
    static std::vector<double> run_state(const batch& g, const pulsarlm::kv& lines, const at::Tensor& rows) {
        at::Tensor tokens = g.tokens.cpu();
        at::Tensor history_positions = g.history_positions.cpu();
        at::Tensor rejected_tokens = g.rejected_tokens.cpu();
        at::Tensor query_starts = g.kind == batch_kind::decode ? at::arange(rows.size(0) + 1, at::kInt)
                                                               : g.cu_seqlens_q.cpu();
        at::Tensor states = lines.k.cpu();
        auto token = tokens.accessor<std::int64_t, 1>();
        auto history_position = history_positions.accessor<std::int64_t, 1>();
        auto rejected = rejected_tokens.accessor<std::int32_t, 1>();
        auto query_start = query_starts.accessor<std::int32_t, 1>();
        auto table = rows.accessor<std::int32_t, 2>();
        auto state = states.accessor<float, 2>();
        const std::int64_t last_slot = g.state_slots - 1;
        std::vector<double> sums;
        for (std::int64_t s = 0; s < rows.size(0); s++) {
            double sum = state[table[s][last_slot - std::min<std::int64_t>(rejected[s], last_slot)]][0];
            const std::int64_t first = history_position[s];
            const std::int64_t end = first + query_start[s + 1] - query_start[s];
            const std::int64_t newest_page_end = end / g.page_size * g.page_size;
            for (std::int32_t row = query_start[s]; row < query_start[s + 1]; row++) {
                sum += static_cast<double>(token[row]);
                sums.push_back(sum);
                const std::int64_t position = first + row - query_start[s];
                if (position + 1 == newest_page_end) {
                    const std::int64_t page = position / g.page_size;
                    state[table[s][last_slot + 1 + page % 2]][0] = static_cast<float>(sum);
                }
            }
            state[table[s][last_slot]][0] = static_cast<float>(sum);
        }
        lines.k.copy_(states);
        return sums;
    }
};

std::vector<cudaStream_t> compute_streams() {
    return {c10::cuda::getCurrentCUDAStream().stream()};
}

std::vector<cudaStream_t> copy_streams() {
    return {c10::cuda::getStreamFromPool(true).stream()};
}

std::vector<pulsarlm::page_shape> page_shapes(const std::vector<pulsarlm::kv_layer>& layers) {
    std::vector<pulsarlm::page_shape> shapes;
    for (const pulsarlm::kv_layer& layer : layers) {
        shapes.push_back(
            pulsarlm::page_shape{
                .n_layers = 1,
                .page_size = page_size,
                .n_kv_heads = layer.n_kv_heads,
                .head_dim = layer.head_dim,
                .dtype = at::kFloat,
                .values_only = layer.values_only || layer.state.has_value(),
                .state_sizes = layer.state ? layer.state->sizes : std::vector<std::int64_t>(),
            }
        );
    }
    return shapes;
}

digest root() {
    return pulsarlm::make_root_digest("sum-model", "1", page_shapes(two_full_layers));
}

digest request_root(const pulsarlm::eviction_policy& policy) {
    return pulsarlm::make_request_root_digest(root(), pulsarlm::to_pages(policy, page_size));
}

// A pool of pool_bytes, or with room for n_lines lines of every layer.
pulsarlm::kv_cache make_cache(
    const std::vector<pulsarlm::kv_layer>& layers,
    std::size_t n_lines,
    std::optional<std::size_t> pool_bytes,
    memory_store& store
) {
    std::vector<pulsarlm::page_shape> shapes = page_shapes(layers);
    std::size_t lines_bytes = 0;
    for (const pulsarlm::page_shape& shape : shapes) {
        lines_bytes += n_lines * pulsarlm::line_bytes(shape);
    }
    return pulsarlm::kv_cache(shapes, pool_bytes.value_or(lines_bytes), at::Device(at::kCUDA, 0), store);
}

struct fixture {
    memory_store backing;
    std::shared_ptr<store_contents> store;
    sum_model model;
    pulsarlm::executor exec;
    pulsarlm::scheduler sched;

    // The executor is laid out for one drafted token, so a recurrent layer keeps its state in the second of two slots;
    // no step drafts.
    static constexpr std::size_t draft_tokens = 1;

    // The pool has room for n_lines lines of every KV layer unless pool_bytes sizes it.
    explicit fixture(
        std::size_t n_lines,
        std::size_t prefill_tokens_per_step = 64,
        std::vector<std::int64_t> stop_ids = {},
        memory_store backing_store = {},
        std::vector<pulsarlm::kv_layer> layers = two_full_layers,
        std::optional<std::size_t> pool_bytes = std::nullopt
    )
    : backing(std::move(backing_store)),
      store(backing.contents),
      model(with_layers(std::move(layers))),
      exec(
          {&model},
          make_cache(model.layers, n_lines, pool_bytes, backing),
          backing,
          compute_streams(),
          copy_streams(),
          {},
          root(),
          std::move(stop_ids),
          std::nullopt,
          draft_tokens
      ),
      sched(exec, prefill_tokens_per_step) {}

    static sum_model with_layers(std::vector<pulsarlm::kv_layer> layers) {
        sum_model m;
        m.layers = std::move(layers);
        return m;
    }

    // The K and V of layer's page hash in the store.
    pulsarlm::kv stored_kv(const digest& hash, std::size_t layer) const {
        return pulsarlm::page_tensors(
            store->pages.at(pulsarlm::store_key(hash, layer)),
            page_shapes(model.layers)[layer]
        );
    }
};

std::vector<digest> keys_of(const std::vector<digest>& hashes, std::size_t layer) {
    std::vector<digest> keys;
    for (const digest& hash : hashes) {
        keys.push_back(pulsarlm::store_key(hash, layer));
    }
    return keys;
}

token_list concat(token_list front, const token_list& back) {
    front.insert(front.end(), back.begin(), back.end());
    return front;
}

std::vector<digest> concat(std::vector<digest> front, const std::vector<digest>& back) {
    front.insert(front.end(), back.begin(), back.end());
    return front;
}

token_list without_last(token_list tokens) {
    tokens.pop_back();
    return tokens;
}

// The first count tokens sum_model samples greedily after context.
token_list continuation(token_list context, std::size_t count) {
    token_list generated;
    for (std::size_t i = 0; i < count; i++) {
        std::int64_t token = std::accumulate(context.begin(), context.end(), std::int64_t{0}) % vocab;
        generated.push_back(token);
        context.push_back(token);
    }
    return generated;
}

pulsarlm::state ended_state(const request_results& result) {
    REQUIRE(result.end.has_value());
    REQUIRE(result.end->has_value());
    return **result.end;
}

request_error ended_error(const request_results& result) {
    REQUIRE(result.end.has_value());
    REQUIRE(!result.end->has_value());
    return result.end->error();
}

std::vector<digest> hashes_of(const pulsarlm::state& s, std::size_t layer = 0) {
    std::vector<digest> hashes;
    for (const pulsarlm::active_page& page : s.layers[layer].pages) {
        hashes.push_back(page.hash);
    }
    return hashes;
}

// Each page's density in the layer's one query head, of pages some query token attended.
std::vector<float> densities_of(const pulsarlm::state& s, std::size_t layer = 0) {
    std::vector<float> densities;
    for (const pulsarlm::active_page& page : s.layers[layer].pages) {
        REQUIRE(page.mass_sum.size() == 1);
        REQUIRE(page.weight_sum > 0.0f);
        densities.push_back(page.mass_sum[0] / page.weight_sum);
    }
    return densities;
}

// Eight pages: a sink page, 7 opening pages 1 and 4, 8 opening pages 2 and 5, and a working page.
const token_list eight_pages{
    100, 101, 102, 103, 7, 104, 105, 106, 8,   107, 108, 109, 110, 111, 112, 113,
    7,   114, 115, 116, 8, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
};

// Token 7 matters in layer 0 and token 8 in layer 1.
bool each_layers_own(std::size_t layer, std::int64_t token) {
    return (layer == 0 && token == 7) || (layer == 1 && token == 8);
}

// The tokens of the pages of tokens, in order.
token_list pages_of(const token_list& tokens, const std::vector<std::size_t>& pages) {
    token_list kept_tokens;
    for (std::size_t page : pages) {
        kept_tokens
            .insert(kept_tokens.end(), tokens.begin() + page * page_size, tokens.begin() + (page + 1) * page_size);
    }
    return kept_tokens;
}

// Every sequence holds every page of hashes, without mass sums.
pulsarlm::state full_layers_state(const std::vector<digest>& hashes, std::size_t n_layers) {
    pulsarlm::layer_sequence sequence;
    for (std::size_t i = 0; i < hashes.size(); i++) {
        sequence.pages.push_back(pulsarlm::active_page{.hash = hashes[i], .index = i});
    }
    return pulsarlm::state{
        .head = hashes.empty() ? std::nullopt : std::optional(hashes.back()),
        .layers = std::vector(n_layers, sequence),
    };
}

// p holds tokens as K.
bool keys_hold(const pulsarlm::kv& p, const token_list& tokens) {
    auto n_tokens = static_cast<std::int64_t>(tokens.size());
    return at::equal(p.k, at::tensor(tokens).to(at::kFloat).view({1, n_tokens, 1, 1}).expand_as(p.k));
}

// p holds tokens as K and their view indices, from first_index on, as V.
bool page_holds(const pulsarlm::kv& p, const token_list& tokens, std::size_t first_index) {
    auto n_tokens = static_cast<std::int64_t>(tokens.size());
    auto first = static_cast<std::int64_t>(first_index);
    at::Tensor v = at::arange(first, first + n_tokens).to(at::kFloat).view({1, n_tokens, 1, 1}).expand_as(p.v);
    return keys_hold(p, tokens) && at::equal(p.v, v);
}

struct submission {
    request r;
    std::shared_ptr<request_results> recorded_results;
    result_writer writer;
};

submission make_submission(request r) {
    auto recorded_results = std::make_shared<request_results>();
    return submission{.r = std::move(r), .recorded_results = recorded_results, .writer = make_writer(recorded_results)};
}

task<void> serve(pulsarlm::scheduler& sched, request r, result_writer writer, std::size_t& remaining) {
    co_await sched.submit(std::move(r), std::move(writer));
    if (--remaining == 0) {
        event_loop::current()->stop();
    }
}

std::vector<request_results> serve_submissions(pulsarlm::scheduler& sched, std::vector<submission> submissions) {
    std::size_t remaining = submissions.size();
    event_loop loop;
    loop.spawn(sched.run());
    for (submission& s : submissions) {
        loop.spawn(serve(sched, std::move(s.r), std::move(s.writer), remaining));
    }
    REQUIRE(loop.run() == nullptr);
    std::vector<request_results> results;
    for (submission& s : submissions) {
        results.push_back(*s.recorded_results);
    }
    return results;
}

std::vector<request_results> serve_all(pulsarlm::scheduler& sched, std::vector<request> requests) {
    std::vector<submission> submissions;
    for (request& r : requests) {
        submissions.push_back(make_submission(std::move(r)));
    }
    return serve_submissions(sched, std::move(submissions));
}

task<void> serve_then(
    pulsarlm::scheduler& sched,
    request first,
    std::function<request(const request_results&)> next,
    std::vector<request_results>& results
) {
    auto first_results = std::make_shared<request_results>();
    co_await sched.submit(std::move(first), make_writer(first_results));
    results.push_back(*first_results);
    auto second_results = std::make_shared<request_results>();
    co_await sched.submit(next(results.front()), make_writer(second_results));
    results.push_back(*second_results);
    event_loop::current()->stop();
}

// Serves first, then the request next makes from first's result.
std::vector<request_results>
serve_in_turn(pulsarlm::scheduler& sched, request first, std::function<request(const request_results&)> next) {
    std::vector<request_results> results;
    event_loop loop;
    loop.spawn(sched.run());
    loop.spawn(serve_then(sched, std::move(first), std::move(next), results));
    REQUIRE(loop.run() == nullptr);
    return results;
}

TEST_CASE("a request from an empty state prefills, decodes and ends with its state") {
    fixture f(16);
    token_list prompt{1, 2, 3, 4, 5, 6};
    std::vector<request_results> results = serve_all(f.sched, {request{.tokens = prompt, .max_tokens = 5}});

    token_list generated = continuation(prompt, 5);
    REQUIRE(results[0].tokens == generated);
    pulsarlm::state end = ended_state(results[0]);
    token_list forwarded = concat(prompt, without_last(generated));
    std::size_t full_tokens = forwarded.size() - forwarded.size() % page_size;
    REQUIRE((hashes_of(end) == pulsarlm::make_page_digests(root(), forwarded, page_size)));
    REQUIRE(end.tail == token_list(forwarded.begin() + full_tokens, forwarded.end()));
    REQUIRE(hashes_of(end).size() == 2);
    for (const pulsarlm::active_page& page : end.layers[0].pages) {
        REQUIRE(page.mass_sum == std::vector<float>{0.0f});
        REQUIRE(page.weight_sum == 0.0f);
    }
    for (std::size_t i = 0; i < hashes_of(end).size(); i++) {
        std::size_t first_index = i * page_size;
        token_list page_tokens(forwarded.begin() + first_index, forwarded.begin() + first_index + page_size);
        REQUIRE(page_holds(f.stored_kv(hashes_of(end)[i], 0), page_tokens, first_index));
        REQUIRE(page_holds(f.stored_kv(hashes_of(end)[i], 1), page_tokens, first_index));
    }
}

TEST_CASE("a step's new pages of every KV layer reach the store in one put") {
    fixture f(16);
    token_list prompt{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<request_results> results = serve_all(f.sched, {request{.tokens = prompt, .max_tokens = 1}});

    REQUIRE(results[0].tokens == continuation(prompt, 1));
    REQUIRE(f.model.forward_count == 1);
    REQUIRE(f.store->puts == 1);
    REQUIRE(f.store->pages.size() == 4);
}

TEST_CASE("the pages a request's steps write and the states it reaches are its lease's") {
    fixture f(16);
    token_list prompt{1, 2, 3, 4, 5, 6, 7, 8};
    serve_all(f.sched, {request{.tokens = prompt, .max_tokens = 1}});

    REQUIRE(f.store->leases == std::vector<lease_id_type>{100});
    REQUIRE(f.store->owners.size() == f.store->pages.size());
    for (const auto& [owner, key] : f.store->owners) {
        REQUIRE(owner == 100);
        REQUIRE(f.store->pages.contains(key));
    }
    REQUIRE(f.store->written_states.size() == 1);
    REQUIRE(f.store->written_states[0].lease_id == 100);
}

TEST_CASE("a request reads the state for its tokens with their page digests") {
    fixture f(16);
    token_list prompt{1, 2, 3, 4, 5, 6, 7, 8, 9};
    serve_all(f.sched, {request{.tokens = prompt, .max_tokens = 1}});

    REQUIRE(f.store->state_reads.size() == 1);
    REQUIRE(f.store->state_reads[0].tokens == prompt);
    REQUIRE((f.store->state_reads[0].page_digests == pulsarlm::make_page_digests(root(), prompt, page_size)));
}

TEST_CASE("a request's page chain starts from its initial digest and continues past its prompt") {
    fixture f(16);
    digest initial{.hi = 3, .lo = 4};
    token_list prompt{1, 2, 3, 4, 5, 6};
    std::vector<request_results> results = serve_all(
        f.sched,
        {request{.initial_digest = initial, .tokens = prompt, .max_tokens = 5}}
    );

    token_list forwarded = concat(prompt, without_last(results[0].tokens));
    std::vector<digest> chain = pulsarlm::make_page_digests(initial, forwarded, page_size);
    REQUIRE(chain.size() == 2);
    REQUIRE((f.store->state_reads[0].page_digests == std::vector{chain[0]}));
    REQUIRE((hashes_of(ended_state(results[0])) == chain));
    REQUIRE(results[0].tokens == continuation(prompt, 5));
}

TEST_CASE("each state reaches the store, and the lease is releases, before the callback that passes it") {
    fixture f(16);
    token_list prompt{1, 2, 3, 4, 5, 6};
    submission s = make_submission(request{.tokens = prompt, .max_tokens = 6, .eviction = {.cycle_tokens = 4}});
    s.recorded_results->observed_store = f.store;
    std::vector<submission> submissions;
    submissions.push_back(std::move(s));
    request_results result = serve_submissions(f.sched, std::move(submissions))[0];

    REQUIRE(result.cycle_ends.size() == 2);
    REQUIRE(result.writes_at_cycle_ends == std::vector<std::size_t>{1, 2});
    REQUIRE(result.writes_at_end == 3);
    REQUIRE(result.releases_at_end == 1);
    REQUIRE(f.store->written_states.size() == 3);
    REQUIRE(f.store->written_states[0].s == result.cycle_ends[0].state);
    REQUIRE(f.store->written_states[1].s == result.cycle_ends[1].state);
    REQUIRE(f.store->written_states[2].s == result.end_state);
}

TEST_CASE("a request continues from the end state of another in the same cache") {
    fixture f(16);
    token_list prompt{1, 2, 3, 4, 5, 6};
    token_list user{7, 8};
    std::vector<request_results> results = serve_in_turn(
        f.sched,
        request{.tokens = prompt, .max_tokens = 5},
        [&](const request_results& first) {
            return request{.tokens = concat(concat(prompt, first.tokens), user), .max_tokens = 4};
        }
    );

    token_list context = concat(concat(prompt, results[0].tokens), user);
    REQUIRE(results[1].tokens == continuation(context, 4));
    REQUIRE(results[1].cached_tokens == 8);
    REQUIRE(
        (hashes_of(ended_state(results[1])) ==
         pulsarlm::make_page_digests(root(), concat(context, without_last(results[1].tokens)), page_size))
    );
    REQUIRE(f.store->fetches.empty());
}

TEST_CASE("a request continues from pages another cache fetches from the store") {
    fixture first_engine(16);
    token_list prompt{1, 2, 3, 4, 5, 6};
    token_list user{7, 8};
    request_results first = serve_all(first_engine.sched, {request{.tokens = prompt, .max_tokens = 5}})[0];
    pulsarlm::state first_end = ended_state(first);

    memory_store shared_store;
    shared_store.contents = first_engine.store;
    fixture second_engine(16, 64, {}, shared_store);
    request_results second = serve_all(
        second_engine.sched,
        {request{.tokens = concat(concat(prompt, first.tokens), user), .max_tokens = 4}}
    )[0];

    REQUIRE(second.tokens == continuation(concat(concat(prompt, first.tokens), user), 4));
    std::vector<digest> keys = concat(keys_of(hashes_of(first_end), 0), keys_of(hashes_of(first_end), 1));
    REQUIRE((first_engine.store->fetches == std::vector<std::vector<digest>>{keys}));
}

TEST_CASE("a request's result counts its tokens and the initial state's full pages it did not forward") {
    fixture f(16);
    token_list prompt{1, 2, 3, 4, 5, 6};
    token_list user{7, 8};
    std::vector<request_results> results = serve_in_turn(
        f.sched,
        request{.tokens = prompt, .max_tokens = 5},
        [&](const request_results& first) {
            return request{.tokens = concat(concat(prompt, first.tokens), user), .max_tokens = 4};
        }
    );

    REQUIRE(results[0].total_tokens == 11);
    REQUIRE(results[0].cached_tokens == 0);
    REQUIRE(results[0].device_cached_tokens == 0);
    REQUIRE(results[1].total_tokens == 8 + 2 + 3 + 4);
    REQUIRE(results[1].cached_tokens == 8);
    REQUIRE(results[1].device_cached_tokens == 8);
}

TEST_CASE("pages fetched from the store count as cached but not device cached") {
    fixture first_engine(16);
    token_list prompt{1, 2, 3, 4, 5, 6};
    request_results first = serve_all(first_engine.sched, {request{.tokens = prompt, .max_tokens = 5}})[0];

    memory_store shared_store;
    shared_store.contents = first_engine.store;
    fixture second_engine(16, 64, {}, shared_store);
    request_results second = serve_all(
        second_engine.sched,
        {request{.tokens = concat(prompt, first.tokens), .max_tokens = 1}}
    )[0];

    REQUIRE(second.cached_tokens == 8);
    REQUIRE(second.device_cached_tokens == 0);
}

TEST_CASE("a state page missing from the store ends the request with miss") {
    fixture f(16);
    token_list prompt{1, 2, 3, 4, 5};
    f.backing.keep_state(
        std::make_shared<const pulsarlm::state>(
            full_layers_state(pulsarlm::make_page_digests(root(), prompt, page_size), 2)
        )
    );
    std::vector<request_results> results = serve_all(f.sched, {request{.tokens = prompt, .max_tokens = 1}});

    REQUIRE(results[0].tokens.empty());
    REQUIRE(ended_error(results[0]) == request_error::miss);
    REQUIRE(f.model.forward_count == 0);
    REQUIRE(f.store->releases == f.store->leases);
}

TEST_CASE("a failed state lookup ends the request with miss and leases nothing") {
    fixture f(16);
    f.store->lookup_fails = true;
    std::vector<request_results> results = serve_all(f.sched, {request{.tokens = {1, 2, 3}, .max_tokens = 1}});

    REQUIRE(results[0].tokens.empty());
    REQUIRE(ended_error(results[0]) == request_error::miss);
    REQUIRE(f.model.forward_count == 0);
    REQUIRE(f.store->state_reads.size() == 1);
    REQUIRE(f.store->leases.empty());
    REQUIRE(f.store->releases.empty());
}

TEST_CASE("a stop token ends the request after it is emitted") {
    token_list prompt{1, 2, 3};
    token_list free_run = continuation(prompt, 4);
    REQUIRE(free_run == token_list{6, 12, 24, 48});
    fixture f(16, 64, {48, 24});
    std::vector<request_results> results = serve_all(f.sched, {request{.tokens = prompt, .max_tokens = 10}});

    REQUIRE(results[0].tokens == token_list{6, 12, 24});
    pulsarlm::state end = ended_state(results[0]);
    REQUIRE((hashes_of(end) == pulsarlm::make_page_digests(root(), token_list{1, 2, 3, 6}, page_size)));
    REQUIRE(end.tail == token_list{12});
}

TEST_CASE("max_tokens 0 forwards the tokens and ends with their state") {
    fixture f(16);
    token_list prompt{1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<request_results> results = serve_all(f.sched, {request{.tokens = prompt, .max_tokens = 0}});

    REQUIRE(results[0].tokens.empty());
    pulsarlm::state end = ended_state(results[0]);
    REQUIRE((hashes_of(end) == pulsarlm::make_page_digests(root(), prompt, page_size)));
    REQUIRE(end.tail == token_list{9});
    REQUIRE(f.store->pages.size() == 4);
}

// The prefill's page goes to the store in a write the write-back does not wait for. The write-back seeing its failure
// is the first after the write ran, which depends on when the loop runs it, and at the latest the one reaching the end
// state, which waits for the sequence's page writes; either ends the request with oom and writes no state.
TEST_CASE("a page write that fails after its write-back ends the request with oom") {
    fixture f(16);
    f.store->failing_put = 1;
    token_list prompt{1, 2, 3, 4, 5};
    std::vector<request_results> results = serve_all(f.sched, {request{.tokens = prompt, .max_tokens = 10}});

    REQUIRE(ended_error(results[0]) == request_error::oom);
    REQUIRE(f.store->releases == f.store->leases);
    REQUIRE(f.store->written_states.empty());
}

TEST_CASE("a request cancelled before it runs ends with cancelled") {
    fixture f(16);
    submission s = make_submission(request{.tokens = {1, 2, 3}, .max_tokens = 10});
    s.recorded_results->cancelled = true;
    std::vector<submission> submissions;
    submissions.push_back(std::move(s));
    std::vector<request_results> results = serve_submissions(f.sched, std::move(submissions));

    REQUIRE(results[0].tokens.empty());
    REQUIRE(ended_error(results[0]) == request_error::cancelled);
    REQUIRE(f.model.forward_count == 0);
    REQUIRE(f.store->releases == f.store->leases);
    REQUIRE(f.store->written_states.empty());
}

// The third forward launches the third step before the second is written back; that write-back sees the cancellation,
// so the third step's token is dropped.
TEST_CASE("a request cancelled while decoding ends with cancelled at the next write-back") {
    fixture f(16);
    token_list prompt{1, 2, 3};
    submission s = make_submission(request{.tokens = prompt, .max_tokens = 10});
    f.model.on_forward = [&f, recorded_results = s.recorded_results] {
        if (f.model.forward_count == 3) {
            recorded_results->cancelled = true;
        }
    };
    std::vector<submission> submissions;
    submissions.push_back(std::move(s));
    std::vector<request_results> results = serve_submissions(f.sched, std::move(submissions));

    REQUIRE(results[0].tokens == continuation(prompt, 2));
    REQUIRE(f.model.forward_count == 3);
    REQUIRE(ended_error(results[0]) == request_error::cancelled);
    REQUIRE(f.store->releases == f.store->leases);
}

// The executor owns a sequence that was not stopped until its own destruction, and the sequence holds the lease.
TEST_CASE("a request's lease is released once, with the executor, when the loop stops while it runs") {
    std::shared_ptr<store_contents> store;
    auto recorded_results = std::make_shared<request_results>();
    {
        fixture f(16);
        store = f.store;
        f.model.on_forward = [] { event_loop::current()->stop(); };
        {
            event_loop loop;
            loop.spawn(f.sched.run());
            loop.spawn(f.sched.submit(request{.tokens = {1, 2, 3}, .max_tokens = 10}, make_writer(recorded_results)));
            REQUIRE(loop.run() == nullptr);
            c10::cuda::getCurrentCUDAStream().synchronize();
            REQUIRE(store->leases.size() == 1);
            REQUIRE(store->releases.empty());
        }
        REQUIRE(ended_error(*recorded_results) == request_error::loop_exited);
        REQUIRE(store->releases.empty());
    }
    REQUIRE(store->releases == store->leases);
}

TEST_CASE("a request that finds too few free lines ends with oom without waiting") {
    fixture f(3);
    token_list first{1, 2, 3, 4, 5};
    token_list second{6, 7, 8, 9, 10};
    std::vector<request_results> results = serve_all(
        f.sched,
        {request{.tokens = first, .max_tokens = 3}, request{.tokens = second, .max_tokens = 3}}
    );

    REQUIRE(results[0].tokens == continuation(first, 3));
    REQUIRE(results[1].tokens.empty());
    REQUIRE(ended_error(results[1]) == request_error::oom);
    REQUIRE(f.model.forward_count == 3);
    std::vector<lease_id_type> releases = f.store->releases;
    std::ranges::sort(releases);
    REQUIRE(releases == std::vector<lease_id_type>{100, 101});
}

TEST_CASE("a request needing more lines than the cache has ends with oom") {
    fixture f(2);
    std::vector<request_results> results = serve_all(f.sched, {request{.tokens = token_list(12, 1), .max_tokens = 1}});

    REQUIRE(results[0].tokens.empty());
    REQUIRE(ended_error(results[0]) == request_error::oom);
    REQUIRE(f.model.forward_count == 0);
}

TEST_CASE("a request that outgrows the cache alone ends with oom") {
    fixture f(2);
    token_list prompt{1, 2, 3, 4};
    std::vector<request_results> results = serve_all(
        f.sched,
        {request{.tokens = prompt, .max_tokens = 10, .eviction = {.cycle_tokens = 4}}}
    );

    REQUIRE(results[0].tokens == continuation(prompt, 5));
    REQUIRE(ended_error(results[0]) == request_error::oom);
}

TEST_CASE("a running request that finds too few free lines for its next cycle ends with oom") {
    fixture f(5);
    token_list first{1, 2, 3, 4};
    token_list second{5, 6, 7, 8};
    std::vector<request_results> results = serve_all(
        f.sched,
        {request{.tokens = first, .max_tokens = 10, .eviction = {.cycle_tokens = 4}},
         request{.tokens = second, .max_tokens = 10, .eviction = {.cycle_tokens = 4}}}
    );

    REQUIRE(results[0].tokens == continuation(first, 10));
    REQUIRE(results[1].tokens == continuation(second, 5));
    REQUIRE(ended_error(results[1]) == request_error::oom);
    REQUIRE(f.model.forward_count == 10);
}

TEST_CASE("a cycle ends every cycle_tokens tokens with the state it reached") {
    fixture f(16);
    token_list prompt{1, 2, 3, 4, 5, 6};
    std::vector<request_results> results = serve_all(
        f.sched,
        {request{.tokens = prompt, .max_tokens = 6, .eviction = {.cycle_tokens = 4}}}
    );

    token_list generated = continuation(prompt, 6);
    REQUIRE(results[0].tokens == generated);
    REQUIRE(f.model.prefill_chunks == std::vector<token_list>{{4}, {2}});
    const std::vector<pulsarlm::cycle_end>& cycle_ends = results[0].cycle_ends;
    REQUIRE(cycle_ends.size() == 2);
    token_list first_pages(prompt.begin(), prompt.begin() + 4);
    REQUIRE((hashes_of(*cycle_ends[0].state) == pulsarlm::make_page_digests(root(), first_pages, page_size)));
    REQUIRE(cycle_ends[0].state->tail.empty());
    token_list forwarded = concat(prompt, token_list(generated.begin(), generated.begin() + 2));
    REQUIRE((hashes_of(*cycle_ends[1].state) == pulsarlm::make_page_digests(root(), forwarded, page_size)));
    REQUIRE(cycle_ends[1].state->tail.empty());
}

TEST_CASE("a request resumed mid-cycle ends its cycles where an uninterrupted run over its tokens does") {
    token_list prompt{1, 2, 3, 4, 5, 6};
    token_list user{7, 8};
    fixture resumed_engine(16);
    std::vector<request_results> resumed = serve_in_turn(
        resumed_engine.sched,
        request{.tokens = prompt, .max_tokens = 2, .eviction = {.cycle_tokens = 8}},
        [&](const request_results& first) {
            return request{
                .tokens = concat(concat(prompt, first.tokens), user),
                .max_tokens = 10,
                .eviction = {.cycle_tokens = 8},
            };
        }
    );
    token_list context = concat(concat(prompt, resumed[0].tokens), user);
    fixture uninterrupted_engine(16);
    request_results uninterrupted = serve_all(
        uninterrupted_engine.sched,
        {request{.tokens = context, .max_tokens = 10, .eviction = {.cycle_tokens = 8}}}
    )[0];

    REQUIRE(resumed[0].cycle_ends.empty());
    REQUIRE(ended_state(resumed[0]).tail.size() == 3);
    REQUIRE(resumed[1].tokens == uninterrupted.tokens);
    REQUIRE(uninterrupted.cycle_ends.size() == 2);
    REQUIRE(resumed[1].cycle_ends.size() == 2);
    for (std::size_t i = 0; i < 2; i++) {
        const pulsarlm::state& resumed_end = *resumed[1].cycle_ends[i].state;
        const pulsarlm::state& uninterrupted_end = *uninterrupted.cycle_ends[i].state;
        REQUIRE((hashes_of(resumed_end) == hashes_of(uninterrupted_end)));
        REQUIRE(resumed_end.tail == uninterrupted_end.tail);
    }
}

TEST_CASE("only the queries of a cycle's last mass_tokens tokens add mass") {
    auto densities_after_first_cycle = [](std::optional<std::size_t> mass_tokens) {
        fixture f(16, 4);
        f.model.is_important = [&f](std::size_t, std::int64_t token) {
            return token == 100 && f.model.forward_count == 1;
        };
        pulsarlm::eviction_policy policy{
            .n_sink = 4,
            .n_working = 4,
            .cycle_tokens = 8,
            .mass_tokens = mass_tokens,
            .eviction_threshold = 0.5,
            .eviction_warmup = 0,
        };
        token_list prompt(eight_pages.begin(), eight_pages.begin() + 8);
        request_results result = serve_all(
            f.sched,
            {request{.tokens = prompt, .max_tokens = 2, .eviction = policy}}
        )[0];
        REQUIRE(f.model.prefill_chunks == std::vector<token_list>{{4}, {4}});
        REQUIRE(result.cycle_ends.size() == 1);
        return densities_of(*result.cycle_ends[0].state);
    };

    REQUIRE(densities_after_first_cycle(4) == std::vector<float>{0.0f, 0.0f});
    REQUIRE(densities_after_first_cycle(std::nullopt)[0] > 0.0f);
}

TEST_CASE("a state before its cycle's mass window holds the eviction state of the cycle's start") {
    fixture f(16, 4);
    f.model.is_important = [](std::size_t, std::int64_t token) { return token == 7; };
    pulsarlm::eviction_policy policy{
        .n_sink = 4,
        .n_working = 16,
        .cycle_tokens = 8,
        .mass_tokens = 4,
        .eviction_threshold = 0.5,
        .eviction_warmup = 0,
    };
    token_list prompt(eight_pages.begin(), eight_pages.begin() + 20);
    request_results result = serve_all(f.sched, {request{.tokens = prompt, .max_tokens = 1, .eviction = policy}})[0];

    REQUIRE(result.cycle_ends.size() == 2);
    const pulsarlm::state& cycle_start = *result.cycle_ends[1].state;
    pulsarlm::state end = ended_state(result);
    for (std::size_t layer = 0; layer < end.layers.size(); layer++) {
        const pulsarlm::layer_sequence& started = cycle_start.layers[layer];
        const pulsarlm::layer_sequence& ended = end.layers[layer];
        REQUIRE(ended.pages.size() == started.pages.size() + 1);
        REQUIRE(std::ranges::equal(std::span(ended.pages).first(started.pages.size()), started.pages));
        REQUIRE(ended.pages.back().weight_sum == 0.0f);
        REQUIRE((ended.density_moments == started.density_moments));
        REQUIRE((ended.score_moments == started.score_moments));
    }
    REQUIRE(cycle_start.layers[0].density_moments[0].count == 1);
}

TEST_CASE("a request resumed mid-cycle reaches the eviction state of an uninterrupted run over its tokens") {
    // The fake model weighs every key of a step by all of the step's queries, so both runs split their steps alike.
    pulsarlm::eviction_policy policy{
        .n_sink = 4,
        .n_working = 4,
        .cycle_tokens = 8,
        .eviction_threshold = 0.5,
        .eviction_warmup = 0,
    };
    auto important = [](std::size_t, std::int64_t token) { return token == 7; };
    token_list prompt(eight_pages.begin(), eight_pages.begin() + 12);
    token_list rest(eight_pages.begin() + 13, eight_pages.end());
    fixture resumed_engine(16, 4);
    resumed_engine.model.is_important = important;
    std::vector<request_results> resumed = serve_in_turn(
        resumed_engine.sched,
        request{.tokens = prompt, .max_tokens = 1, .eviction = policy},
        [&](const request_results& first) {
            return request{.tokens = concat(concat(prompt, first.tokens), rest), .max_tokens = 1, .eviction = policy};
        }
    );
    token_list context = concat(concat(prompt, resumed[0].tokens), rest);
    fixture uninterrupted_engine(16, 4);
    uninterrupted_engine.model.is_important = important;
    request_results uninterrupted = serve_all(
        uninterrupted_engine.sched,
        {request{.tokens = context, .max_tokens = 1, .eviction = policy}}
    )[0];

    REQUIRE(resumed[0].cycle_ends.size() == 1);
    REQUIRE(uninterrupted.cycle_ends.size() == 3);
    REQUIRE(resumed[1].cycle_ends.size() == 2);
    for (std::size_t i = 0; i < 2; i++) {
        const pulsarlm::state& resumed_end = *resumed[1].cycle_ends[i].state;
        const pulsarlm::state& uninterrupted_end = *uninterrupted.cycle_ends[i + 1].state;
        REQUIRE((hashes_of(resumed_end) == hashes_of(uninterrupted_end)));
        std::vector<float> resumed_densities = densities_of(resumed_end);
        std::vector<float> uninterrupted_densities = densities_of(uninterrupted_end);
        for (std::size_t page = 0; page < resumed_densities.size(); page++) {
            REQUIRE_THAT(resumed_densities[page], Catch::Matchers::WithinAbs(uninterrupted_densities[page], 1e-6));
        }
        REQUIRE(resumed_end.layers[0].density_moments[0].count == uninterrupted_end.layers[0].density_moments[0].count);
    }
}

// The first request writes a state at its cycle end 8 and computes page 2, tokens [8, 12), which the second request's
// tokens share before they diverge at 12, where its cycle's mass window begins.
struct diverging_requests {
    request_results first;
    request_results second;
    std::vector<token_list> second_prefill_chunks;
    std::shared_ptr<store_contents> store;
};

const token_list shared_tokens(eight_pages.begin(), eight_pages.begin() + 12);
const token_list diverging_tokens = concat(shared_tokens, {500, 501, 502, 503, 504});

pulsarlm::eviction_policy windowed_policy(std::optional<std::size_t> mass_tokens) {
    return pulsarlm::eviction_policy{
        .n_sink = 4,
        .n_working = 16,
        .cycle_tokens = 8,
        .mass_tokens = mass_tokens,
        .eviction_threshold = 0.5,
        .eviction_warmup = 0,
    };
}

diverging_requests
serve_diverging(const pulsarlm::eviction_policy& policy, std::vector<pulsarlm::kv_layer> layers = two_full_layers) {
    fixture f(16, 4, {}, {}, std::move(layers));
    f.model.is_important = [](std::size_t, std::int64_t token) { return token == 7; };
    std::vector<request_results> results = serve_in_turn(
        f.sched,
        request{.tokens = concat(shared_tokens, {300, 301}), .max_tokens = 2, .eviction = policy},
        [&](const request_results&) {
            f.model.prefill_chunks.clear();
            return request{.tokens = diverging_tokens, .max_tokens = 2, .eviction = policy};
        }
    );
    return diverging_requests{
        .first = results[0],
        .second = results[1],
        .second_prefill_chunks = f.model.prefill_chunks,
        .store = f.store,
    };
}

TEST_CASE("a request takes the hot pages after its state up to its cycle's mass window") {
    bool sliding = GENERATE(false, true);
    std::vector<pulsarlm::kv_layer> layers = two_full_layers;
    if (sliding) {
        layers[1] = pulsarlm::kv_layer{.n_kv_heads = 1, .head_dim = 2, .sliding_window = 9};
    }
    pulsarlm::eviction_policy policy = windowed_policy(4);
    diverging_requests served = serve_diverging(policy, layers);
    fixture uninterrupted_engine(16, 4, {}, {}, layers);
    uninterrupted_engine.model.is_important = [](std::size_t, std::int64_t token) { return token == 7; };
    request_results uninterrupted = serve_all(
        uninterrupted_engine.sched,
        {request{.tokens = diverging_tokens, .max_tokens = 2, .eviction = policy}}
    )[0];

    REQUIRE(served.first.cycle_ends.size() == 1);
    REQUIRE(served.second_prefill_chunks == std::vector<token_list>{{4}});
    REQUIRE(served.second.cached_tokens == 12);
    REQUIRE(served.second.device_cached_tokens == 12);
    REQUIRE(served.second.tokens == uninterrupted.tokens);
    REQUIRE(served.second.cycle_ends.size() == 1);
    REQUIRE(uninterrupted.cycle_ends.size() == 2);
    const pulsarlm::state& reached = *served.second.cycle_ends[0].state;
    const pulsarlm::state& uninterrupted_reached = *uninterrupted.cycle_ends[1].state;
    REQUIRE((reached.head == uninterrupted_reached.head));
    REQUIRE((reached.layers == uninterrupted_reached.layers));

    digest taken_page = pulsarlm::make_page_digests(request_root(policy), shared_tokens, page_size)[2];
    lease_id_type second_lease = served.store->leases[1];
    for (std::size_t layer = 0; layer < 2; layer++) {
        REQUIRE(
            std::ranges::contains(served.store->owners, std::pair(second_lease, pulsarlm::store_key(taken_page, layer)))
        );
    }
}

TEST_CASE("a request takes no hot pages after a state inside its cycle's mass window or with recurrent layers") {
    diverging_requests whole_cycle = serve_diverging(windowed_policy(std::nullopt));
    REQUIRE(whole_cycle.second_prefill_chunks == std::vector<token_list>{{4}, {4}});
    REQUIRE(whole_cycle.second.cached_tokens == 8);

    diverging_requests recurrent = serve_diverging(windowed_policy(4), recurrent_and_full);
    REQUIRE(recurrent.second_prefill_chunks == std::vector<token_list>{{4}, {4}});
    REQUIRE(recurrent.second.cached_tokens == 8);
}

TEST_CASE("a cycle's end evicts the pages scoring below the threshold outside the sink and working pages") {
    fixture f(16);
    f.model.is_important = [](std::size_t, std::int64_t token) { return token == 7; };
    pulsarlm::eviction_policy
        policy{.n_sink = 4, .n_working = 4, .cycle_tokens = 4, .eviction_threshold = 0.5, .eviction_warmup = 0};
    std::vector<request_results> results = serve_all(
        f.sched,
        {request{.tokens = eight_pages, .max_tokens = 2, .eviction = policy}}
    );

    // Each page is sampled once the next one completes. Pages 2 and 3 go once page 5 completes, page 5 once page 6
    // does, so the last prefill step attends pages 0, 1, 4 and 6 and its own page 7, and page 6 goes at its end.
    std::vector<digest> chain = pulsarlm::make_page_digests(request_root(policy), eight_pages, page_size);
    std::int64_t first = continuation(pages_of(eight_pages, {0, 1, 4, 6, 7}), 1)[0];
    token_list kept_tokens = pages_of(eight_pages, {0, 1, 4, 7});
    REQUIRE(results[0].tokens == token_list{first, continuation(concat(kept_tokens, {first}), 1)[0]});
    REQUIRE(results[0].cycle_ends.size() == 8);
    const pulsarlm::cycle_end& prefill_end = results[0].cycle_ends[7];
    REQUIRE((hashes_of(*prefill_end.state) == std::vector{chain[0], chain[1], chain[4], chain[7]}));
    REQUIRE(prefill_end.state->tail.empty());
    pulsarlm::state end = ended_state(results[0]);
    REQUIRE((hashes_of(end) == std::vector{chain[0], chain[1], chain[4], chain[7]}));
    REQUIRE(end.tail == token_list{first});
}

TEST_CASE("a layer evicts nothing by score during its warm-up") {
    fixture f(16);
    f.model.is_important = [](std::size_t, std::int64_t token) { return token == 7; };
    pulsarlm::eviction_policy
        policy{.n_sink = 4, .n_working = 4, .cycle_tokens = 4, .eviction_threshold = 0.5, .eviction_warmup = 4};
    std::vector<request_results> results = serve_all(
        f.sched,
        {request{.tokens = eight_pages, .max_tokens = 2, .eviction = policy}}
    );

    // The prompt's 6 sampled pages are fewer than the warm-up's 4 density samples then 4 score samples.
    REQUIRE(
        (hashes_of(*results[0].cycle_ends.back().state) ==
         pulsarlm::make_page_digests(request_root(policy), eight_pages, page_size))
    );
}

TEST_CASE("a page's mass sums carry over from the cycles before") {
    fixture f(16);
    f.model.is_important = [&f](std::size_t, std::int64_t token) { return token == 7 && f.model.forward_count == 1; };
    token_list prompt(eight_pages.begin(), eight_pages.begin() + 16);
    pulsarlm::eviction_policy policy{
        .n_sink = 4,
        .n_working = 4,
        .cycle_tokens = 8,
        .eviction_threshold = 0.5,
        .eviction_warmup = 0,
        .attention_mass_decay = 0.01
    };
    std::vector<request_results> results = serve_all(
        f.sched,
        {request{.tokens = prompt, .max_tokens = 2, .eviction = policy}}
    );

    REQUIRE(f.model.prefill_chunks == std::vector<token_list>{{8}, {8}});
    REQUIRE(results[0].cycle_ends.size() == 2);
    const pulsarlm::state& reached = *results[0].cycle_ends[1].state;
    REQUIRE((hashes_of(reached) == pulsarlm::make_page_digests(request_root(policy), prompt, page_size)));
    std::vector<float> densities = densities_of(reached);
    REQUIRE(densities.size() == 4);
    double retained = std::pow(0.99, 8.0);
    REQUIRE_THAT(densities[1], Catch::Matchers::WithinAbs(retained / (retained + 1.0), 1e-6));
    REQUIRE((std::vector{densities[0], densities[2], densities[3]} == std::vector{0.0f, 0.0f, 0.0f}));
}

TEST_CASE("a request resumes with the mass sums and moments of its initial state") {
    fixture f(16);
    f.model.is_important = [](std::size_t, std::int64_t token) { return token == 7; };
    token_list prompt(eight_pages.begin(), eight_pages.begin() + 16);
    pulsarlm::eviction_policy
        policy{.n_sink = 4, .n_working = 4, .cycle_tokens = 4, .eviction_threshold = 0.5, .eviction_warmup = 0};
    std::vector<request_results> results = serve_in_turn(
        f.sched,
        request{.tokens = prompt, .max_tokens = 1, .eviction = policy},
        [&](const request_results& first) {
            f.model.is_important = [](std::size_t, std::int64_t) { return false; };
            pulsarlm::state resumed = ended_state(first);
            for (pulsarlm::layer_sequence& sequence : resumed.layers) {
                sequence.density_moments = {pulsarlm::moments{.count = 2, .mean = 0.5, .m2 = 0.25}};
                sequence.score_moments = pulsarlm::moments{.count = 2, .mean = 0.0, .m2 = 1.0};
            }
            f.backing.keep_state(std::make_shared<const pulsarlm::state>(std::move(resumed)));
            return request{
                .tokens = concat(prompt, {first.tokens.back(), 200, 201, 202}),
                .max_tokens = 2,
                .eviction = policy,
            };
        }
    );

    std::vector<digest> chain = pulsarlm::make_page_digests(request_root(policy), prompt, page_size);
    pulsarlm::state first_end = ended_state(results[0]);
    REQUIRE((hashes_of(first_end) == chain));
    REQUIRE(densities_of(first_end) == std::vector<float>{0.0f, 1.0f, 0.0f, 0.0f});
    std::vector<digest> resumed_chain = pulsarlm::make_page_digests(
        request_root(policy),
        concat(prompt, {results[0].tokens.back(), 200, 201, 202}),
        page_size
    );
    const pulsarlm::state& reached = *results[1].cycle_ends[0].state;
    REQUIRE((hashes_of(reached) == std::vector{resumed_chain[0], resumed_chain[1], resumed_chain[4]}));
}

TEST_CASE("a cycle's end evicts the oldest pages down to max_pages") {
    fixture f(16);
    token_list prompt{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    pulsarlm::eviction_policy policy{.n_sink = 0, .n_working = 4, .cycle_tokens = 16, .max_context_window = 28};
    std::vector<request_results> results = serve_all(
        f.sched,
        {request{.tokens = prompt, .max_tokens = 2, .eviction = policy}}
    );

    std::vector<digest> chain = pulsarlm::make_page_digests(request_root(policy), prompt, page_size);
    token_list kept_tokens(prompt.begin() + 8, prompt.end());
    std::int64_t first = continuation(prompt, 1)[0];
    REQUIRE(results[0].tokens == token_list{first, continuation(concat(kept_tokens, {first}), 1)[0]});
    REQUIRE((hashes_of(*results[0].cycle_ends[0].state) == std::vector{chain[2], chain[3]}));
}

TEST_CASE("a prompt longer than the cache fits when each cycle evicts down to max_pages") {
    fixture f(4);
    token_list prompt(32);
    std::iota(prompt.begin(), prompt.end(), 1);
    std::vector<request_results> results = serve_all(
        f.sched,
        {request{
            .tokens = prompt,
            .max_tokens = 2,
            .eviction = {.n_sink = 0, .n_working = 4, .cycle_tokens = 8, .max_context_window = 20}
        }}
    );

    token_list last_two_cycles(prompt.begin() + 16, prompt.end());
    token_list last_cycle(prompt.begin() + 24, prompt.end());
    std::int64_t first = continuation(last_two_cycles, 1)[0];
    REQUIRE(results[0].tokens == token_list{first, continuation(concat(last_cycle, {first}), 1)[0]});
    REQUIRE(results[0].cycle_ends.size() == 4);
    REQUIRE(hashes_of(ended_state(results[0])).size() == 2);
}

TEST_CASE("a request that needs every line of the pool peaks at the whole pool") {
    fixture f(4);
    token_list prompt(32);
    std::iota(prompt.begin(), prompt.end(), 1);
    std::vector<request_results> results = serve_all(
        f.sched,
        {request{
            .tokens = prompt,
            .max_tokens = 2,
            .eviction = {.n_sink = 0, .n_working = 4, .cycle_tokens = 8, .max_context_window = 20}
        }}
    );

    std::size_t pool_bytes = 0;
    for (const pulsarlm::page_shape& shape : page_shapes(two_full_layers)) {
        pool_bytes += 4 * std::bit_ceil(pulsarlm::line_bytes(shape));
    }
    REQUIRE(results[0].peak_device_memory == pool_bytes);
}

TEST_CASE("prefill forwards at most the step's prefill tokens") {
    fixture f(16, 4);
    token_list prompt{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<request_results> results = serve_all(f.sched, {request{.tokens = prompt, .max_tokens = 2}});

    REQUIRE(results[0].tokens == continuation(prompt, 2));
    REQUIRE(f.model.prefill_chunks == std::vector<token_list>{{4}, {4}, {2}});
    REQUIRE(f.model.decode_sizes == std::vector<std::int64_t>{1});
}

TEST_CASE("running requests are forwarded in one step") {
    fixture f(16);
    token_list first{1, 2, 3};
    token_list second{4, 5, 6};
    std::vector<request_results> results = serve_all(
        f.sched,
        {request{.tokens = first, .max_tokens = 3}, request{.tokens = second, .max_tokens = 3}}
    );

    REQUIRE(results[0].tokens == continuation(first, 3));
    REQUIRE(results[1].tokens == continuation(second, 3));
    REQUIRE(f.model.prefill_chunks == std::vector<token_list>{{3, 3}});
    REQUIRE(f.model.decode_sizes == std::vector<std::int64_t>{2, 2});
}

// Each step is launched before the one before it is written back, page completions included: when the n-th step is
// launched, the tokens of the first n - 2 steps have reached the callbacks.
TEST_CASE("each step is launched before the step before it is written back") {
    fixture f(16);
    token_list prompt{1, 2, 3, 4, 5};
    submission s = make_submission(request{.tokens = prompt, .max_tokens = 4});
    std::vector<std::size_t> tokens_at_forward;
    f.model.on_forward = [&tokens_at_forward, recorded_results = s.recorded_results] {
        tokens_at_forward.push_back(recorded_results->tokens.size());
    };
    std::vector<submission> submissions;
    submissions.push_back(std::move(s));
    std::vector<request_results> results = serve_submissions(f.sched, std::move(submissions));

    REQUIRE(results[0].tokens == continuation(prompt, 4));
    REQUIRE(tokens_at_forward == std::vector<std::size_t>{0, 0, 1, 2});
}

// Both requests' decode steps at tail positions 1 and 2 overlap. The first request's stop token, sampled at position
// 1, ends it while the step forwarding that token is launched; that step's token returns to pending, and the second
// request's next step reads its awaited token from the second row of that step's samples.
TEST_CASE("a stop token sampled while the next step runs ends the request with the state it reached") {
    token_list first{1, 2, 3};
    token_list second{5, 0, 0};
    REQUIRE(continuation(first, 3) == token_list{6, 12, 24});
    fixture f(16, 64, {24});
    std::vector<request_results> results = serve_all(
        f.sched,
        {request{.tokens = first, .max_tokens = 10}, request{.tokens = second, .max_tokens = 6}}
    );

    REQUIRE(results[0].tokens == token_list{6, 12, 24});
    pulsarlm::state end = ended_state(results[0]);
    REQUIRE((hashes_of(end) == pulsarlm::make_page_digests(root(), token_list{1, 2, 3, 6}, page_size)));
    REQUIRE(end.tail == token_list{12});
    REQUIRE(results[1].tokens == continuation(second, 6));
    REQUIRE(f.store->releases == f.store->leases);
}

const std::vector<pulsarlm::kv_layer> full_and_sliding{
    pulsarlm::kv_layer{.n_kv_heads = 1, .head_dim = 2, .n_q_heads = 1},
    pulsarlm::kv_layer{.n_kv_heads = 1, .head_dim = 2, .sliding_window = 5},
};

// With a sliding layer, a step completing a page releases the pages that leave the window at its write-back, so the
// next step waits for it: the prefill and the decode at tail position 3 complete pages; the decode steps at tail
// positions 1 and 2 overlap.
TEST_CASE("with a sliding layer, a step after one completing a page waits for its write-back") {
    fixture f(16, 64, {}, {}, full_and_sliding);
    token_list prompt{1, 2, 3, 4, 5};
    submission s = make_submission(request{.tokens = prompt, .max_tokens = 4});
    std::vector<std::size_t> tokens_at_forward;
    f.model.on_forward = [&tokens_at_forward, recorded_results = s.recorded_results] {
        tokens_at_forward.push_back(recorded_results->tokens.size());
    };
    std::vector<submission> submissions;
    submissions.push_back(std::move(s));
    std::vector<request_results> results = serve_submissions(f.sched, std::move(submissions));

    REQUIRE(results[0].tokens == continuation(prompt, 4));
    REQUIRE(tokens_at_forward == std::vector<std::size_t>{0, 1, 1, 3});
}

// Each request holds at most 7 full-attention lines and 3 sliding lines at a time: the window's page and a cycle's 2
// pages. The pool holds those 20 lines; holding passed pages too, one request would end with oom.
TEST_CASE("a sliding layer holds only the pages its window reaches") {
    std::size_t line_bytes = pulsarlm::line_bytes(page_shapes(full_and_sliding).front());
    fixture f(0, 64, {}, {}, full_and_sliding, 20 * line_bytes);
    token_list first(20);
    std::iota(first.begin(), first.end(), 1);
    token_list second(20);
    std::iota(second.begin(), second.end(), 100);
    std::vector<request_results> results = serve_all(
        f.sched,
        {request{.tokens = first, .max_tokens = 6, .eviction = {.cycle_tokens = 8}},
         request{.tokens = second, .max_tokens = 6, .eviction = {.cycle_tokens = 8}}}
    );

    REQUIRE(results[0].tokens == continuation(first, 6));
    REQUIRE(results[1].tokens == continuation(second, 6));
    REQUIRE(f.model.prefill_chunks == std::vector<token_list>{{8, 8}, {8, 8}, {4, 4}});
    REQUIRE(f.model.decode_sizes == std::vector<std::int64_t>(5, 2));
    pulsarlm::state end = ended_state(results[0]);
    token_list forwarded = concat(first, without_last(results[0].tokens));
    REQUIRE((hashes_of(end) == pulsarlm::make_page_digests(root(), forwarded, page_size)));
    for (std::size_t i = 0; i < hashes_of(end).size(); i++) {
        token_list page_tokens(forwarded.begin() + i * page_size, forwarded.begin() + (i + 1) * page_size);
        REQUIRE(keys_hold(f.stored_kv(hashes_of(end)[i], 1), page_tokens));
    }
}

const std::vector<pulsarlm::kv_layer> full_and_wide_sliding{
    pulsarlm::kv_layer{.n_kv_heads = 1, .head_dim = 2, .n_q_heads = 1},
    pulsarlm::kv_layer{.n_kv_heads = 2, .head_dim = 2, .sliding_window = 5},
};

// The request peaks at 8 full-attention lines of 64 bytes and 5 sliding lines of 128 bytes: the window's page and a
// 16-token cycle's 4 pages. A split sizing the sliding layer for ceil(5 / 4) + 1 lines per request would not serve it.
TEST_CASE("the pool gives each line size the lines its layers need") {
    token_list prompt(16);
    std::iota(prompt.begin(), prompt.end(), 1);
    auto serve_in = [&](std::size_t pool_bytes) {
        fixture f(0, 64, {}, {}, full_and_wide_sliding, pool_bytes);
        return serve_all(f.sched, {request{.tokens = prompt, .max_tokens = 16, .eviction = {.cycle_tokens = 16}}})[0];
    };

    request_results served = serve_in(8 * 64 + 5 * 128);
    REQUIRE(served.tokens == continuation(prompt, 16));
    REQUIRE(hashes_of(ended_state(served)).size() == 7);
    REQUIRE(ended_error(serve_in(7 * 64 + 5 * 128)) == request_error::oom);
}

TEST_CASE("a state's sliding layer holds the pages its window reaches from the tail") {
    fixture f(16, 64, {}, {}, full_and_sliding);
    token_list history(26);
    std::iota(history.begin(), history.end(), 1);
    pulsarlm::state end = ended_state(serve_all(f.sched, {request{.tokens = history, .max_tokens = 0}})[0]);

    std::vector<digest> hashes = pulsarlm::make_page_digests(root(), history, page_size);
    REQUIRE(hashes.size() == 6);
    REQUIRE((end.head == hashes.back()));
    REQUIRE((hashes_of(end, 0) == hashes));
    REQUIRE((hashes_of(end, 1) == std::vector{hashes[5]}));
    REQUIRE(end.tail == token_list{25, 26});
}

TEST_CASE("a request resumed on another cache fetches only the window's pages of a sliding layer") {
    fixture first_engine(16, 64, {}, {}, full_and_sliding);
    token_list prompt(20);
    std::iota(prompt.begin(), prompt.end(), 1);
    request_results first = serve_all(first_engine.sched, {request{.tokens = prompt, .max_tokens = 3}})[0];
    pulsarlm::state first_end = ended_state(first);
    REQUIRE(hashes_of(first_end).size() == 5);

    memory_store shared_store;
    shared_store.contents = first_engine.store;
    fixture second_engine(16, 64, {}, shared_store, full_and_sliding);
    token_list user{7, 8};
    request_results second = serve_all(
        second_engine.sched,
        {request{.tokens = concat(concat(prompt, first.tokens), user), .max_tokens = 4}}
    )[0];

    REQUIRE(second.tokens == continuation(concat(concat(prompt, first.tokens), user), 4));
    std::vector<digest> keys = concat(keys_of(hashes_of(first_end), 0), keys_of({hashes_of(first_end)[4]}, 1));
    REQUIRE((first_engine.store->fetches == std::vector<std::vector<digest>>{keys}));
}

TEST_CASE("a sliding layer keeps the pages its window reaches whatever the full-attention layer evicts") {
    std::vector<pulsarlm::kv_layer> layers{
        pulsarlm::kv_layer{.n_kv_heads = 1, .head_dim = 2, .n_q_heads = 1},
        pulsarlm::kv_layer{.n_kv_heads = 1, .head_dim = 2, .sliding_window = 9},
    };
    fixture f(16, 64, {}, {}, layers);
    token_list prompt(16);
    std::iota(prompt.begin(), prompt.end(), 1);
    pulsarlm::eviction_policy policy{.n_sink = 0, .n_working = 4, .cycle_tokens = 16, .max_context_window = 24};
    std::vector<request_results> results = serve_all(
        f.sched,
        {request{.tokens = prompt, .max_tokens = 2, .eviction = policy}}
    );

    std::vector<digest> chain = pulsarlm::make_page_digests(request_root(policy), prompt, page_size);
    token_list kept_tokens(prompt.begin() + 12, prompt.end());
    std::int64_t first = continuation(prompt, 1)[0];
    REQUIRE(results[0].tokens == token_list{first, continuation(concat(kept_tokens, {first}), 1)[0]});
    const pulsarlm::state& reached = *results[0].cycle_ends[0].state;
    REQUIRE((hashes_of(reached, 0) == std::vector{chain[3]}));
    REQUIRE((hashes_of(reached, 1) == std::vector{chain[2], chain[3]}));
}

TEST_CASE("each full-attention layer evicts by its own attention mass") {
    fixture f(16);
    f.model.is_important = each_layers_own;
    pulsarlm::eviction_policy
        policy{.n_sink = 4, .n_working = 4, .cycle_tokens = 4, .eviction_threshold = 0.5, .eviction_warmup = 0};
    std::vector<request_results> results = serve_all(
        f.sched,
        {request{.tokens = eight_pages, .max_tokens = 3, .eviction = policy}}
    );

    // Layer 0, which predicts, evicts as in the single-layer case: its last prefill step attends pages 0, 1, 4, 6 and 7.
    std::vector<digest> chain = pulsarlm::make_page_digests(request_root(policy), eight_pages, page_size);
    std::int64_t first = continuation(pages_of(eight_pages, {0, 1, 4, 6, 7}), 1)[0];
    token_list kept_tokens = pages_of(eight_pages, {0, 1, 4, 7});
    REQUIRE(results[0].tokens == concat({first}, continuation(concat(kept_tokens, {first}), 2)));
    const pulsarlm::state& reached = *results[0].cycle_ends[7].state;
    REQUIRE((hashes_of(reached, 0) == std::vector{chain[0], chain[1], chain[4], chain[7]}));
    REQUIRE((densities_of(reached, 0) == std::vector{0.0f, 1.0f, 1.0f, 0.0f}));
    REQUIRE((hashes_of(reached, 1) == std::vector{chain[0], chain[2], chain[5], chain[7]}));
    REQUIRE((densities_of(reached, 1) == std::vector{0.0f, 1.0f, 1.0f, 0.0f}));
    pulsarlm::state end = ended_state(results[0]);
    REQUIRE((hashes_of(end, 0) == std::vector{chain[0], chain[1], chain[4], chain[7]}));
    REQUIRE((hashes_of(end, 1) == std::vector{chain[0], chain[2], chain[5], chain[7]}));
}

const std::vector<pulsarlm::kv_layer> two_full_and_mirroring{
    pulsarlm::kv_layer{.n_kv_heads = 1, .head_dim = 2, .n_q_heads = 1},
    pulsarlm::kv_layer{.n_kv_heads = 1, .head_dim = 2, .n_q_heads = 1},
    pulsarlm::kv_layer{.n_kv_heads = 1, .head_dim = 2, .n_q_heads = 1, .mirrors_held_pages = true},
};

// As in the test above, layer 0 keeps pages 0, 1, 4 and 7 and layer 1 pages 0, 2, 5 and 7; the mirroring layer measures
// no mass and keeps their union, and the tokens and the attention layers' views are those without it.
TEST_CASE("a mirroring layer holds the pages the other layers hold and changes no position") {
    fixture f(16, 64, {}, {}, two_full_and_mirroring);
    f.model.is_important = each_layers_own;
    pulsarlm::eviction_policy
        policy{.n_sink = 4, .n_working = 4, .cycle_tokens = 4, .eviction_threshold = 0.5, .eviction_warmup = 0};
    std::vector<request_results> results = serve_all(
        f.sched,
        {request{.tokens = eight_pages, .max_tokens = 3, .eviction = policy}}
    );

    std::vector<digest> chain = pulsarlm::make_page_digests(request_root(policy), eight_pages, page_size);
    std::int64_t first = continuation(pages_of(eight_pages, {0, 1, 4, 6, 7}), 1)[0];
    token_list kept_tokens = pages_of(eight_pages, {0, 1, 4, 7});
    REQUIRE(results[0].tokens == concat({first}, continuation(concat(kept_tokens, {first}), 2)));
    pulsarlm::state end = ended_state(results[0]);
    REQUIRE((hashes_of(end, 0) == std::vector{chain[0], chain[1], chain[4], chain[7]}));
    REQUIRE((hashes_of(end, 1) == std::vector{chain[0], chain[2], chain[5], chain[7]}));
    REQUIRE((hashes_of(end, 2) == std::vector{chain[0], chain[1], chain[2], chain[4], chain[5], chain[7]}));
    for (const pulsarlm::cycle_end& ended : results[0].cycle_ends) {
        std::vector<digest> held = hashes_of(*ended.state, 0);
        for (const digest& hash : hashes_of(*ended.state, 1)) {
            if (!std::ranges::contains(held, hash)) {
                held.push_back(hash);
            }
        }
        std::vector<digest> mirrored = hashes_of(*ended.state, 2);
        std::ranges::sort(held, {}, [](const digest& d) { return std::pair(d.hi, d.lo); });
        std::ranges::sort(mirrored, {}, [](const digest& d) { return std::pair(d.hi, d.lo); });
        REQUIRE((mirrored == held));
    }
}

// Layer 0 keeps pages 1 and 4 and layer 1 pages 2 and 5; both drop pages 3 and 6. Pages 1, 2, 4 and 5 stay in the
// union, pages 3 and 6 leave it, so the pages after them renumber, and page 7 sits at the same position in both layers
// despite their different views.
TEST_CASE("a page takes the same position in every layer, its rank among the pages any layer holds") {
    fixture f(16);
    f.model.is_important = each_layers_own;
    std::vector<request_results> results = serve_all(
        f.sched,
        {request{
            .tokens = eight_pages,
            .max_tokens = 2,
            .eviction =
                {.n_sink = 4, .n_working = 4, .cycle_tokens = 4, .eviction_threshold = 0.5, .eviction_warmup = 0},
        }}
    );

    REQUIRE(f.model.decode_page_positions.size() == 1);
    auto positions = f.model.decode_page_positions[0].accessor<std::int32_t, 3>();
    REQUIRE(
        (std::vector{positions[0][0][0], positions[0][0][1], positions[0][0][2], positions[0][0][3]} ==
         std::vector<std::int32_t>{0, 4, 12, 20})
    );
    REQUIRE(
        (std::vector{positions[1][0][0], positions[1][0][1], positions[1][0][2], positions[1][0][3]} ==
         std::vector<std::int32_t>{0, 8, 16, 20})
    );
    std::vector<std::size_t> indices;
    for (const pulsarlm::active_page& page : ended_state(results[0]).layers[1].pages) {
        indices.push_back(page.index);
    }
    REQUIRE((indices == std::vector<std::size_t>{0, 2, 5, 7}));
}

TEST_CASE("a request resumed from a state with different views per layer keeps their pages' positions") {
    fixture f(16);
    f.model.is_important = each_layers_own;
    pulsarlm::eviction_policy
        policy{.n_sink = 4, .n_working = 4, .cycle_tokens = 4, .eviction_threshold = 0.5, .eviction_warmup = 0};
    std::vector<request_results> results = serve_in_turn(
        f.sched,
        request{.tokens = eight_pages, .max_tokens = 2, .eviction = policy},
        [&](const request_results& done) {
            return request{.tokens = concat(eight_pages, done.tokens), .max_tokens = 2, .eviction = policy};
        }
    );

    REQUIRE(f.model.decode_page_positions.size() == 2);
    auto positions = f.model.decode_page_positions.back().accessor<std::int32_t, 3>();
    REQUIRE(
        (std::vector{positions[0][0][0], positions[0][0][1], positions[0][0][2], positions[0][0][3]} ==
         std::vector<std::int32_t>{0, 4, 12, 20})
    );
    REQUIRE(
        (std::vector{positions[1][0][0], positions[1][0][1], positions[1][0][2], positions[1][0][3]} ==
         std::vector<std::int32_t>{0, 8, 16, 20})
    );
}

std::int64_t sum_of_first(const token_list& tokens, std::size_t count) {
    return std::accumulate(tokens.begin(), tokens.begin() + static_cast<std::ptrdiff_t>(count), std::int64_t{0});
}

std::int64_t stored_state(const fixture& f, const digest& hash) {
    return static_cast<std::int64_t>(f.stored_kv(hash, 0).k.item<float>());
}

TEST_CASE("a recurrent layer carries its state across steps and keeps the state at the newest page") {
    fixture f(16, 3, {}, {}, recurrent_and_full);
    token_list prompt{1, 2, 3, 4, 5, 6, 7};
    std::vector<request_results> results = serve_all(f.sched, {request{.tokens = prompt, .max_tokens = 6}});

    token_list generated = continuation(prompt, 6);
    REQUIRE(results[0].tokens == generated);
    REQUIRE(f.model.prefill_chunks == std::vector<token_list>{{3}, {3}});
    pulsarlm::state end = ended_state(results[0]);
    token_list forwarded = concat(prompt, without_last(generated));
    std::vector<digest> hashes = pulsarlm::make_page_digests(root(), forwarded, page_size);
    REQUIRE(hashes.size() == 3);
    REQUIRE((end.head == hashes.back()));
    REQUIRE((hashes_of(end, 1) == hashes));
    REQUIRE((hashes_of(end, 0) == std::vector{hashes.back()}));
    REQUIRE(stored_state(f, hashes.back()) == sum_of_first(forwarded, 12));
    REQUIRE(!f.store->pages.contains(pulsarlm::store_key(hashes[1], 0)));
}

TEST_CASE("a request resumes a recurrent layer's state from another cache") {
    fixture first_engine(16, 64, {}, {}, recurrent_and_full);
    token_list prompt{1, 2, 3, 4, 5, 6};
    request_results first = serve_all(first_engine.sched, {request{.tokens = prompt, .max_tokens = 5}})[0];
    pulsarlm::state first_end = ended_state(first);

    memory_store shared_store;
    shared_store.contents = first_engine.store;
    fixture second_engine(16, 64, {}, shared_store, recurrent_and_full);
    token_list user{7, 8};
    request_results second = serve_all(
        second_engine.sched,
        {request{.tokens = concat(concat(prompt, first.tokens), user), .max_tokens = 4}}
    )[0];

    REQUIRE(second.tokens == continuation(concat(concat(prompt, first.tokens), user), 4));
    std::vector<digest> keys = concat(keys_of({*first_end.head}, 0), keys_of(hashes_of(first_end, 1), 1));
    REQUIRE((first_engine.store->fetches == std::vector<std::vector<digest>>{keys}));
}

// The stop token is sampled while the step forwarding it runs, which updates the recurrent layer's running state and
// is withdrawn; forwarding 96 also completes the page [12, 24, 48, 96] and writes its end state into the boundary line
// the head's is not in. The end state's head and its recurrent state are those of a step-by-step run, so a request
// resumed from them on another cache continues as the uninterrupted tokens do.
TEST_CASE("a recurrent request stopped while its next step runs resumes from the state it reached") {
    token_list prompt{1, 2, 3};
    REQUIRE(continuation(prompt, 5) == token_list{6, 12, 24, 48, 96});
    const std::int64_t stop = GENERATE(24, 96);
    INFO("stop " << stop);
    fixture first_engine(16, 64, {stop}, {}, recurrent_and_full);
    request_results first = serve_all(first_engine.sched, {request{.tokens = prompt, .max_tokens = 10}})[0];
    token_list generated = continuation(prompt, stop == 24 ? 3 : 5);
    REQUIRE(first.tokens == generated);
    pulsarlm::state first_end = ended_state(first);
    REQUIRE((hashes_of(first_end, 1) == pulsarlm::make_page_digests(root(), token_list{1, 2, 3, 6}, page_size)));
    REQUIRE(first_end.tail == token_list(generated.begin() + 1, generated.end() - 1));

    memory_store shared_store;
    shared_store.contents = first_engine.store;
    fixture second_engine(16, 64, {}, shared_store, recurrent_and_full);
    token_list history = concat(concat(prompt, first.tokens), token_list{7, 8});
    request_results second = serve_all(second_engine.sched, {request{.tokens = history, .max_tokens = 4}})[0];

    REQUIRE(second.tokens == continuation(history, 4));
    REQUIRE(second.cached_tokens == page_size);
}

TEST_CASE("a recurrent layer's state covers the pages eviction drops") {
    fixture f(16, 64, {}, {}, recurrent_and_full);
    token_list prompt(16);
    std::iota(prompt.begin(), prompt.end(), 1);
    pulsarlm::eviction_policy policy{.n_sink = 0, .n_working = 4, .cycle_tokens = 16, .max_context_window = 28};
    std::vector<request_results> results = serve_all(
        f.sched,
        {request{.tokens = prompt, .max_tokens = 2, .eviction = policy}}
    );

    REQUIRE(results[0].tokens == continuation(prompt, 2));
    std::vector<digest> chain = pulsarlm::make_page_digests(request_root(policy), prompt, page_size);
    const pulsarlm::state& reached = *results[0].cycle_ends[0].state;
    REQUIRE((hashes_of(reached, 1) == std::vector{chain[2], chain[3]}));
    REQUIRE((hashes_of(reached, 0) == std::vector{chain[3]}));
    REQUIRE(stored_state(f, chain[3]) == sum_of_first(prompt, 16));
}

TEST_CASE("each state a recurrent request reaches has the state at its head in the store") {
    fixture f(16, 64, {}, {}, recurrent_and_full);
    token_list prompt{1, 2, 3, 4, 5, 6};
    std::vector<request_results> results = serve_all(
        f.sched,
        {request{.tokens = prompt, .max_tokens = 7, .eviction = {.cycle_tokens = 2}}}
    );

    token_list history = concat(prompt, results[0].tokens);
    REQUIRE(results[0].tokens == continuation(prompt, 7));
    std::vector<pulsarlm::state> states;
    for (const pulsarlm::cycle_end& ended : results[0].cycle_ends) {
        states.push_back(*ended.state);
    }
    states.push_back(ended_state(results[0]));
    REQUIRE(states.size() == 6);
    for (const pulsarlm::state& reached : states) {
        std::size_t pages = hashes_of(reached, 1).size();
        if (pages > 0) {
            REQUIRE((hashes_of(reached, 0) == std::vector{*reached.head}));
            REQUIRE(stored_state(f, *reached.head) == sum_of_first(history, pages * page_size));
        }
    }
}

}  // namespace
