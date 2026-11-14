#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "pulsarlm/digest.hpp"
#include "pulsarlm/engine/event_loop.hpp"
#include "pulsarlm/engine/eviction.hpp"
#include "pulsarlm/engine/kv_cache.hpp"
#include "pulsarlm/engine/model/interface.hpp"
#include "pulsarlm/engine/request_callbacks.hpp"
#include "pulsarlm/engine/sampling.hpp"

namespace pulsarlm {

// tokens is a history chained from initial_digest, or from the engine's root without it, and holds at least one token.
// Without max_tokens, the request samples until one of the engine's stop tokens.
struct request {
    std::optional<digest> initial_digest;
    std::vector<std::int64_t> tokens;
    sampling_params sampling;
    std::optional<std::size_t> max_tokens;
    eviction_policy eviction;
};

class scheduler {
    enum class exit_reason { finished, cancelled, oom };
    class layer_view;
    class sequence;
    class batch;

    // One model per shard of the cache, each holding its part of every layer.
    std::vector<model*> shards;
    // One layer per KV layer, in the models' KV layer order.
    kv_cache cache;
    page_store& store;
    // Per shard, one kv per KV layer.
    std::vector<std::vector<kv>> lines;
    std::vector<std::optional<std::size_t>> sliding_windows;
    std::vector<bool> recurrent;
    // Per KV layer, the query heads it measures mass and evicts pages in: those of every attention layer reading a
    // full-attention layer, 0 for other layers.
    std::vector<std::size_t> mass_heads;
    // Per shard, per KV layer, the shard's part of mass_heads; the shards' parts in shard order make up the layer's.
    std::vector<std::vector<std::size_t>> shard_mass_heads;
    // One stream per shard.
    std::vector<cudaStream_t> streams;
    // The loops of shards 1 on: shard s's forward runs on worker_loops[s - 1]; shard 0's on the scheduler's loop.
    std::vector<event_loop*> worker_loops;
    std::size_t page_size;
    digest root;
    std::vector<std::int64_t> stop_ids;
    std::size_t prefill_tokens_per_step;
    std::vector<sequence*> running_sequences;
    // Set when a sequence joins the running set.
    std::optional<event> join_event;

    static event& emplace_event(std::optional<event>& waited);
    static void wake(std::optional<event>& waited) noexcept;
    // The tokens of seq's full history pages that every KV layer reading the page holds resident.
    std::size_t device_cached_tokens(const sequence& seq) const;
    // Runs seq from its start to its exit, and writes its end state when it finished.
    task<std::expected<request_result, request_error>> serve(sequence& seq);
    // Per KV layer with pages, lines for its pages, then line_count empty lines; per recurrent layer, a copy of the
    // state at head, or zeros without head, and an empty line. Holds nothing on failure.
    task<std::expected<void, page_error>> take_lines(sequence& seq, std::size_t line_count);
    void release_lines(sequence& seq) noexcept;
    // Drops from each sliding layer's view the pages before the window of the tail's first token, and returns whether
    // any view changed.
    bool release_passed_pages(sequence& seq);
    task<batch> build_batch();
    task<void> launch(batch& b);
    task<void> forward_on_shard(batch& b, std::size_t shard);
    task<void> synchronize_shards();
    task<void> write_back(batch& b);
    // Adds to the moments of each layer that measures mass the full pages outside the sinks that cycle_tokens tokens
    // now follow, given that seq's newest completed full pages just completed.
    void add_sampled_pages(sequence& seq, std::size_t completed);
    // Evicts at the end of seq's cycle and returns whether a view changed.
    bool evict(sequence& seq);
    // Writes the state seq reached to the store and passes it to seq's callbacks.
    task<void> write_cycle_end(sequence& seq);

  public:
    // shards[s] runs on cache shard s with streams[s], shards 1 on on worker_loops[s - 1]; the cache holds one layer
    // per entry of each model's kv_layers(), and the shards agree on which layers are recurrent or sliding. store is the
    // cache's. root is the digest a request's page chain starts from without an initial digest. A step forwards every
    // decoding sequence and at most prefill_tokens_per_step tokens of the others.
    scheduler(
        std::vector<model*>&& shards,
        kv_cache cache,
        page_store& store,
        std::vector<cudaStream_t>&& streams,
        std::vector<event_loop*>&& worker_loops,
        digest root,
        std::vector<std::int64_t>&& stop_ids,
        std::size_t prefill_tokens_per_step
    );

    // Serves r from the newest state the store holds for a prefix of its tokens, then ends writer with its result or an
    // error. Writes the state reached at the end of each eviction cycle to the store, then appends it to writer, and
    // likewise the end state, and releases the state's lease before ending writer. Ends with miss when the store's lookup
    // fails. Ends with oom, never waiting, when the free lines do not cover the state's pages and one cycle's tokens at
    // the start, or the next cycle's tokens while running; the caller resubmits.
    task<void> submit(request r, result_writer writer);
    // Steps the running sequences until the loop stops.
    task<void> run();
    // kv_cache::used_bytes of the scheduler's cache. Safe to call from any thread.
    std::size_t kv_cache_bytes() const noexcept;
};

}  // namespace pulsarlm
