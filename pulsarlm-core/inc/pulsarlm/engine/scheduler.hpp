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

// The tail of initial_state is forwarded again, then tokens; together they hold at least one token. Without
// max_tokens, the request samples until one of the engine's stop tokens. The pages its steps put are owned by id.
struct request {
    state initial_state;
    std::vector<std::int64_t> tokens;
    sampling_params sampling;
    std::optional<std::size_t> max_tokens;
    eviction_policy eviction;
    std::uint64_t id = 0;
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
    // The full pages of KV layer layer's view wholly before the window of a query at view index query.
    std::size_t passed_pages(std::size_t layer, std::size_t query) const noexcept;
    state state_of(const sequence& seq) const;
    // The tokens of seq's full history pages that every KV layer reading the page holds resident.
    std::size_t device_cached_tokens(const sequence& seq) const;
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
    // Returns whether a view changed.
    bool end_cycle(sequence& seq);

  public:
    // shards[s] runs on cache shard s with streams[s], shards 1 on on worker_loops[s - 1]; the cache holds one layer
    // per entry of each model's kv_layers(), and the shards agree on which layers are recurrent or sliding. root is the
    // digest the page chains of a state with no pages start from. A step forwards every decoding sequence and at most
    // prefill_tokens_per_step tokens of the others.
    scheduler(
        std::vector<model*>&& shards,
        kv_cache cache,
        std::vector<cudaStream_t>&& streams,
        std::vector<event_loop*>&& worker_loops,
        digest root,
        std::vector<std::int64_t>&& stop_ids,
        std::size_t prefill_tokens_per_step
    );

    // Serves r, then ends writer with its result or an error. Appends
    // the state to writer at the end of each eviction cycle. Ends with oom, never waiting, when the free lines do not
    // cover the state's pages and one cycle's tokens at the start, or the next cycle's tokens while running; the
    // caller resubmits.
    task<void> submit(request r, result_writer writer);
    // Steps the running sequences until the loop stops.
    task<void> run();
    // Throws std::invalid_argument unless s is a state the scheduler reaches: with one sequence per KV layer, a
    // full-attention layer's ending with head, a sliding layer's ending with head and holding no more pages than its
    // window reaches, a recurrent layer's head alone; each page holding no mass sum or one per query head its layer
    // measures, all finite, and a finite non-negative weight sum, 0 without mass sums; each sequence holding no density
    // moments or one per such head; every moments with a finite mean and a finite non-negative m2, 0 below 2 samples.
    // Safe to call from any thread.
    void validate_state(const state& s) const;
    // kv_cache::used_bytes of the scheduler's cache. Safe to call from any thread.
    std::size_t kv_cache_bytes() const noexcept;
};

}  // namespace pulsarlm
