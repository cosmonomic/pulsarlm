#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "pulsarlm/digest.hpp"
#include "pulsarlm/engine/event_loop.hpp"
#include "pulsarlm/engine/kv_cache.hpp"
#include "pulsarlm/engine/model/interface.hpp"
#include "pulsarlm/engine/page_store.hpp"
#include "pulsarlm/engine/request.hpp"
#include "pulsarlm/engine/request_callbacks.hpp"

namespace pulsarlm {

class sequence;
struct batch_entry;

// Runs sequences on the model: it owns the model shards, the KV cache and the store's side of a request, and does
// whatever touches them, while the scheduler decides what runs.
class executor {
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
    // The loops of shards 1 on: shard s's forward runs on worker_loops[s - 1]; shard 0's on the executor's loop.
    std::vector<event_loop*> worker_loops;
    std::size_t page_size;
    digest root;
    std::vector<std::int64_t> stop_ids;

    // The tokens of seq's full history pages that every KV layer reading the page holds resident.
    std::size_t device_cached_tokens(const sequence& seq) const;
    // Per KV layer with pages, lines for its pages, then line_count empty lines; per recurrent layer, a copy of the
    // state at head, or zeros without head, and an empty line. Holds nothing on failure.
    task<std::expected<void, page_error>> take_lines(sequence& seq, std::size_t line_count);
    void release_lines(sequence& seq) noexcept;
    // Drops from each sliding layer's view the pages before the window of the tail's first token, and returns whether
    // any view changed.
    bool release_passed_pages(sequence& seq);
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

    friend class sequence;

  public:
    // shards[s] runs on cache shard s with streams[s], shards 1 on on worker_loops[s - 1]; the cache holds one layer
    // per entry of each model's kv_layers(), and the shards agree on which layers are recurrent or sliding. store is the
    // cache's. root is the digest a request's page chain starts from without an initial digest.
    executor(
        std::vector<model*>&& shards,
        kv_cache cache,
        page_store& store,
        std::vector<cudaStream_t>&& streams,
        std::vector<event_loop*>&& worker_loops,
        digest root,
        std::vector<std::int64_t>&& stop_ids
    );

    // The sequence of r, resuming the newest state the store holds for a prefix of its tokens, with lines for its
    // first cycle. Fails with miss when the lookup or a page read fails, and with oom when the free lines do not cover
    // the state's pages and one cycle's tokens; its lease is then released.
    task<std::expected<std::unique_ptr<sequence>, request_error>> start(request&& r, result_writer& writer);
    // Runs one step of entries, decoding entries first: the forward on every shard and the sampling, then advancing
    // each sequence, writing its new pages, evicting and writing its state at a cycle end, and taking lines for its
    // next tokens. Sets the exit reason of each sequence that finished, by a stop token or max_tokens, or ran out of
    // lines.
    task<void> step(std::span<const batch_entry> entries);
    // Releases seq's lines, writes its end state when it finished, and releases its lease; seq exited.
    task<std::expected<request_result, request_error>> finish(sequence& seq);
    // kv_cache::used_bytes of the executor's cache. Safe to call from any thread.
    std::size_t kv_cache_bytes() const noexcept;
};

}  // namespace pulsarlm
