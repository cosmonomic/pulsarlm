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
#include "pulsarlm/engine/decode_graphs.hpp"
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
    // One per shard, or none when decode groups forward without graphs.
    std::vector<std::unique_ptr<decode_graphs>> graphs;
    // The page-table width of the decode graphs, the pages of the longest view they cover.
    std::optional<std::size_t> graph_pages;
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
    // Per shard, the tokens the latest launched step sampled, on the shard's device; every shard samples them alike.
    std::vector<at::Tensor> latest_sampled;

    // The tokens of seq's full history pages that every KV layer reading the page holds resident.
    std::size_t device_cached_tokens(const sequence& seq) const;
    // Per KV layer with pages, lines for its pages, then line_count empty lines; per recurrent layer, a copy of the
    // state at head, or zeros without head, and an empty line. Holds nothing on failure.
    task<std::expected<void, page_error>> take_lines(sequence& seq, std::size_t line_count);
    // s, the state the store returned for r, extended by the full pages of r's tokens after its head that every layer
    // with pages caches, up to its cycle's mass window and leaving a token to forward: no query between them adds mass,
    // so it is the state the request reaches there. s itself when it ends inside the window or the model has recurrent
    // layers. digests are the digests of r's full pages.
    std::shared_ptr<const state>
    extend_with_hot_pages(std::shared_ptr<const state>&& s, const request& r, std::span<const digest> digests) const;
    // Makes the store keep, for seq's lease, the pages its views hold from history index first_index on, which the
    // state the store returned does not reference.
    task<std::expected<void, page_error>> own_pages_from(sequence& seq, std::size_t first_index);
    void release_lines(sequence& seq) noexcept;
    // Captures each shard's decode graphs of up to decode_graphs::startup_sequences sequences, every shard at once.
    void capture_decode_graphs();
    // Raises seq's peak_device_memory to the bytes of each device's pool its lines take now.
    void record_device_memory(sequence& seq) const noexcept;
    // Drops from each sliding layer's view the pages before the window of the tail's first token, and returns whether
    // any view changed.
    bool release_passed_pages(sequence& seq);
    // Advances each entry's sequence by its tokens (sequence::advance), records the pages they complete in b, and
    // appends each token the step samples to pending as awaited_token.
    void advance(batch& b);
    task<void> enqueue_on_shard(batch& b, std::size_t shard);
    task<void> synchronize_shards();
    // At the end of seq's cycle, adds to the moments of each layer that measures mass its full pages outside the sinks
    // whose last token is at least cycle_tokens and less than twice cycle_tokens before that end.
    void add_sampled_pages(sequence& seq);
    // Evicts at the end of seq's cycle and returns whether a view changed.
    bool evict(sequence& seq);
    // Writes the state seq reached to the store and passes it to seq's callbacks.
    task<void> write_cycle_end(sequence& seq);

    friend class sequence;

  public:
    // A step queued on every shard and not written back yet.
    class launched_step {
        std::unique_ptr<batch> queued;

        explicit launched_step(std::unique_ptr<batch> queued) noexcept;

        friend class executor;

      public:
        launched_step(launched_step&&) noexcept;
        launched_step& operator=(launched_step&&) noexcept;
        ~launched_step();

        // Whether the step completes no page and ends no cycle (executor::quiet).
        bool quiet() const noexcept;
        // The sequences of the step's entries, in entry order.
        std::vector<sequence*> sequences() const;
    };

    // shards[s] runs on cache shard s with streams[s], shards 1 on on worker_loops[s - 1]; the cache holds one layer
    // per entry of each model's kv_layers(), and the shards agree on which layers are recurrent or sliding. store is the
    // cache's. root is the engine's root digest (make_root_digest). With graph_pages, a decode group whose sequences'
    // tables fit graph_pages pages replays a decode_graphs graph over tables that wide.
    executor(
        std::vector<model*>&& shards,
        kv_cache cache,
        page_store& store,
        std::vector<cudaStream_t>&& streams,
        std::vector<event_loop*>&& worker_loops,
        digest root,
        std::vector<std::int64_t>&& stop_ids,
        std::optional<std::size_t> graph_pages
    );

    // The sequence of r, resuming the newest state the store holds for a prefix of its tokens, with lines for its
    // first cycle. Fails with miss when the lookup or a page read fails, and with oom when the free lines do not cover
    // the state's pages and one cycle's tokens; its lease is then released.
    task<std::expected<std::unique_ptr<sequence>, request_error>> start(request&& r, result_writer& writer);
    // Whether a step of entries completes no page and ends no cycle: its write-back then queues no work on the GPUs
    // and changes no view, so the next step can be launched before it, and its tokens need no awaited token's value.
    bool quiet(std::span<const batch_entry> entries) const noexcept;
    // Launches one step of entries, decoding entries first: advances each sequence by its tokens, then queues the
    // forward and the sampling on every shard without waiting for them. A decoding entry whose token is awaited reads
    // it from the previous launched step's samples on the GPUs, which needs that step to be quiet and this one too.
    task<launched_step> launch(std::vector<batch_entry> entries);
    // Waits for step, then sets each awaited token it sampled, writes the pages it completed, evicts and writes the
    // state at a cycle end, and takes lines for the next tokens. Sets the exit reason of each sequence that finished,
    // by a stop token or max_tokens, or ran out of lines. A sequence stopped by a token while in a later launched step
    // withdraws that step's token (sequence::withdraw_latest_step); entries of sequences that exited before are
    // skipped. Steps are written back in launch order.
    task<void> write_back(launched_step& step);
    // Releases seq's lines, writes its end state when it finished, and releases its lease; seq exited.
    task<std::expected<request_result, request_error>> finish(sequence& seq);
    // kv_cache::used_bytes of the executor's cache. Safe to call from any thread.
    std::size_t kv_cache_bytes() const noexcept;
};

}  // namespace pulsarlm
