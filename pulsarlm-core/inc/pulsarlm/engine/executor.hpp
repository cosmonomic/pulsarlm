#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <deque>
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
#include "pulsarlm/engine/pending_writes.hpp"
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
    // The slots of a recurrent layer keeping a state: one per token a step drafts, and one more.
    std::size_t draft_state_slots;
    // When steps draft, the draft head's KV layer (kv_layer::last_hidden_dim above 0).
    std::optional<std::size_t> draft_layer;
    // Per shard, stop_ids on its device, int64.
    std::vector<at::Tensor> stop_tokens;
    // Per recurrent KV layer, its slots: draft_state_slots for a state, 1 for a window; 0 for a layer with pages.
    std::vector<std::size_t> state_slots;
    std::size_t slots_of(const kv_layer& layer) const noexcept;
    // Per KV layer, whether it mirrors the pages the other attention layers hold (kv_layer::mirrors_held_pages).
    std::vector<bool> mirrors;
    // Per KV layer, the query heads it measures mass and evicts pages in: those of every attention layer reading a
    // full-attention layer, 0 for other layers.
    std::vector<std::size_t> mass_heads;
    // Per shard, per KV layer, the shard's part of mass_heads; the shards' parts in shard order make up the layer's.
    std::vector<std::vector<std::size_t>> shard_mass_heads;
    // One stream per shard.
    std::vector<cudaStream_t> streams;
    // Per shard, the stream a write-back's copies to the store run on, after its step and alongside the next.
    std::vector<cudaStream_t> copy_streams;
    // The loops of shards 1 on: shard s's forward runs on worker_loops[s - 1]; shard 0's on the executor's loop.
    std::vector<event_loop*> worker_loops;
    std::size_t page_size;
    digest root;
    std::vector<std::int64_t> stop_ids;
    // The steps queued on the GPUs and not written back yet, in launch order.
    std::deque<std::unique_ptr<batch>> launched;
    // The page writes write-backs spawned, at most max_store_writes at once.
    pending_writes store_writes;
    static constexpr std::size_t max_store_writes = 8;
    // Per shard, the arrival counters of the steps that run without a graph, grown to the largest step's.
    std::vector<at::Tensor> eager_counters;
    // Per shard, int64 [max_sampled_rows] on its device: the tokens the latest launched step sampled, every shard
    // sampling them alike.
    std::vector<at::Tensor> samples;
    static constexpr std::int64_t max_sampled_rows = 4096;

    // The tokens of seq's full history pages that every KV layer reading the page holds resident.
    std::size_t device_cached_tokens(const sequence& seq) const;
    // Per KV layer with pages, lines for its pages, then line_count empty lines; per recurrent layer, a copy of the
    // state at head, or zeros without head, and two empty lines. Holds nothing on failure.
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
    // After seq's views changed: sets its held pages, then drops from each mirroring layer the pages no other layer
    // holds.
    void update_views(sequence& seq);
    // Whether e's tokens fill a page of its sequence.
    bool completes_page(const batch_entry& e) const noexcept;
    // Moves each entry's tokens to its sequence's tail (sequence::advance), switches the boundary lines of the
    // sequences whose pages it fills, and appends each token the step samples to pending as awaited_token.
    void advance(batch& b);
    task<void> enqueue_on_shard(batch& b, std::size_t shard);
    // Points b's verify entries whose sequences' previous step is launched to that step.
    void link_previous(batch& b) const;
    // Adds the page mass of entry index of b's step, whose sequence kept kept tokens, to its sequence's layers.
    void add_measured_mass(const batch& b, std::size_t index, std::size_t kept);

    // Syncs the pages of hashes, owners and line_ids (kv_cache::sync) on the copy streams, then ends the write in
    // store_writes and in each of writers.
    task<void> async_write_back(
        std::vector<std::vector<digest>> hashes,
        std::vector<std::vector<state::lease_id_type>> owners,
        std::vector<std::vector<std::uint32_t>> line_ids,
        std::vector<std::shared_ptr<pending_writes>> writers
    );
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
    // shards[s] runs on cache shard s with streams[s], shards 1 on on worker_loops[s - 1], and its write-backs' copies
    // to the store on copy_streams[s], which the caller synchronizes before freeing what the copies read; the cache holds one layer
    // per entry of each model's kv_layers(), and the shards agree on which layers are recurrent or sliding. store is the
    // cache's. root is the engine's root digest (make_root_digest). With graph_pages, a decode group whose sequences'
    // tables fit graph_pages pages replays a decode_graphs graph over tables that wide, and so does a verify group. A step drafts
    // draft_tokens tokens per sequence with the shards' draft heads, which they have when draft_tokens is above 0, and
    // the cache's recurrent layers are laid out for them (recurrent_state).
    executor(
        std::vector<model*>&& shards,
        kv_cache cache,
        page_store& store,
        std::vector<cudaStream_t>&& streams,
        std::vector<cudaStream_t>&& copy_streams,
        std::vector<event_loop*>&& worker_loops,
        digest root,
        std::vector<std::int64_t>&& stop_ids,
        std::optional<std::size_t> graph_pages,
        std::size_t draft_tokens
    );
    ~executor();

    // The sequence of r, resuming the newest state the store holds for a prefix of its tokens, with lines for its
    // first cycle. Fails with miss when the lookup or a page read fails, and with oom when the free lines do not cover
    // the state's pages and one cycle's tokens; its lease is then released.
    task<std::expected<std::unique_ptr<sequence>, request_error>> start(request&& r, result_writer& writer);
    // Whether a step of entries ends no cycle and, in a model with sliding layers, completes no page, or in an executor
    // whose steps draft, whether every entry verifies drafts: its write-back
    // then changes no view or table that the next step reads, and copies to the store only lines the next step does not
    // write, so the next step can be launched before it.
    bool quiet(std::span<const batch_entry> entries) const noexcept;
    // Launches one step of entries, decoding entries first, after the launched steps: advances each sequence by its
    // tokens, then queues the forward and the sampling on every shard without waiting for them. A decoding entry whose token is awaited reads
    // it from the previous launched step's samples on the GPUs, which needs that step to be quiet and this one too.
    // A step samples at most max_sampled_rows tokens.
    task<void> launch(std::vector<batch_entry> entries);
    // Writes back the oldest launched step: waits for it, then completes the pages its tokens filled, sets each awaited token it sampled, writes the
    // completed pages, evicts and writes the state at a cycle end, and takes lines for the next tokens. The pages of a
    // sequence writing no state are written by a spawned task; a sequence writing its state waits for its pages' writes
    // first, and a failed write ends its sequence with oom at a later write-back. Sets the exit reason of each sequence that finished,
    // by a stop token or max_tokens, or ran out of lines. A sequence stopped by a token while in a later launched step
    // withdraws that step's token (sequence::withdraw_latest_step); entries of sequences that exited before are
    // skipped.
    task<void> write_back();
    // Releases seq's lines, writes its end state when it finished, and releases its lease; seq exited.
    task<std::expected<request_result, request_error>> finish(sequence& seq);
    // kv_cache::used_bytes of the executor's cache. Safe to call from any thread.
    std::size_t kv_cache_bytes() const noexcept;
};

}  // namespace pulsarlm
