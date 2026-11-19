#pragma once

#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "pulsarlm/digest.hpp"
#include "pulsarlm/engine/event_loop.hpp"
#include "pulsarlm/engine/kv_cache.hpp"
#include "pulsarlm/engine/model/interface.hpp"
#include "pulsarlm/engine/page_store.hpp"
#include "pulsarlm/engine/pending_writes.hpp"
#include "pulsarlm/engine/request.hpp"
#include "pulsarlm/engine/request_callbacks.hpp"
#include "pulsarlm/engine/sequence.hpp"
#include "pulsarlm/engine/step_graphs.hpp"
#include "pulsarlm/slot_map.hpp"

namespace pulsarlm {

// Runs sequences on the model: it owns the model shards, the KV cache and the store's side of a request, and does
// whatever touches them, while the scheduler decides what runs.
class executor {
    // A request being served: its callbacks, and its sequence once the stored state was read.
    struct served_request {
        std::unique_ptr<request_callbacks> callbacks;
        std::optional<sequence> state;
    };

  public:
    using sequence_id = slot_map<served_request>::key_type;
    // A sequence and the tokens it computes in a batch.
    using batch_tokens = std::pair<sequence_id, std::size_t>;

  private:
    class batch;
    struct batch_writes;

    // Names a sequence of an executor without owning it. Each access throws std::out_of_range once the sequence's
    // request was erased.
    class sequence_ref {
        executor* owner;
        sequence_id id;

      public:
        sequence_ref(executor& owner, sequence_id id) noexcept;
        sequence& operator*() const;
        sequence* operator->() const;
        sequence_id key() const noexcept;
        bool operator==(const sequence_ref& other) const noexcept = default;
    };

    // Each request from start until its end.
    slot_map<served_request> requests;

    // One model per shard of the cache, each holding its part of every layer.
    std::vector<model*> shards;
    // One layer per KV layer, in the models' KV layer order.
    kv_cache cache;
    page_store& store;
    // Per shard, one kv per KV layer.
    std::vector<std::vector<kv>> lines;
    // One per shard, or none when decode and verify groups forward without graphs.
    std::vector<std::unique_ptr<step_graphs>> graphs;
    // The page-table width of the graphs, the pages of the longest view they cover.
    std::optional<std::size_t> graph_pages;
    // A recurrent layer's state_slots are draft_state_slots for a state and 1 for a window. A layer's mass_heads are
    // the query heads it measures mass and evicts pages in, those of every attention layer reading a full-attention
    // layer, and 0 for other layers.
    view_layout layout;
    // The slots of a recurrent layer keeping a state: one per token a step drafts, and one more.
    std::size_t draft_state_slots;
    // When steps draft, the draft head's KV layer (kv_layer::last_hidden_dim above 0).
    std::optional<std::size_t> draft_layer;
    // Per shard, stop_ids on its device, int64.
    std::vector<at::Tensor> stop_tokens;
    std::atomic<std::uint64_t> drafted_tokens{0};
    std::atomic<std::uint64_t> accepted_drafts{0};
    std::size_t slots_of(const kv_layer& layer) const noexcept;
    // Per shard, per KV layer, the shard's part of the layer's mass_heads; the shards' parts in shard order make up the
    // layer's.
    std::vector<std::vector<std::size_t>> shard_mass_heads;
    // One stream per shard.
    std::vector<cudaStream_t> streams;
    // Per shard, the stream a write-back's copies to the store run on, after its step and alongside the next.
    std::vector<cudaStream_t> copy_streams;
    // The loops of shards 1 on: shard s's forward runs on worker_loops[s - 1]; shard 0's on the executor's loop.
    std::vector<event_loop*> worker_loops;
    digest root;
    std::vector<std::int64_t> stop_ids;
    // The steps queued on the GPUs and not written back yet, in launch order.
    std::deque<std::unique_ptr<batch>> launched_batches;
    // The page writes write-backs spawned, at most max_store_writes at once.
    pending_writes store_writes;
    static constexpr std::size_t max_store_writes = 8;
    // Per shard, the arrival counters of the steps that run without a graph, grown to the largest step's.
    std::vector<at::Tensor> eager_counters;
    // The most rows of sampled tokens a batch has.
    std::size_t sampled_row_capacity;
    // Per shard, int64 [sampled_row_capacity] on its device: the tokens the latest launched step sampled, every shard
    // sampling them alike.
    std::vector<at::Tensor> samples;

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
    // Captures each shard's graphs of up to step_graphs::startup_sequences sequences, every shard at once.
    void capture_step_graphs();
    // The bytes of each device's pool that seq's lines take now.
    std::size_t device_memory(const sequence& seq) const noexcept;
    // Raises seq's device_memory_peak to its device_memory.
    void record_device_memory(sequence& seq) const noexcept;
    // Drops from each sliding layer's view the pages before the window of the tail's first token, and returns whether
    // any view changed.
    bool release_passed_pages(sequence& seq);
    // After seq's views changed: drops from each mirroring layer the pages no other layer holds.
    void update_views(sequence& seq);
    // Whether token_count more tokens fill a page of seq.
    bool completes_page(const sequence& seq, std::size_t token_count) const noexcept;
    // The sequence of id's request. Throws std::out_of_range once the request was erased, and
    // std::bad_optional_access before start read the stored state.
    sequence& sequence_of(sequence_id id);
    const sequence& sequence_of(sequence_id id) const;
    // Calls on_end of served's callbacks with result and statistics and destroys the callbacks.
    void deliver_end(
        served_request& served,
        std::expected<std::shared_ptr<const state>, request_error>&& result,
        const request_statistics& statistics
    ) noexcept;
    // Delivers result and statistics to id's request (deliver_end), then erases it.
    void end_request(
        sequence_id id,
        std::expected<std::shared_ptr<const state>, request_error>&& result,
        const request_statistics& statistics
    ) noexcept;
    // Moves each entry's tokens to its sequence's tail (sequence::launch_step, sequence::launch_verify_step), switches
    // the boundary lines of the sequences whose pages it fills, and appends each token the step samples to pending as
    // awaited_token.
    void advance(batch& b);
    task<void> enqueue_on_shard(batch& b, std::size_t shard);
    // Points b's verify entries whose sequences' previous step is launched to that step. Throws std::logic_error for
    // such a sequence that is not a verify entry and a drafting entry of that step.
    void link_previous(batch& b) const;
    // Whether a step of entries ends no cycle and, in a model with sliding layers, completes no page, or in an executor
    // whose steps draft, whether every entry verifies drafts: its write-back then changes no view or table that the
    // next step reads, and copies to the store only lines the next step does not write.
    bool entries_allow_overlap(std::span<const batch_tokens> entries) const;
    // Throws std::logic_error unless launch can launch a step of entries now, the limit sampled_row_limit() aside.
    void check_launch(std::span<const batch_tokens> entries) const;

    task<void> join(batch& b);
    // step_duration is the time from b's launch to its write-back.
    batch_writes apply_entries(batch& b, std::chrono::steady_clock::duration step_duration);
    task<void> write_pages_and_states(batch& b, batch_writes&& writes);
    task<void> refresh_views(batch& b);

    // Awaits syncing, the task of a kv_cache::sync on the copy streams, then ends the write in store_writes and in each
    // of writers.
    task<void> async_write_back(
        task<std::expected<void, page_error>> syncing,
        std::vector<std::shared_ptr<pending_writes>> writers
    );
    task<void> synchronize_shards();
    // Evicts at the end of seq's cycle and returns whether a view changed.
    bool evict(sequence& seq);
    // Writes the state seq reached to the store and passes it to callbacks, those of seq's request.
    task<void> write_cycle_end(sequence& seq, request_callbacks& callbacks);

  public:
    // shards[s] runs on cache shard s with streams[s], shards 1 on on worker_loops[s - 1], and its write-backs' copies
    // to the store on copy_streams[s], which the caller synchronizes before freeing what the copies read; the cache holds one layer
    // per entry of each model's kv_layers(), and the shards agree on which layers are recurrent or sliding. store is the
    // cache's. root is the engine's root digest (make_root_digest). With graph_pages, a decode group whose sequences'
    // tables fit graph_pages pages replays a step_graphs graph over tables that wide, and so does a verify group. A step drafts
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
        std::size_t draft_tokens,
        std::size_t sampled_row_limit
    );
    ~executor();

    // Serves r with callbacks, which it takes before it first suspends. Returns the sequence of r, resuming the newest
    // state the store holds for a prefix of its tokens, with lines for its first cycle. Returns std::nullopt after it
    // ended the request through on_end, with value-initialized statistics: with cancelled when the callbacks are
    // cancelled at the start, before the store is read, with miss when the lookup or a page read fails, and with oom
    // when the free lines do not cover the state's pages and one cycle's tokens. The lease of a request that fails is
    // released before on_end.
    task<std::optional<sequence_id>> start(request&& r, std::unique_ptr<request_callbacks>&& callbacks);
    // Whether a step of entries can be launched now, before any write-back: no step is launched or the last launched
    // step allows overlap, and so does a step of entries. A step allows overlap when it ends no cycle and, in a model
    // with sliding layers, completes no page, or in an executor whose steps draft, when every entry verifies drafts.
    bool allows_overlap(std::span<const batch_tokens> entries) const;
    // The launched steps not written back yet, at most 2.
    std::size_t launched_batch_count() const noexcept;
    // The most rows of sampled tokens a batch has.
    std::size_t sampled_row_limit() const noexcept;
    // The rows of a batch's sampled tokens that the sequence takes when it computes token_count tokens.
    std::size_t sampled_row_count(sequence_id id, std::size_t token_count) const;
    // Launches one step of entries, decoding entries first, after the launched steps: advances each sequence by its
    // tokens, then queues the forward and the sampling on every shard without waiting for them. A decoding entry whose token is awaited reads
    // it from the previous launched step's samples on the GPUs, which needs both steps to allow overlap.
    // Throws std::logic_error, which panics the loop, before it changes a sequence or queues work, when entries is
    // empty or names a sequence twice, a token count is 0 or above the sequence's pending tokens, a sequence is not
    // steppable (is_steppable), two steps are launched, a step is launched and allows_overlap(entries) is false, or
    // the step samples more than sampled_row_limit() rows.
    task<void> launch(std::span<const batch_tokens> entries);
    // Writes back the oldest launched step: waits for it, then completes the pages its tokens filled, sets each awaited token it sampled, writes the
    // completed pages, evicts and writes the state at a cycle end, and takes lines for the next tokens. The pages of a
    // sequence writing no state are written by a spawned task; a sequence writing its state waits for its pages' writes
    // first, and a failed write ends its sequence with oom at a later write-back. Sets the exit reason of each sequence that finished,
    // by a stop token or max_tokens, or ran out of lines. A sequence stopped by a token while in a later launched step
    // withdraws that step's token (sequence::withdraw_latest_step); entries of sequences that exited before are
    // skipped. Returns the sequences of the step's entries, in entry order. Throws std::logic_error, which panics the
    // loop, when no step is launched.
    task<std::vector<sequence_id>> write_back();
    // Ends the request of id, whose sequence exited and is in no launched step: releases its lines, writes its end state
    // when it finished, releases its lease, delivers the state or the error and the statistics through on_end, and
    // erases it. Throws std::logic_error for a sequence that has not exited or that a launched step holds.
    task<void> stop(sequence_id id);
    // Releases the lines and the lease of every request's sequence, delivers error and the request's statistics so far
    // through on_end, and erases the request. Called when the loop has exited.
    void end_requests(request_error error) noexcept;
    // Whether the sequence has not exited and no launched step forwards its last tokens.
    bool is_steppable(sequence_id id) const;
    std::size_t pending_token_count(sequence_id id) const;
    // The tokens the sequence's eviction cycle can still forward.
    std::size_t cycle_tokens_left(sequence_id id) const;
    bool has_exited(sequence_id id) const;
    bool is_in_launched_batch(sequence_id id) const;
    // Whether the sequence's callbacks are cancelled.
    bool is_cancelled(sequence_id id) const;
    // Makes a sequence that has not exited exit as cancelled.
    void cancel(sequence_id id);
    // kv_cache::used_bytes of the executor's cache. Safe to call from any thread.
    std::size_t kv_cache_bytes() const noexcept;
    // The drafts the verify steps written back forwarded, and those of them they accepted. Safe to call from any thread.
    std::uint64_t drafted_token_count() const noexcept;
    std::uint64_t accepted_draft_count() const noexcept;
};

}  // namespace pulsarlm
