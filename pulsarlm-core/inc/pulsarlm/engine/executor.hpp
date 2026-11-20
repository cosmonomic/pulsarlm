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
#include <unordered_map>
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
#include "pulsarlm/engine/request_state.hpp"
#include "pulsarlm/engine/request_table.hpp"
#include "pulsarlm/engine/step_graphs.hpp"
#include "pulsarlm/resource.hpp"
#include "pulsarlm/slot_map.hpp"

namespace pulsarlm {

// Runs requests on the model: it owns the model shards, the KV cache and the store's side of a request, and does
// whatever touches them, while the scheduler decides what runs.
class executor {
    // A request the executor runs, from the creation of its request state until its end.
    struct running_request {
        std::unique_ptr<request_callbacks> callbacks;
        request_state state;
    };

  public:
    using request_id_type = slot_map<running_request>::key_type;
    // A request that the scheduler puts into a step, and how many of its pending tokens the step computes.
    struct scheduled_request {
        request_id_type request_id;
        std::size_t token_count;
    };

  private:
    friend struct executor_inspection;
    struct batch_entry;
    class batch;
    struct batch_writes;

    // Each request from the creation of its request state until its end.
    slot_map<running_request> requests;

    // Everything the executor keeps on one GPU across steps.
    class shard {
      public:
        // Holds the shard's part of every layer.
        pulsarlm::model* model;
        // One kv per KV layer.
        std::vector<kv> lines;
        // Per KV layer, the shard's part of the layer's mass_heads; the shards' parts in shard order make up the
        // layer's.
        std::vector<std::size_t> mass_heads;
        // stop_ids on the shard's device, int64.
        at::Tensor stop_tokens;
        cudaStream_t compute_stream;
        // The stream a write-back's copies to the store run on, after its step and alongside the next.
        cudaStream_t copy_stream;
        // The loop the shard's forward runs on, or nullptr for shard 0, whose forward runs on the executor's loop.
        event_loop* worker_loop;
        // None when steps run without graphs.
        std::unique_ptr<step_graphs> graphs;
        // The arrival counters of the steps that run without a graph, grown to the largest step's.
        at::Tensor arrival_counters;
        // Indexed by get_slot. Every shard's steps advance it alike. Its tensors are allocated once, as graphs reads and
        // writes them where they are.
        request_table table;
    };

    // One layer per KV layer, in the models' KV layer order.
    kv_cache cache;
    page_store& store;
    // One per shard of the cache.
    std::vector<shard> shards;
    // The columns of a shard's page_tables (request_table), the pages of the widest view a request holds.
    std::size_t view_page_capacity;
    // The most requests with a record at once, which bounds their slots.
    std::size_t max_running_requests;
    // Per request, the first column of its page tables that the shards' page_tables and page_positions do not hold yet.
    std::unordered_map<request_id_type, std::size_t> stale_tables;
    // The requests whose request_table entries, page tables aside, the shards do not hold as their request states
    // set them, and which no unfinished step holds.
    std::vector<request_id_type> stale_requests;
    // A recurrent layer's state_slots are draft_state_slots for a state and 1 for a window. A layer's mass_heads are
    // the query heads it measures mass and evicts pages in, those of every attention layer reading a full-attention
    // layer, and 0 for other layers.
    view_layout layout;
    // The slots of a recurrent layer keeping a state: one per token a step drafts, and one more.
    std::size_t draft_state_slots;
    // When steps draft, the draft head's KV layer (kv_layer::last_hidden_dim above 0).
    std::optional<std::size_t> draft_layer;
    std::atomic<std::uint64_t> drafted_tokens{0};
    std::atomic<std::uint64_t> accepted_drafts{0};
    std::size_t count_slots(const kv_layer& layer) const noexcept;
    // stream of every shard, in shard order.
    std::vector<cudaStream_t> collect_streams(cudaStream_t shard::* stream) const;
    digest root;
    std::vector<std::int64_t> stop_ids;
    // The steps queued on the GPUs and not written back yet, in launch order.
    std::deque<std::unique_ptr<batch>> launched_batches;
    // The page writes write-backs spawned, at most max_store_writes at once.
    pending_writes store_writes;
    static constexpr std::size_t max_store_writes = 8;
    // The most rows of sampled tokens a batch has.
    std::size_t sampled_row_capacity;

    // The slot of id's request, its index into a shard's request_table.
    static std::size_t get_slot(request_id_type id) noexcept;
    // Marks the columns of id's page tables from first_column on as stale (stale_tables).
    void mark_stale_tables(request_id_type id, std::size_t first_column);
    // The columns of req's page tables once each KV layer holds line_counts[layer] more lines.
    std::size_t table_width_after(const request_state& req, std::span<const std::size_t> line_counts) const;
    // Per KV layer, the lines take_lines gives req for its first cycle.
    std::vector<std::size_t> start_line_counts(const request_state& req) const;
    // Moves stale_tables into b: the cells of each request's stale columns, from its view, that b's enqueue sets.
    void move_stale_tables(batch& b);
    // Moves stale_requests into b: each request's request_table entries from its request state, which b's enqueue
    // sets.
    void move_stale_requests(batch& b);
    // The tokens of req's full history pages that every KV layer reading the page holds resident.
    std::size_t device_cached_tokens(const request_state& req) const;
    // Per KV layer with pages, lines for its pages, then line_count empty lines; per recurrent layer, a copy of the
    // state at head, or zeros without head, and two empty lines. Holds nothing on failure.
    task<std::expected<void, page_error>> take_lines(request_state& req, std::size_t line_count);
    // s, the state the store returned for r, extended by the full pages of r's tokens after its head that every layer
    // with pages caches, up to its cycle's mass window and leaving a token to forward: no query between them adds mass,
    // so it is the state the request reaches there. s itself when it ends inside the window or the model has recurrent
    // layers. digests are the digests of r's full pages.
    std::shared_ptr<const state>
    extend_with_hot_pages(std::shared_ptr<const state>&& s, const request& r, std::span<const digest> digests) const;
    // Makes the store keep, for req's lease, the pages its views hold from history index first_index on, which the
    // state the store returned does not reference.
    task<std::expected<void, page_error>> own_pages_from(request_state& req, std::size_t first_index);
    void release_lines(request_state& req) noexcept;
    // Captures each shard's graphs of up to step_graphs::startup_sequences sequences, every shard at once.
    void capture_step_graphs();
    // The bytes of each device's pool that req's lines take now.
    std::size_t device_memory(const request_state& req) const noexcept;
    // Raises req's device_memory_peak to its device_memory.
    void record_device_memory(request_state& req) const noexcept;
    // Drops from each sliding layer's view the pages before the window of the tail's first token, and returns whether
    // any view changed.
    bool release_passed_pages(request_state& req);
    // After req's views changed: drops from each mirroring layer the pages no other layer holds.
    void update_views(request_state& req);
    // Whether token_count more tokens fill a page of req.
    bool completes_page(const request_state& req, std::size_t token_count) const noexcept;
    // The request state of id's request, which is not erased.
    request_state& get_state(request_id_type id);
    const request_state& get_state(request_id_type id) const;
    const request_table& get_table(std::size_t shard) const;
    // Calls on_end of running's callbacks with result and statistics and destroys the callbacks.
    void deliver_end(
        running_request& running,
        std::expected<std::shared_ptr<const state>, request_error>&& result,
        const request_statistics& statistics
    ) noexcept;
    // Delivers result and statistics to id's request (deliver_end), then erases it.
    void end_request(
        request_id_type id,
        std::expected<std::shared_ptr<const state>, request_error>&& result,
        const request_statistics& statistics
    ) noexcept;
    // Moves each entry's tokens to its request's tail (request_state::launch_step, request_state::launch_verify_step),
    // appends each token the step samples to pending as awaited_token, and plans the draft head's pass.
    void advance(batch& b);
    task<void> enqueue_on_shard(batch& b, std::size_t shard);
    // Whether a step of the requests in scheduled ends no cycle and, in a model with sliding layers, completes no page, or in an executor
    // whose steps draft, whether every entry verifies drafts: its write-back then changes no view or table that the
    // next step reads, and copies to the store only lines the next step does not write.
    bool entries_allow_overlap(std::span<const scheduled_request> scheduled) const;
    // Whether launch can launch a step of the requests in scheduled now, the limit sampled_row_limit() aside: scheduled is not empty and
    // names no request twice, each token count is above 0 and at most the request's pending tokens, each request is
    // steppable (is_steppable), fewer than two steps are launched, and no step is launched or allows_overlap(scheduled).
    bool is_launchable(std::span<const scheduled_request> scheduled) const;

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
    // Evicts at the end of req's cycle and returns whether a view changed.
    bool evict(request_state& req);
    // Writes the state req reached to the store and passes it to callbacks, those of req's request.
    task<void> write_cycle_end(request_state& req, request_callbacks& callbacks);

  public:
    // models[s] runs on cache shard s with streams[s], for s above 0 on worker_loops[s - 1], and its write-backs' copies
    // to the store on copy_streams[s], which the caller synchronizes before freeing what the copies read; the cache holds one layer
    // per entry of each model's kv_layers(), and the shards agree on which layers are recurrent or sliding. store is the
    // cache's. root is the engine's root digest (make_root_digest). A request's view takes at most view_page_capacity
    // pages, and one that would take more ends with context_exceeded. With graph_pages, a step_graphs graph over tables
    // graph_pages wide replays a step of decoding entries alone that all sample and draft nothing (a decode step), and
    // a step of verifying entries alone that all draft (a verify step); any other step runs without a graph. A step
    // drafts draft_tokens tokens per request
    // with the shards' draft heads, which they have when draft_tokens is above 0, and the cache's recurrent layers are
    // laid out for them (recurrent_state). At most max_running_requests requests hold a record at once (start).
    executor(
        std::vector<model*>&& models,
        kv_cache cache,
        page_store& store,
        std::vector<cudaStream_t>&& streams,
        std::vector<cudaStream_t>&& copy_streams,
        std::vector<event_loop*>&& worker_loops,
        digest root,
        std::vector<std::int64_t>&& stop_ids,
        std::size_t view_page_capacity,
        std::optional<std::size_t> graph_pages,
        std::size_t draft_tokens,
        std::size_t sampled_row_limit,
        std::size_t max_running_requests
    );
    ~executor();

    // The callbacks of a request whose state does not exist yet. Destroyed while it still holds them, it ends them
    // with loop_exited.
    using unstarted_callbacks = resource<
        std::unique_ptr<request_callbacks>,
        void (*)(std::unique_ptr<request_callbacks>&) noexcept>;

    // Serves r with callbacks, which stay in the guard until r's request state exists, so a start destroyed while it
    // is suspended before then ends them with loop_exited. Returns the request id of r, resuming the newest state the
    // store holds for a prefix of its tokens, with lines for its first cycle. Returns std::nullopt after it ended the
    // request through on_end, with value-initialized statistics: with cancelled when the callbacks are cancelled at the
    // start, before the store is read, with missing_state when the lookup or a page read fails, with invalid_state when
    // the stored state's head is not a digest of r's full pages, its tail is not a proper prefix of r's tokens after
    // its head, or one of its layers lists a page hash twice, and with oom when the free lines do not cover the state's
    // pages and one cycle's tokens, and with context_exceeded, before it takes lines, when its view with them would take
    // more than view_page_capacity pages. The lease of a request that fails is released before on_end. Fewer than max_running_request_count() requests hold a record at the call.
    task<std::optional<request_id_type>> start(request&& r, unstarted_callbacks callbacks);
    std::size_t max_running_request_count() const noexcept;
    // The requests with a record, from the creation of their request state until their end.
    std::size_t request_count() const noexcept;
    // Whether a step of the requests in scheduled can be launched now, before any write-back: no step is launched or the last launched
    // step allows overlap, and so does a step of the requests in scheduled. A step allows overlap when it ends no cycle and, in a model
    // with sliding layers, completes no page, or in an executor whose steps draft, when every entry verifies drafts.
    bool allows_overlap(std::span<const scheduled_request> scheduled) const;
    // The launched steps not written back yet, at most 2.
    std::size_t launched_batch_count() const noexcept;
    // The most rows of sampled tokens a batch has.
    std::size_t sampled_row_limit() const noexcept;
    // The rows of a batch's sampled tokens that the request takes when it computes token_count tokens.
    std::size_t sampled_row_count(request_id_type id, std::size_t token_count) const;
    // Launches one step of the requests in scheduled after the launched steps, ordered into the runs of one model call
    // (model::batch): advances each request by its tokens, then queues the forward and the sampling on every shard
    // without waiting for them. Each shard's step reads
    // its inputs from the shard's request_table and advances it, so a step reads the tokens and positions the launched
    // steps before it leave.
    // scheduled is not empty and names no request twice, each token count is above 0 and at most the request's pending
    // tokens, each request is steppable (is_steppable), fewer than two steps are launched, no step is launched or
    // allows_overlap(scheduled), and the step samples at most sampled_row_limit() rows.
    task<void> launch(std::span<const scheduled_request> scheduled);
    // Writes back the oldest launched step: waits for it, then completes the pages its tokens filled, sets each awaited token it sampled, writes the
    // completed pages, evicts and writes the state at a cycle end, and takes lines for the next tokens. The pages of a
    // request writing no state are written by a spawned task; a request writing its state waits for its pages' writes
    // first, and a failed write is found at a later write-back. Records on each request a stop token or max_tokens it
    // reached, a failed store write, or a line allocation that failed or a view that would grow past view_page_capacity pages, and sets the exit reason (request_state::exit) of
    // each request with an exit cause that the step leaves in no launched step. A request stopped by a token while in a
    // later launched step withdraws that step's token (request_state::withdraw_latest_step); entries of requests that
    // reached the end or whose store write failed before are skipped. Returns the requests of the step's entries, in
    // entry order. A step is launched.
    task<std::vector<request_id_type>> write_back();
    // Ends the request of id, whose request state exited and is in no launched step: releases its lines, writes its end state
    // when it finished, releases its lease, delivers the state or the error and the statistics through on_end, and
    // erases it.
    task<void> stop(request_id_type id);
    // Releases the lines and the lease of every request, delivers error and the request's statistics so far through
    // on_end, and erases the request. Called when the loop has exited.
    void end_requests(request_error error) noexcept;
    // Whether the request has no exit cause (request_state::has_exit_cause) and no launched step forwards its last
    // tokens.
    bool is_steppable(request_id_type id) const;
    std::size_t pending_token_count(request_id_type id) const;
    // The tokens the request's eviction cycle can still forward.
    std::size_t cycle_tokens_left(request_id_type id) const;
    bool has_exited(request_id_type id) const;
    bool is_in_launched_batch(request_id_type id) const;
    // Whether the request's callbacks are cancelled.
    bool is_cancelled(request_id_type id) const;
    // Cancels a request that has not exited, which exits right away when it is in no launched step.
    void cancel(request_id_type id);
    // kv_cache::used_bytes of the executor's cache. Safe to call from any thread.
    std::size_t kv_cache_bytes() const noexcept;
    // The drafts the verify steps written back forwarded, and those of them they accepted. Safe to call from any thread.
    std::uint64_t drafted_token_count() const noexcept;
    std::uint64_t accepted_draft_count() const noexcept;
};

}  // namespace pulsarlm
