#pragma once

#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
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
        // Whether a step of the request that is not evicted yet may end its cycle under a policy that evicts
        // (step_entry::awaits_write_back), so that no later step of it is prepared before that step's eviction (evict).
        bool awaits_write_back = false;
        // The state the request last stored, which a preemption resumes it from.
        std::shared_ptr<const pulsarlm::state> last_state;
        // Whether the request holds the lines allocate_lines gave it, from then until a preemption takes them.
        bool is_loaded = false;
        // Whether a step's prologue found no lines for the request and nothing to preempt, so that no step of it is
        // prepared before lines are released (end_line_waits).
        bool awaits_released_lines = false;
        // Whether allocate_lines gave the request lines before.
        bool was_loaded = false;
    };

  public:
    using request_id_type = slot_map<running_request>::key_type;
    // A request that the scheduler puts into a step, and how many of its pending tokens the step computes.
    struct scheduled_request {
        request_id_type request_id;
        std::size_t token_count;
    };
    class step;
    class launch_result;

  private:
    friend struct executor_inspection;
    struct step_entry;
    struct step_writes;

    // Each request from the creation of its request state until its end.
    slot_map<running_request> requests;
    // The requests that exited since the last take_exited_requests, in exit order.
    std::vector<request_id_type> exited_requests;

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
        // Indexed by get_slot. Every shard's steps advance it alike. Its tensors are allocated once, as graphs reads
        // and writes them where they are.
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
    // Per request and KV layer, the first column of its page tables that the shards' page_tables and page_positions do
    // not hold yet, none when they hold every column.
    std::unordered_map<request_id_type, std::vector<std::optional<std::size_t>>> stale_tables;
    // The requests whose request_table entries, page tables aside, the shards do not hold as their request states
    // set them, and which no unfinished step holds.
    std::vector<request_id_type> stale_requests;
    // The requests preempted since the last take_preempted_requests, in preemption order.
    std::vector<request_id_type> preempted_requests;
    // The requests that hold the lines allocate_lines gave them (running_request::is_loaded), in the order it gave them.
    std::vector<request_id_type> loaded_requests;
    // The allocate_lines calls that may hold lines for a request not loaded yet.
    std::size_t loading_count = 0;
    // Per request table key_counts cell, as an index into them flattened, the keys a prologue took out of its sliding
    // layer's page tables (compact_sliding_views), negative; not moved into a step yet.
    using key_count_cell = std::int64_t;
    using key_count_shift = std::int32_t;
    std::vector<std::pair<key_count_cell, key_count_shift>> key_count_shifts;
    // Per request table slot, the tokens of the pages a prologue's passes took out of every layer's view
    // (compact_sliding_views), negative, by which its RoPE positions move down; not moved into a step yet.
    using table_slot = std::int64_t;
    using rope_position_shift = std::int64_t;
    std::vector<std::pair<table_slot, rope_position_shift>> rope_position_shifts;
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
    // The page writes write-backs spawned, at most max_store_writes at once.
    pending_writes store_writes;
    static constexpr std::size_t max_store_writes = 8;
    // The most rows of sampled tokens a step has.
    std::size_t sampled_row_capacity;
    // The query token counts that a step which is not uniform (step::is_uniform) pads to, ascending: 1, 2, 3, then
    // each 4 * 2^k and 6 * 2^k below the most query tokens of a step, then that most (make_token_buckets).
    std::vector<std::int64_t> token_buckets;

    // The slot of id's request, its index into a shard's request_table.
    static std::size_t get_slot(request_id_type id) noexcept;
    // The smallest of token_buckets that holds token_count query tokens.
    std::size_t find_token_bucket(std::size_t token_count) const;
    // Marks the columns of id's page tables of every KV layer from first_column on as stale (stale_tables).
    void mark_stale_tables(request_id_type id, std::size_t first_column);
    // Marks the columns of id's page table of KV layer layer from first_column on as stale.
    void mark_stale_layer_table(request_id_type id, std::size_t layer, std::size_t first_column);
    // The columns of req's page tables once each KV layer holds line_counts[layer] more lines.
    std::size_t table_width_after(const request_state& req, std::span<const std::size_t> line_counts) const;
    // Per KV layer, the lines allocate_lines gives req.
    std::vector<std::size_t> start_line_counts(const request_state& req) const;
    // Moves stale_tables into s: the cells of each request's stale columns, from its view, that s's enqueue sets.
    void move_stale_tables(step& s);
    // Moves stale_requests into s: each request's request_table entries from its request state, which s's enqueue
    // sets.
    void move_stale_requests(step& s);
    // Compacts the views of id's request (request_state::compact_views) at its lower bound
    // (request_state::prepared_position_lower_bound), or at the position before it in an executor whose steps draft:
    // takes the pages each sliding layer passed, and those each mirroring layer no longer mirrors, out of that layer's
    // page table from column 0 (mark_stale_layer_table), and queues the shift of its key counts (key_count_shifts).
    // When the held pages shrink, it also marks every layer's page table stale from column 0 for their RoPE positions
    // and queues the shift of its RoPE position (rope_position_shifts).
    void compact_sliding_views(request_id_type id);
    // Moves key_count_shifts and rope_position_shifts into s, whose enqueue applies them.
    void move_table_shifts(step& s);
    // The tokens of req's full history pages that every KV layer reading the page holds resident.
    std::size_t device_cached_tokens(const request_state& req) const;
    // s, the state the store returned for r, extended by the full pages of r's tokens after its head that every layer
    // with pages caches, up to its cycle's mass window and leaving a token to forward: no query between them adds mass,
    // so it is the state the request reaches there. s itself when it ends inside the window or the model has recurrent
    // layers. digests are the digests of r's full pages.
    std::shared_ptr<const state>
    extend_with_hot_pages(std::shared_ptr<const state>&& s, const request& r, std::span<const digest> digests) const;
    // Per KV layer with pages, lines for its pages; per recurrent layer, a copy of the state at head, or zeros without
    // head, and two empty lines. Holds nothing on failure.
    task<std::expected<void, page_error>> take_view_lines(request_state& req);
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
    // After req's views changed: drops from each mirroring layer the pages no other layer holds.
    void update_views(request_state& req);
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
    // Prepares each entry's step in its request (request_state::prepare_step, request_state::prepare_verify_step) and
    // plans the draft head's pass.
    void advance(step& s);
    // Of a step of req computing token_count of its tokens, in an executor whose steps draft: whether it adds the key
    // of the token after them to the draft head's KV layer, and whether it drafts.
    std::pair<bool, bool> plan_drafting(const request_state& req, std::size_t token_count) const;
    // Gives the request of scheduled the lines it lacks for the keys its step writes, after its prepared steps, and
    // marks their columns stale. Returns false, holding none, when its view would grow past view_page_capacity pages,
    // which the request records (request_state::exceed_context), or the lines run out; the request then runs out of
    // lines (request_state::run_out_of_lines) when it has no unfinished step and its lines with the step's would not
    // fit the pool with no line held (kv_cache::can_hold).
    bool add_step_lines(const scheduled_request& scheduled);
    // Gives the request of scheduled its step's lines (add_step_lines), preempting the requests find_preemption_victim
    // names while the lines run out. Returns false when the request leaves the step: add_step_lines gave it an exit
    // cause, or nothing was left to preempt, when it keeps its lines and awaits released lines
    // (running_request::awaits_released_lines) unless it has no unfinished step and alone holds lines
    // (holds_lines_alone), when it preempts itself. lined holds the requests already in the step.
    bool prepare_lines(const scheduled_request& scheduled, std::span<const scheduled_request> lined);
    task<void> enqueue_on_shard(const step& s, launch_result& launched, std::size_t shard);
    // The request to preempt for the lines of the step of the request at requester, when the lines run out: the most
    // recently loaded request (loaded_requests) other than the requester that holds lines, has no unfinished step and
    // no exit cause, and is none of the requests in lined, those already in the step. None when no request qualifies.
    std::optional<request_id_type>
    find_preemption_victim(request_id_type requester, std::span<const scheduled_request> lined) const;
    // Takes the lines of the request at id, which has no unfinished step, rebuilds its request state from its last
    // stored state with every token it sampled as known tokens, and adds it to preempted_requests.
    void preempt(request_id_type id);
    // Clears running_request::awaits_released_lines of every request: at each write-back, preemption, stop, page write
    // end and allocate_lines that fails without an exit, all of which may release lines.
    void end_line_waits() noexcept;
    // Whether no request but the one at id is loaded, no allocate_lines holds lines and no page write is pending.
    bool holds_lines_alone(request_id_type id) const noexcept;
    // Per KV layer, the lines allocate_lines holds at once for req: those of its view's pages, and per recurrent
    // layer its state slots, two empty lines and, with head, the fetched state at head beside its copy.
    std::vector<std::size_t> count_view_lines(const request_state& req) const;
    // Whether prepare can prepare a step of the requests in scheduled, the limit sampled_row_limit() aside: scheduled
    // names no request twice, each token count is above 0 and at most the request's pending tokens, and each request
    // is schedulable (is_schedulable).
    bool is_preparable(std::span<const scheduled_request> scheduled) const;

    task<void> join(const step& s, launch_result& launched);
    // step_duration is the time from s's preparation to its write-back.
    step_writes
    apply_entries(const step& s, const launch_result& launched, std::chrono::steady_clock::duration step_duration);
    task<void> write_pages_and_states(launch_result& launched, step_writes& writes);
    // After the step's pages and states are written: releases the detached lines of its requests
    // (request_state::take_detached_lines) and the lines of the complete pages their sliding layers passed
    // (request_state::drop_passed_pages), and finishes the step in each request.
    void refresh_views(const step& s, const step_writes& writes);
    // Whether the request's callbacks are cancelled.
    bool is_cancelled(request_id_type id) const;
    // Cancels a request that has not exited, which exits right away when no step of it is unfinished.
    void cancel(request_id_type id);
    // Sets the exit reason of id's request (request_state::exit) and adds it to exited_requests.
    void exit_request(request_id_type id);

    // Awaits syncing, the task of a kv_cache::sync on the copy streams, then ends the write in store_writes and in each
    // of writers.
    task<void> async_write_back(
        task<std::expected<void, page_error>> syncing,
        std::vector<std::shared_ptr<pending_writes>> writers
    );
    task<void> synchronize_shards();
    // Evicts at the end of req's cycle and returns whether a view changed.
    bool evict_request(request_state& req);
    // Writes the state running's request reached to the store, keeps it as its last stored state, and passes it to its
    // callbacks.
    task<void> write_cycle_end(running_request& running);

  public:
    // models[s] runs on cache shard s with streams[s], for s above 0 on worker_loops[s - 1], and its write-backs'
    // copies to the store on copy_streams[s], which the caller synchronizes before freeing what the copies read; the
    // cache holds one layer per entry of each model's kv_layers(), and the shards agree on which layers are recurrent
    // or sliding, and the draft head's KV layer is not sliding. store is the cache's. root is the engine's root digest
    // (make_root_digest). A request's view takes at most view_page_capacity pages, and one that would take more ends
    // with context_exceeded. With graph_pages, a step_graphs graph over tables graph_pages wide replays a step of
    // decoding entries alone that all sample and draft nothing (a decode step), and a step of verifying entries alone
    // that all draft (a verify step). Any other step pads its query tokens to their bucket (token_buckets), with graphs
    // or without, and with graph_pages replays the bucket's segment graphs when the model allows their capture
    // (model::allows_segment_capture). A step drafts draft_tokens tokens per request with the shards' draft heads,
    // which they have when draft_tokens is above 0, and the cache's recurrent layers are laid out for them
    // (recurrent_state). At most max_running_requests requests hold a record at once (start), and a step takes at most
    // prefill_tokens_per_step query tokens of prefill chunks (scheduler).
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
        std::size_t max_running_requests,
        std::size_t prefill_tokens_per_step
    );
    ~executor();

    // The callbacks of a request whose state does not exist yet. Destroyed while it still holds them, it ends them
    // with loop_exited.
    using unstarted_callbacks = resource<
        std::unique_ptr<request_callbacks>,
        void (*)(std::unique_ptr<request_callbacks>&) noexcept>;

    // Serves r with callbacks, which stay in the guard until r's request state exists, so a start destroyed while it
    // is suspended before then ends them with loop_exited. Returns the request id of r, resuming the newest state the
    // store holds for a prefix of its tokens, without lines (allocate_lines); each step takes the lines for its tokens
    // when it is prepared. Returns std::nullopt after it ended the request through on_end, with value-initialized
    // statistics: with cancelled when the callbacks are cancelled at the start, before the store is read, with
    // missing_state when the lookup fails, with invalid_state when the stored state's head is not a digest of r's full
    // pages, its tail is not a proper prefix of r's tokens after its head, or one of its layers lists a page hash
    // twice, and with context_exceeded when its view of the state would take more than view_page_capacity pages. The
    // lease of a request that fails is released before on_end. Fewer than max_running_request_count() requests hold a
    // record at the call.
    task<std::optional<request_id_type>> start(request&& r, unstarted_callbacks callbacks);
    // Gives the request of id, which holds no lines (started or preempted), lines for its pages and recurrent states:
    // per KV layer with pages, lines for its view's pages, fetched from the store when not hot; per recurrent layer, a
    // copy of the state at head, or zeros without head, and two empty lines. Returns false holding none when the lines
    // do not fit once no page write is pending; the request then exits with oom (request_state::run_out_of_lines) when
    // they would not fit with no line held (kv_cache::can_hold). A page the store no longer has exits it with
    // missing_state, a store that does not keep the hot pages its view adds to the stored state exits it with oom, and
    // a request whose callbacks are cancelled exits with cancelled.
    task<bool> allocate_lines(request_id_type id);
    // Whether a page write a write-back spawned is pending; a pending one holds the lines of its pages.
    bool has_page_writes() const noexcept;
    // Waits until no page write is pending.
    task<void> wait_for_page_writes();
    // The requests that prologues preempted since the last call, in preemption order; each holds no lines until
    // allocate_lines.
    std::vector<request_id_type> take_preempted_requests();
    std::size_t max_running_request_count() const noexcept;
    // The requests with a record, from the creation of their request state until their end.
    std::size_t request_count() const noexcept;
    // The most rows of sampled tokens a step has.
    std::size_t sampled_row_limit() const noexcept;
    // The rows of a step's sampled tokens that the request takes when it computes token_count tokens.
    std::size_t sampled_row_count(request_id_type id, std::size_t token_count) const;
    // Prepares a step of the requests in scheduled after the prepared steps, ordered into the runs of one model call
    // (model::batch): gives each request the lines its step writes (add_step_lines) and prepares its step in its
    // request state, and takes the pages its sliding layers passed out of its page tables (compact_sliding_views). A
    // request that gets no lines leaves the step, and exits right away when no step of it is unfinished; a step that
    // every request leaves is empty. A request whose step may end its cycle under a policy that evicts is not
    // schedulable until the step's eviction. scheduled names no request
    // twice, each token count is above 0 and at most the request's pending tokens, each request is schedulable
    // (is_schedulable), the step samples at most sampled_row_limit() rows, and at most one prepared step is not written
    // back.
    step prepare(std::span<const scheduled_request> scheduled);
    // Queues s on every shard, after the steps prepared before it, without waiting for it, and returns its outputs.
    // Each shard's step reads its inputs from the shard's request_table and advances it, so a step reads the tokens and
    // positions the steps launched before it leave. An empty step queues nothing.
    task<launch_result> launch(const step& s);
    // Writes back s, after the steps prepared before it, from launched, its launch's result: waits for it, then applies
    // the tokens it forwarded and sampled, delivers them, completes the pages its tokens filled, writes the completed
    // pages, and the state of a request at a cycle end, which the GPU reports, or at its end. The pages of a request
    // writing no state are written by a spawned task; a request writing its state waits for its pages' writes first,
    // and a failed write is found at a later write-back. Records on each request a stop token or max_tokens it reached
    // or a failed store write. The entry of a request that reached the end or whose store write failed at an earlier
    // write-back applies nothing, so a step prepared after the one that reached the end changes no state the request
    // stores. An empty step writes back nothing.
    task<void> write_back(const step& s, launch_result& launched);
    // After s's write-back, from launched, its launch's result: at the cycle end of each of s's requests that reached
    // one (launch_result's cycle end flags) and has no exit cause, samples its pages, evicts under a policy that
    // evicts, writes the state it reached to the store and passes it to its callbacks. Then sets the exit reason
    // (request_state::exit) of each request of s with an exit cause that has no unfinished step left, and cancels each
    // that has not exited and whose callbacks are cancelled (cancel). An empty step evicts nothing.
    task<void> evict(const step& s, const launch_result& launched);
    // Ends the request of id, whose request state exited and has no unfinished step: releases its lines, writes its end
    // state when it finished, releases its lease, delivers the state or the error and the statistics through on_end,
    // and erases it.
    task<void> stop(request_id_type id);
    // Releases the lines and the lease of every request, delivers error and the request's statistics so far through
    // on_end, and erases the request. Called when the loop has exited.
    void end_requests(request_error error) noexcept;
    // Whether a step of the request can be prepared now: it has no exit cause (request_state::has_exit_cause), no
    // prepared step forwards its last tokens, and it awaits no step's eviction
    // (running_request::awaits_write_back). At most one step of the request is unfinished, as at most two steps are
    // unfinished when one is prepared.
    bool is_schedulable(request_id_type id) const;
    std::size_t pending_token_count(request_id_type id) const;
    // The tokens the request's eviction cycle can still forward.
    std::size_t cycle_tokens_left(request_id_type id) const;
    // Whether the request exited (request_state::exit), which leaves it no unfinished step.
    bool has_exited(request_id_type id) const;
    // The requests that exited since the last call, in exit order.
    std::vector<request_id_type> take_exited_requests();
    // kv_cache::used_bytes of the executor's cache. Safe to call from any thread.
    std::size_t kv_cache_bytes() const noexcept;
    // The drafts the verify steps written back forwarded, and those of them they accepted. Safe to call from any
    // thread.
    std::uint64_t drafted_token_count() const noexcept;
    std::uint64_t accepted_draft_count() const noexcept;
};

}  // namespace pulsarlm
