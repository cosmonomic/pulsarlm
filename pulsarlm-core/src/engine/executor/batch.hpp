#pragma once

#include "../pinned_tensor.hpp"

#include "pulsarlm/engine/executor.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pulsarlm {

inline void require(bool condition, const char* message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

// Sets g's page mass_sum and weight_sum for the KV layers that measure mass, those with mass_heads above 0, to zeroed
// parts of one tensor on device, which it returns: per such layer in turn, its mass_sum, then its weight_sum.
// host_cu_view_pages is on the host.
at::Tensor zero_page_mass(
    model::batch& g,
    const at::Tensor& host_cu_view_pages,
    const std::vector<std::size_t>& mass_heads,
    at::Device device
);

// A request's part of a step.
struct executor::batch_entry {
    request_id_type request_id;
    // The tokens the request's query holds: 1, state_slots, or the prefill chunk's length.
    std::size_t query_token_count;
    // The index of the entry's first sample among the step's samples, for an entry that samples.
    std::optional<std::size_t> first_sample_index;
    // Whether the entry may end its request's cycle (request_state::ends_cycle of its query tokens); the write-back
    // takes whether it did from the GPU (cycle_end_flags).
    bool ends_cycle;
    // Whether the step forwards the request's last tokens.
    bool ends_request = false;
    bool is_measured = false;
    // Whether the write-back stores the request's head state.
    bool stores_head_state = false;
    // The request's prepared position bound (request_state::prepared_position_bound) and page-table width
    // (request_state::table_width) at the step's preparation.
    std::size_t start_position = 0;
    std::size_t table_width = 0;

    // The request's rows in the draft pass. logit_row is the row that drafts, before a second-run entry's accepted
    // count - 1 is added, and key_count_offset the request's keys after its rows in the draft head's KV layer
    // less its keys there at the step's start.
    struct draft_rows {
        std::size_t row_count;
        std::size_t logit_row;
        std::size_t key_count_offset;
        bool is_drafting;
    };
    std::optional<draft_rows> draft;

    // Of an entry of the second run, the most tokens it may accept (max_tokens).
    std::size_t max_accepted_token_count = 0;
};

// The tokens each running request forwards in one step, the model inputs for them, and the tokens sampled from the
// step's logits.
class executor::batch {
    // In run order (model::batch): requests whose one pending token decodes, then requests whose pending token and
    // drafts verify, then prefill chunks; within a run in the scheduler's order.
    std::vector<batch_entry> entries;
    std::size_t single_query_request_count = 0;
    std::size_t per_query_state_request_count = 0;
    // Pinned host tensors of the model's inputs: the run counts, page_size, state_slots, cu_seqlens_q and its parts
    // from the later runs, cu_view_pages when the step measures mass, and logit_tokens: the last query token of each
    // entry of the first and third runs and every query token of the second. The other inputs are gathered on each
    // shard from its request_table.
    model::batch inputs;
    // Pinned host int64: the slot of each entry's request (executor::get_slot), and the third run's tokens.
    at::Tensor request_slots;
    at::Tensor prefill_tokens;
    // The page-table width of the inputs gathered without a graph, the widest table among the entries' requests.
    std::size_t table_width = 0;
    // The entries of the model's tokens: the query tokens of a uniform step (is_uniform), else their bucket
    // (executor::find_token_bucket), set at the launch.
    std::size_t padded_token_count = 0;
    // Pinned host fp32, defined for a step that measures mass: the page mass of each KV layer that measures it in
    // turn, laid out as the layer's row of the step's cu_view_pages, its heads innermost; per shard, its heads, then
    // joined over the shards.
    std::vector<at::Tensor> mass_by_shard;
    at::Tensor mass;
    // Per shard, the hidden rows of the step's query tokens, which the draft head's pass reads.
    std::vector<at::Tensor> hidden_by_shard;
    // The time at the start of the executor::launch that launched the batch.
    std::chrono::steady_clock::time_point launch_time;
    // executor::entries_allow_overlap of the step's entries at its launch.
    bool allows_overlap = false;
    // Pinned host tensors, undefined when the step sets none: the cells of each shard's page_tables and page_positions
    // that the step sets before its forward, as indices into them flattened (int64), and their lines and RoPE
    // positions (int32).
    at::Tensor table_cells;
    at::Tensor table_lines;
    at::Tensor table_positions;
    // Pinned host int64, undefined when the step sets none: the slots of each shard's request_table whose entries,
    // page tables aside, the step sets before its forward. stale_entries holds their values as pinned host tensors
    // laid out as the table's, one entry per slot, its page tables undefined.
    at::Tensor stale_slots;
    request_table stale_entries;
    // Pinned host int64, per sample in entry order (one of each entry that samples, but one per query token of an
    // entry of the second run): the slot of its request, its offset among its request's samples, which it adds to the
    // request's draw count, and its index among the logit tokens.
    at::Tensor sample_slots;
    at::Tensor sample_offsets;
    at::Tensor sample_logit_indices;
    // Whether an entry that samples filters (pulsarlm::filters).
    bool filters = false;
    // Pinned host int64, the slots of the entries that sample outside the second run, each taking one draw.
    at::Tensor single_draw_slots;
    // Pinned host int64, one token per sample.
    at::Tensor sampled_tokens;
    // Pinned host tensors in entry order: the tokens each entry computes, 0 for an entry of the second run, whose
    // accepted count is added on the GPU (int64), and its next token, -(r + 1) for the step's sample r, the first
    // sample of an entry of the second run before its accepted count - 1 is added (int64). With cycle_end_flags,
    // which shard 0 copies from the GPU after the step, whether the entry's request reached its cycle end (bool).
    at::Tensor computed_token_counts;
    at::Tensor next_token_sources;
    at::Tensor cycle_end_flags;
    // Pinned host bytes that the pinned host tensors above are views into, except the table cells and stale entries,
    // so that one copy moves them to a shard's device (pack_host_inputs), and per shard that copy, made by the eager
    // path of enqueue.
    at::Tensor host_input_bytes;
    std::vector<at::Tensor> device_input_bytes;
    // Per shard, completing once the step's work on its stream has, and an event recorded after that work.
    std::vector<external_task<cudaError_t>> completions;
    std::vector<at::cuda::CUDAEvent> ends;
    // The slot of each entry's request whose first draft row reads the last_hidden of the position before its first
    // token, and per shard, those hidden vectors gathered at the start of the step from the draft head's KV layer.
    std::vector<std::int64_t> prior_hidden_slots;
    std::vector<at::Tensor> shard_prior_hidden;
    // The draft head's pass, queued after the step's sampling, over the entries with draft rows in entry order: per
    // row, its token, -(r + 1) for the step's sample r, its RoPE position less its request's at the step's start,
    // and its hidden row (draft_head::draft) as an index into a zeros row, then the prior hidden rows, then the hidden
    // rows of the step's query tokens. The rows of an entry of the second run are the state_slots positions after its
    // pending token's, the row after its accepted tokens drafting: its first row's index plus its accepted count - 1.
    // Another request drafts from its last row.
    std::vector<std::int64_t> draft_tokens;
    std::vector<std::int64_t> draft_position_offsets;
    std::vector<std::int64_t> draft_sources;
    // The tokens each drafting request drafts. Draft j after the first sits j positions after the request's last
    // computed token.
    std::size_t draft_count = 0;
    // Per shard, the second run's tokens on its device.
    std::vector<at::Tensor> verify_tokens;
    // Per shard, each second-run entry's accepted tokens (int64 [second-run entries]), and each drafting request's next
    // token then its drafts (int64 [drafting, state_slots]); shard 0's copies to the host.
    std::vector<at::Tensor> accepted_token_counts_by_shard;
    std::vector<at::Tensor> shard_next_tokens;
    at::Tensor accepted_token_counts;
    at::Tensor next_tokens;

    // scheduled with their requests found in owner, ordered into runs, each recorded with whether it samples after its
    // tokens and whether they end its eviction cycle.
    batch(executor& owner, std::span<const scheduled_request> scheduled);

    // Moves the pinned host inputs of the step into host_input_bytes, each at an offset aligned to 16 bytes, and makes
    // each a view of its bytes there.
    void pack_host_inputs();
    // The view of host, a tensor packed into host_input_bytes, in shard's device_input_bytes.
    at::Tensor get_on_device(const at::Tensor& host, std::size_t shard) const;
    // Whether the entry at index belongs to the second run, verifying its drafts.
    bool verifies(std::size_t index) const noexcept;
    // The mass views (model::batch::cu_view_pages) of the entries before index.
    std::size_t count_views_before(std::size_t index) const noexcept;
    // The query tokens of the entries.
    std::size_t count_tokens() const noexcept;
    // The entries of the requests that draft, in entry order.
    std::vector<batch_entry> drafting_entries() const;
    // The host inputs of the step (inputs) and its prefill_tokens, of the entries of owner's requests. An entry of the
    // second run measures each query token's mass apart (model::batch::cu_view_pages), its queries weighing 1 from
    // mass_start_position on: the query tokens its write-back accepts are then weighed as one step's.
    void compose_inputs(const executor& owner);
    // Whether req forwards a query of query_token_count tokens that adds mass.
    static bool measures_mass(const request_state& req, std::size_t query_token_count) noexcept;
    // Pinned host int64, the slot of each entry's request.
    static at::Tensor collect_slots(std::span<const batch_entry> entries);
    // int32 [n_kv_layers, slots, width] on tables' device: the first width columns of the rows of tables, a shard's
    // page_tables or page_positions, at slots, which are on that device.
    static at::Tensor select_tables(const at::Tensor& slots, const at::Tensor& tables, std::size_t width);
    // Sets g's page tables and positions to the rows of table's page_tables and page_positions of the requests'
    // entries, as wide as the widest table among them.
    static void set_page_tables(std::span<const batch_entry> requests, model::batch& g, const request_table& table);
    // Queues on the current stream the copies of the step's table cells and stale entries to the device of table, a
    // shard's, and their writes.
    void write_tables(request_table& table) const;
    // On shard's device, table's, the inputs of the step's sampling: per sample, its request's columns of table, its
    // draw count moved by the sample's offset.
    sampling_inputs gather_sampling_inputs(const request_table& table, std::size_t shard) const;
    // Queues on the current stream the addition to table's draw_counts of the draws the step took: one per entry that
    // samples outside the second run, and a second-run entry's accepted count.
    void advance_draw_counts(request_table& table, std::size_t shard) const;
    // Queues on the current stream, shard's, the gather into shard_prior_hidden of the last_hidden that the entries at
    // prior_hidden_slots read: of the line of draft_layer's view page holding the key before the request's next token
    // in table.
    void gather_prior_hidden(
        const request_table& table,
        std::span<const kv> lines,
        std::size_t draft_layer,
        std::size_t shard,
        at::Device device
    );
    // Queues on the current stream, shard's, the copy of the step's host inputs to its device (device_input_bytes) and
    // the gather of the others from table (gather_step_inputs), the tokens padded to padded_token_count. Returns the
    // inputs on the device and their zeroed page mass, undefined for a step that measures none; the second run's
    // tokens go into verify_tokens. mass_heads is the shard's part of the executor's.
    std::pair<model::batch, at::Tensor> upload_inputs(
        std::span<const kv> lines,
        const std::vector<std::size_t>& mass_heads,
        std::size_t shard,
        const request_table& table
    );
    // Queues on stream shard's writes of the step's table cells and stale entries into table (write_tables), then
    // the gather of the prior hidden vectors from draft_layer's lines, then a replayed verify step or decode step from
    // graphs, or else the step's inputs (upload_inputs), its forward, replayed from graphs' segment graphs when graphs
    // holds those of its bucket, and the copy of its page mass to the host, then
    // the sampling, the drafts, the advance of table's draw counts (advance_draw_counts) and last the advance of its
    // requests (advance_table); shard 0 also queues the copy of the sampled tokens to the host. mass_heads is the
    // shard's part of the executor's. graphs is the shard's, null when steps run without graphs. Every shard's call may
    // run at once, each on its own thread.
    void enqueue(
        model& shard_model,
        std::span<const kv> lines,
        const std::vector<std::size_t>& mass_heads,
        std::optional<std::size_t> draft_layer,
        const at::Tensor& stop_tokens,
        cudaStream_t stream,
        std::size_t shard,
        step_graphs* graphs,
        at::Tensor& counters,
        request_table& table
    );
    // Adds to the draft head's pass the rows of e's request req, at launch before it advances: those of an entry of the
    // second run, with verifies, its state_slots rows, another's from its first position without one through its
    // tokens and, with next_row, its next token, the request drafting draft_count tokens after them with drafts. The
    // entry's hidden rows start at first_hidden_row of the pass's hidden rows. An entry whose first draft row reads the
    // last_hidden of the position before its first token reads the prior hidden vector at prior_hidden_index.
    void plan_draft_rows(
        batch_entry& e,
        const request_state& req,
        bool verifies,
        std::size_t first_hidden_row,
        std::optional<std::size_t> prior_hidden_index,
        bool next_row,
        bool drafts
    );
    // Queues on stream shard, after the step's sampling into samples, the second-run entries' acceptance of their
    // samples, stop_tokens ending a run (int64), and the draft head's pass over draft_layer's view in table; shard 0
    // also queues the copies of the accepted counts and next tokens to the host.
    void enqueue_drafts(
        model& shard_model,
        std::span<const kv> lines,
        const at::Tensor& samples,
        const at::Tensor& stop_tokens,
        std::size_t shard,
        at::Tensor& counters,
        const request_table& table,
        std::size_t draft_layer
    );
    // Of the requests that draft, in entry order, after the draft head's first pass: each one's next token then its
    // first draft, the hidden rows its second draft reads, and the RoPE position and the draft head KV layer's key
    // count of its last computed token, after which its later drafts sit.
    struct first_draft {
        std::vector<at::Tensor> tokens;
        at::Tensor hidden;
        at::Tensor last_positions;
        at::Tensor last_key_counts;
    };
    // On device, shard's, pulsarlm::count_accepted_tokens of the second-run entries' samples, each entry accepting at
    // most its max_accepted_token_count; empty without such entries.
    at::Tensor count_accepted_tokens(
        const at::Tensor& samples,
        const at::Tensor& stop_tokens,
        std::size_t shard,
        at::Device device
    ) const;
    std::optional<first_draft> enqueue_first_draft_pass(
        model& shard_model,
        std::span<const kv> lines,
        const at::Tensor& samples,
        const at::Tensor& accepted_counts,
        std::size_t shard,
        at::Tensor& counters,
        const request_table& table,
        std::size_t draft_layer
    );
    void enqueue_later_draft_passes(
        model& shard_model,
        std::span<const kv> lines,
        const first_draft& first,
        std::size_t shard,
        at::Tensor& counters,
        const request_table& table
    );
    // Whether the step replays a step_graphs verify step: every entry belongs to the second run and drafts.
    bool verify_step_replays() const noexcept;
    // Whether the step is uniform, run unpadded and with graphs from a step_graphs verify step or, in an executor
    // whose steps do not draft (drafts false), decode step.
    bool is_uniform(bool drafts) const noexcept;
    // Whether the step replays a step_graphs decode step when it drafts nothing: every entry belongs to the first run
    // and samples.
    bool decode_step_replays() const noexcept;
    // Queues on device, shard's, the replay of the step's verify step from graphs, which gathers its inputs from the
    // shard's request table and advances it, then the copies of its page mass to the host and, on shard 0, of its
    // outputs. mass_heads is the shard's part of the executor's.
    void enqueue_verify_step(
        const std::vector<std::size_t>& mass_heads,
        std::size_t shard,
        at::Device device,
        step_graphs& graphs
    );
    // enqueue_verify_step for the step's decode step.
    void enqueue_decode_step(
        const std::vector<std::size_t>& mass_heads,
        std::size_t shard,
        at::Device device,
        step_graphs& graphs
    );
    // Queues on device, shard's, after everything of the step that reads table, the advance of each entry's request
    // in table (advance_requests) by its computed tokens, its next token from samples or the host, and the drafts of
    // the drafting requests; shard 0 also queues the copy of cycle_end_flags to the host.
    void advance_table(request_table& table, const at::Tensor& samples, std::size_t shard, at::Device device);
    // Joins the shards' page mass, each shard's heads after the previous shards', once their copies completed.
    void join_mass(std::span<const shard> shards, const view_layout& layout);
    // Per KV layer, the mass_sum the step gave each page of a view, the layer's mass_heads values per page, and their
    // weight_sum; empty for a layer that measures none.
    struct layer_step_mass {
        std::span<const float> mass_sum;
        std::span<const float> weight_sum;
    };
    // The page mass of the mass view view of the step from mass, laid out as join_mass leaves it.
    std::vector<layer_step_mass> view_mass(std::size_t view, const view_layout& layout) const;
    // Adds the page mass of the entry at index, which measures mass and whose request req accepted accepted_count
    // tokens from history position start_position, to req's layers. layout is the executor's.
    void add_measured_mass(
        std::size_t index,
        request_state& req,
        std::size_t start_position,
        std::size_t accepted_count,
        const view_layout& layout
    ) const;
    // Whether the entry at index reached its request's cycle end, as the GPU reported in cycle_end_flags.
    bool reaches_cycle_end(std::size_t index) const noexcept;

    friend class executor;
};

struct executor::batch_writes {
    // The pages of the requests reaching a state, written now with state_writers' states, and those a spawned task
    // writes.
    std::vector<std::vector<digest>> new_hashes;
    std::vector<std::vector<state::lease_id_type>> new_owners;
    std::vector<std::vector<std::uint32_t>> new_lines;
    std::vector<std::vector<digest>> spawned_hashes;
    std::vector<std::vector<state::lease_id_type>> spawned_owners;
    std::vector<std::vector<std::uint32_t>> spawned_lines;
    std::vector<std::shared_ptr<pending_writes>> spawned_writers;
    std::vector<request_state*> page_writers;
    std::vector<request_state*> state_reachers;
    std::vector<request_state*> state_writers;
};

}  // namespace pulsarlm
