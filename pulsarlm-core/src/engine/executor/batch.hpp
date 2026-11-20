#pragma once

#include "../pinned_tensor.hpp"

#include "pulsarlm/engine/executor.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>

#include <array>
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

// Sets on_device's page mass_sum and weight_sum for the KV layers that measure mass, those with mass_heads above 0, to
// zeroed parts of one tensor, which it returns: per such layer in turn, its mass_sum, then its weight_sum.
// host_cu_view_pages is on the host.
at::Tensor zero_page_mass(
    model::batch& on_device,
    const at::Tensor& host_cu_view_pages,
    const std::vector<std::size_t>& mass_heads
);

// The tokens each running request forwards in one step, the model inputs for them, and the tokens sampled from the
// step's logits.
class executor::batch {
    // One request's part of the batch, as recorded at its launch.
    struct entry {
        request_ref req;
        std::size_t token_count;
        bool is_sampling;
        bool ends_cycle;
        bool ends_request = false;
        bool is_measured = false;
        bool stores_head_state = false;
        std::size_t start_position = 0;
        std::size_t end_position = 0;
        std::size_t sampled_position = 0;
        std::size_t first_sample_row = 0;
        // Of an entry whose first draft row reads the last_hidden of the position before its first token, the index
        // of that hidden vector among the step's prior_hidden_slots.
        std::optional<std::size_t> prior_hidden_index;

        // The request's rows in the draft pass. logit_row is the row that drafts, before a verify entry's accepted
        // count - 1 is added, and key_count_offset the request's keys after its rows in the draft head's KV layer
        // less its keys there at the step's start.
        struct draft_rows {
            std::size_t row_count;
            std::size_t logit_row;
            std::size_t key_count_offset;
            bool is_drafting;
        };
        std::optional<draft_rows> draft;

        // Of a verify entry, the most tokens it may accept (max_tokens).
        std::size_t max_accepted_token_count = 0;
    };

    struct group {
        // Empty when the batch has no group of the kind.
        std::vector<entry> entries;
        // Pinned host tensors: a prefill group's tokens, cu_seqlens_q unless the group decodes, and cu_view_pages when
        // it measures mass. The other inputs are gathered on each shard from its request_table.
        model::batch inputs;
        // Pinned host int64, the slot of each entry's request (executor::slot_of).
        at::Tensor slots;
        // The page-table width, and whether the group's forward replays a graph.
        std::size_t table_width = 0;
        bool is_replayed = false;
        // Pinned host fp32, defined for a group that measures mass: the page mass of each KV layer that measures it in
        // turn, laid out as the layer's row of the group's cu_view_pages, its heads innermost; per shard, its heads,
        // then joined over the shards.
        std::vector<at::Tensor> mass_by_shard;
        at::Tensor mass;
        // Per shard, the group's hidden rows, which the draft head's pass reads.
        std::vector<at::Tensor> hidden_by_shard;
    };

    // Each decode entry forwards its request's only pending token, a verify entry with its drafts after it.
    group decode_group{.inputs = {.kind = model::batch_kind::decode}};
    group verify_group{.inputs = {.kind = model::batch_kind::verify}};
    group prefill_group{.inputs = {.kind = model::batch_kind::prefill}};
    // The time at the start of the executor::launch that launched the batch.
    std::chrono::steady_clock::time_point launch_time;
    // The tokens a verify entry forwards and samples after: its pending token and its drafts.
    std::size_t verify_rows = 0;
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
    // Pinned host int64, per row sampled in entry order (a row of each entry that samples, but a verify entry's
    // verify_rows): the slot of its request, and its offset among its request's rows, which it adds to the request's
    // draw count.
    at::Tensor sample_slots;
    at::Tensor sample_offsets;
    // Whether an entry that samples filters (pulsarlm::filters).
    bool filters = false;
    // Pinned host int64, the slots of the entries that sample outside the verify group, each taking one draw.
    at::Tensor single_draw_slots;
    // Pinned host int64, one token per row sampled.
    at::Tensor sampled_tokens;
    // Pinned host tensors in entry order: the slot of each entry's request (int64), the tokens it computes, 0 for a
    // verify entry, whose accepted count is added on the GPU (int64), and its next token, -(r + 1) for row r of the
    // step's samples, a verify entry's row before its accepted count - 1 is added (int64). With cycle_end_flags, which
    // shard 0 copies from the GPU after the step, whether the entry's request reached its cycle end (bool).
    at::Tensor entry_slots;
    at::Tensor computed_token_counts;
    at::Tensor next_token_sources;
    at::Tensor cycle_end_flags;
    // Per shard, completing once the step's work on its stream has, and an event recorded after that work.
    std::vector<external_task<cudaError_t>> completions;
    std::vector<at::cuda::CUDAEvent> ends;
    // The slot of each entry's request whose first draft row reads the last_hidden of the position before its first
    // token, at its prior_hidden_index, and per shard, those hidden vectors gathered at the start of the step from the
    // draft head's KV layer.
    std::vector<std::int64_t> prior_hidden_slots;
    std::vector<at::Tensor> shard_prior_hidden;
    // The draft head's pass, queued after the step's sampling, over the entries with draft rows in entry order: per
    // row, its token, -(r + 1) for the step's sampled row r, its RoPE position less its request's at the step's start,
    // and its hidden row (draft_head::draft) as an index into a zeros row, then the prior hidden rows, then the hidden
    // rows of the decode, verify and prefill groups. A verify entry's rows are the verify_rows positions after its
    // pending token's, the row after its accepted tokens drafting: its first row's index plus its accepted count - 1.
    // Another request drafts from its last row.
    std::vector<std::int64_t> draft_tokens;
    std::vector<std::int64_t> draft_position_offsets;
    std::vector<std::int64_t> draft_sources;
    // The tokens each drafting request drafts. Draft j after the first sits j positions after the request's last
    // computed token.
    std::size_t draft_count = 0;
    // Per shard, the verify group's tokens on its device.
    std::vector<at::Tensor> verify_tokens;
    // Per shard, each verify entry's accepted tokens (int64 [verify entries]), and each drafting request's next token
    // then its drafts (int64 [drafting, verify_rows]); shard 0's copies to the host.
    std::vector<at::Tensor> accepted_token_counts_by_shard;
    std::vector<at::Tensor> shard_next_tokens;
    at::Tensor accepted_token_counts;
    at::Tensor next_tokens;

    // scheduled with their requests found in owner, each recorded with whether it samples after its tokens and
    // whether they end its eviction cycle. With owner's graph_pages, a decode or verify group takes tables
    // graph_pages wide and replays a graph.
    batch(executor& owner, std::span<const scheduled_request> scheduled);

    // The groups in entry order: decode, verify, prefill.
    std::array<group*, 3> groups() noexcept;
    std::array<const group*, 3> groups() const noexcept;
    // The entries of the requests that draft, in entry order.
    std::vector<entry> drafting_entries() const;
    // The host inputs of group (group::inputs). A verify group measures each row's mass apart
    // (model::batch::cu_view_pages), its queries weighing 1 from mass_start_position on: the rows its write-back
    // accepts are then weighed as one step's.
    static model::batch
    make_group(std::span<const entry> group, std::size_t page_size, std::size_t state_slots, model::batch_kind kind);
    // The tokens e forwards: its pending tokens', and its drafts.
    static std::size_t forwarded(const entry& e) noexcept;
    // The widest table among the group's requests.
    static std::size_t table_width(std::span<const entry> group) noexcept;
    static bool measures_mass(std::span<const entry> group) noexcept;
    // Whether e forwards a query that adds mass.
    static bool measures_mass(const entry& e) noexcept;
    // Pinned host int64, the slot of each entry's request.
    static at::Tensor request_slots(std::span<const entry> group);
    // int32 [n_kv_layers, slots, width] on tables' device: the first width columns of the rows of tables, a shard's
    // page_tables or page_positions, at slots, which are on that device.
    static at::Tensor select_tables(const at::Tensor& slots, const at::Tensor& tables, std::size_t width);
    // Sets g's page tables and positions to group's rows of table's page_tables and page_positions, as wide as the
    // widest table among its requests.
    static void set_page_tables(std::span<const entry> group, model::batch& g, const request_table& table);
    // Queues on the current stream the copies of the step's table cells and stale entries to the device of table, a
    // shard's, and their writes.
    void write_tables(request_table& table) const;
    // On device, table's, the inputs of the step's sampling: per row sampled, its request's columns of table, its
    // draw count moved by the row's offset.
    sampling_inputs gather_sampling_inputs(const request_table& table, at::Device device) const;
    // Queues on the current stream the addition to table's draw_counts of the draws the step took: one per entry that
    // samples outside the verify group, and a verify entry's accepted count.
    void advance_draw_counts(request_table& table, std::size_t shard, at::Device device) const;
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
    // Queues on the current stream, shard's, the copies of g's host inputs to its device and the gather of the others
    // from table (gather_group_inputs). Returns the inputs on the device and their zeroed page mass, undefined for a
    // group that measures none. mass_heads is the shard's part of the executor's.
    std::pair<model::batch, at::Tensor> upload_inputs(
        const group& g,
        std::span<const kv> lines,
        const std::vector<std::size_t>& mass_heads,
        std::size_t shard,
        const request_table& table
    );
    // Queues on stream shard's writes of the step's table cells and stale entries into table (write_tables), then
    // the gather of the prior hidden vectors from draft_layer's lines, then per group its inputs (upload_inputs), its
    // forward and the copy of its page mass to the host, then the sampling, the drafts, the advance of table's draw
    // counts (advance_draw_counts) and last the advance of its requests (advance_table); shard 0 also queues the copy
    // of the sampled tokens to the host. mass_heads is the shard's part of the executor's. graphs is the shard's when a
    // group is replayed. Every shard's call may run at once, each on its own thread.
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
    // Adds to the draft head's pass the rows of e's request, at launch before it advances: those of an entry of the
    // verify group, with verifies, its verify_rows rows, another's from its first position without one through its
    // tokens and, with next_row, its next token, the request drafting draft_count tokens after them with drafts. The
    // entry's hidden rows start at first_hidden_row of the pass's hidden rows.
    void plan_draft_rows(entry& e, bool verifies, std::size_t first_hidden_row, bool next_row, bool drafts);
    // Queues on stream shard, after the step's sampling into samples, the verify entries' acceptance of their samples,
    // stop_tokens ending a run (int64), and the draft head's pass over draft_layer's view in table; shard 0 also
    // queues the copies of the accepted counts and next tokens to the host.
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
    // On device, shard's, pulsarlm::count_accepted_tokens of the verify entries' rows of samples, each entry accepting at
    // most its max_accepted_token_count; empty without verify entries.
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
    // Whether the step replays a step_graphs verify step: every entry verifies and drafts, and the verify group
    // replays.
    bool steps_replay() const noexcept;
    // Queues on device, shard's, the replay of its verify step from graphs, on_device being the verify group and
    // verify_mass its page mass, sampled by inputs gathered from table and drafting at the positions of table's
    // entries in draft_layer, then the copies of its outputs to the host on shard 0. Returns the step's samples.
    at::Tensor enqueue_step(
        const model::batch& on_device,
        const at::Tensor& verify_mass,
        std::size_t shard,
        at::Device device,
        step_graphs* graphs,
        const request_table& table,
        std::size_t draft_layer
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
    // The page mass of the view view of group from mass, laid out as join_mass leaves it.
    static std::vector<layer_step_mass>
    view_mass(const model::batch& group, const at::Tensor& mass, std::size_t view, const view_layout& layout);
    // Adds the page mass of the entry at position of g, which measures mass and whose request accepted accepted_count
    // tokens, to its request's layers. layout is the executor's.
    void add_measured_mass(
        const group& g,
        std::size_t position,
        std::size_t accepted_count,
        const view_layout& layout
    ) const;

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
