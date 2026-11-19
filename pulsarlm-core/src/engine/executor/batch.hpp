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

// The tokens each running sequence forwards in one step, the model inputs for them, and the tokens sampled from the
// step's logits.
class executor::batch {
    // One sequence's part of the batch, as recorded at its launch.
    struct entry {
        sequence_ref seq;
        std::size_t token_count;
        bool is_sampling;
        bool ends_cycle;
        bool ends_sequence = false;
        bool is_measured = false;
        bool stores_head_state = false;
        std::size_t start_position = 0;
        std::size_t end_position = 0;
        std::size_t sampled_position = 0;
        std::size_t first_sample_row = 0;
        std::optional<std::size_t> prior_row;

        // The sequence's rows in the draft pass. logit_row is the row that drafts, before a verify entry's accepted
        // count - 1 is added, and key_count the sequence's keys after its rows in the draft head's KV layer.
        struct draft_rows {
            std::size_t row_count;
            std::size_t logit_row;
            std::size_t key_count;
            bool is_drafting;
        };
        std::optional<draft_rows> draft;

        // Of a verify entry, the most tokens it may accept (max_tokens), and its sequence's measured_start.
        std::size_t max_accepted_token_count = 0;
        std::size_t measured_start = 0;
        // Of a verify entry whose sequence's previous step is launched and not written back, its position among
        // that step's verify entries and among its drafting sequences: its tokens are that step's next tokens, and its
        // positions lie that step's accepted count - 1 after those its inputs hold.
        std::optional<std::size_t> previous_verify_row;
        std::optional<std::size_t> previous_drafting_row;
    };

    struct group {
        // Empty when the batch has no group of the kind.
        std::vector<entry> entries;
        // Pinned host tensors.
        model::batch inputs;
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

    // Each decode entry forwards its sequence's only pending token, a verify entry with its drafts after it.
    group decode_group{.inputs = {.kind = model::batch_kind::decode}};
    group verify_group{.inputs = {.kind = model::batch_kind::verify}};
    group prefill_group{.inputs = {.kind = model::batch_kind::prefill}};
    // The time at the start of the executor::launch that launched the batch.
    std::chrono::steady_clock::time_point launch_time;
    // The tokens a verify entry forwards and samples after: its pending token and its drafts.
    std::size_t verify_rows = 0;
    // Whether a decode entry's token is awaited, -(r + 1) in the decode group's tokens for row r of the previous
    // launched step's samples.
    bool awaits_tokens = false;
    // executor::entries_allow_overlap of the step's entries at its launch.
    bool allows_overlap = false;
    // Per row sampled, in entry order: a row of each entry that samples but a verify entry's verify_rows, the params
    // and draw.
    std::vector<sampling_params> sample_params;
    std::vector<sampling_draw> sample_draws;
    // Pinned host int64, one token per row sampled.
    at::Tensor sampled_tokens;
    // Per shard, completing once the step's work on its stream has, and an event recorded after that work.
    std::vector<external_task<cudaError_t>> completions;
    std::vector<at::cuda::CUDAEvent> ends;
    // Per entry whose draft rows start at its first token after an earlier step forwarded the position before it,
    // the line of the draft head's KV layer holding that position, and per shard, those lines' last_hidden as of the
    // step's launch; the entry's prior_row is its row.
    std::vector<std::int64_t> prior_lines;
    std::vector<at::Tensor> shard_prior_hidden;
    // The draft head's pass, queued after the step's sampling, over the entries with draft rows in entry order: per
    // row, its token, -(r + 1) for the step's sampled row r, its RoPE position, and its hidden row (draft_head::draft)
    // as an index into a zeros row, then the prior hidden rows, then the hidden rows of the decode, verify and prefill
    // groups. A verify entry's rows are the verify_rows positions after its pending token's, the row after its accepted
    // tokens drafting: its first row's index plus its accepted count - 1. Another sequence drafts from its last row.
    std::vector<std::int64_t> draft_tokens;
    std::vector<std::int64_t> draft_positions;
    std::vector<std::int64_t> draft_sources;
    // Per draft after the first, the RoPE position and key count of each sequence that drafts, in entry order, before a
    // verify entry's accepted count is added.
    std::vector<std::vector<std::int64_t>> later_positions;
    std::vector<std::vector<std::int32_t>> later_key_counts;
    // The step the verify entries' previous_verify_row and previous_drafting_row index.
    const batch* previous_batch = nullptr;
    // Per shard, the verify group's tokens on its device.
    std::vector<at::Tensor> verify_tokens;
    // Per shard, each verify entry's accepted tokens (int64 [verify entries]), and each drafting sequence's next token
    // then its drafts (int64 [drafting, verify_rows]); shard 0's copies to the host.
    std::vector<at::Tensor> accepted_token_counts_by_shard;
    std::vector<at::Tensor> shard_next_tokens;
    at::Tensor accepted_token_counts;
    at::Tensor next_tokens;

    // step_entries with their sequences found in owner, each recorded with whether it samples after its tokens and
    // whether they end its eviction cycle. With owner's graph_pages, a decode or verify group whose tables fit
    // graph_pages pages takes tables that wide and replays a graph.
    batch(executor& owner, std::span<const batch_tokens> step_entries);

    // The groups in entry order: decode, verify, prefill.
    std::array<group*, 3> groups() noexcept;
    std::array<const group*, 3> groups() const noexcept;
    // The entries of the sequences that draft, in entry order.
    std::vector<entry> drafting_entries() const;
    // A verify group measures each row's mass apart (model::batch::cu_view_pages), its queries weighing 1 from
    // measured_start on: the rows its write-back accepts are then weighed as one step's.
    static model::batch
    make_group(std::span<const entry> group, std::size_t page_size, std::size_t state_slots, model::batch_kind kind);
    // The tokens e forwards: its pending tokens', and its drafts.
    static std::size_t forwarded(const entry& e) noexcept;
    // The widest table among the group's sequences.
    static std::size_t table_width(std::span<const entry> group) noexcept;
    static bool measures_mass(std::span<const entry> group) noexcept;
    // Whether e forwards a query that adds mass.
    static bool measures_mass(const entry& e) noexcept;
    // The group's tables of lines, or of positions with positions, on shard's device, as model::batch lays them out.
    static at::Tensor page_tables(
        std::span<const entry> group,
        std::size_t n_kv_layers,
        bool positions,
        std::size_t shard,
        std::size_t width
    );
    static void set_page_tables(std::span<const entry> group, model::batch& g, std::size_t n_layers, std::size_t shard);
    // Queues on the current stream, shard's, the copies of g's inputs to its device, with its tables, the filling of a
    // decode group's awaited tokens from samples and a verify group's tokens of the previous step. Returns the inputs
    // on the device and their zeroed page mass, undefined for a group that measures none. mass_heads is the shard's
    // part of the executor's.
    std::pair<model::batch, at::Tensor> upload_inputs(
        const group& g,
        std::span<const kv> lines,
        const std::vector<std::size_t>& mass_heads,
        std::size_t shard,
        const at::Tensor& samples
    );
    // Queues on stream shard's copy of the prior hidden rows from draft_layer's lines, then per group its inputs
    // (upload_inputs), its forward and the copy of its page mass to the host, then the sampling into samples; shard 0
    // also queues the copy of the sampled tokens to the host. mass_heads is the shard's part of the executor's. graphs
    // is the shard's when a group is replayed. Every shard's call may run at once, each on its own thread.
    void enqueue(
        model& shard_model,
        std::span<const kv> lines,
        const std::vector<std::size_t>& mass_heads,
        std::optional<std::size_t> draft_layer,
        const at::Tensor& stop_tokens,
        cudaStream_t stream,
        std::size_t shard,
        decode_graphs* graphs,
        const at::Tensor& samples,
        at::Tensor& counters
    );
    // Adds to the draft head's pass the rows of e's sequence, at launch before it advances: those of an entry of the
    // verify group, with verifies, its verify_rows rows, another's from its first position without one through its
    // tokens and, with next_row, its next token, the sequence drafting draft_count tokens after them with drafts. The
    // entry's hidden rows start at first_hidden_row of the pass's hidden rows.
    void plan_draft_rows(
        entry& e,
        bool verifies,
        std::size_t first_hidden_row,
        bool next_row,
        bool drafts,
        std::size_t draft_layer,
        std::size_t draft_count
    );
    // Queues on stream shard, after the step's sampling into samples, the verify entries' acceptance of their samples,
    // stop_tokens ending a run (int64), and the draft head's pass; shard 0 also queues the copies of the accepted
    // counts and next tokens to the host.
    void enqueue_drafts(
        model& shard_model,
        std::span<const kv> lines,
        const at::Tensor& samples,
        const at::Tensor& stop_tokens,
        std::size_t shard,
        at::Tensor& counters
    );
    // Of the sequences that draft, in entry order, after the draft head's first pass: each one's next token then its
    // first draft, the hidden rows its second draft reads, and the shift of its later positions and key counts.
    struct first_draft {
        std::vector<at::Tensor> tokens;
        at::Tensor hidden;
        at::Tensor later_shift;
    };
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
        const at::Tensor& shift,
        std::size_t shard,
        at::Tensor& counters
    );
    void enqueue_later_draft_passes(
        model& shard_model,
        std::span<const kv> lines,
        const first_draft& first,
        std::size_t shard,
        at::Tensor& counters
    );
    // Whether the step replays a decode_graphs verify step: every entry verifies and drafts, and the verify group
    // replays.
    bool steps_replay() const noexcept;
    // Queues on device, shard's, the replay of its verify step from graphs, on_device being the verify group and
    // verify_mass its page mass, then the copies of its outputs: into samples, and on shard 0 to the host.
    void enqueue_step(
        const model::batch& on_device,
        const at::Tensor& verify_mass,
        std::size_t shard,
        at::Device device,
        decode_graphs* graphs,
        const at::Tensor& samples
    );
    // Takes into g, the verify group on shard's device, the previous step's next tokens of the entries whose previous
    // step is launched, and moves their positions by its accepted count - 1.
    void take_previous_tokens(model::batch& g, std::size_t shard, at::Device device) const;
    // On device, shard's, per row sampled, the tokens the previous step of a verify entry's sequence accepts, by which its
    // draw counts lag; 0 for the other rows.
    at::Tensor previous_draw_shifts(std::size_t shard, at::Device device) const;
    // On device, shard's, each verify entry's accepted count - 1 of its previous step, 0 without one, int64 [verify
    // entries].
    at::Tensor previous_shift(std::size_t shard, at::Device device) const;
    // Joins the shards' page mass, each shard's heads after the previous shards', once their copies completed.
    void join_mass(const std::vector<std::vector<std::size_t>>& shard_mass_heads, const view_layout& layout);
    // Per KV layer, the mass_sum the step gave each page of a view, the layer's mass_heads values per page, and their
    // weight_sum; empty for a layer that measures none.
    struct layer_step_mass {
        std::span<const float> mass_sum;
        std::span<const float> weight_sum;
    };
    // The page mass of the view view of group from mass, laid out as join_mass leaves it.
    static std::vector<layer_step_mass>
    view_mass(const model::batch& group, const at::Tensor& mass, std::size_t view, const view_layout& layout);
    // Adds the page mass of the entry at position of g, which measures mass and whose sequence accepted accepted_count
    // tokens, to its sequence's layers. layout is the executor's.
    void add_measured_mass(
        const group& g,
        std::size_t position,
        std::size_t accepted_count,
        const view_layout& layout
    ) const;

    friend class executor;
};

struct executor::batch_writes {
    // The pages of the sequences reaching a state, written now with state_writers' states, and those a spawned task
    // writes.
    std::vector<std::vector<digest>> new_hashes;
    std::vector<std::vector<state::lease_id_type>> new_owners;
    std::vector<std::vector<std::uint32_t>> new_lines;
    std::vector<std::vector<digest>> spawned_hashes;
    std::vector<std::vector<state::lease_id_type>> spawned_owners;
    std::vector<std::vector<std::uint32_t>> spawned_lines;
    std::vector<std::shared_ptr<pending_writes>> spawned_writers;
    std::vector<sequence*> page_writers;
    std::vector<sequence*> state_reachers;
    std::vector<sequence*> state_writers;
};

}  // namespace pulsarlm
