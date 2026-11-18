#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAGraph.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "pulsarlm/engine/model/interface.hpp"
#include "pulsarlm/engine/sampling.hpp"

namespace pulsarlm {

// Replays one GPU's decode forward from CUDA graphs over page tables of one fixed width, a graph per number of decoding
// sequences, whether the step measures mass and whether the group is a verify group of state_slots tokens per sequence
// or a decode group. A verify group's forward reads nothing back to the host either. The model's decode forward reads nothing back to the host and launches
// the same kernels for every group of one such shape. Each graph first fills the group's awaited tokens from samples
// (fill_awaited_tokens_cuda).
class decode_graphs {
    struct captured_forward {
        at::cuda::CUDAGraph graph;
        group_batch inputs;
        at::Tensor logits;
        // Undefined for a model without a draft head.
        at::Tensor hidden;
    };
    // The sequences of a group, whether it measures mass, and whether it is a verify group.
    using shape = std::tuple<std::int64_t, bool, bool>;
    // A draft pass of the model's draft head over a group of state_slots rows per sequence, the first pass after a
    // verify group, or of one row per sequence.
    struct captured_draft {
        at::cuda::CUDAGraph graph;
        group_batch inputs;
        at::Tensor hidden_rows;
        draft_output output;
    };
    // The sequences of a draft pass, and whether each has state_slots rows.
    using draft_shape = std::pair<std::int64_t, bool>;
    // A verify step: the forward of a verify group, its sampling, its acceptance of the drafts and the draft head's
    // passes after it. Its inputs are the verify group's, each row's sampling inputs, each sequence's most tokens kept,
    // and the draft passes' RoPE positions and key counts before the kept counts are added: per row of the first pass
    // [rows] and per sequence [sequences], then per later draft [drafts - 1, sequences] each.
    struct captured_step {
        at::cuda::CUDAGraph graph;
        group_batch inputs;
        sampling_inputs sampling;
        at::Tensor max_kept;
        at::Tensor draft_positions;
        at::Tensor draft_key_counts;
        at::Tensor later_positions;
        at::Tensor later_key_counts;
        group_batch first_pass;
        group_batch later_pass;
        // Each row's sample, each sequence's kept count and its next token then its drafts.
        at::Tensor samples;
        at::Tensor kept;
        at::Tensor next_tokens;
    };
    // The sequences of a verify step, whether it measures mass, and whether its sampling filters.
    using step_shape = std::tuple<std::int64_t, bool, bool>;

    model& shard_model;
    std::vector<kv> lines;
    at::Tensor samples;
    std::int64_t width;
    std::int64_t page_size;
    std::int64_t state_slots;
    // Per KV layer, the query heads whose mass this GPU measures, 0 for a layer that measures none.
    std::vector<std::size_t> mass_heads;
    // Per KV layer, mass_sum then weight_sum rows for startup_sequences sequences' views, shared by the graphs of up to
    // that many sequences; graphs of more sequences hold their own.
    std::vector<at::Tensor> shared_mass_sum;
    std::vector<at::Tensor> shared_weight_sum;
    std::map<shape, std::unique_ptr<captured_forward>> forwards;
    std::map<draft_shape, std::unique_ptr<captured_draft>> drafts;
    std::map<step_shape, std::unique_ptr<captured_step>> steps;
    // int64, the tokens that end a run of kept drafts.
    at::Tensor stop_tokens;

    // Inputs of a group of sequences on the lines' device, the mass rows of up to startup_sequences sequences in the
    // shared buffers, with the group's own arrival counters.
    group_batch make_inputs(std::int64_t sequences, bool measures_mass, bool verifies) const;
    std::unique_ptr<captured_forward> capture(std::int64_t sequences, bool measures_mass, bool verifies);
    // The captured forward of g's shape, captured first if it is not yet, with g's inputs copied into it and replayed.
    const captured_forward& replay(const group_batch& g, bool verifies);
    // Inputs of a draft pass of sequences, with state_slots rows each or one, and its hidden rows.
    std::pair<group_batch, at::Tensor> make_draft_inputs(std::int64_t sequences, bool verify_rows) const;
    std::unique_ptr<captured_draft> capture_draft(std::int64_t sequences, bool verify_rows);
    std::unique_ptr<captured_step> capture_step(std::int64_t sequences, bool measures_mass, bool filters);

  public:
    // The decoding sequences whose graphs capture_startup records.
    static constexpr std::int64_t startup_sequences = 8;

    // Graphs over page tables width pages wide, of pages page_size tokens long, a verify step's run of kept drafts
    // ending at stop_tokens (int64 on the lines' device), with state_slots slots
    // (group_batch::state_slots). mass_heads holds, per entry of the model's kv_layers(), the query heads
    // whose mass this GPU measures. lines is on the current device, and so is samples, int64, which holds the previous
    // step's sampled tokens at each replay.
    decode_graphs(
        model& shard_model,
        std::span<const kv> lines,
        std::size_t width,
        std::size_t page_size,
        std::size_t state_slots,
        std::vector<std::size_t> mass_heads,
        at::Tensor samples,
        at::Tensor stop_tokens
    );

    std::size_t page_width() const noexcept;
    // Captures the graphs of 1 to startup_sequences sequences, with mass when a layer measures it and without, decode
    // groups and with state_slots above 1 verify groups, on the current stream, which is not the default stream, after
    // an eager forward of each that zeroes line 0 of each layer and writes into it. Every GPU of the model runs it at
    // once, before any sequence holds a line.
    void capture_startup();
    // model::forward's decode logits and hidden rows of g, its awaited tokens filled from samples, on the current
    // stream, valid until the next call, with g's page mass written as model::forward writes it. g's page tables are
    // page_width() wide; a shape not captured yet is captured first.
    std::pair<at::Tensor, at::Tensor> decode(const group_batch& g);
    // decode's verify logits and hidden rows of g, a verify group of state_slots tokens per sequence.
    std::pair<at::Tensor, at::Tensor> verify(const group_batch& g);
    // draft_head::draft of g and hidden, g holding state_slots rows per sequence with cu_seqlens_q and a logit row per
    // sequence, or with verify_rows false one row per sequence; valid until the next call. The model has a draft head,
    // and g's page tables are page_width() wide.
    draft_output draft(const group_batch& g, const at::Tensor& hidden, bool verify_rows);
    // The verify step of g, a verify group of state_slots tokens per sequence whose sequences all draft: its rows'
    // samples by sampling, each sequence's kept count, at most max_kept (int64 [sequences]), and its next token then
    // its drafts [sequences, state_slots], valid until the next call, with g's page mass written as model::forward
    // writes it. A sequence keeps its pending token and its drafts while sample t equals draft t + 1, then the sample
    // after them, through a stop token. The draft passes' RoPE positions and key counts are draft_positions and
    // draft_key_counts, later_positions and later_key_counts (captured_step) before the kept counts are added.
    std::tuple<at::Tensor, at::Tensor, at::Tensor> verify_step(
        const group_batch& g,
        const sampling_inputs& sampling,
        const at::Tensor& max_kept,
        const at::Tensor& draft_positions,
        const at::Tensor& draft_key_counts,
        const at::Tensor& later_positions,
        const at::Tensor& later_key_counts
    );
};

}  // namespace pulsarlm
