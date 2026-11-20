#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAGraph.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "pulsarlm/engine/model/interface.hpp"
#include "pulsarlm/engine/sampling.hpp"

namespace pulsarlm {

// Replays one GPU's decode groups, verify groups and verify steps from CUDA graphs over page tables of one fixed width,
// a graph per number of sequences and per whether the step measures mass. A verify group holds state_slots tokens per
// sequence. A verify step is the forward of a verify group, its sampling, its acceptance of the drafts and the draft
// head's passes after it. The model's decode and verify forwards read nothing back to the host and launch the same
// kernels for every group of one such shape.
class step_graphs {
    struct captured_forward {
        at::cuda::CUDAGraph graph;
        model::batch inputs;
        at::Tensor logits;
        // Undefined for a model without a draft head.
        at::Tensor hidden;
    };
    // A verify step. Its inputs are the verify group's, each row's sampling inputs, each sequence's most tokens
    // accepted, and the draft passes' RoPE positions and key counts before the accepted counts are added: per row of
    // the first pass [rows] and per sequence [sequences], then per later draft [drafts - 1, sequences] each.
    struct captured_step {
        at::cuda::CUDAGraph graph;
        model::batch inputs;
        sampling_inputs sampling;
        at::Tensor max_accepted_token_counts;
        at::Tensor draft_positions;
        at::Tensor draft_key_counts;
        at::Tensor later_positions;
        at::Tensor later_key_counts;
        model::batch first_pass;
        model::batch later_pass;
        // Each row's sample, each sequence's accepted count and its next token then its drafts.
        at::Tensor samples;
        at::Tensor accepted_token_counts;
        at::Tensor next_tokens;
    };
    using sequence_count_type = std::int64_t;
    using measures_mass_type = bool;
    using group_shape = std::pair<sequence_count_type, measures_mass_type>;

    model& shard_model;
    std::vector<kv> lines;
    std::int64_t width;
    std::int64_t page_size;
    std::int64_t state_slots;
    // Per KV layer, the query heads whose mass this GPU measures, 0 for a layer that measures none.
    std::vector<std::size_t> mass_heads;
    // Per KV layer, mass_sum then weight_sum rows for startup_sequences sequences' views, shared by the decode graphs
    // of up to that many sequences; the other graphs hold their own.
    std::vector<at::Tensor> shared_mass_sum;
    std::vector<at::Tensor> shared_weight_sum;
    std::map<group_shape, std::unique_ptr<captured_forward>> decode_forwards;
    std::map<group_shape, std::unique_ptr<captured_forward>> verify_forwards;
    // Verify steps whose sampling takes the arg max or the Gumbel race over all tokens.
    std::map<group_shape, std::unique_ptr<captured_step>> unfiltered_steps;
    // Verify steps whose sampling filters by top_k or top_p first.
    std::map<group_shape, std::unique_ptr<captured_step>> filtered_steps;
    // int64, the tokens that end a run of accepted drafts.
    at::Tensor stop_tokens;

    static group_shape shape_of(const model::batch& g);
    // Inputs of a group of sequences of kind, decode or verify, on the lines' device, with the group's own arrival
    // counters. The mass rows of a decode group of up to startup_sequences sequences are in the shared buffers.
    model::batch make_group_inputs(std::int64_t sequences, bool measures_mass, model::batch_kind kind) const;
    // Inputs of a draft pass of sequences with rows_per_sequence rows each.
    model::batch
    make_draft_pass_inputs(std::int64_t sequences, std::int64_t rows_per_sequence, model::batch_kind kind) const;
    // Inputs of the draft pass after a verify group, state_slots rows per sequence and a logit row per sequence.
    model::batch make_first_pass_inputs(std::int64_t sequences) const;
    // Inputs of a draft pass of one row per sequence.
    model::batch make_later_pass_inputs(std::int64_t sequences) const;
    // Zeros in the dtype of the draft head's KV layer, [row_count, hidden_dim].
    at::Tensor make_draft_hidden_rows(std::int64_t row_count) const;
    // Sampling inputs of row_count rows that all take the arg max.
    sampling_inputs make_sampling_inputs(std::int64_t row_count) const;
    // Records into the capture in progress the forward of captured's inputs.
    void record_forward(captured_forward& captured);
    std::unique_ptr<captured_forward> capture_decode(std::int64_t sequences, bool measures_mass);
    std::unique_ptr<captured_forward> capture_verify(std::int64_t sequences, bool measures_mass);
    // A verify step that samples by sampling, whose filters is the captured graph's.
    std::unique_ptr<captured_step> capture_step(std::int64_t sequences, bool measures_mass, sampling_inputs&& sampling);
    std::unique_ptr<captured_step> capture_unfiltered_step(std::int64_t sequences, bool measures_mass);
    std::unique_ptr<captured_step> capture_filtered_step(std::int64_t sequences, bool measures_mass);
    // Replays captured, of g's shape, with g's inputs.
    model::output replay(captured_forward& captured, const model::batch& g);

  public:
    // The decoding sequences whose graphs capture_startup records.
    static constexpr std::int64_t startup_sequences = 8;

    struct verify_step_outputs {
        // Each row's sample, int64 [sequences * state_slots].
        at::Tensor samples;
        // Each sequence's accepted token count, int64 [sequences].
        at::Tensor accepted_token_counts;
        // Each sequence's next token then its drafts, int64 [sequences, state_slots].
        at::Tensor next_tokens;
    };

    // Graphs over page tables width pages wide, of pages page_size tokens long, a verify step's run of accepted drafts
    // ending at stop_tokens (int64 on the lines' device), with state_slots slots
    // (model::batch::state_slots). mass_heads holds, per entry of the model's kv_layers(), the query heads
    // whose mass this GPU measures. lines is on the current device.
    step_graphs(
        model& shard_model,
        std::span<const kv> lines,
        std::size_t width,
        std::size_t page_size,
        std::size_t state_slots,
        std::vector<std::size_t> mass_heads,
        at::Tensor stop_tokens
    );

    std::size_t page_width() const noexcept;
    // Captures the graphs of 1 to startup_sequences sequences, with mass when a layer measures it and without: decode
    // groups, with state_slots above 1 verify groups, and with a draft head too the verify steps that do not filter.
    // It runs on the current stream, which is not the default stream, after an eager forward of each kind that zeroes
    // line 0 of each layer and writes into it. Every GPU of the model runs it at once, before any request holds a
    // line.
    void capture_startup();
    // model::forward's decode logits and hidden rows of g on the current stream, valid until the next call, with g's
    // page mass written as model::forward writes it. g's page tables are page_width() wide; a shape not captured yet is
    // captured first.
    model::output decode(const model::batch& g);
    // decode's verify logits and hidden rows of g, a verify group of state_slots tokens per sequence.
    model::output verify(const model::batch& g);
    // The verify step of g, a verify group of state_slots tokens per sequence whose sequences all draft, sampled by
    // sampling, each sequence accepting at most max_accepted_token_counts (int64 [sequences]) tokens. The outputs are
    // valid until the next call, and g's page mass is written as model::forward writes it. A sequence accepts its
    // pending token and its drafts while sample t equals draft t + 1, then the sample after them, through a stop
    // token. The draft passes' RoPE positions and key counts are draft_positions and draft_key_counts, later_positions
    // and later_key_counts (captured_step) before the accepted counts are added. The model has a draft head.
    verify_step_outputs verify_step(
        const model::batch& g,
        const sampling_inputs& sampling,
        const at::Tensor& max_accepted_token_counts,
        const at::Tensor& draft_positions,
        const at::Tensor& draft_key_counts,
        const at::Tensor& later_positions,
        const at::Tensor& later_key_counts
    );
};

}  // namespace pulsarlm
