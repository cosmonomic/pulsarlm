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
#include "pulsarlm/engine/request_table.hpp"
#include "pulsarlm/engine/sampling.hpp"

namespace pulsarlm {

// Replays one GPU's decode steps and verify steps from CUDA graphs over page tables of one fixed width, a graph per
// number of requests and per whether the step measures mass. Each graph first gathers its step's inputs from the
// request table at the slots of the step's requests (gather_step_inputs), at the addresses the table's tensors had at
// construction. A decode step is the forward of requests with one query token each that all sample, its sampling, and
// the advance of its requests' draw counts and entries in the table (advance_requests). A verify step is the forward of
// requests with state_slots query tokens each that keep a recurrent state and a mass view per query token, its
// sampling, its acceptance of the drafts, the draft head's passes after it, and the advance of its requests' draw
// counts and entries. The model's forwards of such steps read nothing back to the host and launch the same kernels for
// every step of one shape.
class step_graphs {
    struct captured_decode_step {
        at::cuda::CUDAGraph graph;
        // int64 [sequences], the slots of the step's requests, which each replay sets.
        at::Tensor request_slots;
        model::batch inputs;
        at::Tensor samples;
        at::Tensor ends_cycle;
    };
    // The inputs of a verify step are its requests' slots and each request's most tokens accepted.
    struct captured_verify_step {
        at::cuda::CUDAGraph graph;
        at::Tensor request_slots;
        at::Tensor max_accepted_token_counts;
        model::batch inputs;
        model::batch first_pass;
        model::batch later_pass;
        at::Tensor samples;
        at::Tensor accepted_token_counts;
        at::Tensor next_tokens;
        at::Tensor ends_cycle;
    };
    using sequence_count_type = std::int64_t;
    using measures_mass_type = bool;
    using step_shape = std::pair<sequence_count_type, measures_mass_type>;

    model& shard_model;
    std::vector<kv> lines;
    std::int64_t width;
    std::int64_t page_size;
    std::int64_t state_slots;
    // Per KV layer, the query heads whose mass this GPU measures, 0 for a layer that measures none.
    std::vector<std::size_t> mass_heads;
    // Per KV layer, mass_sum then weight_sum rows for startup_sequences sequences' views, shared by the decode steps
    // of up to that many sequences; the other graphs hold their own.
    std::vector<at::Tensor> shared_mass_sum;
    std::vector<at::Tensor> shared_weight_sum;
    // Steps whose sampling takes the arg max or the Gumbel race over all tokens.
    std::map<step_shape, std::unique_ptr<captured_decode_step>> unfiltered_decode_steps;
    std::map<step_shape, std::unique_ptr<captured_verify_step>> unfiltered_verify_steps;
    // Steps whose sampling filters by top_k or top_p first.
    std::map<step_shape, std::unique_ptr<captured_decode_step>> filtered_decode_steps;
    std::map<step_shape, std::unique_ptr<captured_verify_step>> filtered_verify_steps;
    // int64, the tokens that end a run of accepted drafts.
    at::Tensor stop_tokens;
    request_table table;
    // int64 [0], the tokens of a step without prefill chunks.
    at::Tensor no_prefill_tokens;

    // The index among the lines of the draft head's KV layer, the one that keeps last hidden vectors.
    std::size_t find_draft_layer() const;
    // int64 zeros [sequences] on the lines' device.
    at::Tensor make_request_slots(std::int64_t sequences) const;
    // The inputs that gather_step_inputs does not fill of a step of sequences, each with one query token or, with
    // verifies, state_slots query tokens that keep a recurrent state and a mass view each, on the lines' device, with
    // the step's own arrival counters. The mass rows of a decode step of up to startup_sequences sequences are in the
    // shared buffers.
    model::batch make_step_inputs(std::int64_t sequences, bool measures_mass, bool verifies) const;
    // Inputs of a draft pass of sequences with rows_per_sequence rows each, prefill chunks for more than one row.
    model::batch make_draft_pass_inputs(std::int64_t sequences, std::int64_t rows_per_sequence) const;
    // Inputs of the draft pass after a verify step, state_slots rows per sequence and a logit token per sequence.
    model::batch make_first_pass_inputs(std::int64_t sequences) const;
    // Inputs of a draft pass of one row per sequence.
    model::batch make_later_pass_inputs(std::int64_t sequences) const;
    // Zeros in the dtype of the draft head's KV layer, [row_count, hidden_dim].
    at::Tensor make_draft_hidden_rows(std::int64_t row_count) const;
    // The sampling inputs of rows taken by the requests at sample_slots (int64 [rows]) in the table, each row's draw
    // at its entry of draw_counts (int64 [rows]).
    sampling_inputs
    gather_sampling_inputs(const at::Tensor& sample_slots, const at::Tensor& draw_counts, bool filters) const;
    // Gathers inputs, a step of the requests at request_slots without prefill chunks, from the table
    // (gather_step_inputs).
    void gather_inputs(const at::Tensor& request_slots, model::batch& inputs) const;
    std::unique_ptr<captured_decode_step> capture_decode_step(std::int64_t sequences, bool measures_mass, bool filters);
    std::unique_ptr<captured_verify_step> capture_verify_step(std::int64_t sequences, bool measures_mass, bool filters);

  public:
    // The decoding sequences whose graphs capture_startup records.
    static constexpr std::int64_t startup_sequences = 8;

    struct decode_step_outputs {
        // Each request's sample, int64 [sequences].
        at::Tensor samples;
        // Whether each request's history position reached its cycle end after the advance, bool [sequences].
        at::Tensor ends_cycle;
    };

    struct verify_step_outputs {
        // Each row's sample, int64 [sequences * state_slots].
        at::Tensor samples;
        // Each sequence's accepted token count, int64 [sequences].
        at::Tensor accepted_token_counts;
        // Each sequence's next token then its drafts, int64 [sequences, state_slots].
        at::Tensor next_tokens;
        // Whether each request's history position reached its cycle end after the advance, bool [sequences].
        at::Tensor ends_cycle;
    };

    // Graphs over page tables width pages wide, of pages page_size tokens long, a verify step's run of accepted drafts
    // ending at stop_tokens (int64 on the lines' device), with state_slots slots
    // (model::batch::state_slots). mass_heads holds, per entry of the model's kv_layers(), the query heads
    // whose mass this GPU measures. lines and table are on the current device; the graphs read and write table's
    // tensors where they are at construction, so these are never reallocated, and table.tokens has state_slots
    // columns.
    step_graphs(
        model& shard_model,
        std::span<const kv> lines,
        std::size_t width,
        std::size_t page_size,
        std::size_t state_slots,
        std::vector<std::size_t> mass_heads,
        at::Tensor stop_tokens,
        request_table table
    );

    std::size_t page_width() const noexcept;
    // Captures the graphs of 1 to startup_sequences sequences, with mass when a layer measures it and without: with a
    // draft head and state_slots above 1 the verify steps that do not filter, and otherwise the decode steps that do
    // not filter. It runs on the current stream, which is not the default stream, after an eager forward of the
    // captured kind, its inputs gathered from slot 0 of the table, that zeroes line 0 of each layer and writes into it.
    // Every GPU of the model runs it at once, before any request holds a line or a slot.
    void capture_startup();
    // The decode step of the requests at request_slots (int64), which all sample, sampled as filters says
    // (pulsarlm::filters). cu_view_pages (int32) is the step's model::batch::cu_view_pages, undefined when it measures
    // no mass; page_mass_sum and page_weight_sum are its model::batch::page_mass_sum and page_weight_sum, into which
    // the step's page mass is written as model::forward writes it. request_slots and cu_view_pages are in pinned host
    // memory or on the lines' device. A shape not captured yet is captured first. After the step each request's draw
    // count in the table is one more, and its entry is advanced (advance_requests) by its one computed token, no
    // rejected tokens and its sample as its next token. The outputs are valid until the next call.
    decode_step_outputs decode_step(
        const at::Tensor& request_slots,
        const at::Tensor& cu_view_pages,
        std::span<const at::Tensor> page_mass_sum,
        std::span<const at::Tensor> page_weight_sum,
        bool filters
    );
    // The verify step of the requests at request_slots, state_slots query tokens per request, whose requests all draft,
    // sampled as filters says, each request accepting at most max_accepted_token_counts (int64 [sequences], in pinned
    // host memory or on the lines' device) tokens, its page mass written as decode_step writes it. A
    // request accepts its pending token and its drafts while sample t equals draft t + 1, then the sample after them,
    // through a stop token. After the step each request's draw count in the table is more by its accepted count, and
    // its entry is advanced by its accepted count, state_slots less that count rejected, and its next token and drafts.
    // The outputs are valid until the next call. The model has a draft head.
    verify_step_outputs verify_step(
        const at::Tensor& request_slots,
        const at::Tensor& cu_view_pages,
        std::span<const at::Tensor> page_mass_sum,
        std::span<const at::Tensor> page_weight_sum,
        bool filters,
        const at::Tensor& max_accepted_token_counts
    );
};

}  // namespace pulsarlm
