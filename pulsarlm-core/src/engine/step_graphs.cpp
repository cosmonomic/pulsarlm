#include "pulsarlm/engine/step_graphs.hpp"

#include "executor/request_table_kernels.hpp"

#include <ATen/ATen.h>

#include <algorithm>
#include <ranges>
#include <cassert>
#include <cstddef>
#include <utility>

namespace pulsarlm {

namespace {

void copy_input(const at::Tensor& buffer, const at::Tensor& source) {
    assert(buffer.sizes() == source.sizes());
    buffer.copy_(source, /*non_blocking=*/true);
}

// Copies a step's composition into a captured step's buffers, slot_buffer and buffers: the slots of its requests and,
// when it measures mass, its cu_view_pages, zeroing the mass rows of its views in buffers, into which the graph adds
// the step's mass.
void copy_composition(
    const at::Tensor& slot_buffer,
    const model::batch& buffers,
    const at::Tensor& request_slots,
    const at::Tensor& cu_view_pages,
    std::span<const at::Tensor> page_mass_sum,
    std::span<const at::Tensor> page_weight_sum
) {
    assert(cu_view_pages.defined() == buffers.cu_view_pages.defined());
    assert(page_mass_sum.size() == page_weight_sum.size());
    copy_input(slot_buffer, request_slots);
    if (cu_view_pages.defined()) {
        copy_input(buffers.cu_view_pages, cu_view_pages);
    }
    for (std::size_t layer = 0; layer < page_mass_sum.size(); layer++) {
        if (page_mass_sum[layer].defined()) {
            buffers.page_mass_sum[layer].narrow(0, 0, page_mass_sum[layer].size(0)).zero_();
            buffers.page_weight_sum[layer].narrow(0, 0, page_weight_sum[layer].size(0)).zero_();
        }
    }
}

// Copies into page_mass_sum and page_weight_sum the mass a replay added to the rows of the step's views in buffers.
void copy_page_mass(
    std::span<const at::Tensor> page_mass_sum,
    std::span<const at::Tensor> page_weight_sum,
    const model::batch& buffers
) {
    for (std::size_t layer = 0; layer < page_mass_sum.size(); layer++) {
        if (page_mass_sum[layer].defined()) {
            page_mass_sum[layer].copy_(buffers.page_mass_sum[layer].narrow(0, 0, page_mass_sum[layer].size(0)));
            page_weight_sum[layer].copy_(buffers.page_weight_sum[layer].narrow(0, 0, page_weight_sum[layer].size(0)));
        }
    }
}

}  // namespace

step_graphs::step_graphs(
    model& shard_model,
    std::span<const kv> lines,
    std::size_t width,
    std::size_t page_size,
    std::size_t state_slots,
    std::vector<std::size_t> mass_heads,
    at::Tensor stop_tokens,
    request_table table,
    std::vector<std::int64_t> bucket_token_counts
)
: shard_model(shard_model),
  lines(lines.begin(), lines.end()),
  width(static_cast<std::int64_t>(width)),
  page_size(static_cast<std::int64_t>(page_size)),
  state_slots(static_cast<std::int64_t>(state_slots)),
  mass_heads(std::move(mass_heads)),
  stop_tokens(std::move(stop_tokens)),
  table(std::move(table)),
  no_prefill_tokens(at::empty({0}, lines.front().k.options().dtype(at::kLong))),
  bucket_token_counts(std::move(bucket_token_counts)),
  segment_pool(at::cuda::graph_pool_handle()) {
    assert(std::ranges::is_sorted(this->bucket_token_counts));
    assert(this->table.tokens.size(1) == this->state_slots && this->width <= this->table.page_tables.size(2));
    auto floats = this->lines.front().k.options().dtype(at::kFloat);
    const std::int64_t rows = startup_sequences * this->width;
    for (std::size_t heads : this->mass_heads) {
        shared_mass_sum.push_back(
            heads > 0 ? at::empty({rows, static_cast<std::int64_t>(heads)}, floats) : at::Tensor()
        );
        shared_weight_sum.push_back(heads > 0 ? at::empty({rows}, floats) : at::Tensor());
    }
}

std::size_t step_graphs::page_width() const noexcept {
    return static_cast<std::size_t>(width);
}

std::size_t step_graphs::find_draft_layer() const {
    auto with_hidden = std::ranges::find_if(lines, [](const kv& line) { return line.last_hidden.defined(); });
    TORCH_CHECK(with_hidden != lines.end(), "step_graphs: no KV layer keeps the draft head's hidden rows");
    return static_cast<std::size_t>(with_hidden - lines.begin());
}

at::Tensor step_graphs::make_request_slots(std::int64_t sequences) const {
    return at::zeros({sequences}, lines.front().k.options().dtype(at::kLong));
}

model::batch step_graphs::make_step_inputs(std::int64_t sequences, bool measures_mass, bool verifies) const {
    auto device = lines.front().k.options();
    auto ints = device.dtype(at::kInt);
    const auto layers = static_cast<std::int64_t>(lines.size());
    const std::int64_t rows = verifies ? state_slots : 1;
    const std::int64_t tokens = sequences * rows;
    const at::Tensor query_starts = at::arange(sequences + 1, ints) * rows;
    model::batch inputs{
        .single_query_request_count = verifies ? 0 : sequences,
        .per_query_state_request_count = verifies ? sequences : 0,
        .page_size = page_size,
        .state_slots = state_slots,
        .cu_seqlens_q = query_starts,
        .cu_seqlens_q_from_second_run = verifies ? query_starts : at::zeros({1}, ints),
        .cu_seqlens_q_from_third_run = at::zeros({1}, ints),
        .logit_tokens = at::arange(tokens, device.dtype(at::kLong)),
        .arrival_counters = at::zeros({shard_model.arrival_counter_count(sequences, tokens)}, ints),
    };
    if (measures_mass) {
        // Each view is the one page its key lies on; a verify step's views are its query tokens'.
        const std::int64_t views = tokens;
        inputs.cu_view_pages = at::arange(views + 1, ints).unsqueeze(0).repeat({layers, 1});
        const std::int64_t mass_rows = views * width;
        auto floats = device.dtype(at::kFloat);
        for (std::size_t layer = 0; layer < mass_heads.size(); layer++) {
            if (mass_heads[layer] == 0) {
                inputs.page_mass_sum.emplace_back();
                inputs.page_weight_sum.emplace_back();
            } else if (sequences <= startup_sequences && !verifies) {
                inputs.page_mass_sum.push_back(shared_mass_sum[layer].narrow(0, 0, mass_rows));
                inputs.page_weight_sum.push_back(shared_weight_sum[layer].narrow(0, 0, mass_rows));
            } else {
                inputs.page_mass_sum.push_back(
                    at::empty({mass_rows, static_cast<std::int64_t>(mass_heads[layer])}, floats)
                );
                inputs.page_weight_sum.push_back(at::empty({mass_rows}, floats));
            }
        }
    }
    return inputs;
}

model::batch step_graphs::make_draft_pass_inputs(std::int64_t sequences, std::int64_t rows_per_sequence) const {
    auto device = lines.front().k.options();
    auto ints = device.dtype(at::kInt);
    auto longs = device.dtype(at::kLong);
    const auto layers = static_cast<std::int64_t>(lines.size());
    const std::int64_t tokens = sequences * rows_per_sequence;
    const bool single_queries = rows_per_sequence == 1;
    const at::Tensor query_starts = at::arange(sequences + 1, ints) * rows_per_sequence;
    return model::batch{
        .single_query_request_count = single_queries ? sequences : 0,
        .tokens = at::zeros({tokens}, longs),
        .pos = at::zeros({tokens}, longs),
        .page_tables = at::zeros({layers, sequences, width}, ints),
        .page_size = page_size,
        .state_slots = state_slots,
        .cu_seqlens_q = query_starts,
        .cu_seqlens_q_from_second_run = single_queries ? at::zeros({1}, ints) : query_starts,
        .cu_seqlens_q_from_third_run = single_queries ? at::zeros({1}, ints) : query_starts,
        .key_counts = at::full({layers, sequences}, rows_per_sequence, ints),
        .page_positions = at::zeros({layers, sequences, width}, ints),
        .logit_tokens = at::arange(sequences, longs) * rows_per_sequence + (rows_per_sequence - 1),
        .arrival_counters = at::zeros({shard_model.arrival_counter_count(sequences, tokens)}, ints),
    };
}

model::batch step_graphs::make_first_pass_inputs(std::int64_t sequences) const {
    return make_draft_pass_inputs(sequences, state_slots);
}

model::batch step_graphs::make_later_pass_inputs(std::int64_t sequences) const {
    return make_draft_pass_inputs(sequences, 1);
}

at::Tensor step_graphs::make_draft_hidden_rows(std::int64_t row_count) const {
    const at::Tensor& last_hidden = lines[find_draft_layer()].last_hidden;
    return at::zeros({row_count, shard_model.drafter()->hidden_dim()}, last_hidden.options());
}

sampling_inputs
step_graphs::gather_sampling_inputs(const at::Tensor& sample_slots, const at::Tensor& draw_counts, bool filters) const {
    return sampling_inputs{
        .temperatures = table.temperatures.index_select(0, sample_slots),
        .top_k = table.top_k.index_select(0, sample_slots),
        .top_p = table.top_p.index_select(0, sample_slots),
        .draws = at::stack({table.seeds.index_select(0, sample_slots), draw_counts}),
        .filters = filters,
    };
}

void step_graphs::gather_inputs(const at::Tensor& request_slots, model::batch& inputs) const {
    const std::int64_t token_count = inputs.single_query_request_count +
        inputs.per_query_state_request_count * inputs.state_slots;
    gather_step_inputs(table, request_slots, page_width(), token_count, token_count, no_prefill_tokens, inputs);
}

std::unique_ptr<step_graphs::captured_decode_step>
step_graphs::capture_decode_step(std::int64_t sequences, bool measures_mass, bool filters) {
    auto captured = std::make_unique<captured_decode_step>();
    captured->request_slots = make_request_slots(sequences);
    captured->inputs = make_step_inputs(sequences, measures_mass, false);
    const at::Tensor& slots = captured->request_slots;
    // Thread-local capture leaves the other GPUs' threads free to launch while this one captures.
    captured->graph.capture_begin({0, 0}, cudaStreamCaptureModeThreadLocal);
    gather_inputs(slots, captured->inputs);
    const sampling_inputs sampling = gather_sampling_inputs(slots, table.draw_counts.index_select(0, slots), filters);
    graph_segments segments(sequences);
    model::output output = shard_model.forward(captured->inputs, lines, segments);
    at::Tensor samples = sample(output.logits, sampling);
    const at::Tensor ones = at::ones_like(slots);
    table.draw_counts.index_add_(0, slots, ones);
    at::Tensor next = at::zeros({sequences, table.tokens.size(1)}, samples.options());
    next.select(1, 0).copy_(samples);
    at::Tensor ends_cycle = at::empty({sequences}, slots.options().dtype(at::kBool));
    advance_requests(table, slots, ones, at::zeros_like(slots), ones, next, ends_cycle);
    captured->graph.capture_end();
    captured->samples = samples;
    captured->ends_cycle = ends_cycle;
    return captured;
}

std::unique_ptr<step_graphs::captured_verify_step>
step_graphs::capture_verify_step(std::int64_t sequences, bool measures_mass, bool filters) {
    auto captured = std::make_unique<captured_verify_step>();
    const auto longs = lines.front().k.options().dtype(at::kLong);
    const std::int64_t rows = state_slots;
    const std::int64_t tokens = sequences * rows;
    const std::int64_t later = state_slots - 2;
    captured->request_slots = make_request_slots(sequences);
    captured->inputs = make_step_inputs(sequences, measures_mass, true);
    captured->first_pass = make_first_pass_inputs(sequences);
    captured->later_pass = make_later_pass_inputs(sequences);
    const at::Tensor& slots = captured->request_slots;
    model::batch& inputs = captured->inputs;
    const auto layers = static_cast<std::int64_t>(lines.size());
    const auto draft_layer = static_cast<std::int64_t>(find_draft_layer());
    auto every_layer = [&](const at::Tensor& counts) {
        return counts.to(at::kInt).unsqueeze(0).expand({layers, sequences});
    };
    draft_head& head = *shard_model.drafter();
    captured->graph.capture_begin({0, 0}, cudaStreamCaptureModeThreadLocal);
    gather_inputs(slots, inputs);
    const at::Tensor sample_slots = slots.unsqueeze(1).expand({sequences, rows}).reshape({tokens});
    const at::Tensor draw_offsets = at::arange(rows, longs).repeat({sequences});
    const sampling_inputs sampling = gather_sampling_inputs(
        sample_slots,
        table.draw_counts.index_select(0, sample_slots) + draw_offsets,
        filters
    );
    graph_segments verified_segments(tokens);
    model::output verified_rows = shard_model.forward(inputs, lines, verified_segments);
    at::Tensor samples = sample(verified_rows.logits, sampling);
    at::Tensor accepted_counts = count_accepted_tokens(
        samples.view({sequences, rows}),
        inputs.tokens.view({sequences, rows}).narrow(1, 1, rows - 1),
        stop_tokens,
        table.remaining_tokens.index_select(0, slots).clamp_min(1)
    );
    // The RoPE position of each request's pending token, and its keys in the draft head's KV layer before it.
    const at::Tensor positions = table.rope_positions.index_select(0, slots);
    const at::Tensor keys = table.key_counts.index_select(0, slots).select(1, draft_layer).to(at::kLong);
    at::Tensor drafting_rows = at::arange(sequences, longs) * rows + accepted_counts - 1;
    model::batch& first = captured->first_pass;
    first.tokens.copy_(samples);
    first.pos.copy_((positions.unsqueeze(1) + at::arange(1, rows + 1, longs)).reshape({tokens}));
    first.page_tables.copy_(inputs.page_tables);
    first.page_positions.copy_(inputs.page_positions);
    first.key_counts.copy_(every_layer(keys + (rows + 1)));
    first.logit_tokens.copy_(drafting_rows);
    graph_segments first_segments(tokens);
    model::output drafted_rows = head.draft(first, verified_rows.hidden, lines, first_segments);
    at::Tensor token = drafted_rows.logits.argmax(-1);
    at::Tensor previous_hidden = drafted_rows.hidden.index_select(0, drafting_rows);
    std::vector<at::Tensor> next{samples.index_select(0, drafting_rows), token};
    model::batch& one_row = captured->later_pass;
    for (std::int64_t j = 0; j < later; j++) {
        one_row.tokens.copy_(token);
        one_row.pos.copy_(positions + accepted_counts + (j + 1));
        one_row.page_tables.copy_(inputs.page_tables);
        one_row.page_positions.copy_(inputs.page_positions);
        one_row.key_counts.copy_(every_layer(keys + accepted_counts + (j + 2)));
        graph_segments later_segments(sequences);
        model::output following = head.draft(one_row, previous_hidden, lines, later_segments);
        token = following.logits.argmax(-1);
        previous_hidden = following.hidden;
        next.push_back(token);
    }
    at::Tensor next_tokens = at::stack(next, 1);
    table.draw_counts.index_add_(0, slots, accepted_counts);
    at::Tensor ends_cycle = at::empty({sequences}, slots.options().dtype(at::kBool));
    advance_requests(table, slots, accepted_counts, rows - accepted_counts, accepted_counts, next_tokens, ends_cycle);
    captured->graph.capture_end();
    captured->samples = samples;
    captured->accepted_token_counts = accepted_counts;
    captured->next_tokens = next_tokens;
    captured->ends_cycle = ends_cycle;
    return captured;
}

// A capture cannot create what the forward creates at its first run, such as cuBLAS handles and RoPE tables, so an eager
// forward of the captured kind runs first, and an eager draft pass of each kind before the verify steps. Their tables
// point every view at line 0, which is zeroed first so that the forward reads no uninitialized memory; no request holds
// a line or a slot yet, and the forwards write nothing into the table.
void step_graphs::capture_startup() {
    for (const kv& layer : lines) {
        for (const at::Tensor& part : {layer.k, layer.v, layer.index_keys, layer.last_hidden}) {
            if (part.defined()) {
                part.select(0, 0).zero_();
            }
        }
    }
    std::vector<bool> mass_options{false};
    if (std::ranges::any_of(mass_heads, [](std::size_t heads) { return heads > 0; })) {
        mass_options.push_back(true);
    }
    const bool drafts = state_slots > 1 && shard_model.drafter() != nullptr;
    for (bool measures_mass : mass_options) {
        model::batch inputs = make_step_inputs(1, measures_mass, drafts);
        gather_inputs(make_request_slots(1), inputs);
        graph_segments segments(inputs.tokens.size(0));
        shard_model.forward(inputs, lines, segments);
    }
    if (drafts) {
        graph_segments first_segments(state_slots);
        shard_model.drafter()
            ->draft(make_first_pass_inputs(1), make_draft_hidden_rows(state_slots), lines, first_segments);
        graph_segments later_segments(1);
        shard_model.drafter()->draft(make_later_pass_inputs(1), make_draft_hidden_rows(1), lines, later_segments);
    }
    for (std::int64_t sequences = 1; sequences <= startup_sequences; sequences++) {
        for (bool measures_mass : mass_options) {
            const step_shape shape{sequences, measures_mass};
            if (drafts && !unfiltered_verify_steps.contains(shape)) {
                unfiltered_verify_steps.emplace(shape, capture_verify_step(sequences, measures_mass, false));
            }
            if (!drafts && !unfiltered_decode_steps.contains(shape)) {
                unfiltered_decode_steps.emplace(shape, capture_decode_step(sequences, measures_mass, false));
            }
        }
    }
    for (std::int64_t token_count : std::views::reverse(bucket_token_counts)) {
        if (shard_model.allows_segment_capture(token_count) && !segment_steps.contains(token_count)) {
            warm_up_bucket(token_count);
            segment_steps.emplace(token_count, capture_segments(token_count));
        }
    }
}

std::int64_t step_graphs::count_most_logit_tokens(std::int64_t token_count) const {
    return std::min(token_count, table.history_positions.size(0) * state_slots);
}

void step_graphs::warm_up_bucket(std::int64_t token_count) {
    const auto device = lines.front().k.options();
    const auto ints = device.dtype(at::kInt);
    const auto longs = device.dtype(at::kLong);
    const std::int64_t requests = (token_count + page_size - 1) / page_size;
    const at::Tensor query_starts = at::minimum(
        at::arange(requests + 1, ints) * page_size,
        at::scalar_tensor(token_count, ints)
    );
    model::batch inputs{
        .page_size = page_size,
        .state_slots = state_slots,
        .cu_seqlens_q = query_starts,
        .cu_seqlens_q_from_second_run = query_starts,
        .cu_seqlens_q_from_third_run = query_starts,
        .logit_tokens = query_starts.narrow(0, 1, requests).to(at::kLong) - 1,
        .arrival_counters = at::zeros({shard_model.arrival_counter_count(requests, token_count)}, ints),
    };
    gather_step_inputs(
        table,
        make_request_slots(requests),
        page_width(),
        token_count,
        token_count,
        at::zeros({token_count}, longs),
        inputs
    );
    graph_segments segments(token_count);
    shard_model.forward(inputs, lines, segments);
}

// The calls between segments do not run during the capture, so it writes into no line.
std::unique_ptr<step_graphs::captured_segments> step_graphs::capture_segments(std::int64_t token_count) {
    const auto device = lines.front().k.options();
    const auto ints = device.dtype(at::kInt);
    const auto longs = device.dtype(at::kLong);
    auto captured = std::make_unique<captured_segments>();
    const at::Tensor query_starts = at::tensor({0, static_cast<std::int32_t>(token_count)}, at::kInt)
                                        .to(device.device());
    const std::int64_t max_requests = table.history_positions.size(0);
    captured->inputs = model::batch{
        .tokens = at::zeros({token_count}, longs),
        .pos = at::zeros({token_count}, longs),
        .page_size = page_size,
        .state_slots = state_slots,
        .cu_seqlens_q = query_starts,
        .cu_seqlens_q_from_second_run = query_starts,
        .cu_seqlens_q_from_third_run = query_starts,
        .logit_tokens = at::zeros({count_most_logit_tokens(token_count)}, longs),
        .arrival_counters = at::zeros({shard_model.arrival_counter_count(max_requests, token_count)}, ints),
    };
    graph_segments segments(token_count, captured->capture, segment_pool);
    model::output output = shard_model.forward(captured->inputs, lines, segments);
    segments.finish_capture();
    captured->logits = graph_segments::make_unowned_view(output.logits);
    captured->hidden = graph_segments::make_unowned_view(output.hidden);
    return captured;
}

bool step_graphs::has_segments(std::int64_t bucket_token_count) const {
    return segment_steps.contains(bucket_token_count);
}

model::output step_graphs::replay_segments(const model::batch& inputs, std::int64_t token_count) {
    captured_segments& captured = *segment_steps.at(inputs.tokens.size(0));
    model::batch& replayed = captured.inputs;
    copy_input(replayed.tokens, inputs.tokens);
    copy_input(replayed.pos, inputs.pos);
    replayed.single_query_request_count = inputs.single_query_request_count;
    replayed.per_query_state_request_count = inputs.per_query_state_request_count;
    replayed.page_tables = inputs.page_tables;
    replayed.history_positions = inputs.history_positions;
    replayed.rejected_tokens = inputs.rejected_tokens;
    replayed.page_size = inputs.page_size;
    replayed.state_slots = inputs.state_slots;
    replayed.cu_seqlens_q = inputs.cu_seqlens_q;
    replayed.cu_seqlens_q_from_second_run = inputs.cu_seqlens_q_from_second_run;
    replayed.cu_seqlens_q_from_third_run = inputs.cu_seqlens_q_from_third_run;
    replayed.kept_state_counts = inputs.kept_state_counts;
    replayed.key_counts = inputs.key_counts;
    replayed.page_positions = inputs.page_positions;
    replayed.cu_view_pages = inputs.cu_view_pages;
    replayed.page_mass_sum = inputs.page_mass_sum;
    replayed.page_weight_sum = inputs.page_weight_sum;
    replayed.query_weights = inputs.query_weights;
    replayed.logit_tokens = inputs.logit_tokens;
    graph_segments::replay(captured.capture, token_count);
    return model::output{
        .logits = captured.logits.narrow(0, 0, inputs.logit_tokens.size(0)),
        .hidden = captured.hidden,
    };
}

step_graphs::decode_step_outputs step_graphs::decode_step(
    const at::Tensor& request_slots,
    const at::Tensor& cu_view_pages,
    std::span<const at::Tensor> page_mass_sum,
    std::span<const at::Tensor> page_weight_sum,
    bool filters
) {
    const step_shape shape{request_slots.size(0), cu_view_pages.defined()};
    auto& steps = filters ? filtered_decode_steps : unfiltered_decode_steps;
    auto found = steps.find(shape);
    if (found == steps.end()) {
        found = steps.emplace(shape, capture_decode_step(shape.first, shape.second, filters)).first;
    }
    captured_decode_step& captured = *found->second;
    copy_composition(
        captured.request_slots,
        captured.inputs,
        request_slots,
        cu_view_pages,
        page_mass_sum,
        page_weight_sum
    );
    captured.graph.replay();
    copy_page_mass(page_mass_sum, page_weight_sum, captured.inputs);
    return decode_step_outputs{.samples = captured.samples, .ends_cycle = captured.ends_cycle};
}

step_graphs::verify_step_outputs step_graphs::verify_step(
    const at::Tensor& request_slots,
    const at::Tensor& cu_view_pages,
    std::span<const at::Tensor> page_mass_sum,
    std::span<const at::Tensor> page_weight_sum,
    bool filters
) {
    assert(shard_model.drafter() != nullptr);
    const step_shape shape{request_slots.size(0), cu_view_pages.defined()};
    auto& steps = filters ? filtered_verify_steps : unfiltered_verify_steps;
    auto found = steps.find(shape);
    if (found == steps.end()) {
        found = steps.emplace(shape, capture_verify_step(shape.first, shape.second, filters)).first;
    }
    captured_verify_step& captured = *found->second;
    copy_composition(
        captured.request_slots,
        captured.inputs,
        request_slots,
        cu_view_pages,
        page_mass_sum,
        page_weight_sum
    );
    captured.graph.replay();
    copy_page_mass(page_mass_sum, page_weight_sum, captured.inputs);
    return verify_step_outputs{
        .samples = captured.samples,
        .accepted_token_counts = captured.accepted_token_counts,
        .next_tokens = captured.next_tokens,
        .ends_cycle = captured.ends_cycle,
    };
}

}  // namespace pulsarlm
