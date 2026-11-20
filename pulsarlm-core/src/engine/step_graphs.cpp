#include "pulsarlm/engine/step_graphs.hpp"

#include "executor/request_table_kernels.hpp"

#include <ATen/ATen.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <utility>

namespace pulsarlm {

namespace {

void copy_input(const at::Tensor& buffer, const at::Tensor& source) {
    assert(buffer.sizes() == source.sizes());
    buffer.copy_(source, /*non_blocking=*/true);
}

// Copies a group's composition into a captured group's buffers, slot_buffer and buffers: the slots of its requests and,
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

// Copies into page_mass_sum and page_weight_sum the mass a replay added to the rows of the group's views in buffers.
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
    request_table table
)
: shard_model(shard_model),
  lines(lines.begin(), lines.end()),
  width(static_cast<std::int64_t>(width)),
  page_size(static_cast<std::int64_t>(page_size)),
  state_slots(static_cast<std::int64_t>(state_slots)),
  mass_heads(std::move(mass_heads)),
  stop_tokens(std::move(stop_tokens)),
  table(std::move(table)) {
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

model::batch step_graphs::make_group_inputs(std::int64_t sequences, bool measures_mass, model::batch_kind kind) const {
    assert(kind == model::batch_kind::decode || kind == model::batch_kind::verify);
    auto device = lines.front().k.options();
    auto ints = device.dtype(at::kInt);
    const auto layers = static_cast<std::int64_t>(lines.size());
    const bool verifies = kind == model::batch_kind::verify;
    const std::int64_t rows = verifies ? state_slots : 1;
    const std::int64_t tokens = sequences * rows;
    model::batch inputs{
        .kind = kind,
        .page_size = page_size,
        .state_slots = state_slots,
        .arrival_counters = at::zeros({shard_model.arrival_counter_count(sequences, tokens)}, ints),
    };
    if (verifies) {
        inputs.cu_seqlens_q = at::arange(sequences + 1, ints) * rows;
    }
    if (measures_mass) {
        // Each view is the one page its key lies on; a verify group's views are its rows'.
        const std::int64_t views = verifies ? tokens : sequences;
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

model::batch step_graphs::make_draft_pass_inputs(
    std::int64_t sequences,
    std::int64_t rows_per_sequence,
    model::batch_kind kind
) const {
    auto device = lines.front().k.options();
    auto ints = device.dtype(at::kInt);
    auto longs = device.dtype(at::kLong);
    const auto layers = static_cast<std::int64_t>(lines.size());
    const std::int64_t tokens = sequences * rows_per_sequence;
    return model::batch{
        .kind = kind,
        .tokens = at::zeros({tokens}, longs),
        .pos = at::zeros({tokens}, longs),
        .page_tables = at::zeros({layers, sequences, width}, ints),
        .page_size = page_size,
        .state_slots = state_slots,
        .key_counts = at::full({layers, sequences}, rows_per_sequence, ints),
        .page_positions = at::zeros({layers, sequences, width}, ints),
        .arrival_counters = at::zeros({shard_model.arrival_counter_count(sequences, tokens)}, ints),
    };
}

model::batch step_graphs::make_first_pass_inputs(std::int64_t sequences) const {
    model::batch inputs = make_draft_pass_inputs(sequences, state_slots, model::batch_kind::prefill);
    inputs.cu_seqlens_q = at::arange(sequences + 1, inputs.page_tables.options()) * state_slots;
    inputs.logit_rows = at::arange(sequences, inputs.tokens.options()) * state_slots;
    return inputs;
}

model::batch step_graphs::make_later_pass_inputs(std::int64_t sequences) const {
    return make_draft_pass_inputs(sequences, 1, model::batch_kind::decode);
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

std::unique_ptr<step_graphs::captured_forward>
step_graphs::capture_forward(std::int64_t sequences, bool measures_mass, model::batch_kind kind) {
    auto captured = std::make_unique<captured_forward>();
    captured->request_slots = make_request_slots(sequences);
    captured->inputs = make_group_inputs(sequences, measures_mass, kind);
    // Thread-local capture leaves the other GPUs' threads free to launch while this one captures.
    captured->graph.capture_begin({0, 0}, cudaStreamCaptureModeThreadLocal);
    gather_group_inputs(table, captured->request_slots, page_width(), captured->inputs);
    model::output output = shard_model.forward(captured->inputs, lines);
    captured->graph.capture_end();
    captured->logits = output.logits;
    captured->hidden = output.hidden;
    return captured;
}

std::unique_ptr<step_graphs::captured_decode_step>
step_graphs::capture_decode_step(std::int64_t sequences, bool measures_mass, bool filters) {
    auto captured = std::make_unique<captured_decode_step>();
    captured->request_slots = make_request_slots(sequences);
    captured->inputs = make_group_inputs(sequences, measures_mass, model::batch_kind::decode);
    const at::Tensor& slots = captured->request_slots;
    captured->graph.capture_begin({0, 0}, cudaStreamCaptureModeThreadLocal);
    gather_group_inputs(table, slots, page_width(), captured->inputs);
    const sampling_inputs sampling = gather_sampling_inputs(slots, table.draw_counts.index_select(0, slots), filters);
    model::output output = shard_model.forward(captured->inputs, lines);
    at::Tensor samples = sample(output.logits, sampling);
    const at::Tensor ones = at::ones_like(slots);
    table.draw_counts.index_add_(0, slots, ones);
    at::Tensor next = at::zeros({sequences, table.tokens.size(1)}, samples.options());
    next.select(1, 0).copy_(samples);
    at::Tensor ends_cycle = at::empty({sequences}, slots.options().dtype(at::kBool));
    advance_requests(table, slots, ones, at::zeros_like(slots), next, ends_cycle);
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
    captured->max_accepted_token_counts = at::full({sequences}, rows, longs);
    captured->inputs = make_group_inputs(sequences, measures_mass, model::batch_kind::verify);
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
    gather_group_inputs(table, slots, page_width(), inputs);
    const at::Tensor sample_slots = slots.unsqueeze(1).expand({sequences, rows}).reshape({tokens});
    const at::Tensor draw_offsets = at::arange(rows, longs).repeat({sequences});
    const sampling_inputs sampling = gather_sampling_inputs(
        sample_slots,
        table.draw_counts.index_select(0, sample_slots) + draw_offsets,
        filters
    );
    model::output verified_rows = shard_model.forward(inputs, lines);
    at::Tensor samples = sample(verified_rows.logits, sampling);
    at::Tensor accepted_counts = count_accepted_tokens(
        samples.view({sequences, rows}),
        inputs.tokens.view({sequences, rows}).narrow(1, 1, rows - 1),
        stop_tokens,
        captured->max_accepted_token_counts
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
    first.logit_rows.copy_(drafting_rows);
    model::output drafted_rows = head.draft(first, verified_rows.hidden, lines);
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
        model::output following = head.draft(one_row, previous_hidden, lines);
        token = following.logits.argmax(-1);
        previous_hidden = following.hidden;
        next.push_back(token);
    }
    at::Tensor next_tokens = at::stack(next, 1);
    table.draw_counts.index_add_(0, slots, accepted_counts);
    at::Tensor ends_cycle = at::empty({sequences}, slots.options().dtype(at::kBool));
    advance_requests(table, slots, accepted_counts, rows - accepted_counts, next_tokens, ends_cycle);
    captured->graph.capture_end();
    captured->samples = samples;
    captured->accepted_token_counts = accepted_counts;
    captured->next_tokens = next_tokens;
    captured->ends_cycle = ends_cycle;
    return captured;
}

// A capture cannot create what the forward creates at its first run, such as cuBLAS handles and RoPE tables, so an eager
// forward of each kind runs first, and an eager draft pass of each kind before the verify steps. Their tables point
// every view at line 0, which is zeroed first so that the forward reads no uninitialized memory; no request holds a
// line or a slot yet, and the forwards write nothing into the table.
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
    const bool verifies = state_slots > 1;
    const bool drafts = verifies && shard_model.drafter() != nullptr;
    std::vector<model::batch_kind> kinds{model::batch_kind::decode};
    if (verifies) {
        kinds.push_back(model::batch_kind::verify);
    }
    for (model::batch_kind kind : kinds) {
        for (bool measures_mass : mass_options) {
            model::batch inputs = make_group_inputs(1, measures_mass, kind);
            gather_group_inputs(table, make_request_slots(1), page_width(), inputs);
            shard_model.forward(inputs, lines);
        }
    }
    if (drafts) {
        shard_model.drafter()->draft(make_first_pass_inputs(1), make_draft_hidden_rows(state_slots), lines);
        shard_model.drafter()->draft(make_later_pass_inputs(1), make_draft_hidden_rows(1), lines);
    }
    for (std::int64_t sequences = 1; sequences <= startup_sequences; sequences++) {
        for (bool measures_mass : mass_options) {
            const group_shape shape{sequences, measures_mass};
            if (!decode_forwards.contains(shape)) {
                decode_forwards.emplace(shape, capture_forward(sequences, measures_mass, model::batch_kind::decode));
            }
            if (verifies && !verify_forwards.contains(shape)) {
                verify_forwards.emplace(shape, capture_forward(sequences, measures_mass, model::batch_kind::verify));
            }
            if (drafts && !unfiltered_verify_steps.contains(shape)) {
                unfiltered_verify_steps.emplace(shape, capture_verify_step(sequences, measures_mass, false));
            }
            if (!drafts && !unfiltered_decode_steps.contains(shape)) {
                unfiltered_decode_steps.emplace(shape, capture_decode_step(sequences, measures_mass, false));
            }
        }
    }
}

model::output step_graphs::replay(
    captured_forward& captured,
    const at::Tensor& request_slots,
    const at::Tensor& cu_view_pages,
    std::span<const at::Tensor> page_mass_sum,
    std::span<const at::Tensor> page_weight_sum
) {
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
    return model::output{.logits = captured.logits, .hidden = captured.hidden};
}

model::output step_graphs::decode(
    const at::Tensor& request_slots,
    const at::Tensor& cu_view_pages,
    std::span<const at::Tensor> page_mass_sum,
    std::span<const at::Tensor> page_weight_sum
) {
    const group_shape shape{request_slots.size(0), cu_view_pages.defined()};
    auto found = decode_forwards.find(shape);
    if (found == decode_forwards.end()) {
        found = decode_forwards.emplace(shape, capture_forward(shape.first, shape.second, model::batch_kind::decode))
                    .first;
    }
    return replay(*found->second, request_slots, cu_view_pages, page_mass_sum, page_weight_sum);
}

step_graphs::verify_outputs step_graphs::verify(
    const at::Tensor& request_slots,
    const at::Tensor& cu_view_pages,
    std::span<const at::Tensor> page_mass_sum,
    std::span<const at::Tensor> page_weight_sum
) {
    const group_shape shape{request_slots.size(0), cu_view_pages.defined()};
    auto found = verify_forwards.find(shape);
    if (found == verify_forwards.end()) {
        found = verify_forwards.emplace(shape, capture_forward(shape.first, shape.second, model::batch_kind::verify))
                    .first;
    }
    captured_forward& captured = *found->second;
    model::output output = replay(captured, request_slots, cu_view_pages, page_mass_sum, page_weight_sum);
    return verify_outputs{.logits = output.logits, .hidden = output.hidden, .tokens = captured.inputs.tokens};
}

step_graphs::decode_step_outputs step_graphs::decode_step(
    const at::Tensor& request_slots,
    const at::Tensor& cu_view_pages,
    std::span<const at::Tensor> page_mass_sum,
    std::span<const at::Tensor> page_weight_sum,
    bool filters
) {
    const group_shape shape{request_slots.size(0), cu_view_pages.defined()};
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
    bool filters,
    const at::Tensor& max_accepted_token_counts
) {
    assert(shard_model.drafter() != nullptr);
    const group_shape shape{request_slots.size(0), cu_view_pages.defined()};
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
    copy_input(captured.max_accepted_token_counts, max_accepted_token_counts);
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
