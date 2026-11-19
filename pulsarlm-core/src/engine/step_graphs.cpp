#include "pulsarlm/engine/step_graphs.hpp"

#include "pulsarlm/kernels/sampling.hpp"

#include <ATen/ATen.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <utility>

namespace pulsarlm {

namespace {

void copy_input(const at::Tensor& buffer, const at::Tensor& source) {
    assert(buffer.sizes() == source.sizes());
    buffer.copy_(source);
}

// Copies g's inputs into buffers, a captured group's, and zeroes the mass rows of g's views in buffers, into which the
// graph adds the step's mass.
void copy_group_inputs(const model::batch& buffers, const model::batch& g) {
    assert(g.page_size == buffers.page_size && g.state_slots == buffers.state_slots);
    assert(g.cu_view_pages.defined() == buffers.cu_view_pages.defined());
    copy_input(buffers.tokens, g.tokens);
    copy_input(buffers.pos, g.pos);
    copy_input(buffers.page_tables, g.page_tables);
    copy_input(buffers.history_positions, g.history_positions);
    copy_input(buffers.rejected_tokens, g.rejected_tokens);
    copy_input(buffers.key_counts, g.key_counts);
    copy_input(buffers.page_positions, g.page_positions);
    if (g.cu_view_pages.defined()) {
        copy_input(buffers.cu_view_pages, g.cu_view_pages);
        copy_input(buffers.query_weights, g.query_weights);
    }
    for (std::size_t layer = 0; layer < g.page_mass_sum.size(); layer++) {
        if (g.page_mass_sum[layer].defined()) {
            buffers.page_mass_sum[layer].narrow(0, 0, g.page_mass_sum[layer].size(0)).zero_();
            buffers.page_weight_sum[layer].narrow(0, 0, g.page_weight_sum[layer].size(0)).zero_();
        }
    }
}

// Copies into g the mass a replay added to the rows of g's views in buffers.
void copy_page_mass(const model::batch& g, const model::batch& buffers) {
    for (std::size_t layer = 0; layer < g.page_mass_sum.size(); layer++) {
        if (g.page_mass_sum[layer].defined()) {
            g.page_mass_sum[layer].copy_(buffers.page_mass_sum[layer].narrow(0, 0, g.page_mass_sum[layer].size(0)));
            g.page_weight_sum[layer].copy_(
                buffers.page_weight_sum[layer].narrow(0, 0, g.page_weight_sum[layer].size(0))
            );
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
    at::Tensor samples,
    at::Tensor stop_tokens
)
: shard_model(shard_model),
  lines(lines.begin(), lines.end()),
  samples(std::move(samples)),
  width(static_cast<std::int64_t>(width)),
  page_size(static_cast<std::int64_t>(page_size)),
  state_slots(static_cast<std::int64_t>(state_slots)),
  mass_heads(std::move(mass_heads)),
  stop_tokens(std::move(stop_tokens)) {
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

step_graphs::group_shape step_graphs::shape_of(const model::batch& g) {
    return {g.page_tables.size(1), g.cu_view_pages.defined()};
}

model::batch step_graphs::make_group_inputs(std::int64_t sequences, bool measures_mass, model::batch_kind kind) const {
    assert(kind == model::batch_kind::decode || kind == model::batch_kind::verify);
    auto device = lines.front().k.options();
    auto ints = device.dtype(at::kInt);
    auto longs = device.dtype(at::kLong);
    const auto layers = static_cast<std::int64_t>(lines.size());
    const bool verifies = kind == model::batch_kind::verify;
    const std::int64_t rows = verifies ? state_slots : 1;
    const std::int64_t tokens = sequences * rows;
    model::batch inputs{
        .kind = kind,
        .tokens = at::zeros({tokens}, longs),
        .pos = at::zeros({tokens}, longs),
        .page_tables = at::zeros({layers, sequences, width}, ints),
        .history_positions = at::zeros({sequences}, longs),
        .rejected_tokens = at::zeros({sequences}, ints),
        .page_size = page_size,
        .state_slots = state_slots,
        .key_counts = at::full({layers, sequences}, rows, ints),
        .page_positions = at::zeros({layers, sequences, width}, ints),
        .arrival_counters = at::zeros({shard_model.arrival_counter_count(sequences, tokens)}, ints),
    };
    if (verifies) {
        inputs.cu_seqlens_q = at::arange(sequences + 1, ints) * rows;
    }
    if (measures_mass) {
        // Each view is the one page its key lies on; a verify group's views are its rows'.
        const std::int64_t views = verifies ? tokens : sequences;
        inputs.cu_view_pages = at::arange(views + 1, ints).unsqueeze(0).repeat({layers, 1});
        inputs.query_weights = at::zeros({tokens}, device.dtype(at::kFloat));
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
    auto with_hidden = std::ranges::find_if(lines, [](const kv& line) { return line.last_hidden.defined(); });
    TORCH_CHECK(with_hidden != lines.end(), "step_graphs: no KV layer keeps the draft head's hidden rows");
    return at::zeros({row_count, shard_model.drafter()->hidden_dim()}, with_hidden->last_hidden.options());
}

sampling_inputs step_graphs::make_sampling_inputs(std::int64_t row_count) const {
    const auto options = lines.front().k.options();
    return sampling_inputs{
        .temperatures = at::zeros({row_count}, options.dtype(at::kFloat)),
        .top_k = at::ones({row_count}, options.dtype(at::kLong)),
        .top_p = at::full({row_count}, 2.0f, options.dtype(at::kFloat)),
        .draws = at::zeros({2, row_count}, options.dtype(at::kLong)),
        .filters = false,
    };
}

void step_graphs::record_forward(captured_forward& captured) {
    model::output output = shard_model.forward(captured.inputs, lines);
    captured.logits = output.logits;
    captured.hidden = output.hidden;
}

std::unique_ptr<step_graphs::captured_forward> step_graphs::capture_decode(std::int64_t sequences, bool measures_mass) {
    auto captured = std::make_unique<captured_forward>();
    captured->inputs = make_group_inputs(sequences, measures_mass, model::batch_kind::decode);
    // Thread-local capture leaves the other GPUs' threads free to launch while this one captures.
    captured->graph.capture_begin({0, 0}, cudaStreamCaptureModeThreadLocal);
    fill_awaited_tokens_cuda(captured->inputs.tokens, samples);
    record_forward(*captured);
    captured->graph.capture_end();
    return captured;
}

std::unique_ptr<step_graphs::captured_forward> step_graphs::capture_verify(std::int64_t sequences, bool measures_mass) {
    auto captured = std::make_unique<captured_forward>();
    captured->inputs = make_group_inputs(sequences, measures_mass, model::batch_kind::verify);
    captured->graph.capture_begin({0, 0}, cudaStreamCaptureModeThreadLocal);
    record_forward(*captured);
    captured->graph.capture_end();
    return captured;
}

// A capture cannot create what the forward creates at its first run, such as cuBLAS handles and RoPE tables, so an eager
// forward of each kind runs first, and an eager draft pass of each kind before the verify steps. Their tables point
// every view at line 0, which is zeroed first so that the forward reads no uninitialized memory; no request holds a
// line yet.
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
    for (bool measures_mass : mass_options) {
        shard_model.forward(make_group_inputs(1, measures_mass, model::batch_kind::decode), lines);
    }
    if (verifies) {
        for (bool measures_mass : mass_options) {
            shard_model.forward(make_group_inputs(1, measures_mass, model::batch_kind::verify), lines);
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
                decode_forwards.emplace(shape, capture_decode(sequences, measures_mass));
            }
            if (verifies && !verify_forwards.contains(shape)) {
                verify_forwards.emplace(shape, capture_verify(sequences, measures_mass));
            }
            if (drafts && !unfiltered_steps.contains(shape)) {
                unfiltered_steps.emplace(shape, capture_unfiltered_step(sequences, measures_mass));
            }
        }
    }
}

model::output step_graphs::replay(captured_forward& captured, const model::batch& g) {
    copy_group_inputs(captured.inputs, g);
    captured.graph.replay();
    copy_page_mass(g, captured.inputs);
    return model::output{.logits = captured.logits, .hidden = captured.hidden};
}

model::output step_graphs::decode(const model::batch& g) {
    const group_shape shape = shape_of(g);
    auto found = decode_forwards.find(shape);
    if (found == decode_forwards.end()) {
        found = decode_forwards.emplace(shape, capture_decode(shape.first, shape.second)).first;
    }
    return replay(*found->second, g);
}

model::output step_graphs::verify(const model::batch& g) {
    const group_shape shape = shape_of(g);
    auto found = verify_forwards.find(shape);
    if (found == verify_forwards.end()) {
        found = verify_forwards.emplace(shape, capture_verify(shape.first, shape.second)).first;
    }
    return replay(*found->second, g);
}

std::unique_ptr<step_graphs::captured_step>
step_graphs::capture_step(std::int64_t sequences, bool measures_mass, sampling_inputs&& sampling) {
    auto captured = std::make_unique<captured_step>();
    captured->inputs = make_group_inputs(sequences, measures_mass, model::batch_kind::verify);
    captured->sampling = std::move(sampling);
    const auto longs = lines.front().k.options().dtype(at::kLong);
    const std::int64_t rows = state_slots;
    const std::int64_t tokens = sequences * rows;
    const std::int64_t later = state_slots - 2;
    captured->max_accepted_token_counts = at::full({sequences}, rows, longs);
    captured->draft_positions = at::zeros({tokens}, longs);
    captured->draft_key_counts = at::full({sequences}, rows + 1, longs);
    captured->later_positions = at::zeros({later, sequences}, longs);
    captured->later_key_counts = at::ones({later, sequences}, longs);
    captured->first_pass = make_first_pass_inputs(sequences);
    captured->later_pass = make_later_pass_inputs(sequences);
    const model::batch& inputs = captured->inputs;
    const auto layers = static_cast<std::int64_t>(lines.size());
    auto every_layer = [&](const at::Tensor& counts) {
        return counts.to(at::kInt).unsqueeze(0).expand({layers, sequences});
    };
    draft_head& head = *shard_model.drafter();
    captured->graph.capture_begin({0, 0}, cudaStreamCaptureModeThreadLocal);
    model::output verified_rows = shard_model.forward(inputs, lines);
    at::Tensor samples = sample(verified_rows.logits, captured->sampling);
    at::Tensor accepted_counts = count_accepted_tokens(
        samples.view({sequences, rows}),
        inputs.tokens.view({sequences, rows}).narrow(1, 1, rows - 1),
        stop_tokens,
        captured->max_accepted_token_counts
    );
    at::Tensor drafting_rows = at::arange(sequences, longs) * rows + accepted_counts - 1;
    model::batch& first = captured->first_pass;
    first.tokens.copy_(samples);
    first.pos.copy_(captured->draft_positions);
    first.page_tables.copy_(inputs.page_tables);
    first.page_positions.copy_(inputs.page_positions);
    first.key_counts.copy_(every_layer(captured->draft_key_counts));
    first.logit_rows.copy_(drafting_rows);
    model::output drafted_rows = head.draft(first, verified_rows.hidden, lines);
    at::Tensor token = drafted_rows.logits.argmax(-1);
    at::Tensor previous_hidden = drafted_rows.hidden.index_select(0, drafting_rows);
    std::vector<at::Tensor> next{samples.index_select(0, drafting_rows), token};
    model::batch& one_row = captured->later_pass;
    for (std::int64_t j = 0; j < later; j++) {
        one_row.tokens.copy_(token);
        one_row.pos.copy_(captured->later_positions[j] + accepted_counts);
        one_row.page_tables.copy_(inputs.page_tables);
        one_row.page_positions.copy_(inputs.page_positions);
        one_row.key_counts.copy_(every_layer(captured->later_key_counts[j] + accepted_counts));
        model::output following = head.draft(one_row, previous_hidden, lines);
        token = following.logits.argmax(-1);
        previous_hidden = following.hidden;
        next.push_back(token);
    }
    captured->samples = samples;
    captured->accepted_token_counts = accepted_counts;
    captured->next_tokens = at::stack(next, 1);
    captured->graph.capture_end();
    return captured;
}

std::unique_ptr<step_graphs::captured_step>
step_graphs::capture_unfiltered_step(std::int64_t sequences, bool measures_mass) {
    return capture_step(sequences, measures_mass, make_sampling_inputs(sequences * state_slots));
}

std::unique_ptr<step_graphs::captured_step>
step_graphs::capture_filtered_step(std::int64_t sequences, bool measures_mass) {
    sampling_inputs sampling = make_sampling_inputs(sequences * state_slots);
    sampling.filters = true;
    return capture_step(sequences, measures_mass, std::move(sampling));
}

step_graphs::verify_step_outputs step_graphs::verify_step(
    const model::batch& g,
    const sampling_inputs& sampling,
    const at::Tensor& max_accepted_token_counts,
    const at::Tensor& draft_positions,
    const at::Tensor& draft_key_counts,
    const at::Tensor& later_positions,
    const at::Tensor& later_key_counts
) {
    assert(shard_model.drafter() != nullptr);
    const group_shape shape = shape_of(g);
    captured_step* captured = nullptr;
    if (sampling.filters) {
        auto found = filtered_steps.find(shape);
        if (found == filtered_steps.end()) {
            found = filtered_steps.emplace(shape, capture_filtered_step(shape.first, shape.second)).first;
        }
        captured = found->second.get();
    } else {
        auto found = unfiltered_steps.find(shape);
        if (found == unfiltered_steps.end()) {
            found = unfiltered_steps.emplace(shape, capture_unfiltered_step(shape.first, shape.second)).first;
        }
        captured = found->second.get();
    }
    copy_group_inputs(captured->inputs, g);
    copy_input(captured->sampling.temperatures, sampling.temperatures);
    copy_input(captured->sampling.top_k, sampling.top_k);
    copy_input(captured->sampling.top_p, sampling.top_p);
    copy_input(captured->sampling.draws, sampling.draws);
    copy_input(captured->max_accepted_token_counts, max_accepted_token_counts);
    copy_input(captured->draft_positions, draft_positions);
    copy_input(captured->draft_key_counts, draft_key_counts);
    copy_input(captured->later_positions, later_positions);
    copy_input(captured->later_key_counts, later_key_counts);
    captured->graph.replay();
    copy_page_mass(g, captured->inputs);
    return verify_step_outputs{
        .samples = captured->samples,
        .accepted_token_counts = captured->accepted_token_counts,
        .next_tokens = captured->next_tokens,
    };
}

}  // namespace pulsarlm
