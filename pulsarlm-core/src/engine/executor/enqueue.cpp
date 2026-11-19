#include "batch.hpp"

#include "pulsarlm/engine/executor.hpp"
#include "pulsarlm/engine/sampling.hpp"
#include "pulsarlm/kernels/sampling.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace pulsarlm {

namespace {

at::Tensor to_device(const at::Tensor& host, at::Device device) {
    return host.defined() ? host.to(host.options().device(device), /*non_blocking=*/true) : host;
}

model::batch to_device(const model::batch& host, at::Device device) {
    return model::batch{
        .kind = host.kind,
        .tokens = to_device(host.tokens, device),
        .pos = to_device(host.pos, device),
        .history_positions = to_device(host.history_positions, device),
        .rejected_tokens = to_device(host.rejected_tokens, device),
        .page_size = host.page_size,
        .state_slots = host.state_slots,
        .cu_seqlens_q = to_device(host.cu_seqlens_q, device),
        .key_counts = to_device(host.key_counts, device),
        .cu_view_pages = to_device(host.cu_view_pages, device),
        .query_weights = to_device(host.query_weights, device),
    };
}

}  // namespace

std::pair<model::batch, at::Tensor> executor::batch::upload_inputs(
    const group& g,
    std::span<const kv> lines,
    const std::vector<std::size_t>& mass_heads,
    std::size_t shard,
    const at::Tensor& samples
) {
    at::Device device = lines.front().k.device();
    model::batch on_device = to_device(g.inputs, device);
    on_device.page_tables = page_tables(g.entries, lines.size(), false, shard, g.table_width);
    on_device.page_positions = page_tables(g.entries, lines.size(), true, shard, g.table_width);
    if (&g == &decode_group) {
        if (awaits_tokens && !g.is_replayed) {
            fill_awaited_tokens_cuda(on_device.tokens, samples);
        }
    } else if (&g == &verify_group) {
        if (previous_batch) {
            take_previous_tokens(on_device, shard, device);
        }
        verify_tokens[shard] = on_device.tokens;
    } else {
        const at::Tensor& cu_seqlens_q = on_device.cu_seqlens_q;
        on_device.logit_rows = cu_seqlens_q.narrow(0, 1, cu_seqlens_q.size(0) - 1).to(at::kLong) - 1;
    }
    at::Tensor mass;
    if (g.inputs.cu_view_pages.defined()) {
        mass = zero_page_mass(on_device, g.inputs.cu_view_pages, mass_heads);
    }
    return {std::move(on_device), std::move(mass)};
}

void executor::batch::enqueue(
    model& shard_model,
    std::span<const kv> lines,
    const std::vector<std::size_t>& mass_heads,
    std::optional<std::size_t> draft_layer,
    const at::Tensor& stop_tokens,
    cudaStream_t stream,
    std::size_t shard,
    step_graphs* graphs,
    const at::Tensor& samples,
    at::Tensor& counters
) {
    at::Device device = lines.front().k.device();
    c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(stream, device.index()));
    if (draft_layer && !prior_lines.empty()) {
        at::Tensor line_ids = at::tensor(prior_lines, at::TensorOptions().dtype(at::kLong)).to(device, true);
        shard_prior_hidden[shard] = lines[*draft_layer].last_hidden.index_select(0, line_ids);
    }
    if (steps_replay()) {
        auto [on_device, mass] = upload_inputs(verify_group, lines, mass_heads, shard, samples);
        enqueue_step(on_device, mass, shard, device, graphs, samples);
        return;
    }
    // A replayed group reads its graph's own counters.
    std::int64_t needed = 0;
    for (const group* g : groups()) {
        if (!g->entries.empty() && !g->is_replayed) {
            needed = std::max(
                needed,
                shard_model
                    .arrival_counter_count(static_cast<std::int64_t>(g->entries.size()), g->inputs.tokens.size(0))
            );
        }
    }
    if (!counters.defined() || counters.numel() < needed) {
        counters = at::zeros({needed}, at::device(device).dtype(at::kInt));
    }
    std::vector<at::Tensor> rows;
    for (group* g : groups()) {
        if (g->entries.empty()) {
            continue;
        }
        auto [on_device, mass] = upload_inputs(*g, lines, mass_heads, shard, samples);
        on_device.arrival_counters = counters;
        model::output output;
        if (!g->is_replayed) {
            output = shard_model.forward(on_device, lines);
        } else if (g == &decode_group) {
            output = graphs->decode(on_device);
        } else {
            output = graphs->verify(on_device);
        }
        g->hidden_by_shard[shard] = output.hidden;
        if (mass.defined()) {
            g->mass_by_shard[shard] = pinned(mass.sizes(), at::kFloat);
            g->mass_by_shard[shard].copy_(mass, /*non_blocking=*/true);
        }
        const auto entry_rows = static_cast<std::int64_t>(g == &verify_group ? verify_rows : 1);
        for (std::size_t position = 0; position < g->entries.size(); position++) {
            if (g->entries[position].is_sampling) {
                rows.push_back(output.logits.narrow(0, static_cast<std::int64_t>(position) * entry_rows, entry_rows));
            }
        }
    }
    if (!rows.empty()) {
        at::Tensor tokens = previous_batch
            ? sample(at::cat(rows), sample_params, sample_draws, previous_draw_shifts(shard, device))
            : sample(at::cat(rows), sample_params, sample_draws);
        samples.narrow(0, 0, tokens.size(0)).copy_(tokens);
        if (shard == 0) {
            sampled_tokens = pinned({tokens.size(0)}, at::kLong);
            sampled_tokens.copy_(tokens, /*non_blocking=*/true);
        }
    }
    if (draft_layer && !draft_tokens.empty()) {
        enqueue_drafts(shard_model, lines, samples, stop_tokens, shard, counters);
    }
}

at::Tensor executor::batch::previous_draw_shifts(std::size_t shard, at::Device device) const {
    const auto longs = at::TensorOptions().dtype(at::kLong);
    std::vector<std::int64_t> rows;
    std::vector<std::int64_t> verify_entries;
    for (std::size_t v = 0; v < verify_group.entries.size(); v++) {
        const entry& e = verify_group.entries[v];
        if (!e.previous_verify_row) {
            continue;
        }
        for (std::size_t j = 0; j < verify_rows; j++) {
            rows.push_back(static_cast<std::int64_t>(e.first_sample_row + j));
            verify_entries.push_back(static_cast<std::int64_t>(v));
        }
    }
    at::Tensor shifts = at::zeros(
        {static_cast<std::int64_t>(sample_params.size())},
        at::device(device).dtype(at::kLong)
    );
    if (rows.empty()) {
        return shifts;
    }
    at::Tensor accepted_before = previous_shift(shard, device) + 1;
    return shifts.index_copy(
        0,
        at::tensor(rows, longs).to(device, /*non_blocking=*/true),
        accepted_before.index_select(0, at::tensor(verify_entries, longs).to(device, /*non_blocking=*/true))
    );
}

bool executor::batch::steps_replay() const noexcept {
    return verify_group.is_replayed && decode_group.entries.empty() && prefill_group.entries.empty() &&
        std::ranges::all_of(verify_group.entries, [](const entry& e) { return e.draft && e.draft->is_drafting; });
}

void executor::batch::enqueue_step(
    const model::batch& on_device,
    const at::Tensor& verify_mass,
    std::size_t shard,
    at::Device device,
    step_graphs* graphs,
    const at::Tensor& samples
) {
    const auto longs = at::TensorOptions().dtype(at::kLong);
    auto on_device_tensor = [&](const auto& values) {
        return at::tensor(values, longs).to(device, /*non_blocking=*/true);
    };
    const auto rows = static_cast<std::int64_t>(verify_rows);
    const auto sequences = static_cast<std::int64_t>(verify_group.entries.size());
    sampling_inputs sampling = sampling_inputs_of(sample_params, sample_draws, device);
    at::Tensor shift = at::zeros({sequences}, at::device(device).dtype(at::kLong));
    if (previous_batch) {
        sampling.draws.select(0, 1).add_(previous_draw_shifts(shard, device));
        shift = previous_shift(shard, device);
    }
    std::vector<std::int64_t> later_positions_flat;
    std::vector<std::int64_t> later_key_counts_flat;
    for (std::size_t j = 0; j < later_positions.size(); j++) {
        later_positions_flat.insert(later_positions_flat.end(), later_positions[j].begin(), later_positions[j].end());
        later_key_counts_flat
            .insert(later_key_counts_flat.end(), later_key_counts[j].begin(), later_key_counts[j].end());
    }
    const auto later = static_cast<std::int64_t>(later_positions.size());
    std::vector<std::int64_t> max_accepted_token_counts;
    std::vector<std::int64_t> key_counts;
    for (const entry& e : verify_group.entries) {
        max_accepted_token_counts.push_back(static_cast<std::int64_t>(e.max_accepted_token_count));
        key_counts.push_back(static_cast<std::int64_t>(e.draft->key_count));
    }
    const step_graphs::verify_step_outputs outputs = graphs->verify_step(
        on_device,
        sampling,
        on_device_tensor(max_accepted_token_counts),
        on_device_tensor(draft_positions) + at::repeat_interleave(shift, rows),
        on_device_tensor(key_counts) + shift,
        on_device_tensor(later_positions_flat).view({later, sequences}) + shift,
        on_device_tensor(later_key_counts_flat).view({later, sequences}) + shift
    );
    accepted_token_counts_by_shard[shard] = outputs.accepted_token_counts;
    shard_next_tokens[shard] = outputs.next_tokens;
    samples.narrow(0, 0, outputs.samples.size(0)).copy_(outputs.samples);
    if (verify_mass.defined()) {
        verify_group.mass_by_shard[shard] = pinned(verify_mass.sizes(), at::kFloat);
        verify_group.mass_by_shard[shard].copy_(verify_mass, /*non_blocking=*/true);
    }
    if (shard == 0) {
        sampled_tokens = pinned(outputs.samples.sizes(), at::kLong);
        sampled_tokens.copy_(outputs.samples, /*non_blocking=*/true);
        accepted_token_counts = pinned(outputs.accepted_token_counts.sizes(), at::kLong);
        accepted_token_counts.copy_(outputs.accepted_token_counts, /*non_blocking=*/true);
        next_tokens = pinned(outputs.next_tokens.sizes(), at::kLong);
        next_tokens.copy_(outputs.next_tokens, /*non_blocking=*/true);
    }
}

void executor::batch::take_previous_tokens(model::batch& g, std::size_t shard, at::Device device) const {
    const auto longs = at::TensorOptions().dtype(at::kLong);
    const auto rows = static_cast<std::int64_t>(verify_rows);
    const auto verifying = static_cast<std::int64_t>(verify_group.entries.size());
    std::vector<std::int64_t> entries_with_previous;
    std::vector<std::int64_t> drafting_rows;
    std::vector<std::int64_t> verify_mass_start_positions;
    for (std::size_t v = 0; v < verify_group.entries.size(); v++) {
        const entry& e = verify_group.entries[v];
        if (e.previous_drafting_row) {
            entries_with_previous.push_back(static_cast<std::int64_t>(v));
            drafting_rows.push_back(static_cast<std::int64_t>(*e.previous_drafting_row));
        }
        verify_mass_start_positions.push_back(static_cast<std::int64_t>(e.mass_start_position));
    }
    g.tokens.view({verifying, rows})
        .index_copy_(
            0,
            at::tensor(entries_with_previous, longs).to(device, /*non_blocking=*/true),
            previous_batch->shard_next_tokens[shard]
                .index_select(0, at::tensor(drafting_rows, longs).to(device, /*non_blocking=*/true))
        );
    at::Tensor shift = previous_shift(shard, device);
    g.pos += at::repeat_interleave(shift, rows);
    g.key_counts += shift.to(at::kInt).unsqueeze(0);
    g.history_positions += shift;
    g.rejected_tokens -= shift.to(at::kInt);
    if (g.query_weights.defined()) {
        at::Tensor row_positions = at::repeat_interleave(g.history_positions, rows) +
            at::arange(rows, g.history_positions.options()).repeat({verifying});
        at::Tensor mass_start_positions = at::tensor(verify_mass_start_positions, longs)
                                              .to(device, /*non_blocking=*/true);
        g.query_weights = row_positions.ge(at::repeat_interleave(mass_start_positions, rows)).to(at::kFloat);
    }
}

at::Tensor executor::batch::previous_shift(std::size_t shard, at::Device device) const {
    at::Tensor shift = at::zeros(
        {static_cast<std::int64_t>(verify_group.entries.size())},
        at::device(device).dtype(at::kLong)
    );
    std::vector<std::int64_t> entries_with_previous;
    std::vector<std::int64_t> previous_rows;
    for (std::size_t v = 0; v < verify_group.entries.size(); v++) {
        if (const std::optional<std::size_t>& previous_row = verify_group.entries[v].previous_verify_row) {
            entries_with_previous.push_back(static_cast<std::int64_t>(v));
            previous_rows.push_back(static_cast<std::int64_t>(*previous_row));
        }
    }
    if (entries_with_previous.empty()) {
        return shift;
    }
    const auto longs = at::TensorOptions().dtype(at::kLong);
    at::Tensor accepted_before = previous_batch->accepted_token_counts_by_shard[shard].index_select(
        0,
        at::tensor(previous_rows, longs).to(device, /*non_blocking=*/true)
    );
    return shift
        .index_copy(0, at::tensor(entries_with_previous, longs).to(device, /*non_blocking=*/true), accepted_before - 1);
}

void executor::batch::enqueue_drafts(
    model& shard_model,
    std::span<const kv> lines,
    const at::Tensor& samples,
    const at::Tensor& stop_tokens,
    std::size_t shard,
    at::Tensor& counters
) {
    at::Device device = lines.front().k.device();
    at::Tensor shift = previous_shift(shard, device);
    at::Tensor accepted_counts = count_accepted_tokens(samples, stop_tokens, shard, device);
    accepted_token_counts_by_shard[shard] = accepted_counts;
    const std::optional<first_draft>
        first = enqueue_first_draft_pass(shard_model, lines, samples, accepted_counts, shift, shard, counters);
    if (first) {
        enqueue_later_draft_passes(shard_model, lines, *first, shard, counters);
    }
    if (shard == 0) {
        accepted_token_counts = pinned(accepted_counts.sizes(), at::kLong);
        accepted_token_counts.copy_(accepted_counts, /*non_blocking=*/true);
        if (first) {
            next_tokens = pinned(shard_next_tokens[shard].sizes(), at::kLong);
            next_tokens.copy_(shard_next_tokens[shard], /*non_blocking=*/true);
        }
    }
}

namespace {

template <typename values_type>
at::Tensor tensor_on_device(const values_type& values, at::TensorOptions options, at::Device device) {
    return at::tensor(values, options).to(device, /*non_blocking=*/true);
}

// Every KV layer's row of a group's key counts, as the draft head reads only its own.
at::Tensor layer_key_counts(std::size_t n_layers, const at::Tensor& counts) {
    return counts.to(at::kInt).unsqueeze(0).expand({static_cast<std::int64_t>(n_layers), -1}).contiguous();
}

void set_arrival_counters(
    model& shard_model,
    at::Tensor& counters,
    at::Device device,
    model::batch& g,
    std::int64_t sequences
) {
    const std::int64_t needed = shard_model.arrival_counter_count(sequences, g.tokens.size(0));
    if (!counters.defined() || counters.numel() < needed) {
        counters = at::zeros({needed}, at::device(device).dtype(at::kInt));
    }
    g.arrival_counters = counters;
}

}  // namespace

void executor::batch::set_page_tables(
    std::span<const entry> group,
    model::batch& g,
    std::size_t n_layers,
    std::size_t shard
) {
    const std::size_t width = table_width(group);
    g.page_tables = page_tables(group, n_layers, false, shard, width);
    g.page_positions = page_tables(group, n_layers, true, shard, width);
}

at::Tensor executor::batch::count_accepted_tokens(
    const at::Tensor& samples,
    const at::Tensor& stop_tokens,
    std::size_t shard,
    at::Device device
) const {
    const auto verifying = static_cast<std::int64_t>(verify_group.entries.size());
    if (verifying == 0) {
        return at::empty({0}, at::device(device).dtype(at::kLong));
    }
    const auto longs = at::TensorOptions().dtype(at::kLong);
    const auto rows = static_cast<std::int64_t>(verify_rows);
    std::vector<std::int64_t> sample_indices;
    std::vector<std::int64_t> max_accepted_token_counts;
    for (const entry& e : verify_group.entries) {
        for (std::size_t j = 0; j < verify_rows; j++) {
            sample_indices.push_back(static_cast<std::int64_t>(e.first_sample_row + j));
        }
        max_accepted_token_counts.push_back(static_cast<std::int64_t>(e.max_accepted_token_count));
    }
    return pulsarlm::count_accepted_tokens(
        samples.index_select(0, tensor_on_device(sample_indices, longs, device)).view({verifying, rows}),
        verify_tokens[shard].view({verifying, rows}).narrow(1, 1, rows - 1),
        stop_tokens,
        tensor_on_device(max_accepted_token_counts, longs, device)
    );
}

std::optional<executor::batch::first_draft> executor::batch::enqueue_first_draft_pass(
    model& shard_model,
    std::span<const kv> lines,
    const at::Tensor& samples,
    const at::Tensor& accepted_counts,
    const at::Tensor& shift,
    std::size_t shard,
    at::Tensor& counters
) {
    at::Device device = lines.front().k.device();
    draft_head& head = *shard_model.drafter();
    const auto n_layers = lines.size();
    const auto longs = at::TensorOptions().dtype(at::kLong);
    const auto ints = at::TensorOptions().dtype(at::kInt);

    const auto verifying = static_cast<std::int64_t>(verify_group.entries.size());
    std::vector<at::Tensor> sources;
    if (shard_prior_hidden[shard].defined()) {
        sources.push_back(shard_prior_hidden[shard]);
    }
    for (const group* g : groups()) {
        if (g->hidden_by_shard[shard].defined()) {
            sources.push_back(g->hidden_by_shard[shard]);
        }
    }
    sources.insert(sources.begin(), at::zeros({1, head.hidden_dim()}, sources.back().options()));
    // Per entry with draft rows, its position among the verify entries, -1 for another.
    std::vector<entry> draft_entries;
    std::vector<std::int64_t> verify_indices;
    std::vector<std::int64_t> drafting_indices;
    std::vector<std::int32_t> starts{0};
    std::vector<std::int64_t> entry_rows;
    std::vector<std::int64_t> draft_key_counts;
    std::vector<std::int64_t> draft_logit_rows;
    for (const group* g : groups()) {
        for (std::size_t position = 0; position < g->entries.size(); position++) {
            const entry& e = g->entries[position];
            if (!e.draft) {
                continue;
            }
            if (e.draft->is_drafting) {
                drafting_indices.push_back(static_cast<std::int64_t>(draft_entries.size()));
            }
            draft_entries.push_back(e);
            verify_indices.push_back(g == &verify_group ? static_cast<std::int64_t>(position) : -1);
            starts.push_back(starts.back() + static_cast<std::int32_t>(e.draft->row_count));
            entry_rows.push_back(static_cast<std::int64_t>(e.draft->row_count));
            draft_key_counts.push_back(static_cast<std::int64_t>(e.draft->key_count));
            draft_logit_rows.push_back(static_cast<std::int64_t>(e.draft->logit_row));
        }
    }
    // Per draft entry, the accepted count - 1 of a verify entry, and the previous step's shift of its positions.
    const auto draft_count = static_cast<std::int64_t>(draft_entries.size());
    at::Tensor verify_index = tensor_on_device(verify_indices, longs, device);
    at::Tensor is_verify = verify_index.ge(0);
    at::Tensor clamped = verify_index.clamp_min(0);
    at::Tensor accepted_after = verifying > 0 ? at::where(is_verify, accepted_counts.index_select(0, clamped) - 1, 0)
                                              : at::zeros({draft_count}, at::device(device).dtype(at::kLong));
    at::Tensor entry_shift = verifying > 0 ? at::where(is_verify, shift.index_select(0, clamped), 0)
                                           : at::zeros({draft_count}, at::device(device).dtype(at::kLong));
    at::Tensor row_shift = at::repeat_interleave(
        entry_shift,
        tensor_on_device(entry_rows, longs, device),
        /*dim=*/0,
        static_cast<std::int64_t>(draft_tokens.size())
    );
    model::batch rows{
        .kind = model::batch_kind::prefill,
        .tokens = tensor_on_device(draft_tokens, longs, device),
        .pos = tensor_on_device(draft_positions, longs, device) + row_shift,
        .cu_seqlens_q = tensor_on_device(starts, ints, device),
        .key_counts = layer_key_counts(n_layers, tensor_on_device(draft_key_counts, longs, device) + entry_shift),
        .logit_rows = tensor_on_device(draft_logit_rows, longs, device) + accepted_after,
    };
    fill_awaited_tokens_cuda(rows.tokens, samples);
    set_page_tables(draft_entries, rows, n_layers, shard);
    at::Tensor hidden = at::cat(sources).index_select(0, tensor_on_device(draft_sources, longs, device));
    set_arrival_counters(shard_model, counters, device, rows, draft_count);
    model::output drafted_rows = head.draft(rows, hidden, lines);
    if (drafting_indices.empty()) {
        return std::nullopt;
    }
    at::Tensor drafting_entries_index = tensor_on_device(drafting_indices, longs, device);
    at::Tensor drafting_rows = rows.logit_rows.index_select(0, drafting_entries_index);
    at::Tensor later_shift = (accepted_after + 1).mul(is_verify.to(at::kLong)) + entry_shift;
    later_shift = later_shift.index_select(0, drafting_entries_index);
    at::Tensor token = drafted_rows.logits.index_select(0, drafting_entries_index).argmax(-1);
    at::Tensor previous_hidden = drafted_rows.hidden.index_select(0, drafting_rows);
    return first_draft{
        .tokens = {rows.tokens.index_select(0, drafting_rows), token},
        .hidden = previous_hidden,
        .later_shift = later_shift,
    };
}

void executor::batch::enqueue_later_draft_passes(
    model& shard_model,
    std::span<const kv> lines,
    const first_draft& first,
    std::size_t shard,
    at::Tensor& counters
) {
    at::Device device = lines.front().k.device();
    draft_head& head = *shard_model.drafter();
    const auto n_layers = lines.size();
    const auto longs = at::TensorOptions().dtype(at::kLong);

    const std::vector<entry> drafting = drafting_entries();
    at::Tensor token = first.tokens.back();
    at::Tensor previous_hidden = first.hidden;
    std::vector<at::Tensor> tokens = first.tokens;
    for (std::size_t j = 0; j < later_positions.size(); j++) {
        model::batch one_row{
            .kind = model::batch_kind::decode,
            .tokens = token,
            .pos = tensor_on_device(later_positions[j], longs, device) + first.later_shift,
            .key_counts = layer_key_counts(
                n_layers,
                tensor_on_device(later_key_counts[j], longs, device) + first.later_shift
            ),
        };
        set_page_tables(drafting, one_row, n_layers, shard);
        set_arrival_counters(shard_model, counters, device, one_row, static_cast<std::int64_t>(drafting.size()));
        model::output next = head.draft(one_row, previous_hidden, lines);
        token = next.logits.argmax(-1);
        previous_hidden = next.hidden;
        tokens.push_back(token);
    }
    shard_next_tokens[shard] = at::stack(tokens, 1);
}

}  // namespace pulsarlm
