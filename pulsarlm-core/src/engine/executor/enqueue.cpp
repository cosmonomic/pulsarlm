#include "batch.hpp"
#include "request_table_kernels.hpp"

#include "pulsarlm/engine/executor.hpp"
#include "pulsarlm/engine/sampling.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

namespace pulsarlm {

namespace {

at::Tensor to_device(const at::Tensor& host, at::Device device) {
    return host.defined() ? host.to(host.options().device(device), /*non_blocking=*/true) : host;
}

template <typename values_type>
at::Tensor tensor_on_device(const values_type& values, at::TensorOptions options, at::Device device) {
    return at::tensor(values, options).to(device, /*non_blocking=*/true);
}

// encoded's tokens, each -(r + 1) replaced by row r of samples.
at::Tensor resolve_tokens(const at::Tensor& encoded, const at::Tensor& samples) {
    if (!samples.defined() || samples.numel() == 0) {
        return encoded;
    }
    return at::where(encoded < 0, samples.index_select(0, (-encoded - 1).clamp_min(0)), encoded);
}

// Every KV layer's row of a draft pass's key counts, as the draft head reads only its own.
at::Tensor layer_key_counts(std::size_t n_layers, const at::Tensor& counts) {
    return counts.to(at::kInt).unsqueeze(0).expand({static_cast<std::int64_t>(n_layers), -1}).contiguous();
}

// int64 [slots]: the keys of KV layer layer's view in table before the next token of each request at slots.
at::Tensor layer_keys(const request_table& table, const at::Tensor& slots, std::size_t layer) {
    return table.key_counts.index_select(0, slots).select(1, static_cast<std::int64_t>(layer)).to(at::kLong);
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

std::pair<model::batch, at::Tensor> executor::batch::upload_inputs(
    std::span<const kv> lines,
    const std::vector<std::size_t>& mass_heads,
    std::size_t shard,
    const request_table& table
) {
    at::Device device = lines.front().k.device();
    model::batch on_device{
        .single_query_request_count = inputs.single_query_request_count,
        .per_query_state_request_count = inputs.per_query_state_request_count,
        .page_size = inputs.page_size,
        .state_slots = inputs.state_slots,
        .cu_seqlens_q = to_device(inputs.cu_seqlens_q, device),
        .cu_seqlens_q_from_second_run = to_device(inputs.cu_seqlens_q_from_second_run, device),
        .cu_seqlens_q_from_third_run = to_device(inputs.cu_seqlens_q_from_third_run, device),
        .cu_view_pages = to_device(inputs.cu_view_pages, device),
        .logit_tokens = to_device(inputs.logit_tokens, device),
    };
    gather_step_inputs(
        table,
        request_slots.to(device, /*non_blocking=*/true),
        table_width,
        static_cast<std::int64_t>(count_tokens()),
        to_device(prefill_tokens, device),
        on_device
    );
    if (per_query_state_request_count > 0) {
        verify_tokens[shard] = on_device.tokens.narrow(
            0,
            inputs.single_query_request_count,
            inputs.per_query_state_request_count * inputs.state_slots
        );
    }
    at::Tensor mass;
    if (inputs.cu_view_pages.defined()) {
        mass = zero_page_mass(on_device, inputs.cu_view_pages, mass_heads, device);
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
    at::Tensor& counters,
    request_table& table
) {
    at::Device device = lines.front().k.device();
    c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(stream, device.index()));
    write_tables(table);
    if (draft_layer && !prior_hidden_slots.empty()) {
        gather_prior_hidden(table, lines, *draft_layer, shard, device);
    }
    if (graphs != nullptr && verify_step_replays()) {
        enqueue_verify_step(mass_heads, shard, device, *graphs);
        return;
    }
    if (graphs != nullptr && !draft_layer && decode_step_replays()) {
        enqueue_decode_step(mass_heads, shard, device, *graphs);
        return;
    }
    const std::int64_t needed = shard_model.arrival_counter_count(
        static_cast<std::int64_t>(entries.size()),
        static_cast<std::int64_t>(count_tokens())
    );
    if (!counters.defined() || counters.numel() < needed) {
        counters = at::zeros({needed}, at::device(device).dtype(at::kInt));
    }
    auto [on_device, step_mass] = upload_inputs(lines, mass_heads, shard, table);
    on_device.arrival_counters = counters;
    const model::output output = shard_model.forward(on_device, lines);
    hidden_by_shard[shard] = output.hidden;
    if (step_mass.defined()) {
        mass_by_shard[shard] = pinned(step_mass.sizes(), at::kFloat);
        mass_by_shard[shard].copy_(step_mass, /*non_blocking=*/true);
    }
    at::Tensor samples;
    if (sample_slots.numel() > 0) {
        samples = sample(
            output.logits.index_select(0, sample_logit_indices.to(device, /*non_blocking=*/true)),
            gather_sampling_inputs(table, device)
        );
        if (shard == 0) {
            sampled_tokens = pinned({samples.size(0)}, at::kLong);
            sampled_tokens.copy_(samples, /*non_blocking=*/true);
        }
    }
    if (draft_layer && !draft_tokens.empty()) {
        enqueue_drafts(shard_model, lines, samples, stop_tokens, shard, counters, table, *draft_layer);
    }
    advance_draw_counts(table, shard, device);
    advance_table(table, samples, shard, device);
}

void executor::batch::write_tables(request_table& table) const {
    const at::Device device = table.page_tables.device();
    if (table_cells.defined()) {
        const at::Tensor cells = table_cells.to(device, /*non_blocking=*/true);
        table.page_tables.view({-1}).index_copy_(0, cells, table_lines.to(device, /*non_blocking=*/true));
        table.page_positions.view({-1}).index_copy_(0, cells, table_positions.to(device, /*non_blocking=*/true));
    }
    if (stale_slots.defined()) {
        const at::Tensor slots = stale_slots.to(device, /*non_blocking=*/true);
        for (at::Tensor request_table::* column : {
                 &request_table::history_positions,
                 &request_table::rope_positions,
                 &request_table::key_counts,
                 &request_table::rejected_tokens,
                 &request_table::tokens,
                 &request_table::cycle_end_positions,
                 &request_table::mass_start_positions,
                 &request_table::attention_mass_decays,
                 &request_table::seeds,
                 &request_table::draw_counts,
                 &request_table::temperatures,
                 &request_table::top_k,
                 &request_table::top_p,
             }) {
            (table.*column).index_copy_(0, slots, (stale_entries.*column).to(device, /*non_blocking=*/true));
        }
    }
}

sampling_inputs executor::batch::gather_sampling_inputs(const request_table& table, at::Device device) const {
    const at::Tensor slots = sample_slots.to(device, /*non_blocking=*/true);
    const at::Tensor offsets = sample_offsets.to(device, /*non_blocking=*/true);
    return sampling_inputs{
        .temperatures = table.temperatures.index_select(0, slots),
        .top_k = table.top_k.index_select(0, slots),
        .top_p = table.top_p.index_select(0, slots),
        .draws = at::stack({table.seeds.index_select(0, slots), table.draw_counts.index_select(0, slots) + offsets}),
        .filters = filters,
    };
}

void executor::batch::advance_draw_counts(request_table& table, std::size_t shard, at::Device device) const {
    if (single_draw_slots.numel() > 0) {
        const at::Tensor slots = single_draw_slots.to(device, /*non_blocking=*/true);
        table.draw_counts.index_add_(0, slots, at::ones_like(slots));
    }
    if (per_query_state_request_count > 0) {
        const at::Tensor verifying_slots = request_slots.narrow(
            0,
            static_cast<std::int64_t>(single_query_request_count),
            static_cast<std::int64_t>(per_query_state_request_count)
        );
        table.draw_counts
            .index_add_(0, verifying_slots.to(device, /*non_blocking=*/true), accepted_token_counts_by_shard[shard]);
    }
}

void executor::batch::gather_prior_hidden(
    const request_table& table,
    std::span<const kv> lines,
    std::size_t draft_layer,
    std::size_t shard,
    at::Device device
) {
    const kv& layer_lines = lines[draft_layer];
    const at::Tensor slots = tensor_on_device(prior_hidden_slots, at::TensorOptions().dtype(at::kLong), device);
    const at::Tensor pages = at::floor_divide(layer_keys(table, slots, draft_layer) - 1, layer_lines.k.size(1));
    const at::Tensor line_ids = table.page_tables.select(1, static_cast<std::int64_t>(draft_layer))
                                    .index_select(0, slots)
                                    .gather(1, pages.unsqueeze(1))
                                    .squeeze(1)
                                    .to(at::kLong);
    shard_prior_hidden[shard] = layer_lines.last_hidden.index_select(0, line_ids);
}

void executor::batch::advance_table(
    request_table& table,
    const at::Tensor& samples,
    std::size_t shard,
    at::Device device
) {
    const auto entry_count = static_cast<std::int64_t>(entries.size());
    const auto first_verify = static_cast<std::int64_t>(single_query_request_count);
    const auto verifying = static_cast<std::int64_t>(per_query_state_request_count);
    at::Tensor computed_counts = computed_token_counts.to(device, /*non_blocking=*/true);
    at::Tensor rejected_counts = at::zeros({entry_count}, computed_counts.options());
    at::Tensor next_sources = next_token_sources.to(device, /*non_blocking=*/true);
    if (verifying > 0) {
        const at::Tensor& accepted_counts = accepted_token_counts_by_shard[shard];
        computed_counts.narrow(0, first_verify, verifying).copy_(accepted_counts);
        rejected_counts.narrow(0, first_verify, verifying).copy_(inputs.state_slots - accepted_counts);
        next_sources.narrow(0, first_verify, verifying).sub_(accepted_counts - 1);
    }
    at::Tensor next = at::zeros({entry_count, table.tokens.size(1)}, computed_counts.options());
    next.select(1, 0).copy_(resolve_tokens(next_sources, samples));
    if (shard_next_tokens[shard].defined()) {
        std::vector<std::int64_t> drafting_positions;
        for (std::size_t index = 0; index < entries.size(); index++) {
            if (entries[index].draft && entries[index].draft->is_drafting) {
                drafting_positions.push_back(static_cast<std::int64_t>(index));
            }
        }
        next.index_copy_(
            0,
            tensor_on_device(drafting_positions, at::TensorOptions().dtype(at::kLong), device),
            shard_next_tokens[shard]
        );
    }
    at::Tensor ends_cycle = at::empty({entry_count}, computed_counts.options().dtype(at::kBool));
    advance_requests(
        table,
        request_slots.to(device, /*non_blocking=*/true),
        computed_counts,
        rejected_counts,
        next,
        ends_cycle
    );
    if (shard == 0) {
        cycle_end_flags = pinned({entry_count}, at::kBool);
        cycle_end_flags.copy_(ends_cycle, /*non_blocking=*/true);
    }
}

bool executor::batch::verify_step_replays() const noexcept {
    return per_query_state_request_count == entries.size() &&
        std::ranges::all_of(entries, [](const batch_entry& e) { return e.draft && e.draft->is_drafting; });
}

bool executor::batch::decode_step_replays() const noexcept {
    return single_query_request_count == entries.size() &&
        std::ranges::all_of(entries, [](const batch_entry& e) { return e.first_sample_index.has_value(); });
}

void executor::batch::enqueue_verify_step(
    const std::vector<std::size_t>& mass_heads,
    std::size_t shard,
    at::Device device,
    step_graphs& graphs
) {
    model::batch page_mass_rows;
    at::Tensor step_mass;
    if (inputs.cu_view_pages.defined()) {
        step_mass = zero_page_mass(page_mass_rows, inputs.cu_view_pages, mass_heads, device);
    }
    std::vector<std::int64_t> max_accepted_token_counts;
    for (const batch_entry& e : entries) {
        max_accepted_token_counts.push_back(static_cast<std::int64_t>(e.max_accepted_token_count));
    }
    const step_graphs::verify_step_outputs outputs = graphs.verify_step(
        request_slots,
        inputs.cu_view_pages,
        page_mass_rows.page_mass_sum,
        page_mass_rows.page_weight_sum,
        filters,
        pinned_values(max_accepted_token_counts)
    );
    if (step_mass.defined()) {
        mass_by_shard[shard] = pinned(step_mass.sizes(), at::kFloat);
        mass_by_shard[shard].copy_(step_mass, /*non_blocking=*/true);
    }
    if (shard == 0) {
        sampled_tokens = pinned(outputs.samples.sizes(), at::kLong);
        sampled_tokens.copy_(outputs.samples, /*non_blocking=*/true);
        accepted_token_counts = pinned(outputs.accepted_token_counts.sizes(), at::kLong);
        accepted_token_counts.copy_(outputs.accepted_token_counts, /*non_blocking=*/true);
        next_tokens = pinned(outputs.next_tokens.sizes(), at::kLong);
        next_tokens.copy_(outputs.next_tokens, /*non_blocking=*/true);
        cycle_end_flags = pinned(outputs.ends_cycle.sizes(), at::kBool);
        cycle_end_flags.copy_(outputs.ends_cycle, /*non_blocking=*/true);
    }
}

void executor::batch::enqueue_decode_step(
    const std::vector<std::size_t>& mass_heads,
    std::size_t shard,
    at::Device device,
    step_graphs& graphs
) {
    model::batch page_mass_rows;
    at::Tensor step_mass;
    if (inputs.cu_view_pages.defined()) {
        step_mass = zero_page_mass(page_mass_rows, inputs.cu_view_pages, mass_heads, device);
    }
    const step_graphs::decode_step_outputs outputs = graphs.decode_step(
        request_slots,
        inputs.cu_view_pages,
        page_mass_rows.page_mass_sum,
        page_mass_rows.page_weight_sum,
        filters
    );
    if (step_mass.defined()) {
        mass_by_shard[shard] = pinned(step_mass.sizes(), at::kFloat);
        mass_by_shard[shard].copy_(step_mass, /*non_blocking=*/true);
    }
    if (shard == 0) {
        sampled_tokens = pinned(outputs.samples.sizes(), at::kLong);
        sampled_tokens.copy_(outputs.samples, /*non_blocking=*/true);
        cycle_end_flags = pinned(outputs.ends_cycle.sizes(), at::kBool);
        cycle_end_flags.copy_(outputs.ends_cycle, /*non_blocking=*/true);
    }
}

void executor::batch::enqueue_drafts(
    model& shard_model,
    std::span<const kv> lines,
    const at::Tensor& samples,
    const at::Tensor& stop_tokens,
    std::size_t shard,
    at::Tensor& counters,
    const request_table& table,
    std::size_t draft_layer
) {
    at::Device device = lines.front().k.device();
    at::Tensor accepted_counts = count_accepted_tokens(samples, stop_tokens, shard, device);
    accepted_token_counts_by_shard[shard] = accepted_counts;
    const std::optional<first_draft> first = enqueue_first_draft_pass(
        shard_model,
        lines,
        samples,
        accepted_counts,
        shard,
        counters,
        table,
        draft_layer
    );
    if (first) {
        enqueue_later_draft_passes(shard_model, lines, *first, shard, counters, table);
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

void executor::batch::set_page_tables(
    std::span<const batch_entry> requests,
    model::batch& g,
    const request_table& table
) {
    std::size_t width = 0;
    for (const batch_entry& e : requests) {
        width = std::max(width, e.table_width);
    }
    const at::Tensor slots = collect_slots(requests).to(table.page_tables.device(), /*non_blocking=*/true);
    g.page_tables = select_tables(slots, table.page_tables, width);
    g.page_positions = select_tables(slots, table.page_positions, width);
}

at::Tensor executor::batch::count_accepted_tokens(
    const at::Tensor& samples,
    const at::Tensor& stop_tokens,
    std::size_t shard,
    at::Device device
) const {
    const auto verifying = static_cast<std::int64_t>(per_query_state_request_count);
    if (verifying == 0) {
        return at::empty({0}, at::device(device).dtype(at::kLong));
    }
    const auto longs = at::TensorOptions().dtype(at::kLong);
    const std::int64_t rows = inputs.state_slots;
    std::vector<std::int64_t> sample_indices;
    std::vector<std::int64_t> max_accepted_token_counts;
    for (const batch_entry& e : std::span(entries).subspan(single_query_request_count, per_query_state_request_count)) {
        for (std::int64_t j = 0; j < rows; j++) {
            sample_indices.push_back(static_cast<std::int64_t>(*e.first_sample_index) + j);
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
    std::size_t shard,
    at::Tensor& counters,
    const request_table& table,
    std::size_t draft_layer
) {
    at::Device device = lines.front().k.device();
    draft_head& head = *shard_model.drafter();
    const auto n_layers = lines.size();
    const auto longs = at::TensorOptions().dtype(at::kLong);
    const auto ints = at::TensorOptions().dtype(at::kInt);

    const auto verifying = static_cast<std::int64_t>(per_query_state_request_count);
    std::vector<at::Tensor> sources;
    if (shard_prior_hidden[shard].defined()) {
        sources.push_back(shard_prior_hidden[shard]);
    }
    if (hidden_by_shard[shard].defined()) {
        sources.push_back(hidden_by_shard[shard]);
    }
    sources.insert(sources.begin(), at::zeros({1, head.hidden_dim()}, sources.back().options()));
    // Per entry with draft rows, its position in the second run, -1 for another, and the tokens it computes, 0 for an
    // entry of the second run, whose accepted count is added on the GPU.
    std::vector<batch_entry> draft_entries;
    std::vector<std::int64_t> verify_indices;
    std::vector<std::int64_t> computed_counts;
    std::vector<std::int64_t> drafting_indices;
    std::vector<std::int32_t> starts{0};
    std::vector<std::int64_t> entry_rows;
    std::vector<std::int64_t> key_count_offsets;
    std::vector<std::int64_t> draft_logit_rows;
    for (std::size_t index = 0; index < entries.size(); index++) {
        const batch_entry& e = entries[index];
        if (!e.draft) {
            continue;
        }
        if (e.draft->is_drafting) {
            drafting_indices.push_back(static_cast<std::int64_t>(draft_entries.size()));
        }
        draft_entries.push_back(e);
        const bool verifying_entry = verifies(index);
        verify_indices.push_back(
            verifying_entry ? static_cast<std::int64_t>(index - single_query_request_count) : std::int64_t{-1}
        );
        computed_counts.push_back(verifying_entry ? 0 : static_cast<std::int64_t>(e.query_token_count));
        starts.push_back(starts.back() + static_cast<std::int32_t>(e.draft->row_count));
        entry_rows.push_back(static_cast<std::int64_t>(e.draft->row_count));
        key_count_offsets.push_back(static_cast<std::int64_t>(e.draft->key_count_offset));
        draft_logit_rows.push_back(static_cast<std::int64_t>(e.draft->logit_row));
    }
    // Per draft entry, its request's RoPE position and draft head KV layer's keys at the step's start, and the accepted
    // count - 1 of an entry of the second run.
    const auto draft_count_of_entries = static_cast<std::int64_t>(draft_entries.size());
    const at::Tensor slots = collect_slots(draft_entries).to(device, /*non_blocking=*/true);
    const at::Tensor positions = table.rope_positions.index_select(0, slots);
    const at::Tensor keys = layer_keys(table, slots, draft_layer);
    at::Tensor verify_index = tensor_on_device(verify_indices, longs, device);
    at::Tensor is_verify = verify_index.ge(0);
    at::Tensor accepted_after = verifying > 0
        ? at::where(is_verify, accepted_counts.index_select(0, verify_index.clamp_min(0)) - 1, 0)
        : at::zeros({draft_count_of_entries}, at::device(device).dtype(at::kLong));
    const at::Tensor query_starts = tensor_on_device(starts, ints, device);
    model::batch rows{
        .tokens = resolve_tokens(tensor_on_device(draft_tokens, longs, device), samples),
        .pos = at::repeat_interleave(
                   positions,
                   tensor_on_device(entry_rows, longs, device),
                   /*dim=*/0,
                   static_cast<std::int64_t>(draft_tokens.size())
               ) +
            tensor_on_device(draft_position_offsets, longs, device),
        .cu_seqlens_q = query_starts,
        .cu_seqlens_q_from_second_run = query_starts,
        .cu_seqlens_q_from_third_run = query_starts,
        .key_counts = layer_key_counts(n_layers, keys + tensor_on_device(key_count_offsets, longs, device)),
        .logit_tokens = tensor_on_device(draft_logit_rows, longs, device) + accepted_after,
    };
    set_page_tables(draft_entries, rows, table);
    at::Tensor hidden = at::cat(sources).index_select(0, tensor_on_device(draft_sources, longs, device));
    set_arrival_counters(shard_model, counters, device, rows, draft_count_of_entries);
    model::output drafted_rows = head.draft(rows, hidden, lines);
    if (drafting_indices.empty()) {
        return std::nullopt;
    }
    at::Tensor drafting_entries_index = tensor_on_device(drafting_indices, longs, device);
    at::Tensor drafting_rows = rows.logit_tokens.index_select(0, drafting_entries_index);
    at::Tensor computed = at::where(is_verify, accepted_after + 1, tensor_on_device(computed_counts, longs, device));
    at::Tensor token = drafted_rows.logits.index_select(0, drafting_entries_index).argmax(-1);
    at::Tensor previous_hidden = drafted_rows.hidden.index_select(0, drafting_rows);
    return first_draft{
        .tokens = {rows.tokens.index_select(0, drafting_rows), token},
        .hidden = previous_hidden,
        .last_positions = (positions + computed).index_select(0, drafting_entries_index),
        .last_key_counts = (keys + computed).index_select(0, drafting_entries_index),
    };
}

void executor::batch::enqueue_later_draft_passes(
    model& shard_model,
    std::span<const kv> lines,
    const first_draft& first,
    std::size_t shard,
    at::Tensor& counters,
    const request_table& table
) {
    at::Device device = lines.front().k.device();
    draft_head& head = *shard_model.drafter();
    const auto n_layers = lines.size();

    const std::vector<batch_entry> drafting = drafting_entries();
    const auto drafting_count = static_cast<std::int64_t>(drafting.size());
    const at::Tensor query_starts = at::arange(drafting_count + 1, at::device(device).dtype(at::kInt));
    const at::Tensor no_later_queries = at::zeros({1}, at::device(device).dtype(at::kInt));
    const at::Tensor logit_tokens = at::arange(drafting_count, at::device(device).dtype(at::kLong));
    at::Tensor token = first.tokens.back();
    at::Tensor previous_hidden = first.hidden;
    std::vector<at::Tensor> tokens = first.tokens;
    for (std::int64_t j = 1; j < static_cast<std::int64_t>(draft_count); j++) {
        model::batch one_row{
            .single_query_request_count = drafting_count,
            .tokens = token,
            .pos = first.last_positions + j,
            .cu_seqlens_q = query_starts,
            .cu_seqlens_q_from_second_run = no_later_queries,
            .cu_seqlens_q_from_third_run = no_later_queries,
            .key_counts = layer_key_counts(n_layers, first.last_key_counts + j + 1),
            .logit_tokens = logit_tokens,
        };
        set_page_tables(drafting, one_row, table);
        set_arrival_counters(shard_model, counters, device, one_row, static_cast<std::int64_t>(drafting.size()));
        model::output next = head.draft(one_row, previous_hidden, lines);
        token = next.logits.argmax(-1);
        previous_hidden = next.hidden;
        tokens.push_back(token);
    }
    shard_next_tokens[shard] = at::stack(tokens, 1);
}

}  // namespace pulsarlm
