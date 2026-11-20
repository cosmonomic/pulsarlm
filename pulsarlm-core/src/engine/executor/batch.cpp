#include "batch.hpp"

#include "pulsarlm/engine/executor.hpp"

#include <ATen/ATen.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace pulsarlm {

// A request with more than one pending token prefills; any other verifies its drafts or decodes its pending token.
executor::batch::batch(executor& owner, std::span<const scheduled_request> scheduled) {
    std::vector<batch_entry> verifying_entries;
    std::vector<batch_entry> prefilling_entries;
    for (const auto& [id, token_count] : scheduled) {
        const request_state& req = owner.get_state(id);
        const bool prefills = req.unprepared_token_count() > 1;
        const bool verifies = !prefills && req.prepared_draft_count() > 0;
        assert(!prefills || req.prepared_draft_count() == 0);
        const std::size_t query_token_count = token_count + req.prepared_draft_count();
        std::vector<batch_entry>& run = prefills ? prefilling_entries : verifies ? verifying_entries : entries;
        run.push_back(
            batch_entry{
                .request_id = id,
                .query_token_count = query_token_count,
                .first_sample_index = std::nullopt,
                .ends_cycle = req.ends_cycle(query_token_count),
                .table_width = req.table_width(),
            }
        );
    }
    single_query_request_count = entries.size();
    per_query_state_request_count = verifying_entries.size();
    entries.insert(entries.end(), verifying_entries.begin(), verifying_entries.end());
    entries.insert(entries.end(), prefilling_entries.begin(), prefilling_entries.end());
    for (const batch_entry& e : entries) {
        table_width = std::max(table_width, e.table_width);
    }
    assert(table_width <= owner.view_page_capacity);
    compose_inputs(owner);

    std::vector<std::int64_t> slots;
    std::vector<std::int64_t> offsets;
    std::vector<std::int64_t> logit_indices;
    std::vector<std::int64_t> single_draws;
    std::int64_t first_logit_index = 0;
    for (std::size_t index = 0; index < entries.size(); index++) {
        batch_entry& e = entries[index];
        const request_state& req = owner.get_state(e.request_id);
        const bool verifying_entry = verifies(index);
        const std::size_t token_count = verifying_entry ? 1 : e.query_token_count;
        const std::size_t logit_token_count = verifying_entry ? e.query_token_count : 1;
        assert(!verifying_entry || e.query_token_count == owner.draft_state_slots);
        if (token_count == req.unprepared_token_count() && req.may_sample()) {
            e.first_sample_index = slots.size();
            const auto slot = static_cast<std::int64_t>(get_slot(e.request_id));
            filters |= pulsarlm::filters(req.sampling_parameters());
            for (std::size_t offset = 0; offset < logit_token_count; offset++) {
                slots.push_back(slot);
                offsets.push_back(static_cast<std::int64_t>(offset));
                logit_indices.push_back(first_logit_index + static_cast<std::int64_t>(offset));
            }
            if (!verifying_entry) {
                single_draws.push_back(slot);
            }
        }
        assert(!verifying_entry || e.first_sample_index);
        first_logit_index += static_cast<std::int64_t>(logit_token_count);
    }
    sample_slots = pinned_values(slots);
    sample_offsets = pinned_values(offsets);
    sample_logit_indices = pinned_values(logit_indices);
    single_draw_slots = pinned_values(single_draws);
    std::vector<std::int64_t> computed_counts;
    std::vector<std::int64_t> next_sources;
    for (std::size_t index = 0; index < entries.size(); index++) {
        const batch_entry& e = entries[index];
        const request_state& req = owner.get_state(e.request_id);
        const std::size_t next_prompt_token = req.prepared_prompt_token_count() + e.query_token_count;
        computed_counts.push_back(verifies(index) ? 0 : static_cast<std::int64_t>(e.query_token_count));
        if (e.first_sample_index) {
            next_sources.push_back(-static_cast<std::int64_t>(*e.first_sample_index) - 1);
        } else {
            next_sources.push_back(
                next_prompt_token < req.prompt_token_count() ? req.prompt_tokens()[next_prompt_token] : 0
            );
        }
    }
    computed_token_counts = pinned_values(computed_counts);
    next_token_sources = pinned_values(next_sources);
    pack_host_inputs();
}

void executor::batch::pack_host_inputs() {
    constexpr std::size_t alignment = 16;
    const std::vector<at::Tensor*> packed{
        &inputs.cu_seqlens_q,
        &inputs.cu_seqlens_q_from_second_run,
        &inputs.cu_seqlens_q_from_third_run,
        &inputs.cu_view_pages,
        &inputs.logit_tokens,
        &request_slots,
        &prefill_tokens,
        &sample_slots,
        &sample_offsets,
        &sample_logit_indices,
        &single_draw_slots,
        &computed_token_counts,
        &next_token_sources,
    };
    std::vector<std::size_t> offsets;
    std::size_t byte_count = 0;
    for (const at::Tensor* host : packed) {
        offsets.push_back(byte_count);
        if (host->defined()) {
            byte_count += (host->nbytes() + alignment - 1) / alignment * alignment;
        }
    }
    host_input_bytes = pinned({static_cast<std::int64_t>(byte_count)}, at::kByte);
    for (std::size_t index = 0; index < packed.size(); index++) {
        at::Tensor& host = *packed[index];
        if (!host.defined()) {
            continue;
        }
        at::Tensor view = host_input_bytes.narrow(0, static_cast<std::int64_t>(offsets[index]), host.nbytes())
                              .view(host.scalar_type())
                              .view(host.sizes());
        view.copy_(host);
        host = view;
    }
}

at::Tensor executor::batch::get_on_device(const at::Tensor& host, std::size_t shard) const {
    const at::Tensor& device_bytes = device_input_bytes[shard];
    if (host.numel() == 0) {
        return at::empty(host.sizes(), host.options().device(device_bytes.device()).pinned_memory(false));
    }
    const auto offset = static_cast<std::int64_t>(
        static_cast<const std::byte*>(host.const_data_ptr()) -
        static_cast<const std::byte*>(host_input_bytes.const_data_ptr())
    );
    assert(offset >= 0 && offset + static_cast<std::int64_t>(host.nbytes()) <= host_input_bytes.numel());
    return device_bytes.narrow(0, offset, static_cast<std::int64_t>(host.nbytes()))
        .view(host.scalar_type())
        .view(host.sizes());
}

bool executor::batch::reaches_cycle_end(std::size_t index) const noexcept {
    assert(index < static_cast<std::size_t>(cycle_end_flags.numel()));
    return cycle_end_flags.const_data_ptr<bool>()[index];
}

bool executor::batch::verifies(std::size_t index) const noexcept {
    return index >= single_query_request_count && index < single_query_request_count + per_query_state_request_count;
}

std::size_t executor::batch::count_views_before(std::size_t index) const noexcept {
    const std::size_t step_requests = single_query_request_count + per_query_state_request_count;
    if (index <= single_query_request_count) {
        return index;
    }
    const auto state_slots = static_cast<std::size_t>(inputs.state_slots);
    if (index <= step_requests) {
        return single_query_request_count + (index - single_query_request_count) * state_slots;
    }
    return single_query_request_count + per_query_state_request_count * state_slots + index - step_requests;
}

std::size_t executor::batch::count_tokens() const noexcept {
    std::size_t token_count = 0;
    for (const batch_entry& e : entries) {
        token_count += e.query_token_count;
    }
    return token_count;
}

std::vector<executor::batch_entry> executor::batch::drafting_entries() const {
    std::vector<batch_entry> drafting;
    for (const batch_entry& e : entries) {
        if (e.draft && e.draft->is_drafting) {
            drafting.push_back(e);
        }
    }
    return drafting;
}

void executor::batch::compose_inputs(const executor& owner) {
    const std::size_t n_layers = owner.layout.layers.size();
    const auto n_seqs = static_cast<std::int64_t>(entries.size());
    const auto n_kv_layers = static_cast<std::int64_t>(n_layers);
    const auto single_queries = static_cast<std::int64_t>(single_query_request_count);
    const std::size_t step_requests = single_query_request_count + per_query_state_request_count;
    inputs = model::batch{
        .single_query_request_count = single_queries,
        .per_query_state_request_count = static_cast<std::int64_t>(per_query_state_request_count),
        .page_size = static_cast<std::int64_t>(owner.layout.page_size),
        .state_slots = static_cast<std::int64_t>(owner.draft_state_slots),
    };
    std::vector<std::int64_t> tokens;
    for (const batch_entry& e : std::span(entries).subspan(step_requests)) {
        const request_state& req = owner.get_state(e.request_id);
        const std::span<const std::int64_t> chunk = req.prompt_tokens().subspan(
            req.prepared_prompt_token_count(),
            e.query_token_count
        );
        tokens.insert(tokens.end(), chunk.begin(), chunk.end());
    }
    prefill_tokens = pinned_values(tokens);
    request_slots = collect_slots(entries);

    inputs.cu_seqlens_q = pinned({n_seqs + 1}, at::kInt);
    auto* query_start = inputs.cu_seqlens_q.mutable_data_ptr<std::int32_t>();
    query_start[0] = 0;
    for (std::size_t s = 0; s < entries.size(); s++) {
        query_start[s + 1] = query_start[s] + static_cast<std::int32_t>(entries[s].query_token_count);
    }
    auto query_starts_from = [&](std::size_t first) {
        at::Tensor starts = pinned({n_seqs - static_cast<std::int64_t>(first) + 1}, at::kInt);
        auto* start = starts.mutable_data_ptr<std::int32_t>();
        for (std::size_t s = first; s <= entries.size(); s++) {
            start[s - first] = query_start[s] - query_start[first];
        }
        return starts;
    };
    inputs.cu_seqlens_q_from_second_run = query_starts_from(single_query_request_count);
    inputs.cu_seqlens_q_from_third_run = query_starts_from(step_requests);

    std::vector<std::int64_t> logit_tokens;
    for (std::size_t s = 0; s < entries.size(); s++) {
        const std::int32_t first = verifies(s) ? query_start[s] : query_start[s + 1] - 1;
        for (std::int32_t token = first; token < query_start[s + 1]; token++) {
            logit_tokens.push_back(token);
        }
    }
    inputs.logit_tokens = pinned_values(logit_tokens);

    const bool step_measures_mass = std::ranges::any_of(entries, [&](const batch_entry& e) {
        return measures_mass(owner.get_state(e.request_id), e.query_token_count);
    });
    if (step_measures_mass) {
        const auto views = static_cast<std::int64_t>(count_views_before(entries.size()));
        inputs.cu_view_pages = pinned({n_kv_layers, views + 1}, at::kInt);
        auto* cu_view_pages = inputs.cu_view_pages.mutable_data_ptr<std::int32_t>();
        for (std::size_t layer = 0; layer < n_layers; layer++) {
            std::int32_t* row = cu_view_pages + layer * static_cast<std::size_t>(views + 1);
            row[0] = 0;
            std::size_t view = 0;
            for (std::size_t s = 0; s < entries.size(); s++) {
                const auto pages = static_cast<std::int32_t>(
                    owner.get_state(entries[s].request_id).views()[layer].lines.size()
                );
                for (std::size_t i = 0; i < (verifies(s) ? entries[s].query_token_count : 1); i++, view++) {
                    row[view + 1] = row[view] + pages;
                }
            }
        }
    }
}

bool executor::batch::measures_mass(const request_state& req, std::size_t query_token_count) noexcept {
    return req.eviction_policy_in_pages().can_evict() &&
        req.prepared_position_bound() + query_token_count > req.mass_start_position();
}

at::Tensor executor::batch::collect_slots(std::span<const batch_entry> entries) {
    std::vector<std::int64_t> slots;
    for (const batch_entry& e : entries) {
        slots.push_back(static_cast<std::int64_t>(get_slot(e.request_id)));
    }
    return pinned_values(slots);
}

at::Tensor executor::batch::select_tables(const at::Tensor& slots, const at::Tensor& tables, std::size_t width) {
    return tables.narrow(2, 0, static_cast<std::int64_t>(width)).index_select(0, slots).transpose(0, 1).contiguous();
}

void executor::batch::plan_draft_rows(
    batch_entry& e,
    const request_state& req,
    bool verifies,
    std::size_t first_hidden_row,
    std::optional<std::size_t> prior_hidden_index,
    bool next_row,
    bool drafts
) {
    const auto sampled = static_cast<std::int64_t>(e.first_sample_index.value_or(0));
    const std::size_t first_draft_row = draft_tokens.size();
    if (verifies) {
        assert(!prior_hidden_index && e.first_sample_index);
        for (std::size_t j = 0; j < e.query_token_count; j++) {
            draft_tokens.push_back(-(sampled + static_cast<std::int64_t>(j)) - 1);
            draft_position_offsets.push_back(static_cast<std::int64_t>(j + 1));
            draft_sources.push_back(static_cast<std::int64_t>(first_hidden_row + j));
        }
        e.draft = batch_entry::draft_rows{
            .row_count = e.query_token_count,
            .logit_row = first_draft_row,
            .key_count_offset = e.query_token_count + 1,
            .is_drafting = drafts,
        };
        return;
    }
    const std::size_t start = req.prepared_position_bound();
    const std::size_t first = req.reads_last_hidden() ? start : start + 1;
    const std::size_t next = start + e.query_token_count;
    const std::size_t end = next_row ? next + 1 : next;
    assert(next_row || !drafts);
    // The token at offset i from the step's first, a prompt token or the sample of a step written back.
    auto token_at = [&](std::size_t i) {
        const std::size_t prompt_index = req.prepared_prompt_token_count() + i;
        if (prompt_index < req.prompt_token_count()) {
            return req.prompt_tokens()[prompt_index];
        }
        assert(i == 0 && req.unfinished_step_count() == 0 && start == req.history_position());
        return *req.pending_token();
    };
    for (std::size_t t = first; t < end; t++) {
        if (t < next) {
            draft_tokens.push_back(token_at(t - start));
        } else {
            draft_tokens.push_back(e.first_sample_index ? -sampled - 1 : token_at(e.query_token_count));
        }
        draft_position_offsets.push_back(static_cast<std::int64_t>(t - start));
        if (t == 0) {
            draft_sources.push_back(0);
        } else if (t == start) {
            assert(prior_hidden_index);
            draft_sources.push_back(static_cast<std::int64_t>(1 + *prior_hidden_index));
        } else {
            draft_sources.push_back(static_cast<std::int64_t>(first_hidden_row + t - 1 - start));
        }
    }
    if (end == first) {
        return;
    }
    e.draft = batch_entry::draft_rows{
        .row_count = end - first,
        .logit_row = first_draft_row + end - first - 1,
        .key_count_offset = end - start,
        .is_drafting = drafts,
    };
}

}  // namespace pulsarlm
