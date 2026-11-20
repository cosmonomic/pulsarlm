#include "batch.hpp"

#include "pulsarlm/engine/executor.hpp"

#include <ATen/ATen.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace pulsarlm {

executor::batch::batch(executor& owner, std::span<const scheduled_request> scheduled) {
    for (const auto& [id, token_count] : scheduled) {
        request_state& req = owner.state_of(id);
        group& g = req.pending_token_list().size() > 1 ? prefill_group
            : !req.draft_tokens().empty()              ? verify_group
                                                       : decode_group;
        g.entries.push_back(
            entry{
                .req = request_ref(owner, id),
                .token_count = token_count,
                .is_sampling = token_count == req.pending_token_list().size() && req.may_sample(),
                .ends_cycle = req.ends_cycle(token_count),
            }
        );
    }
    if (!verify_group.entries.empty()) {
        verify_rows = forwarded(verify_group.entries.front());
        assert(std::ranges::all_of(verify_group.entries, [&](const entry& e) {
            return e.is_sampling && forwarded(e) == verify_rows;
        }));
    }
    for (group* g : groups()) {
        if (g->entries.empty()) {
            continue;
        }
        g->inputs = make_group(g->entries, owner.layout.page_size, owner.draft_state_slots, g->inputs.kind);
        g->slots = request_slots(g->entries);
        g->table_width = table_width(g->entries);
        assert(g->table_width <= owner.view_page_capacity);
        if (g != &prefill_group && owner.graph_pages) {
            g->is_replayed = true;
            g->table_width = *owner.graph_pages;
        }
    }
    std::vector<std::int64_t> slots;
    std::vector<std::int64_t> offsets;
    std::vector<std::int64_t> single_draws;
    for (group* g : groups()) {
        for (entry& e : g->entries) {
            e.first_sample_row = slots.size();
            if (!e.is_sampling) {
                continue;
            }
            const auto slot = static_cast<std::int64_t>(slot_of(e.req.key()));
            filters |= pulsarlm::filters(e.req->sampling_parameters());
            for (std::size_t offset = 0; offset < (g == &verify_group ? verify_rows : 1); offset++) {
                slots.push_back(slot);
                offsets.push_back(static_cast<std::int64_t>(offset));
            }
            if (g != &verify_group) {
                single_draws.push_back(slot);
            }
        }
    }
    sample_slots = pinned_values(slots);
    sample_offsets = pinned_values(offsets);
    single_draw_slots = pinned_values(single_draws);
    std::vector<std::int64_t> entry_request_slots;
    std::vector<std::int64_t> computed_counts;
    std::vector<std::int64_t> next_sources;
    for (const group* g : groups()) {
        for (const entry& e : g->entries) {
            const std::span<const std::int64_t> pending = e.req->pending_token_list();
            entry_request_slots.push_back(static_cast<std::int64_t>(slot_of(e.req.key())));
            computed_counts.push_back(g == &verify_group ? 0 : static_cast<std::int64_t>(e.token_count));
            if (e.is_sampling) {
                next_sources.push_back(-static_cast<std::int64_t>(e.first_sample_row) - 1);
            } else {
                next_sources.push_back(e.token_count < pending.size() ? pending[e.token_count] : 0);
            }
        }
    }
    entry_slots = pinned_values(entry_request_slots);
    computed_token_counts = pinned_values(computed_counts);
    next_token_sources = pinned_values(next_sources);
}

std::array<executor::batch::group*, 3> executor::batch::groups() noexcept {
    return {&decode_group, &verify_group, &prefill_group};
}

std::array<const executor::batch::group*, 3> executor::batch::groups() const noexcept {
    return {&decode_group, &verify_group, &prefill_group};
}

std::vector<executor::batch::entry> executor::batch::drafting_entries() const {
    std::vector<entry> drafting;
    for (const group* g : groups()) {
        for (const entry& e : g->entries) {
            if (e.draft && e.draft->is_drafting) {
                drafting.push_back(e);
            }
        }
    }
    return drafting;
}

std::size_t executor::batch::forwarded(const entry& e) noexcept {
    return e.token_count + e.req->draft_tokens().size();
}

model::batch executor::batch::make_group(
    std::span<const entry> group,
    std::size_t page_size,
    std::size_t state_slots,
    model::batch_kind kind
) {
    std::size_t token_count = 0;
    for (const entry& e : group) {
        token_count += forwarded(e);
    }
    std::size_t n_layers = group.front().req->views().size();
    auto n_tokens = static_cast<std::int64_t>(token_count);
    auto n_seqs = static_cast<std::int64_t>(group.size());
    auto n_kv_layers = static_cast<std::int64_t>(n_layers);
    model::batch g{
        .kind = kind,
        .page_size = static_cast<std::int64_t>(page_size),
        .state_slots = static_cast<std::int64_t>(state_slots),
    };
    if (kind == model::batch_kind::prefill) {
        g.tokens = pinned({n_tokens}, at::kLong);
        auto* tokens = g.tokens.mutable_data_ptr<std::int64_t>();
        for (const entry& e : group) {
            const std::span<const std::int64_t> pending = e.req->pending_token_list().first(e.token_count);
            assert(!std::ranges::contains(pending, request_state::awaited_token));
            tokens = std::ranges::copy(pending, tokens).out;
        }
    }
    if (kind != model::batch_kind::decode) {
        g.cu_seqlens_q = pinned({n_seqs + 1}, at::kInt);
        auto* query_start = g.cu_seqlens_q.mutable_data_ptr<std::int32_t>();
        query_start[0] = 0;
        for (std::size_t s = 0; s < group.size(); s++) {
            query_start[s + 1] = query_start[s] + static_cast<std::int32_t>(forwarded(group[s]));
        }
    }
    if (measures_mass(group)) {
        // A verify group's views are its rows'.
        const bool per_row = kind == model::batch_kind::verify;
        const std::int64_t views = per_row ? n_tokens : n_seqs;
        g.cu_view_pages = pinned({n_kv_layers, views + 1}, at::kInt);
        auto* cu_view_pages = g.cu_view_pages.mutable_data_ptr<std::int32_t>();
        for (std::size_t layer = 0; layer < n_layers; layer++) {
            std::int32_t* row = cu_view_pages + layer * static_cast<std::size_t>(views + 1);
            row[0] = 0;
            std::size_t view = 0;
            for (const entry& e : group) {
                const auto pages = static_cast<std::int32_t>(e.req->views()[layer].lines.size());
                for (std::size_t i = 0; i < (per_row ? forwarded(e) : 1); i++, view++) {
                    row[view + 1] = row[view] + pages;
                }
            }
        }
    }
    return g;
}

std::size_t executor::batch::table_width(std::span<const entry> group) noexcept {
    std::size_t width = 0;
    for (const entry& e : group) {
        width = std::max(width, e.req->table_width());
    }
    return width;
}

bool executor::batch::measures_mass(std::span<const entry> group) noexcept {
    return std::ranges::any_of(group, [](const entry& e) { return measures_mass(e); });
}

// A request whose previous step is launched may lie up to that step's drafts further on.
bool executor::batch::measures_mass(const entry& e) noexcept {
    const std::size_t lag = e.req->unfinished_step_count() > 0 ? e.req->draft_tokens().size() : 0;
    return e.req->eviction_policy_in_pages().can_evict() &&
        e.req->history_position() + forwarded(e) + lag > e.req->mass_start_position();
}

at::Tensor executor::batch::request_slots(std::span<const entry> group) {
    std::vector<std::int64_t> slots;
    for (const entry& e : group) {
        slots.push_back(static_cast<std::int64_t>(slot_of(e.req.key())));
    }
    return pinned_values(slots);
}

at::Tensor executor::batch::select_tables(const at::Tensor& slots, const at::Tensor& tables, std::size_t width) {
    return tables.narrow(2, 0, static_cast<std::int64_t>(width)).index_select(0, slots).transpose(0, 1).contiguous();
}

void executor::batch::plan_draft_rows(
    entry& e,
    bool verifies,
    std::size_t first_hidden_row,
    bool next_row,
    bool drafts
) {
    const request_state& req = *e.req;
    const auto sampled = static_cast<std::int64_t>(e.first_sample_row);
    const std::size_t first_draft_row = draft_tokens.size();
    if (verifies) {
        assert(!e.prior_hidden_index);
        for (std::size_t j = 0; j < verify_rows; j++) {
            draft_tokens.push_back(-(sampled + static_cast<std::int64_t>(j)) - 1);
            draft_position_offsets.push_back(static_cast<std::int64_t>(j + 1));
            draft_sources.push_back(static_cast<std::int64_t>(first_hidden_row + j));
        }
        e.draft = entry::draft_rows{
            .row_count = verify_rows,
            .logit_row = first_draft_row,
            .key_count_offset = verify_rows + 1,
            .is_drafting = drafts,
        };
        return;
    }
    const std::size_t start = req.history_position();
    const std::size_t first = req.reads_last_hidden() ? start : start + 1;
    const std::size_t next = start + e.token_count;
    const std::size_t end = next_row ? next + 1 : next;
    assert(next_row || !drafts);
    for (std::size_t t = first; t < end; t++) {
        if (t < next) {
            draft_tokens.push_back(req.pending_token_list()[t - start]);
        } else {
            draft_tokens.push_back(e.is_sampling ? -sampled - 1 : req.pending_token_list()[e.token_count]);
        }
        draft_position_offsets.push_back(static_cast<std::int64_t>(t - start));
        if (t == 0) {
            draft_sources.push_back(0);
        } else if (t == start) {
            assert(e.prior_hidden_index);
            draft_sources.push_back(static_cast<std::int64_t>(1 + *e.prior_hidden_index));
        } else {
            draft_sources.push_back(static_cast<std::int64_t>(first_hidden_row + t - 1 - start));
        }
    }
    if (end == first) {
        return;
    }
    e.draft = entry::draft_rows{
        .row_count = end - first,
        .logit_row = first_draft_row + end - first - 1,
        .key_count_offset = end - start,
        .is_drafting = drafts,
    };
}

}  // namespace pulsarlm
