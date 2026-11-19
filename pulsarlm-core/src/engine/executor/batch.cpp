#include "batch.hpp"

#include "pulsarlm/engine/executor.hpp"
#include "pulsarlm/kernels/ops.hpp"

#include <ATen/ATen.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace pulsarlm {

executor::batch::batch(executor& owner, std::span<const batch_tokens> step_entries) {
    bool has_prefill = false;
    for (const auto& [id, token_count] : step_entries) {
        sequence& seq = owner.sequences.at(id);
        has_prefill = has_prefill || seq.pending_token_list().size() > 1;
        group& g = has_prefill ? prefill_group : seq.draft_tokens().empty() ? decode_group : verify_group;
        g.entries.push_back(
            entry{
                .seq = sequence_ref(owner, id),
                .token_count = token_count,
                .is_sampling = token_count == seq.pending_token_list().size() && seq.may_sample(),
                .ends_cycle = seq.ends_cycle(token_count),
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
        g->table_width = table_width(g->entries);
        if (g != &prefill_group && owner.graph_pages && g->table_width <= *owner.graph_pages) {
            g->is_replayed = true;
            g->table_width = *owner.graph_pages;
        }
    }
    awaits_tokens = std::ranges::any_of(decode_group.entries, [](const entry& e) {
        return e.seq->pending_token_list().front() == sequence::awaited_token;
    });
    // A verify entry's draws advance by the tokens its write-back accepts.
    for (group* g : groups()) {
        for (entry& e : g->entries) {
            e.first_sample_row = sample_params.size();
            if (!e.is_sampling) {
                continue;
            }
            const sampling_draw next_draw = e.seq->next_draw();
            for (std::size_t row = 0; row < (g == &verify_group ? verify_rows : 1); row++) {
                sample_params.push_back(e.seq->sampling_parameters());
                sample_draws.push_back(sampling_draw{.seed = next_draw.seed, .count = next_draw.count + row});
            }
        }
    }
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
    return e.token_count + e.seq->draft_tokens().size();
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
    std::size_t n_layers = group.front().seq->views().size();
    auto n_tokens = static_cast<std::int64_t>(token_count);
    auto n_seqs = static_cast<std::int64_t>(group.size());
    auto n_kv_layers = static_cast<std::int64_t>(n_layers);
    model::batch g{
        .kind = kind,
        .tokens = pinned({n_tokens}, at::kLong),
        .pos = pinned({n_tokens}, at::kLong),
    };
    at::Tensor key_counts = pinned({n_kv_layers, n_seqs}, at::kInt);
    at::Tensor query_starts = pinned({n_seqs + 1}, at::kInt);

    auto* tokens = g.tokens.mutable_data_ptr<std::int64_t>();
    auto* positions = g.pos.mutable_data_ptr<std::int64_t>();
    auto* key_count = key_counts.mutable_data_ptr<std::int32_t>();
    auto* query_start = query_starts.mutable_data_ptr<std::int32_t>();
    std::size_t first_row = 0;
    query_start[0] = 0;
    for (std::size_t s = 0; s < group.size(); s++) {
        const sequence& seq = *group[s].seq;
        std::size_t first_position = seq.next_position();
        const std::size_t count = forwarded(group[s]);
        for (std::size_t i = 0; i < count; i++) {
            const std::int64_t token = i < group[s].token_count ? seq.pending_token_list()[i]
                                                                : seq.draft_tokens()[i - group[s].token_count];
            tokens[first_row + i] = token == sequence::awaited_token
                ? -static_cast<std::int64_t>(seq.awaited_sample_row()) - 1
                : token;
            positions[first_row + i] = static_cast<std::int64_t>(first_position + i);
        }
        for (std::size_t layer = 0; layer < n_layers; layer++) {
            std::size_t start = seq.next_index(layer);
            key_count[layer * group.size() + s] = static_cast<std::int32_t>(start + count);
        }
        first_row += count;
        query_start[s + 1] = static_cast<std::int32_t>(first_row);
    }
    g.key_counts = key_counts;
    if (kind != model::batch_kind::decode) {
        g.cu_seqlens_q = query_starts;
    }
    g.history_positions = pinned({n_seqs}, at::kLong);
    auto* history_position = g.history_positions.mutable_data_ptr<std::int64_t>();
    for (std::size_t s = 0; s < group.size(); s++) {
        history_position[s] = static_cast<std::int64_t>(group[s].seq->history_position());
    }
    g.rejected_tokens = pinned({n_seqs}, at::kInt);
    auto* rejected = g.rejected_tokens.mutable_data_ptr<std::int32_t>();
    for (std::size_t s = 0; s < group.size(); s++) {
        rejected[s] = static_cast<std::int32_t>(group[s].seq->rejected_token_count());
    }
    g.page_size = static_cast<std::int64_t>(page_size);
    g.state_slots = static_cast<std::int64_t>(state_slots);
    if (measures_mass(group)) {
        // A verify group's views are its rows'.
        const bool per_row = kind == model::batch_kind::verify;
        const std::int64_t views = per_row ? n_tokens : n_seqs;
        g.cu_view_pages = pinned({n_kv_layers, views + 1}, at::kInt);
        g.query_weights = pinned({n_tokens}, at::kFloat);
        auto* cu_view_pages = g.cu_view_pages.mutable_data_ptr<std::int32_t>();
        for (std::size_t layer = 0; layer < n_layers; layer++) {
            std::int32_t* row = cu_view_pages + layer * static_cast<std::size_t>(views + 1);
            row[0] = 0;
            std::size_t view = 0;
            for (const entry& e : group) {
                const auto pages = static_cast<std::int32_t>(e.seq->views()[layer].lines.size());
                for (std::size_t i = 0; i < (per_row ? forwarded(e) : 1); i++, view++) {
                    row[view + 1] = row[view] + pages;
                }
            }
        }
        std::span<float> weights(g.query_weights.mutable_data_ptr<float>(), token_count);
        std::size_t first = 0;
        for (const entry& e : group) {
            const std::size_t count = forwarded(e);
            const std::size_t start = e.seq->history_position();
            if (per_row) {
                for (std::size_t i = 0; i < count; i++) {
                    weights[first + i] = start + i >= e.seq->measured_start() ? 1.0f : 0.0f;
                }
            } else {
                e.seq->fill_query_weights(start, weights.subspan(first, count));
            }
            first += count;
        }
    }
    return g;
}

std::size_t executor::batch::table_width(std::span<const entry> group) noexcept {
    std::size_t width = 0;
    for (const entry& e : group) {
        width = std::max(width, e.seq->table_width());
    }
    return width;
}

bool executor::batch::measures_mass(std::span<const entry> group) noexcept {
    return std::ranges::any_of(group, [](const entry& e) { return measures_mass(e); });
}

// A sequence whose previous step is launched may lie up to that step's drafts further on.
bool executor::batch::measures_mass(const entry& e) noexcept {
    const std::size_t lag = e.seq->launched_step_count() > 0 ? e.seq->draft_tokens().size() : 0;
    return e.seq->eviction_policy_in_pages().can_evict() &&
        e.seq->history_position() + forwarded(e) + lag > e.seq->measured_start();
}

at::Tensor executor::batch::page_tables(
    std::span<const entry> group,
    std::size_t n_kv_layers,
    bool positions,
    std::size_t shard,
    std::size_t width
) {
    std::vector<at::Tensor> tables;
    std::vector<std::size_t> page_counts;
    for (const entry& e : group) {
        tables.push_back(positions ? e.seq->position_table(shard) : e.seq->line_table(shard));
        page_counts.push_back(e.seq->table_width());
    }
    return gather_page_tables_cuda(
        tables,
        page_counts,
        static_cast<std::int64_t>(n_kv_layers),
        static_cast<std::int64_t>(width)
    );
}

void executor::batch::plan_draft_rows(
    entry& e,
    bool verifies,
    std::size_t first_hidden_row,
    bool next_row,
    bool drafts,
    std::size_t draft_layer,
    std::size_t draft_count
) {
    const sequence& seq = *e.seq;
    const std::size_t start = seq.history_position();
    const std::size_t next_index = seq.next_index(draft_layer);
    const std::size_t next_position = seq.next_position();
    const auto sampled = static_cast<std::int64_t>(e.first_sample_row);
    const std::size_t first_draft_row = draft_tokens.size();
    // The position after the row that drafts, before a verify entry's accepted count is added.
    std::size_t drafted_after = 0;
    if (verifies) {
        for (std::size_t j = 0; j < verify_rows; j++) {
            draft_tokens.push_back(-(sampled + static_cast<std::int64_t>(j)) - 1);
            draft_positions.push_back(static_cast<std::int64_t>(next_position + j + 1));
            draft_sources.push_back(static_cast<std::int64_t>(first_hidden_row + j));
        }
        e.draft = entry::draft_rows{
            .row_count = verify_rows,
            .logit_row = first_draft_row,
            .key_count = next_index + verify_rows + 1,
            .is_drafting = drafts,
        };
    } else {
        const std::size_t first = seq.end_of_draft_rows();
        const std::size_t next = start + e.token_count;
        const std::size_t end = next_row ? next + 1 : next;
        assert((first == start || first == start + 1) && (next_row || !drafts));
        for (std::size_t t = first; t < end; t++) {
            if (t < next) {
                draft_tokens.push_back(seq.pending_token_list()[t - start]);
            } else {
                draft_tokens.push_back(e.is_sampling ? -sampled - 1 : seq.pending_token_list()[e.token_count]);
            }
            draft_positions.push_back(static_cast<std::int64_t>(next_position + t - start));
            if (t == 0) {
                draft_sources.push_back(0);
            } else if (t == start) {
                assert(e.prior_row);
                draft_sources.push_back(static_cast<std::int64_t>(1 + *e.prior_row));
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
            .key_count = next_index + end - start,
            .is_drafting = drafts,
        };
        drafted_after = e.token_count;
    }
    if (drafts) {
        later_positions.resize(draft_count - 1);
        later_key_counts.resize(draft_count - 1);
        for (std::size_t j = 1; j < draft_count; j++) {
            later_positions[j - 1].push_back(static_cast<std::int64_t>(next_position + drafted_after + j));
            later_key_counts[j - 1].push_back(static_cast<std::int32_t>(next_index + drafted_after + j + 1));
        }
    }
}

}  // namespace pulsarlm
