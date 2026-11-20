#include "pulsarlm/engine/request_state.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace pulsarlm {

namespace {

std::size_t ceil_div(std::size_t n, std::size_t d) {
    return (n + d - 1) / d;
}

}  // namespace

std::vector<std::uint32_t> layer_view::remove(std::span<const std::size_t> view_indices) {
    std::vector<std::uint32_t> removed_lines;
    std::size_t kept_count = 0;
    std::size_t next_removed = 0;
    for (std::size_t page = 0; page < pages.size(); page++) {
        if (next_removed < view_indices.size() && view_indices[next_removed] == page) {
            removed_lines.push_back(lines[page]);
            next_removed++;
            continue;
        }
        pages[kept_count] = pages[page];
        indices[kept_count] = indices[page];
        lines[kept_count] = lines[page];
        std::copy_n(mass_sum.begin() + page * mass_heads, mass_heads, mass_sum.begin() + kept_count * mass_heads);
        weight_sum[kept_count] = weight_sum[page];
        kept_count++;
    }
    assert(next_removed == view_indices.size());
    pages.erase(pages.begin() + kept_count, pages.end());
    indices.erase(indices.begin() + kept_count, indices.end());
    lines.erase(lines.begin() + kept_count, lines.begin() + kept_count + view_indices.size());
    mass_sum.erase(
        mass_sum.begin() + kept_count * mass_heads,
        mass_sum.begin() + (kept_count + view_indices.size()) * mass_heads
    );
    weight_sum.erase(weight_sum.begin() + kept_count, weight_sum.begin() + kept_count + view_indices.size());
    return removed_lines;
}

void layer_view::append_lines(std::span<const std::uint32_t> added) {
    lines.insert(lines.end(), added.begin(), added.end());
    mass_sum.resize(lines.size() * mass_heads, 0.0f);
    weight_sum.resize(lines.size(), 0.0f);
}

std::size_t layer_view::count_held_lines() const noexcept {
    return lines.size() - static_cast<std::size_t>(std::ranges::count(lines, no_line));
}

request_state::request_state(
    request&& r,
    std::shared_ptr<const state>&& initial_state,
    state::lease_id_type lease_id,
    std::vector<digest>&& digests,
    const digest& root,
    const view_layout& layout
)
: held_lease_id(lease_id),
  sampling(r.sampling),
  seed(sampling_seed(r.sampling.seed)),
  max_tokens(r.max_tokens),
  prompt(std::move(r.tokens)),
  root(root),
  head(initial_state->head),
  prompt_digests(std::move(digests)),
  layout(layout),
  layers(layout.layers.size()),
  eviction(to_pages(r.eviction, layout.page_size)) {
    std::size_t resumed_page_count = 0;
    if (head) {
        auto head_digest = std::ranges::find(prompt_digests, *head);
        assert(head_digest != prompt_digests.end());
        resumed_page_count = static_cast<std::size_t>(head_digest - prompt_digests.begin()) + 1;
    }
    prompt_digests.erase(prompt_digests.begin(), prompt_digests.begin() + resumed_page_count);
    const std::span<const std::int64_t> unstored_tokens = std::span(prompt).subspan(
        resumed_page_count * layout.page_size
    );
    assert(initial_state->tail.size() < unstored_tokens.size());
    assert(std::ranges::equal(initial_state->tail, unstored_tokens.first(initial_state->tail.size())));
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        layers[layer].mass_heads = layout.layers[layer].mass_heads;
        layers[layer].density_moments.resize(layout.layers[layer].mass_heads);
    }
    for (std::size_t layer = 0; layer < initial_state->layers.size(); layer++) {
        if (layout.layers[layer].is_recurrent) {
            continue;
        }
        layer_view& view = layers[layer];
        const layer_sequence& sequence = initial_state->layers[layer];
        for (const active_page& page : sequence.pages) {
            view.pages.push_back(page.hash);
            view.indices.push_back(page.index);
            if (page.mass_sum.empty()) {
                view.mass_sum.resize(view.mass_sum.size() + view.mass_heads, 0.0f);
            } else {
                view.mass_sum.insert(view.mass_sum.end(), page.mass_sum.begin(), page.mass_sum.end());
            }
            view.weight_sum.push_back(page.weight_sum);
        }
        if (!sequence.density_moments.empty()) {
            view.density_moments = sequence.density_moments;
        }
        view.score_moments = sequence.score_moments;
    }
    for (const layer_sequence& sequence : initial_state->layers) {
        if (!sequence.pages.empty()) {
            history_pages = sequence.pages.back().index + 1;
        }
    }
    assert(history_pages == resumed_page_count);
    cached_count = history_pages * layout.page_size;
    position_bound = cached_count;
    prepared_prompt_tokens = cached_count;
    std::size_t resumed_end = cached_count + initial_state->tail.size();
    next_cycle_end = (resumed_end / eviction.cycle_tokens + 1) * eviction.cycle_tokens;
    std::size_t window_start = next_cycle_end - eviction.mass_tokens;
    mass_end_position = resumed_end > window_start ? resumed_end : next_cycle_end - eviction.cycle_tokens;
    resumed_mass_end_position = mass_end_position;
    update_held();
}

std::size_t request_state::next_index(std::size_t layer) const noexcept {
    return layers[layer].pages.size() * layout.page_size + tail.size();
}

std::size_t request_state::next_position() const noexcept {
    return held_pages.size() * layout.page_size + tail.size();
}

std::size_t request_state::page_position(std::size_t layer, std::size_t page) const {
    const layer_view& view = layers[layer];
    if (page >= view.pages.size()) {
        return (held_pages.size() + page - view.pages.size()) * layout.page_size;
    }
    auto rank = std::ranges::lower_bound(held_pages, view.indices[page]) - held_pages.begin();
    return static_cast<std::size_t>(rank) * layout.page_size;
}

void request_state::update_held() {
    held_pages.clear();
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        if (!layout.layers[layer].is_recurrent && !layout.layers[layer].mirrors_held_pages) {
            held_pages.insert(held_pages.end(), layers[layer].indices.begin(), layers[layer].indices.end());
        }
    }
    std::ranges::sort(held_pages);
    held_pages.erase(std::ranges::unique(held_pages).begin(), held_pages.end());
}

bool request_state::has_exact_prepared_counts() const noexcept {
    return written_back_step_count >= bounded_count_steps;
}

bool request_state::may_sample() const noexcept {
    return !max_tokens || !has_exact_prepared_counts() || prepared_sample_count() < *max_tokens;
}

std::size_t request_state::complete_pages() {
    std::size_t full_page_count = tail.size() / layout.page_size;
    if (full_page_count == 0) {
        return 0;
    }
    std::vector<digest> full_page_digests;
    for (std::size_t page = 0; page < full_page_count; page++) {
        if (next_prompt_digest < prompt_digests.size()) {
            full_page_digests.push_back(prompt_digests[next_prompt_digest++]);
        } else {
            std::span<const std::int64_t> tokens = std::span(tail).subspan(page * layout.page_size, layout.page_size);
            const digest& parent = full_page_digests.empty() ? head.value_or(root) : full_page_digests.back();
            full_page_digests.push_back(make_digest(parent, tokens));
        }
    }
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        if (!layout.layers[layer].is_recurrent) {
            layer_view& view = layers[layer];
            view.pages.insert(view.pages.end(), full_page_digests.begin(), full_page_digests.end());
            for (std::size_t i = 0; i < full_page_count; i++) {
                view.indices.push_back(history_pages + i);
            }
        }
    }
    if (std::ranges::contains(layout.layers, false, &view_layout::layer::is_recurrent)) {
        for (std::size_t i = 0; i < full_page_count; i++) {
            held_pages.push_back(history_pages + i);
        }
    }
    history_pages += full_page_count;
    head = full_page_digests.back();
    tail.erase(tail.begin(), tail.begin() + full_page_count * layout.page_size);
    if (std::ranges::contains(layout.layers, true, &view_layout::layer::is_recurrent)) {
        state_pending = true;
    }
    return full_page_count;
}

std::size_t request_state::head_state_line_index(std::size_t layer) const {
    assert(layout.layers[layer].is_recurrent && history_pages > 0);
    return layout.layers[layer].state_slots + (history_pages - 1) % 2;
}

std::uint32_t request_state::head_state_line(std::size_t layer) const {
    const std::uint32_t line = layers[layer].lines[head_state_line_index(layer)];
    assert(line != layer_view::no_line);
    return line;
}

std::uint32_t request_state::release_head_state_line(std::size_t layer) {
    const std::size_t parity = (history_pages - 1) % 2;
    for (std::size_t page = history_position() / layout.page_size;
         (page + 1) * layout.page_size <= prepared_position_bound();
         page++) {
        assert(page % 2 != parity);
    }
    const std::uint32_t line = std::exchange(layers[layer].lines[head_state_line_index(layer)], layer_view::no_line);
    assert(line != layer_view::no_line);
    return line;
}

void request_state::store_head_state() noexcept {
    state_pending = false;
}

std::size_t request_state::passed_pages(std::size_t layer, std::size_t query) const noexcept {
    const std::optional<std::size_t>& window = layout.layers[layer].sliding_window;
    if (!window || query + 1 <= *window) {
        return 0;
    }
    return (query + 1 - *window) / layout.page_size;
}

state request_state::to_state() const {
    state s{.head = head, .layers = std::vector<layer_sequence>(layers.size()), .tail = tail};
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        std::vector<active_page>& pages = s.layers[layer].pages;
        if (layout.layers[layer].is_recurrent) {
            if (head) {
                pages.push_back(active_page{.hash = *head, .index = history_pages - 1});
            }
            continue;
        }
        const layer_view& view = layers[layer];
        for (std::size_t page = passed_pages(layer, view.pages.size() * layout.page_size); page < view.pages.size();
             page++) {
            auto mass_sum = std::span(view.mass_sum).subspan(page * view.mass_heads, view.mass_heads);
            pages.push_back(
                active_page{
                    .hash = view.pages[page],
                    .index = view.indices[page],
                    .mass_sum = {mass_sum.begin(), mass_sum.end()},
                    .weight_sum = view.weight_sum[page],
                }
            );
        }
        s.layers[layer].density_moments = view.density_moments;
        s.layers[layer].score_moments = view.score_moments;
    }
    return s;
}

std::size_t request_state::history_position() const noexcept {
    return history_pages * layout.page_size + tail.size();
}

std::size_t request_state::cycle_end_position() const noexcept {
    return next_cycle_end;
}

void request_state::start_next_cycle() noexcept {
    next_cycle_end += eviction.cycle_tokens;
}

std::size_t request_state::mass_start_position() const noexcept {
    return std::max(next_cycle_end - eviction.mass_tokens, resumed_mass_end_position);
}

void request_state::fill_query_weights(std::size_t first, std::span<float> weights) const noexcept {
    std::size_t unmeasured = std::min(weights.size(), std::max(mass_start_position(), first) - first);
    std::fill_n(weights.begin(), unmeasured, 0.0f);
    query_weights(weights.subspan(unmeasured), eviction.attention_mass_decay);
}

bool request_state::ends_cycle(std::size_t token_count) const noexcept {
    return prepared_position_bound() + token_count >= next_cycle_end;
}

std::size_t request_state::cycle_tokens_left() const noexcept {
    assert(prepared_position_bound() <= next_cycle_end);
    std::size_t left = next_cycle_end - prepared_position_bound();
    if (max_tokens) {
        // Every token sampled from here is forwarded but the last.
        const std::size_t samples_left = *max_tokens - std::min(prepared_sample_count(), *max_tokens);
        left = std::min(left, unprepared_token_count() + (samples_left > 0 ? samples_left - 1 : 0));
    }
    return left;
}

std::size_t request_state::table_width() const noexcept {
    std::size_t width = 0;
    for (const layer_view& view : layers) {
        width = std::max(width, view.lines.size());
    }
    return width;
}

// The pages an unfinished verify step fills are known only at its write-back.
std::vector<std::size_t> request_state::find_unheld_state_lines(std::size_t layer, std::size_t key_count) const {
    const std::vector<std::uint32_t>& lines = layers[layer].lines;
    const std::size_t state_slots = layout.layers[layer].state_slots;
    assert(
        lines.size() == state_slots + 2 &&
        !std::ranges::contains(std::span(lines).first(state_slots), layer_view::no_line)
    );
    const std::size_t end = prepared_position_bound() + key_count;
    std::vector<std::size_t> view_indices;
    for (std::size_t page = history_position() / layout.page_size; (page + 1) * layout.page_size <= end; page++) {
        const std::size_t view_index = state_slots + page % 2;
        if (lines[view_index] == layer_view::no_line && !std::ranges::contains(view_indices, view_index)) {
            view_indices.push_back(view_index);
        }
    }
    return view_indices;
}

std::vector<std::size_t> request_state::missing_lines(std::span<const std::size_t> key_counts) const {
    const std::size_t unwritten_back_tokens = prepared_position_bound() - history_position();
    std::vector<std::size_t> counts;
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        const std::vector<std::uint32_t>& lines = layers[layer].lines;
        if (!layout.layers[layer].is_recurrent) {
            std::size_t needed = ceil_div(
                next_index(layer) + unwritten_back_tokens + key_counts[layer],
                layout.page_size
            );
            counts.push_back(needed > lines.size() ? needed - lines.size() : 0);
            continue;
        }
        counts.push_back(find_unheld_state_lines(layer, key_counts[layer]).size());
    }
    return counts;
}

std::span<const layer_view> request_state::views() const noexcept {
    return layers;
}

std::span<const std::int64_t> request_state::prompt_tokens() const noexcept {
    return prompt;
}

std::span<const std::int64_t> request_state::tail_tokens() const noexcept {
    return tail;
}

std::optional<std::int64_t> request_state::pending_token() const noexcept {
    const std::size_t position = history_position();
    return position < prompt.size() ? std::optional(prompt[position]) : pending_sample;
}

std::span<const std::int64_t> request_state::draft_tokens() const noexcept {
    return drafts;
}

std::size_t request_state::prepared_draft_count() const noexcept {
    return prepared_drafts;
}

const sampling_params& request_state::sampling_parameters() const noexcept {
    return sampling;
}

sampling_draw request_state::next_draw() const noexcept {
    return sampling_draw{.seed = seed, .count = delivered_tokens};
}

const std::optional<std::size_t>& request_state::token_limit() const noexcept {
    return max_tokens;
}

// Once every prepared step is written back, the steps' counts are exact.
std::size_t request_state::prepared_sample_count() const noexcept {
    return unfinished_step_count() == 0 ? delivered_tokens : prepared_samples;
}

std::size_t request_state::delivered_token_count() const noexcept {
    return delivered_tokens;
}

std::size_t request_state::prepared_position_bound() const noexcept {
    return unfinished_step_count() == 0 ? history_position() : position_bound;
}

std::size_t request_state::prepared_prompt_token_count() const noexcept {
    return prepared_prompt_tokens;
}

std::size_t request_state::unprepared_token_count() const noexcept {
    return prepared_prompt_tokens < prompt.size() ? prompt.size() - prepared_prompt_tokens : 1;
}

std::size_t request_state::prompt_token_count() const noexcept {
    return prompt.size();
}

std::size_t request_state::cached_token_count() const noexcept {
    return cached_count;
}

std::size_t request_state::device_cached_token_count() const noexcept {
    return device_cached_count;
}

std::size_t request_state::device_memory_peak() const noexcept {
    return peak_device_memory;
}

bool request_state::has_ended_prefill() const noexcept {
    return has_prefill_ended;
}

std::size_t request_state::prefill_step_count() const noexcept {
    return prefill_steps;
}

std::chrono::steady_clock::duration request_state::prefill_step_duration() const noexcept {
    return prefill_duration;
}

std::chrono::steady_clock::duration request_state::step_duration() const noexcept {
    return total_step_duration;
}

std::size_t request_state::drafted_token_count() const noexcept {
    return drafted_tokens;
}

std::size_t request_state::accepted_draft_count() const noexcept {
    return accepted_drafts;
}

std::size_t request_state::history_page_count() const noexcept {
    return history_pages;
}

const std::optional<digest>& request_state::head_digest() const noexcept {
    return head;
}

std::span<const std::size_t> request_state::held_page_indices() const noexcept {
    return held_pages;
}

bool request_state::has_pending_state() const noexcept {
    return state_pending;
}

const std::shared_ptr<pending_writes>& request_state::pending_page_writes() const noexcept {
    return page_writes;
}

bool request_state::has_exited() const noexcept {
    return end_reason.has_value();
}

std::optional<request_state::exit_reason> request_state::reason_of_exit() const noexcept {
    return end_reason;
}

bool request_state::has_exit_cause() const noexcept {
    return has_reached_end || has_failed_write || has_exceeded_context || is_out_of_lines || is_cancelled;
}

bool request_state::discards_steps() const noexcept {
    return has_reached_end || has_failed_write;
}

bool request_state::has_failed_store_write() const noexcept {
    return has_failed_write;
}

std::size_t request_state::unfinished_step_count() const noexcept {
    return prepared_step_count - written_back_step_count;
}

bool request_state::has_prepared_last_step() const noexcept {
    return last_step_prepared;
}

std::size_t request_state::rejected_token_count() const noexcept {
    return rejected_tokens;
}

bool request_state::reads_last_hidden() const noexcept {
    return next_step_reads_last_hidden;
}

const page_eviction_policy& request_state::eviction_policy_in_pages() const noexcept {
    return eviction;
}

state::lease_id_type request_state::lease_id() const noexcept {
    return held_lease_id;
}

std::vector<std::size_t> request_state::pages_to_evict(std::size_t layer) const {
    const layer_view& view = layers[layer];
    return pulsarlm::pages_to_evict(
        std::span(view.mass_sum).first(view.pages.size() * view.mass_heads),
        std::span(view.weight_sum).first(view.pages.size()),
        view.mass_heads,
        view.density_moments,
        view.score_moments,
        layout.page_size,
        tail.size(),
        eviction
    );
}

void request_state::prepare_step(
    std::size_t token_count,
    bool samples,
    std::size_t draft_count,
    bool continues_drafting
) {
    assert(!last_step_prepared && prepared_drafts == 0 && token_count > 0 && token_count <= unprepared_token_count());
    assert(
        unfinished_step_count() > 0 || (position_bound >= history_position() && prepared_samples >= delivered_tokens)
    );
    const std::size_t start = prepared_position_bound();
    const std::size_t samples_before = prepared_sample_count();
    const bool exact = has_exact_prepared_counts();
    if (prepared_prompt_tokens < prompt.size()) {
        prepared_prompt_tokens += token_count;
    }
    position_bound = start + token_count;
    prepared_samples = samples_before + (samples ? 1 : 0);
    prepared_step_count++;
    if (!exact) {
        bounded_count_steps = prepared_step_count;
    }
    last_step_prepared = samples ? exact && !may_sample() : prepared_prompt_tokens == prompt.size();
    next_step_reads_last_hidden = !continues_drafting;
    prepared_drafts = draft_count;
}

void request_state::prepare_verify_step(std::size_t draft_count) {
    assert(!last_step_prepared && prepared_drafts > 0 && prepared_prompt_tokens == prompt.size());
    assert(
        unfinished_step_count() > 0 || (position_bound >= history_position() && prepared_samples >= delivered_tokens)
    );
    const std::size_t verified_count = prepared_drafts + 1;
    position_bound = prepared_position_bound() + verified_count;
    prepared_samples = prepared_sample_count() + verified_count;
    prepared_step_count++;
    bounded_count_steps = prepared_step_count;
    next_step_reads_last_hidden = false;
    prepared_drafts = draft_count;
}

void request_state::apply_step(
    std::size_t forwarded_count,
    std::span<const std::int64_t> emitted,
    std::span<const std::int64_t> step_drafts,
    std::size_t rejected_count
) {
    std::size_t forwarded_emitted = 0;
    for (std::size_t i = 0; i < forwarded_count; i++) {
        const std::size_t position = history_position();
        if (position < prompt.size()) {
            tail.push_back(prompt[position]);
        } else if (pending_sample) {
            tail.push_back(*pending_sample);
            pending_sample.reset();
        } else {
            assert(forwarded_emitted < emitted.size());
            tail.push_back(emitted[forwarded_emitted++]);
        }
    }
    assert(emitted.size() - forwarded_emitted <= 1);
    if (forwarded_emitted < emitted.size()) {
        assert(!pending_sample);
        pending_sample = emitted.back();
    }
    delivered_tokens += emitted.size();
    drafts.assign(step_drafts.begin(), step_drafts.end());
    rejected_tokens = rejected_count;
}

void request_state::record_verified_drafts(std::size_t drafted_count, std::size_t accepted_count) noexcept {
    drafted_tokens += drafted_count;
    accepted_drafts += accepted_count;
}

void request_state::finish_step() noexcept {
    assert(unfinished_step_count() > 0);
    written_back_step_count++;
}

void request_state::reach_end() noexcept {
    has_reached_end = true;
}

void request_state::fail_store_write() noexcept {
    has_failed_write = true;
}

void request_state::run_out_of_lines() noexcept {
    is_out_of_lines = true;
}

void request_state::exceed_context() noexcept {
    has_exceeded_context = true;
}

void request_state::cancel() noexcept {
    is_cancelled = true;
}

void request_state::record_step(std::chrono::steady_clock::duration duration) noexcept {
    if (!has_prefill_ended) {
        prefill_steps++;
        prefill_duration += duration;
    }
    total_step_duration += duration;
}

void request_state::end_prefill() noexcept {
    has_prefill_ended = true;
}

void request_state::add_lines(std::size_t layer, std::span<const std::uint32_t> added) {
    layers[layer].append_lines(added);
}

void request_state::add_step_lines(
    std::span<const std::size_t> key_counts,
    std::span<const std::vector<std::uint32_t>> added
) {
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        if (!layout.layers[layer].is_recurrent) {
            layers[layer].append_lines(added[layer]);
            continue;
        }
        const std::vector<std::size_t> view_indices = find_unheld_state_lines(layer, key_counts[layer]);
        assert(view_indices.size() == added[layer].size());
        for (std::size_t i = 0; i < view_indices.size(); i++) {
            layers[layer].lines[view_indices[i]] = added[layer][i];
        }
    }
}

std::vector<std::uint32_t> request_state::remove_pages(std::size_t layer, std::span<const std::size_t> view_indices) {
    std::vector<std::uint32_t> removed_lines = layers[layer].remove(view_indices);
    update_held();
    return removed_lines;
}

std::vector<std::vector<std::uint32_t>> request_state::take_lines() {
    std::vector<std::vector<std::uint32_t>> taken_lines;
    for (layer_view& view : layers) {
        std::vector<std::uint32_t>& lines = taken_lines.emplace_back(std::exchange(view.lines, {}));
        std::erase(lines, layer_view::no_line);
    }
    return taken_lines;
}

void request_state::add_step_mass(
    std::span<const std::span<const float>> step_mass_sums,
    std::span<const std::span<const float>> step_weight_sums,
    std::size_t end
) {
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        layer_view& view = layers[layer];
        if (view.mass_heads > 0) {
            const std::size_t measured_lines = std::min(view.weight_sum.size(), step_weight_sums[layer].size());
            pulsarlm::add_step_mass(
                std::span(view.mass_sum).first(measured_lines * view.mass_heads),
                std::span(view.weight_sum).first(measured_lines),
                step_mass_sums[layer].first(measured_lines * view.mass_heads),
                step_weight_sums[layer].first(measured_lines),
                end - mass_end_position,
                eviction.attention_mass_decay
            );
        }
    }
    mass_end_position = end;
}

void request_state::sample_pages() {
    if (!eviction.can_evict()) {
        return;
    }
    for (layer_view& view : layers) {
        if (view.mass_heads == 0) {
            continue;
        }
        for (std::size_t page = 0; page < view.pages.size(); page++) {
            std::size_t index = view.indices[page];
            std::size_t page_end = (index + 1) * layout.page_size;
            if (index < eviction.n_sink || page_end + eviction.cycle_tokens > next_cycle_end ||
                page_end + 2 * eviction.cycle_tokens <= next_cycle_end) {
                continue;
            }
            add_sampled_page(
                view.density_moments,
                view.score_moments,
                std::span(view.mass_sum).subspan(page * view.mass_heads, view.mass_heads),
                view.weight_sum[page],
                eviction.eviction_warmup
            );
        }
    }
}

void request_state::exit() noexcept {
    assert(has_exit_cause() && unfinished_step_count() == 0 && !end_reason);
    if (has_failed_write) {
        end_reason = exit_reason::oom;
    } else if (has_reached_end) {
        end_reason = exit_reason::finished;
    } else if (has_exceeded_context) {
        end_reason = exit_reason::context_exceeded;
    } else if (is_out_of_lines) {
        end_reason = exit_reason::oom;
    } else {
        end_reason = exit_reason::cancelled;
    }
}

void request_state::record_device_cached_tokens(std::size_t count) noexcept {
    device_cached_count = count;
}

void request_state::record_device_memory(std::size_t bytes) noexcept {
    peak_device_memory = std::max(peak_device_memory, bytes);
}

}  // namespace pulsarlm
