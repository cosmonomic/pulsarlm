#include "pulsarlm/engine/sequence.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

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

at::Tensor pinned(at::IntArrayRef sizes, at::ScalarType dtype) {
    return at::empty(sizes, at::TensorOptions().dtype(dtype).pinned_memory(true));
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

sequence::sequence(
    request&& r,
    std::shared_ptr<const state>&& initial_state,
    state::lease_id_type lease_id,
    std::vector<digest>&& digests,
    const digest& root,
    const view_layout& layout,
    result_writer& writer,
    std::span<const at::Device> devices
)
: request_writer(writer),
  held_lease_id(lease_id),
  sampling(r.sampling),
  seed(sampling_seed(r.sampling.seed)),
  max_tokens(r.max_tokens),
  root(root),
  head(initial_state->head),
  prompt_digests(std::move(digests)),
  layout(layout),
  layers(layout.layers.size()),
  eviction(to_pages(r.eviction, layout.page_size)) {
    for (const at::Device& device : devices) {
        tables.push_back(
            at::zeros({static_cast<std::int64_t>(layers.size()), 0}, at::TensorOptions().dtype(at::kInt).device(device))
        );
        position_tables.push_back(at::zeros_like(tables.back()));
    }
    std::size_t resumed_page_count = 0;
    if (head) {
        auto head_digest = std::ranges::find(prompt_digests, *head);
        assert(head_digest != prompt_digests.end());
        resumed_page_count = static_cast<std::size_t>(head_digest - prompt_digests.begin()) + 1;
    }
    prompt_digests.erase(prompt_digests.begin(), prompt_digests.begin() + resumed_page_count);
    pending_tokens.assign(r.tokens.begin() + resumed_page_count * layout.page_size, r.tokens.end());
    assert(initial_state->tail.size() < pending_tokens.size());
    assert(std::ranges::equal(initial_state->tail, std::span(pending_tokens).first(initial_state->tail.size())));
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
    cached_count = history_pages * layout.page_size;
    std::size_t resumed_end = cached_count + initial_state->tail.size();
    next_cycle_end = (resumed_end / eviction.cycle_tokens + 1) * eviction.cycle_tokens;
    std::size_t window_start = next_cycle_end - eviction.mass_tokens;
    measured_end = resumed_end > window_start ? resumed_end : next_cycle_end - eviction.cycle_tokens;
    prompt_count = r.tokens.size();
    draft_rows_end = cached_count;
    update_held();
}

std::size_t sequence::next_index(std::size_t layer) const noexcept {
    return layers[layer].pages.size() * layout.page_size + tail.size();
}

std::size_t sequence::next_position() const noexcept {
    return held_pages.size() * layout.page_size + tail.size();
}

std::size_t sequence::page_position(std::size_t layer, std::size_t page) const {
    const layer_view& view = layers[layer];
    if (page >= view.pages.size()) {
        return (held_pages.size() + page - view.pages.size()) * layout.page_size;
    }
    auto rank = std::ranges::lower_bound(held_pages, view.indices[page]) - held_pages.begin();
    return static_cast<std::size_t>(rank) * layout.page_size;
}

void sequence::update_held() {
    held_pages.clear();
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        if (!layout.layers[layer].is_recurrent && !layout.layers[layer].mirrors_held_pages) {
            held_pages.insert(held_pages.end(), layers[layer].indices.begin(), layers[layer].indices.end());
        }
    }
    std::ranges::sort(held_pages);
    held_pages.erase(std::ranges::unique(held_pages).begin(), held_pages.end());
}

bool sequence::may_sample() const noexcept {
    return !max_tokens || generated_count < *max_tokens;
}

void sequence::advance(std::size_t count) {
    assert(count <= pending_tokens.size());
    tail.insert(tail.end(), pending_tokens.begin(), pending_tokens.begin() + count);
    pending_tokens.erase(pending_tokens.begin(), pending_tokens.begin() + count);
}

std::size_t sequence::complete_pages(std::size_t end) {
    std::size_t full_page_count = std::min(tail.size(), end - history_pages * layout.page_size) / layout.page_size;
    if (full_page_count == 0) {
        return 0;
    }
    std::vector<digest> full_page_digests;
    for (std::size_t page = 0; page < full_page_count; page++) {
        if (next_prompt_digest < prompt_digests.size()) {
            full_page_digests.push_back(prompt_digests[next_prompt_digest++]);
        } else {
            std::span<const std::int64_t> tokens = std::span(tail).subspan(page * layout.page_size, layout.page_size);
            assert(!std::ranges::contains(tokens, awaited_token));
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

std::size_t sequence::head_state_line_index(std::size_t layer) const {
    assert(layout.layers[layer].is_recurrent && history_pages > 0);
    return layout.layers[layer].state_slots + (history_pages - 1) % 2;
}

std::uint32_t sequence::head_state_line(std::size_t layer) const {
    return layers[layer].lines[head_state_line_index(layer)];
}

std::uint32_t sequence::replace_head_state_line(std::size_t layer, std::uint32_t line) {
    return std::exchange(layers[layer].lines[head_state_line_index(layer)], line);
}

void sequence::store_head_state() noexcept {
    state_pending = false;
}

void sequence::accept_tokens(std::size_t start, std::span<const std::int64_t> accepted_tokens) {
    assert(!accepted_tokens.empty() && start >= history_pages * layout.page_size);
    auto after = tail.begin() + static_cast<std::ptrdiff_t>(start - history_pages * layout.page_size) + 1;
    tail.insert(after, accepted_tokens.begin(), accepted_tokens.end() - 1);
    resolve_token(start + accepted_tokens.size(), accepted_tokens.back());
}

void sequence::withdraw_latest_verify_step() {
    assert(!tail.empty());
    pending_tokens.assign(1, tail.back());
    tail.pop_back();
    drafts.clear();
    unapplied_shift = 0;
    last_step_launched = false;
}

void sequence::resolve_token(std::size_t index, std::int64_t token) {
    std::size_t offset = index - history_pages * layout.page_size;
    std::int64_t& awaited = offset < tail.size() ? tail[offset] : pending_tokens[offset - tail.size()];
    assert(awaited == awaited_token);
    awaited = token;
}

void sequence::withdraw_latest_step() {
    assert(!tail.empty() && pending_tokens.size() <= 1);
    if (!pending_tokens.empty()) {
        generated_count--;
        draw_count--;
    }
    pending_tokens.assign(1, tail.back());
    tail.pop_back();
    last_step_launched = false;
}

std::size_t sequence::passed_pages(std::size_t layer, std::size_t query) const noexcept {
    const std::optional<std::size_t>& window = layout.layers[layer].sliding_window;
    if (!window || query + 1 <= *window) {
        return 0;
    }
    return (query + 1 - *window) / layout.page_size;
}

state sequence::to_state() const {
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

std::size_t sequence::history_position() const noexcept {
    return history_pages * layout.page_size + tail.size();
}

void sequence::start_next_cycle() noexcept {
    next_cycle_end += eviction.cycle_tokens;
}

std::size_t sequence::measured_start() const noexcept {
    return std::max(next_cycle_end - eviction.mass_tokens, measured_end);
}

void sequence::fill_query_weights(std::size_t first, std::span<float> weights) const noexcept {
    std::size_t unmeasured = std::min(weights.size(), std::max(measured_start(), first) - first);
    std::fill_n(weights.begin(), unmeasured, 0.0f);
    query_weights(weights.subspan(unmeasured), eviction.attention_mass_decay);
}

bool sequence::ends_cycle(std::size_t token_count) const noexcept {
    return history_position() + token_count >= next_cycle_end;
}

std::size_t sequence::cycle_tokens_left() const noexcept {
    std::size_t left = next_cycle_end - history_position();
    if (max_tokens) {
        // Every token sampled from here is forwarded but the last.
        std::size_t samples_left = *max_tokens - generated_count;
        left = std::min(left, pending_tokens.size() + (samples_left > 0 ? samples_left - 1 : 0));
    }
    return left;
}

std::size_t sequence::table_width() const noexcept {
    std::size_t width = 0;
    for (const layer_view& view : layers) {
        width = std::max(width, view.lines.size());
    }
    return width;
}

void sequence::upload_table(std::size_t first_page, std::span<const cudaStream_t> streams) {
    auto n_layers = static_cast<std::int64_t>(layers.size());
    auto width = static_cast<std::int64_t>(table_width());
    if (tables.front().size(1) < width) {
        std::int64_t capacity = std::max<std::int64_t>(width, 2 * tables.front().size(1));
        for (std::size_t shard = 0; shard < tables.size(); shard++) {
            tables[shard] = at::empty({n_layers, capacity}, tables[shard].options());
            position_tables[shard] = at::empty_like(tables[shard]);
        }
        first_page = 0;
    }
    auto page_count = width;
    auto first = static_cast<std::int64_t>(first_page);
    if (first == page_count) {
        return;
    }
    at::Tensor host_lines = pinned({n_layers, page_count - first}, at::kInt);
    at::Tensor host_positions = pinned({n_layers, page_count - first}, at::kInt);
    auto* line_entries = host_lines.mutable_data_ptr<std::int32_t>();
    auto* position_entries = host_positions.mutable_data_ptr<std::int32_t>();
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        const layer_view& view = layers[layer];
        for (std::int64_t page = first; page < page_count; page++) {
            auto index = static_cast<std::size_t>(page);
            bool used = index < view.lines.size();
            *line_entries++ = used ? static_cast<std::int32_t>(view.lines[index]) : 0;
            *position_entries++ = used && !layout.layers[layer].is_recurrent
                ? static_cast<std::int32_t>(page_position(layer, index))
                : 0;
        }
    }
    for (std::size_t shard = 0; shard < tables.size(); shard++) {
        c10::cuda::CUDAStreamGuard guard(
            c10::cuda::getStreamFromExternal(streams[shard], tables[shard].device().index())
        );
        tables[shard].narrow(1, first, page_count - first).copy_(host_lines, /*non_blocking=*/true);
        position_tables[shard].narrow(1, first, page_count - first).copy_(host_positions, /*non_blocking=*/true);
    }
}

std::size_t sequence::cycle_lines() const noexcept {
    assert(tail.empty());
    return ceil_div(cycle_tokens_left(), layout.page_size);
}

std::vector<std::size_t> sequence::missing_lines() const {
    std::size_t left = cycle_tokens_left();
    std::vector<std::size_t> counts;
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        std::size_t held_count = layers[layer].lines.size();
        std::size_t needed = layout.layers[layer].is_recurrent ? layout.layers[layer].state_slots + 2
                                                               : ceil_div(next_index(layer) + left, layout.page_size);
        counts.push_back(needed > held_count ? needed - held_count : 0);
    }
    return counts;
}

std::span<const layer_view> sequence::views() const noexcept {
    return layers;
}

std::span<const std::int64_t> sequence::tail_tokens() const noexcept {
    return tail;
}

std::span<const std::int64_t> sequence::pending_token_list() const noexcept {
    return pending_tokens;
}

std::span<const std::int64_t> sequence::draft_tokens() const noexcept {
    return drafts;
}

const sampling_params& sequence::sampling_parameters() const noexcept {
    return sampling;
}

sampling_draw sequence::next_draw() const noexcept {
    return sampling_draw{.seed = seed, .count = draw_count};
}

const std::optional<std::size_t>& sequence::token_limit() const noexcept {
    return max_tokens;
}

std::size_t sequence::generated_token_count() const noexcept {
    return generated_count;
}

std::size_t sequence::delivered_token_count() const noexcept {
    return delivered_count;
}

std::size_t sequence::prompt_token_count() const noexcept {
    return prompt_count;
}

std::size_t sequence::cached_token_count() const noexcept {
    return cached_count;
}

std::size_t sequence::device_cached_token_count() const noexcept {
    return device_cached_count;
}

std::size_t sequence::device_memory_peak() const noexcept {
    return peak_device_memory;
}

bool sequence::has_ended_prefill() const noexcept {
    return has_prefill_ended;
}

std::size_t sequence::prefill_step_count() const noexcept {
    return prefill_steps;
}

std::chrono::steady_clock::duration sequence::prefill_step_duration() const noexcept {
    return prefill_duration;
}

std::chrono::steady_clock::duration sequence::step_duration() const noexcept {
    return total_step_duration;
}

std::size_t sequence::drafted_token_count() const noexcept {
    return drafted_tokens;
}

std::size_t sequence::accepted_draft_count() const noexcept {
    return accepted_drafts;
}

std::size_t sequence::history_page_count() const noexcept {
    return history_pages;
}

const std::optional<digest>& sequence::head_digest() const noexcept {
    return head;
}

std::span<const std::size_t> sequence::held_page_indices() const noexcept {
    return held_pages;
}

bool sequence::has_pending_state() const noexcept {
    return state_pending;
}

const std::shared_ptr<pending_writes>& sequence::pending_page_writes() const noexcept {
    return page_writes;
}

const at::Tensor& sequence::line_table(std::size_t shard) const {
    return tables[shard];
}

const at::Tensor& sequence::position_table(std::size_t shard) const {
    return position_tables[shard];
}

bool sequence::has_exited() const noexcept {
    return end_reason.has_value();
}

std::optional<sequence::exit_reason> sequence::reason_of_exit() const noexcept {
    return end_reason;
}

std::size_t sequence::launched_step_count() const noexcept {
    return steps_in_flight;
}

bool sequence::has_launched_last_step() const noexcept {
    return last_step_launched;
}

std::size_t sequence::awaited_sample_row() const noexcept {
    return awaited_row;
}

std::size_t sequence::rejected_token_count() const noexcept {
    return rejected_tokens;
}

std::size_t sequence::end_of_draft_rows() const noexcept {
    return draft_rows_end;
}

std::size_t sequence::unapplied_position_shift() const noexcept {
    return unapplied_shift;
}

const page_eviction_policy& sequence::eviction_policy_in_pages() const noexcept {
    return eviction;
}

state::lease_id_type sequence::lease_id() const noexcept {
    return held_lease_id;
}

result_writer& sequence::writer() const noexcept {
    return request_writer;
}

std::vector<std::size_t> sequence::pages_to_evict(std::size_t layer) const {
    const layer_view& view = layers[layer];
    return pulsarlm::pages_to_evict(
        std::span(view.mass_sum).first(view.pages.size() * view.mass_heads),
        std::span(view.weight_sum).first(view.pages.size()),
        view.mass_heads,
        view.density_moments,
        view.score_moments,
        layout.page_size,
        eviction
    );
}

void sequence::launch_step(
    std::size_t token_count,
    const std::optional<std::size_t>& sample_row,
    std::size_t draft_row_count,
    std::size_t draft_count
) {
    rejected_tokens = 0;
    advance(token_count);
    if (sample_row) {
        pending_tokens.push_back(awaited_token);
        awaited_row = *sample_row;
        generated_count++;
        draw_count++;
    }
    last_step_launched = sample_row ? !may_sample() : pending_tokens.empty();
    draft_rows_end += draft_row_count;
    drafts.assign(draft_count, awaited_token);
    steps_in_flight++;
}

void sequence::launch_verify_step(std::size_t token_count, std::size_t draft_count) {
    advance(token_count);
    pending_tokens.push_back(awaited_token);
    rejected_tokens = drafts.size();
    drafts.assign(draft_count, awaited_token);
    steps_in_flight++;
}

void sequence::apply_sample(std::size_t position, std::int64_t token) {
    resolve_token(position, token);
    delivered_count++;
}

void sequence::apply_accepted_tokens(
    std::size_t start,
    std::span<const std::int64_t> accepted_tokens,
    std::size_t verified_count
) {
    const bool has_later_step = steps_in_flight > 1;
    const std::size_t accepted_count = accepted_tokens.size();
    accept_tokens(start, accepted_tokens);
    draw_count += accepted_count;
    generated_count += accepted_count;
    delivered_count += accepted_count;
    drafted_tokens += verified_count - 1;
    accepted_drafts += accepted_count - 1;
    unapplied_shift = has_later_step ? accepted_count - 1 : 0;
    if (!has_later_step) {
        rejected_tokens = verified_count - accepted_count;
        draft_rows_end = start + accepted_count + 1;
    }
}

void sequence::finish_step() noexcept {
    assert(steps_in_flight > 0);
    steps_in_flight--;
}

void sequence::record_step(std::chrono::steady_clock::duration duration) noexcept {
    if (!has_prefill_ended) {
        prefill_steps++;
        prefill_duration += duration;
    }
    total_step_duration += duration;
}

void sequence::end_prefill() noexcept {
    has_prefill_ended = true;
}

void sequence::set_drafts(std::span<const std::int64_t> tokens) {
    drafts.assign(tokens.begin(), tokens.end());
}

void sequence::add_lines(std::size_t layer, std::span<const std::uint32_t> added) {
    layers[layer].append_lines(added);
}

std::vector<std::uint32_t> sequence::remove_pages(std::size_t layer, std::span<const std::size_t> view_indices) {
    std::vector<std::uint32_t> removed_lines = layers[layer].remove(view_indices);
    update_held();
    return removed_lines;
}

std::vector<std::vector<std::uint32_t>> sequence::take_lines() {
    std::vector<std::vector<std::uint32_t>> taken_lines;
    for (layer_view& view : layers) {
        taken_lines.push_back(std::exchange(view.lines, {}));
    }
    return taken_lines;
}

void sequence::add_step_mass(
    std::span<const std::span<const float>> step_mass_sums,
    std::span<const std::span<const float>> step_weight_sums,
    std::size_t end
) {
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        layer_view& view = layers[layer];
        if (view.mass_heads > 0) {
            pulsarlm::add_step_mass(
                view.mass_sum,
                view.weight_sum,
                step_mass_sums[layer].first(view.mass_sum.size()),
                step_weight_sums[layer].first(view.weight_sum.size()),
                end - measured_end,
                eviction.attention_mass_decay
            );
        }
    }
    measured_end = end;
}

void sequence::sample_pages() {
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

void sequence::exit(exit_reason reason) noexcept {
    end_reason = reason;
}

void sequence::record_device_cached_tokens(std::size_t count) noexcept {
    device_cached_count = count;
}

void sequence::record_device_memory(std::size_t bytes) noexcept {
    peak_device_memory = std::max(peak_device_memory, bytes);
}

}  // namespace pulsarlm
