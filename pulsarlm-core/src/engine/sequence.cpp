#include "pulsarlm/engine/sequence.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cassert>
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

lease_guard::lease_guard(page_store& store, state::lease_id_type lease_id) noexcept
: store(store),
  lease_id(lease_id) {}

lease_guard::~lease_guard() {
    release();
}

state::lease_id_type lease_guard::id() const noexcept {
    return lease_id;
}

void lease_guard::release() noexcept {
    if (held) {
        held = false;
        store.release(lease_id);
    }
}

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
    page_store& store,
    state::lease_id_type lease_id,
    std::vector<digest>&& digests,
    const digest& root,
    const executor& owner,
    result_writer& writer,
    std::span<const at::Device> devices
)
: writer(writer),
  lease(store, lease_id),
  last_state(std::move(initial_state)),
  sampling(r.sampling),
  seed(sampling_seed(r.sampling.seed)),
  max_tokens(r.max_tokens),
  root(root),
  head(last_state->head),
  prompt_digests(std::move(digests)),
  recurrent(owner.recurrent),
  sliding_windows(owner.sliding_windows),
  page_size(owner.page_size),
  layers(owner.recurrent.size()),
  eviction(to_pages(r.eviction, owner.page_size)) {
    for (const at::Device& device : devices) {
        tables.push_back(
            at::zeros(
                {static_cast<std::int64_t>(recurrent.size()), 0},
                at::TensorOptions().dtype(at::kInt).device(device)
            )
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
    pending_tokens.assign(r.tokens.begin() + resumed_page_count * page_size, r.tokens.end());
    assert(last_state->tail.size() < pending_tokens.size());
    assert(std::ranges::equal(last_state->tail, std::span(pending_tokens).first(last_state->tail.size())));
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        layers[layer].mass_heads = owner.mass_heads[layer];
        layers[layer].density_moments.resize(owner.mass_heads[layer]);
    }
    for (std::size_t layer = 0; layer < last_state->layers.size(); layer++) {
        if (recurrent[layer]) {
            continue;
        }
        layer_view& view = layers[layer];
        const layer_sequence& sequence = last_state->layers[layer];
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
    for (const layer_sequence& sequence : last_state->layers) {
        if (!sequence.pages.empty()) {
            history_pages = sequence.pages.back().index + 1;
        }
    }
    cached_count = history_pages * page_size;
    prompt_count = r.tokens.size();
    update_held();
}

std::size_t sequence::next_index(std::size_t layer) const noexcept {
    return layers[layer].pages.size() * page_size + tail.size();
}

std::size_t sequence::next_position() const noexcept {
    return held_pages.size() * page_size + tail.size();
}

std::size_t sequence::page_position(std::size_t layer, std::size_t page) const {
    const layer_view& view = layers[layer];
    if (page >= view.pages.size()) {
        return (held_pages.size() + page - view.pages.size()) * page_size;
    }
    auto rank = std::ranges::lower_bound(held_pages, view.indices[page]) - held_pages.begin();
    return static_cast<std::size_t>(rank) * page_size;
}

void sequence::update_held() {
    held_pages.clear();
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        if (!recurrent[layer]) {
            held_pages.insert(held_pages.end(), layers[layer].indices.begin(), layers[layer].indices.end());
        }
    }
    std::ranges::sort(held_pages);
    held_pages.erase(std::ranges::unique(held_pages).begin(), held_pages.end());
}

bool sequence::may_sample() const noexcept {
    return !max_tokens || generated_count < *max_tokens;
}

std::size_t sequence::advance(std::size_t count) {
    assert(count <= pending_tokens.size());
    tail.insert(tail.end(), pending_tokens.begin(), pending_tokens.begin() + count);
    pending_tokens.erase(pending_tokens.begin(), pending_tokens.begin() + count);
    covered_count += count;
    std::size_t full_page_count = tail.size() / page_size;
    if (full_page_count == 0) {
        return 0;
    }
    std::vector<digest> full_page_digests;
    for (std::size_t page = 0; page < full_page_count; page++) {
        if (next_prompt_digest < prompt_digests.size()) {
            full_page_digests.push_back(prompt_digests[next_prompt_digest++]);
        } else {
            const digest& parent = full_page_digests.empty() ? head.value_or(root) : full_page_digests.back();
            full_page_digests.push_back(make_digest(parent, std::span(tail).subspan(page * page_size, page_size)));
        }
    }
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        if (!recurrent[layer]) {
            layer_view& view = layers[layer];
            view.pages.insert(view.pages.end(), full_page_digests.begin(), full_page_digests.end());
            for (std::size_t i = 0; i < full_page_count; i++) {
                view.indices.push_back(history_pages + i);
            }
        }
    }
    if (std::ranges::contains(recurrent, false)) {
        for (std::size_t i = 0; i < full_page_count; i++) {
            held_pages.push_back(history_pages + i);
        }
    }
    history_pages += full_page_count;
    head = full_page_digests.back();
    tail.erase(tail.begin(), tail.begin() + full_page_count * page_size);
    return full_page_count;
}

std::size_t sequence::passed_pages(std::size_t layer, std::size_t query) const noexcept {
    const std::optional<std::size_t>& window = sliding_windows[layer];
    if (!window || query + 1 <= *window) {
        return 0;
    }
    return (query + 1 - *window) / page_size;
}

state sequence::to_state() const {
    state s{.head = head, .layers = std::vector<layer_sequence>(layers.size()), .tail = tail};
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        std::vector<active_page>& pages = s.layers[layer].pages;
        if (recurrent[layer]) {
            if (head) {
                pages.push_back(active_page{.hash = *head, .index = history_pages - 1});
            }
            continue;
        }
        const layer_view& view = layers[layer];
        for (std::size_t page = passed_pages(layer, view.pages.size() * page_size); page < view.pages.size(); page++) {
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

void sequence::start_cycle() noexcept {
    cycle_start = covered_count;
}

bool sequence::ends_cycle(std::size_t token_count) const noexcept {
    return covered_count + token_count - cycle_start >= eviction.cycle_tokens;
}

std::size_t sequence::cycle_tokens_left() const noexcept {
    std::size_t left = eviction.cycle_tokens - (covered_count - cycle_start);
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
    auto page_count = static_cast<std::int64_t>(table_width());
    if (tables.front().size(1) < page_count) {
        std::int64_t capacity = std::max<std::int64_t>(page_count, 2 * tables.front().size(1));
        for (std::size_t shard = 0; shard < tables.size(); shard++) {
            tables[shard] = at::empty({n_layers, capacity}, tables[shard].options());
            position_tables[shard] = at::empty_like(tables[shard]);
        }
        first_page = 0;
    }
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
            *position_entries++ = used && !recurrent[layer] ? static_cast<std::int32_t>(page_position(layer, index))
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
    return ceil_div(cycle_tokens_left(), page_size);
}

std::vector<std::size_t> sequence::missing_lines() const {
    std::size_t left = cycle_tokens_left();
    std::vector<std::size_t> counts;
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        std::size_t held_count = layers[layer].lines.size();
        std::size_t needed = recurrent[layer] ? 2 : ceil_div(next_index(layer) + left, page_size);
        counts.push_back(needed > held_count ? needed - held_count : 0);
    }
    return counts;
}

}  // namespace pulsarlm
