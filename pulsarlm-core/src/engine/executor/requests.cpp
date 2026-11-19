#include "batch.hpp"

#include "pulsarlm/engine/executor.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

namespace pulsarlm {

namespace {

bool lists_page_twice(const layer_sequence& layer) {
    std::unordered_set<digest, digest_hash> hashes;
    return !std::ranges::all_of(layer.pages, [&](const active_page& page) { return hashes.insert(page.hash).second; });
}

request_statistics statistics_of(const request_state& req) {
    return request_statistics{
        .total_token_count = req.prompt_token_count() + req.delivered_token_count(),
        .cached_token_count = req.cached_token_count(),
        .device_cached_token_count = req.device_cached_token_count(),
        .peak_device_memory = req.device_memory_peak(),
        .drafted_token_count = req.drafted_token_count(),
        .accepted_draft_count = req.accepted_draft_count(),
        .step_duration = req.step_duration(),
    };
}

}  // namespace

task<std::optional<executor::request_id_type>> executor::start(request&& r, unstarted_callbacks callbacks) {
    if (callbacks.value()->cancelled()) {
        callbacks.release()->on_end(std::unexpected(request_error::cancelled), request_statistics{});
        co_return std::nullopt;
    }
    assert(
        !r.tokens.empty() && r.eviction.n_working >= 1 && r.eviction.cycle_tokens >= 1 &&
        r.eviction.attention_mass_decay > 0.0 && r.eviction.attention_mass_decay <= 1.0 &&
        r.eviction.eviction_threshold >= 0.0 && r.eviction.eviction_threshold <= 1.0
    );
    digest chain_root = r.initial_digest.value_or(
        make_request_root_digest(root, to_pages(r.eviction, layout.page_size))
    );
    std::vector<digest> digests = make_page_digests(chain_root, r.tokens, layout.page_size);
    std::expected<std::pair<std::shared_ptr<const state>, state::lease_id_type>, page_error>
        lookup = co_await store.read_state(r.tokens, digests);
    if (!lookup) {
        callbacks.release()->on_end(std::unexpected(request_error::missing_state), request_statistics{});
        co_return std::nullopt;
    }
    auto& [stored_state, lease_id] = *lookup;
    std::size_t stored_pages = 0;
    if (stored_state->head) {
        auto head_digest = std::ranges::find(digests, *stored_state->head);
        if (head_digest == digests.end()) {
            store.release(lease_id);
            callbacks.release()->on_end(std::unexpected(request_error::invalid_state), request_statistics{});
            co_return std::nullopt;
        }
        stored_pages = static_cast<std::size_t>(head_digest - digests.begin()) + 1;
    }
    std::span<const std::int64_t> unstored_tokens = std::span(r.tokens).subspan(stored_pages * layout.page_size);
    if (stored_state->tail.size() >= unstored_tokens.size() ||
        !std::ranges::equal(stored_state->tail, unstored_tokens.first(stored_state->tail.size())) ||
        std::ranges::any_of(stored_state->layers, lists_page_twice)) {
        store.release(lease_id);
        callbacks.release()->on_end(std::unexpected(request_error::invalid_state), request_statistics{});
        co_return std::nullopt;
    }
    std::shared_ptr<const state> initial_state = extend_with_hot_pages(std::move(stored_state), r, digests);
    std::vector<at::Device> devices;
    for (const shard& s : shards) {
        devices.push_back(s.lines.front().k.device());
    }
    request_state started(
        std::move(r),
        std::move(initial_state),
        lease_id,
        std::move(digests),
        chain_root,
        layout,
        std::span<const at::Device>(devices)
    );
    const request_id_type id = requests.emplace(callbacks.release(), std::move(started));
    request_state& req = requests[id].state;
    // take_lines holds the resident pages before it first suspends, so they are the ones counted here.
    req.record_device_cached_tokens(device_cached_tokens(req));
    std::expected<void, page_error> line_allocation = co_await take_lines(req, req.cycle_lines());
    if (!line_allocation) {
        release_lines(req);
        store.release(req.lease_id());
        end_request(
            id,
            std::unexpected(
                line_allocation.error() == page_error::miss ? request_error::missing_state : request_error::oom
            ),
            request_statistics{}
        );
        co_return std::nullopt;
    }
    if (req.history_page_count() > stored_pages && !co_await own_pages_from(req, stored_pages)) {
        release_lines(req);
        store.release(req.lease_id());
        end_request(id, std::unexpected(request_error::oom), request_statistics{});
        co_return std::nullopt;
    }
    req.upload_table(0, streams_of(&shard::compute_stream));
    co_return id;
}

void executor::deliver_end(
    running_request& running,
    std::expected<std::shared_ptr<const state>, request_error>&& result,
    const request_statistics& statistics
) noexcept {
    std::unique_ptr<request_callbacks> ended = std::move(running.callbacks);
    ended->on_end(std::move(result), statistics);
}

void executor::end_request(
    request_id_type id,
    std::expected<std::shared_ptr<const state>, request_error>&& result,
    const request_statistics& statistics
) noexcept {
    if (running_request* running = requests.find(id)) {
        deliver_end(*running, std::move(result), statistics);
        requests.erase(id);
    }
}

void executor::end_requests(request_error error) noexcept {
    for (running_request& running : requests) {
        release_lines(running.state);
        store.release(running.state.lease_id());
        deliver_end(running, std::unexpected(error), statistics_of(running.state));
    }
    requests.clear();
}

std::shared_ptr<const state> executor::extend_with_hot_pages(
    std::shared_ptr<const state>&& s,
    const request& r,
    std::span<const digest> digests
) const {
    if (std::ranges::contains(layout.layers, true, &view_layout::layer::is_recurrent)) {
        return std::move(s);
    }
    std::size_t head_pages = 0;
    if (s->head) {
        auto head_digest = std::ranges::find(digests, *s->head);
        assert(head_digest != digests.end());
        head_pages = static_cast<std::size_t>(head_digest - digests.begin()) + 1;
    }
    page_eviction_policy policy = to_pages(r.eviction, layout.page_size);
    std::size_t resumed_end = head_pages * layout.page_size + s->tail.size();
    std::size_t cycle_end = (resumed_end / policy.cycle_tokens + 1) * policy.cycle_tokens;
    std::size_t window_start = cycle_end - policy.mass_tokens;
    if (resumed_end > window_start) {
        return std::move(s);
    }
    std::size_t page_limit = std::min(window_start, r.tokens.size() - 1) / layout.page_size;
    std::size_t end_page = head_pages;
    while (end_page < page_limit &&
           std::ranges::all_of(
               std::views::iota(std::size_t{0}, cache.layer_count()),
               [&](std::size_t layer) { return cache.contains(layer, digests[end_page]); }
           )) {
        end_page++;
    }
    if (end_page == head_pages) {
        return std::move(s);
    }
    state extended{.head = digests[end_page - 1], .layers = s->layers, .tail = {}};
    extended.layers.resize(cache.layer_count());
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        std::vector<active_page>& pages = extended.layers[layer].pages;
        for (std::size_t index = head_pages; index < end_page; index++) {
            pages.push_back(active_page{.hash = digests[index], .index = index});
        }
        const std::optional<std::size_t>& window = layout.layers[layer].sliding_window;
        std::size_t next_query = pages.size() * layout.page_size;
        if (window && next_query + 1 > *window) {
            std::size_t passed = std::min((next_query + 1 - *window) / layout.page_size, pages.size());
            pages.erase(pages.begin(), pages.begin() + static_cast<std::ptrdiff_t>(passed));
        }
    }
    return std::make_shared<const state>(std::move(extended));
}

task<std::expected<void, page_error>> executor::own_pages_from(request_state& req, std::size_t first_index) {
    std::size_t n_layers = cache.layer_count();
    std::vector<std::vector<digest>> hashes(n_layers);
    std::vector<std::vector<state::lease_id_type>> owners(n_layers);
    std::vector<std::vector<std::uint32_t>> line_ids(n_layers);
    for (std::size_t layer = 0; layer < n_layers; layer++) {
        const layer_view& view = req.views()[layer];
        for (std::size_t page = 0; page < view.pages.size(); page++) {
            if (view.indices[page] >= first_index) {
                hashes[layer].push_back(view.pages[page]);
                owners[layer].push_back(req.lease_id());
                line_ids[layer].push_back(view.lines[page]);
            }
        }
    }
    co_return co_await cache.sync(hashes, owners, line_ids, streams_of(&shard::compute_stream));
}

task<void> executor::stop(request_id_type id) {
    request_state& req = state_of(id);
    assert(req.has_exited() && req.launched_step_count() == 0);
    release_lines(req);
    std::expected<std::shared_ptr<const state>, request_error> result = std::unexpected(request_error::cancelled);
    switch (*req.reason_of_exit()) {
    case request_state::exit_reason::finished: {
        auto end_state = std::make_shared<const state>(req.to_state());
        co_await store.write_state(req.lease_id(), std::shared_ptr<const state>(end_state));
        result = std::move(end_state);
        break;
    }
    case request_state::exit_reason::cancelled:
        result = std::unexpected(request_error::cancelled);
        break;
    case request_state::exit_reason::oom:
        result = std::unexpected(request_error::oom);
        break;
    }
    store.release(req.lease_id());
    end_request(id, std::move(result), statistics_of(req));
}

std::size_t executor::device_cached_tokens(const request_state& req) const {
    std::vector<bool> page_fetched(req.history_page_count(), false);
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        const layer_view& view = req.views()[layer];
        if (layout.layers[layer].is_recurrent) {
            if (req.head_digest() && !cache.contains(layer, *req.head_digest())) {
                page_fetched[req.history_page_count() - 1] = true;
            }
            continue;
        }
        for (std::size_t i = 0; i < view.pages.size(); i++) {
            if (!cache.contains(layer, view.pages[i])) {
                page_fetched[view.indices[i]] = true;
            }
        }
    }
    return (req.history_page_count() - static_cast<std::size_t>(std::ranges::count(page_fetched, true))) *
        layout.page_size;
}

task<std::expected<void, page_error>> executor::take_lines(request_state& req, std::size_t line_count) {
    std::vector<std::span<const digest>> held_hashes(cache.layer_count());
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        if (!layout.layers[layer].is_recurrent) {
            held_hashes[layer] = req.views()[layer].pages;
        } else if (req.head_digest()) {
            held_hashes[layer] = std::span(&*req.head_digest(), 1);
        }
    }
    // A get failing with miss releases the lines it held_hashes while it fetched.
    std::expected<std::vector<std::vector<std::uint32_t>>, page_error> page_lines = co_await cache.get(
        held_hashes,
        streams_of(&shard::compute_stream)
    );
    if (!page_lines) {
        co_return std::unexpected(page_lines.error());
    }
    std::vector<std::size_t> counts(cache.layer_count(), line_count);
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        if (layout.layers[layer].is_recurrent) {
            // The stored state's copy takes the last slot.
            counts[layer] = layout.layers[layer].state_slots + 2 - ((*page_lines)[layer].empty() ? 0 : 1);
        }
    }
    std::expected<std::vector<std::vector<std::uint32_t>>, page_error> allocation = cache.allocate(counts);
    std::vector<std::vector<std::uint32_t>> states(cache.layer_count());
    bool copied = allocation.has_value();
    for (std::size_t layer = 0; layer < cache.layer_count() && copied; layer++) {
        if (layout.layers[layer].is_recurrent && !(*page_lines)[layer].empty()) {
            std::expected<std::vector<std::uint32_t>, page_error>
                state_copy = cache.clone(layer, (*page_lines)[layer].front(), 1, streams_of(&shard::compute_stream));
            copied = state_copy.has_value();
            if (copied) {
                states[layer] = std::move(*state_copy);
            }
        }
    }
    if (!copied) {
        for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
            cache.release(layer, (*page_lines)[layer]);
            cache.release(layer, states[layer]);
            if (allocation) {
                cache.release(layer, (*allocation)[layer]);
            }
        }
        co_return std::unexpected(page_error::oom);
    }
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        std::span<const std::uint32_t> allocated = (*allocation)[layer];
        if (layout.layers[layer].is_recurrent) {
            const std::size_t first_slots = layout.layers[layer].state_slots - 1;
            req.add_lines(layer, allocated.first(first_slots));
            allocated = allocated.subspan(first_slots);
            if (states[layer].empty()) {
                for (std::size_t shard = 0; shard < shards.size(); shard++) {
                    c10::cuda::CUDAStreamGuard guard(
                        c10::cuda::getStreamFromExternal(
                            shards[shard].compute_stream,
                            shards[shard].lines[layer].k.device().index()
                        )
                    );
                    shards[shard].lines[layer].k.select(0, allocated.front()).zero_();
                }
                req.add_lines(layer, allocated.first(1));
                allocated = allocated.subspan(1);
            } else {
                req.add_lines(layer, states[layer]);
            }
        } else {
            req.add_lines(layer, (*page_lines)[layer]);
        }
        req.add_lines(layer, allocated);
    }
    // The copies of the states at head read the fetched lines.
    co_await synchronize_shards();
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        if (layout.layers[layer].is_recurrent) {
            cache.release(layer, (*page_lines)[layer]);
        }
    }
    record_device_memory(req);
    co_return std::expected<void, page_error>{};
}

std::size_t executor::device_memory(const request_state& req) const noexcept {
    std::size_t bytes = 0;
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        bytes += req.views()[layer].lines.size() * cache.line_block_bytes(layer);
    }
    return bytes;
}

void executor::record_device_memory(request_state& req) const noexcept {
    req.record_device_memory(device_memory(req));
}

void executor::release_lines(request_state& req) noexcept {
    const std::vector<std::vector<std::uint32_t>> taken_lines = req.take_lines();
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        cache.release(layer, taken_lines[layer]);
    }
}

}  // namespace pulsarlm
