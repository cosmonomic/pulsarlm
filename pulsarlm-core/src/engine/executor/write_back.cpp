#include "step.hpp"

#include "pulsarlm/engine/executor.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pulsarlm {

// A request ends with oom when the sync of its new pages fails, as a state names only pages the store has. The step
// that last read the lines of pages that left a view has completed, so they are released right after the sync. A
// recurrent layer's state at head reaches the store at the resume points, cycle ends and the end, that find it pending.
task<void> executor::write_back(const step& s, launch_result& launched) {
    end_line_waits();
    if (s.entries.empty()) {
        co_return;
    }
    co_await join(s, launched);
    step_writes writes = apply_entries(s, launched, std::chrono::steady_clock::now() - s.prepare_time);
    co_await write_pages_and_states(launched, writes);
    refresh_views(s, writes);
    end_line_waits();
}

// The step's write-back came before, so every page it completed is in the views and its pages and states are written.
task<void> executor::evict(const step& s, const launch_result& launched) {
    for (std::size_t index = 0; index < s.entries.size(); index++) {
        const step_entry& e = s.entries[index];
        request_state& req = get_state(e.request_id);
        if (req.has_exit_cause() || !launched.reaches_cycle_end(index)) {
            continue;
        }
        req.sample_pages();
        const bool views_changed = evict_request(req);
        if (views_changed) {
            update_views(req);
        }
        co_await write_cycle_end(requests[e.request_id]);
        if (views_changed) {
            assert(req.unfinished_step_count() == 0);
            stale_requests.push_back(e.request_id);
            mark_stale_tables(e.request_id, 0);
        }
    }
    for (const step_entry& e : s.entries) {
        running_request& running = requests[e.request_id];
        if (e.awaits_write_back) {
            running.awaits_write_back = false;
        }
        if (running.state.has_exit_cause() && running.state.unfinished_step_count() == 0) {
            exit_request(e.request_id);
        }
        if (!running.state.has_exited() && is_cancelled(e.request_id)) {
            cancel(e.request_id);
        }
    }
}

task<void> executor::join(const step& s, launch_result& launched) {
    for (external_task<cudaError_t>& completion : launched.completions) {
        cudaError_t status = co_await std::move(completion);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("write_back: ") + cudaGetErrorString(status));
        }
    }
    s.join_mass(launched, shards, layout);
}

executor::step_writes executor::apply_entries(
    const step& s,
    const launch_result& launched,
    std::chrono::steady_clock::duration step_duration
) {
    std::size_t n_layers = cache.layer_count();
    step_writes writes{
        .new_hashes = std::vector<std::vector<digest>>(n_layers),
        .new_owners = std::vector<std::vector<state::lease_id_type>>(n_layers),
        .new_lines = std::vector<std::vector<std::uint32_t>>(n_layers),
        .spawned_hashes = std::vector<std::vector<digest>>(n_layers),
        .spawned_owners = std::vector<std::vector<state::lease_id_type>>(n_layers),
        .spawned_lines = std::vector<std::vector<std::uint32_t>>(n_layers),
    };
    const std::int64_t* next_sampled = launched.sampled_tokens.defined()
        ? launched.sampled_tokens.const_data_ptr<std::int64_t>()
        : nullptr;
    // Each drafting request's next token then its drafts.
    std::unordered_map<const request_state*, std::span<const std::int64_t>> drafted;
    const std::vector<step_entry> drafting_entries = s.drafting_entries();
    for (std::size_t d = 0; d < drafting_entries.size(); d++) {
        const auto row = static_cast<std::size_t>(launched.next_tokens.size(1));
        drafted.emplace(
            &get_state(drafting_entries[d].request_id),
            std::span(launched.next_tokens.const_data_ptr<std::int64_t>() + d * row, row)
        );
    }
    for (std::size_t index = 0; index < s.entries.size(); index++) {
        const step_entry& e = s.entries[index];
        const bool verifies = s.verifies(index);
        request_state& req = get_state(e.request_id);
        request_callbacks& callbacks = *requests[e.request_id].callbacks;
        std::span<const std::int64_t> samples;
        if (e.first_sample_index) {
            samples = std::span(next_sampled, verifies ? e.query_token_count : 1);
            next_sampled += samples.size();
        }
        if (req.discards_steps()) {
            continue;
        }
        if (req.pending_page_writes()->any_failed()) {
            req.fail_store_write();
            continue;
        }
        // The tokens the step accepts, and of its samples those it emits.
        const std::size_t start = req.history_position();
        std::size_t accepted_count = e.query_token_count;
        std::span<const std::int64_t> emitted = samples;
        if (verifies) {
            accepted_count = static_cast<std::size_t>(
                launched.accepted_token_counts.const_data_ptr<std::int64_t>()[index - s.single_query_request_count]
            );
            emitted = samples.first(accepted_count);
            req.record_verified_drafts(e.query_token_count - 1, accepted_count - 1);
            drafted_tokens.fetch_add(e.query_token_count - 1, std::memory_order_relaxed);
            accepted_drafts.fetch_add(accepted_count - 1, std::memory_order_relaxed);
        }
        const std::size_t end_position = start + accepted_count;
        const bool ends_cycle = launched.reaches_cycle_end(index);
        assert(e.start_position >= start && (!ends_cycle || e.ends_cycle));
        assert(ends_cycle == (end_position >= req.find_cycle_end(start)));
        assert(verifies || e.start_position != start || ends_cycle == e.ends_cycle);
        std::span<const std::int64_t> step_drafts;
        if (auto found = drafted.find(&req); found != drafted.end()) {
            step_drafts = found->second.subspan(1);
        }
        req.apply_step(accepted_count, emitted, step_drafts, verifies ? e.query_token_count - accepted_count : 0);
        std::size_t completed_count = req.complete_pages();
        record_device_memory(req);
        if (e.is_measured) {
            s.add_measured_mass(index, req, start, accepted_count, launched.mass, layout);
        }
        const bool is_prefill_step = !req.has_ended_prefill();
        req.record_step(step_duration);
        if (is_prefill_step && end_position >= req.prompt_token_count()) {
            req.end_prefill();
            callbacks.on_prefill(
                prefill_statistics{
                    .cached_token_count = req.cached_token_count(),
                    .device_cached_token_count = req.device_cached_token_count(),
                    .computed_token_count = req.prompt_token_count() - req.cached_token_count(),
                    .step_count = req.prefill_step_count(),
                    .step_duration = req.prefill_step_duration(),
                    .device_memory = device_memory(req),
                }
            );
        }
        // The first token a step delivers carries the step's counts and its duration; the prefill statistics hold
        // the duration of a prefill step.
        decode_statistics token_statistics{
            .device_memory = device_memory(req),
            .drafted_token_count = verifies ? e.query_token_count - 1 : 0,
            .accepted_draft_count = verifies ? accepted_count - 1 : 0,
            .step_duration = is_prefill_step ? std::chrono::steady_clock::duration::zero() : step_duration,
        };
        for (std::int64_t token : emitted) {
            callbacks.on_decode(token, token_statistics);
            token_statistics.drafted_token_count = 0;
            token_statistics.accepted_draft_count = 0;
            token_statistics.step_duration = std::chrono::steady_clock::duration::zero();
        }
        const std::optional<std::size_t>& token_limit = req.token_limit();
        const bool reaches_end = e.ends_request ||
            (!emitted.empty() && std::ranges::contains(stop_ids, emitted.back())) ||
            (token_limit && req.delivered_token_count() >= *token_limit);
        if (reaches_end) {
            req.reach_end();
        }
        const bool reaches_state = ends_cycle || reaches_end;
        if (reaches_state) {
            writes.state_reachers.push_back(&req);
            if (req.has_pending_state()) {
                writes.state_writers.push_back(&req);
            }
        }
        if (completed_count > 0) {
            auto& hashes = reaches_state ? writes.new_hashes : writes.spawned_hashes;
            auto& owners = reaches_state ? writes.new_owners : writes.spawned_owners;
            auto& line_ids = reaches_state ? writes.new_lines : writes.spawned_lines;
            for (std::size_t layer = 0; layer < n_layers; layer++) {
                if (layout.layers[layer].is_recurrent) {
                    continue;
                }
                const layer_view& view = req.views()[layer];
                std::size_t first_new_page = view.pages.size() - completed_count;
                hashes[layer].insert(hashes[layer].end(), view.pages.begin() + first_new_page, view.pages.end());
                owners[layer].resize(hashes[layer].size(), req.lease_id());
                line_ids[layer].insert(
                    line_ids[layer].end(),
                    view.lines.begin() + first_new_page,
                    view.lines.begin() + view.pages.size()
                );
            }
            if (reaches_state) {
                writes.page_writers.push_back(&req);
            } else {
                writes.spawned_writers.push_back(req.pending_page_writes());
            }
        }
    }
    for (request_state* req : writes.state_writers) {
        for (std::size_t layer = 0; layer < n_layers; layer++) {
            if (layout.layers[layer].is_recurrent) {
                writes.new_hashes[layer].push_back(*req->head_digest());
                writes.new_owners[layer].push_back(req->lease_id());
                writes.new_lines[layer].push_back(req->head_state_line(layer));
            }
        }
    }
    return writes;
}

task<void> executor::write_pages_and_states(launch_result& launched, step_writes& writes) {
    std::size_t n_layers = cache.layer_count();
    // The copies wait for this step only, its draft pass included, not for the steps launched after it, which write
    // none of the copied lines.
    for (std::size_t shard = 0; shard < shards.size(); shard++) {
        at::Device device = shards[shard].lines.front().k.device();
        launched.ends[shard].block(c10::cuda::getStreamFromExternal(shards[shard].copy_stream, device.index()));
    }
    if (!writes.spawned_writers.empty()) {
        while (store_writes.size() >= max_store_writes) {
            co_await store_writes.wait();
        }
        store_writes.start();
        for (const std::shared_ptr<pending_writes>& writer : writes.spawned_writers) {
            writer->start();
        }
        // kv_cache::sync holds the lines before the write-back below releases the ones that left a view.
        event_loop::current()->spawn(async_write_back(
            cache.sync(
                writes.spawned_hashes,
                writes.spawned_owners,
                writes.spawned_lines,
                collect_streams(&shard::copy_stream)
            ),
            std::move(writes.spawned_writers)
        ));
    }
    // A state names only pages the store has.
    for (request_state* req : writes.state_reachers) {
        co_await req->pending_page_writes()->wait();
        if (req->pending_page_writes()->any_failed()) {
            req->fail_store_write();
        }
    }
    std::expected<void, page_error> sync_result = co_await cache.sync(
        writes.new_hashes,
        writes.new_owners,
        writes.new_lines,
        collect_streams(&shard::copy_stream)
    );
    if (!sync_result) {
        for (request_state* req : writes.page_writers) {
            req->fail_store_write();
        }
        for (request_state* req : writes.state_writers) {
            req->fail_store_write();
        }
    } else {
        for (request_state* req : writes.state_writers) {
            if (req->has_failed_store_write()) {
                continue;
            }
            req->store_head_state();
            // The synced line keeps the stored state as a cached page; the prologue of a step that writes the state at
            // the end of a page of its parity gives the view another.
            for (std::size_t layer = 0; layer < n_layers; layer++) {
                if (layout.layers[layer].is_recurrent) {
                    const std::uint32_t stored_line = req->release_head_state_line(layer);
                    cache.release(layer, std::span(&stored_line, 1));
                }
            }
        }
    }
}

// The step that wrote the detached lines has completed, and the copy of a stored one has been synced.
void executor::refresh_views(const step& s, const step_writes& writes) {
    for (const step_entry& e : s.entries) {
        request_state& req = get_state(e.request_id);
        const std::vector<std::vector<std::uint32_t>> detached = req.take_detached_lines();
        const std::vector<std::vector<std::uint32_t>> released = req.take_released_lines();
        for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
            cache.release(layer, detached[layer]);
            cache.release(layer, released[layer]);
        }
        for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
            if (layout.layers[layer].sliding_window) {
                cache.release(layer, req.drop_passed_pages(layer));
            }
        }
        req.finish_step();
        if (!req.has_exit_cause() && std::ranges::contains(writes.state_writers, &req)) {
            for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
                if (layout.layers[layer].is_recurrent) {
                    mark_stale_layer_table(e.request_id, layer, layout.layers[layer].state_slots);
                }
            }
        }
    }
}

task<void> executor::async_write_back(
    task<std::expected<void, page_error>> syncing,
    std::vector<std::shared_ptr<pending_writes>> writers
) {
    std::expected<void, page_error> written = co_await std::move(syncing);
    for (const std::shared_ptr<pending_writes>& writer : writers) {
        writer->finish(written.has_value());
    }
    store_writes.finish(written.has_value());
    end_line_waits();
}

// The step that last used the evicted lines has completed, so they are released right away.
bool executor::evict_request(request_state& req) {
    if (!req.eviction_policy_in_pages().can_evict()) {
        return false;
    }
    bool evicted = false;
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        if (layout.layers[layer].mass_heads == 0) {
            continue;
        }
        std::vector<std::size_t> pages = req.pages_to_evict(layer);
        if (!pages.empty()) {
            cache.release(layer, req.remove_pages(layer, pages));
            evicted = true;
        }
    }
    return evicted;
}

task<void> executor::write_cycle_end(running_request& running) {
    auto cycle_end_state = std::make_shared<const state>(running.state.to_state());
    co_await store.write_state(running.state.lease_id(), std::shared_ptr<const state>(cycle_end_state));
    running.last_state = cycle_end_state;
    running.callbacks->on_cycle(cycle_end{.state = std::move(cycle_end_state)});
}

void executor::update_views(request_state& req) {
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        if (!layout.layers[layer].mirrors_held_pages) {
            continue;
        }
        const layer_view& view = req.views()[layer];
        std::vector<std::size_t> unheld;
        for (std::size_t page = 0; page < view.pages.size(); page++) {
            if (!std::ranges::binary_search(req.held_page_indices(), view.indices[page])) {
                unheld.push_back(page);
            }
        }
        if (!unheld.empty()) {
            cache.release(layer, req.remove_pages(layer, unheld));
        }
    }
}

task<void> executor::synchronize_shards() {
    for (const shard& s : shards) {
        co_await synchronize(s.compute_stream);
    }
}

}  // namespace pulsarlm
