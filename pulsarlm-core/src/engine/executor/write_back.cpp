#include "batch.hpp"

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
task<std::vector<executor::request_id_type>> executor::write_back() {
    assert(!launched_batches.empty());
    std::unique_ptr<batch> written = std::move(launched_batches.front());
    launched_batches.pop_front();
    batch& b = *written;
    co_await join(b);
    batch_writes writes = apply_entries(b, std::chrono::steady_clock::now() - b.launch_time);
    co_await write_pages_and_states(b, std::move(writes));
    co_await refresh_views(b);
    std::vector<request_id_type> written_requests;
    for (const batch::group* g : std::as_const(b).groups()) {
        for (const batch::entry& e : g->entries) {
            if (e.req->has_exit_cause() && e.req->unfinished_step_count() == 0) {
                e.req->exit();
            }
            written_requests.push_back(e.req.key());
        }
    }
    co_return written_requests;
}

task<void> executor::join(batch& b) {
    for (external_task<cudaError_t>& completion : b.completions) {
        cudaError_t status = co_await std::move(completion);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("write_back: ") + cudaGetErrorString(status));
        }
    }
    b.join_mass(shards, layout);
}

executor::batch_writes executor::apply_entries(batch& b, std::chrono::steady_clock::duration step_duration) {
    std::size_t n_layers = cache.layer_count();
    batch_writes writes{
        .new_hashes = std::vector<std::vector<digest>>(n_layers),
        .new_owners = std::vector<std::vector<state::lease_id_type>>(n_layers),
        .new_lines = std::vector<std::vector<std::uint32_t>>(n_layers),
        .spawned_hashes = std::vector<std::vector<digest>>(n_layers),
        .spawned_owners = std::vector<std::vector<state::lease_id_type>>(n_layers),
        .spawned_lines = std::vector<std::vector<std::uint32_t>>(n_layers),
    };
    const std::int64_t* next_sampled = b.sampled_tokens.defined() ? b.sampled_tokens.const_data_ptr<std::int64_t>()
                                                                  : nullptr;
    // Each drafting request's next token then its drafts.
    std::unordered_map<const request_state*, std::span<const std::int64_t>> drafted;
    const std::vector<batch::entry> drafting_entries = b.drafting_entries();
    for (std::size_t d = 0; d < drafting_entries.size(); d++) {
        const auto row = static_cast<std::size_t>(b.next_tokens.size(1));
        drafted.emplace(
            &*drafting_entries[d].req,
            std::span(b.next_tokens.const_data_ptr<std::int64_t>() + d * row, row)
        );
    }
    for (batch::group* g : b.groups()) {
        const bool verifies = g == &b.verify_group;
        for (std::size_t position = 0; position < g->entries.size(); position++) {
            batch::entry& e = g->entries[position];
            request_state& req = *e.req;
            request_callbacks& callbacks = *requests[e.req.key()].callbacks;
            std::span<const std::int64_t> samples;
            if (e.is_sampling) {
                samples = std::span(next_sampled, verifies ? b.verify_rows : 1);
                next_sampled += samples.size();
            }
            if (req.discards_steps()) {
                continue;
            }
            if (req.pending_page_writes()->any_failed()) {
                req.fail_store_write();
                continue;
            }
            // The tokens the step accepts, and of its samples those it emits. A step launched after this one took its next
            // tokens on the GPU and lags its accepted count - 1.
            const bool next_launched = req.unfinished_step_count() > 1;
            std::size_t accepted_count = batch::forwarded(e);
            std::span<const std::int64_t> emitted = samples;
            if (verifies) {
                accepted_count = static_cast<std::size_t>(
                    b.accepted_token_counts.const_data_ptr<std::int64_t>()[position]
                );
                emitted = samples.first(accepted_count);
                const std::size_t start = e.start_position + req.unapplied_position_shift();
                req.apply_accepted_tokens(start, emitted, b.verify_rows);
                drafted_tokens.fetch_add(b.verify_rows - 1, std::memory_order_relaxed);
                accepted_drafts.fetch_add(accepted_count - 1, std::memory_order_relaxed);
                e.start_position = start;
                e.end_position = start + accepted_count;
            }
            std::size_t completed_count = req.complete_pages(e.end_position);
            if (e.is_measured) {
                b.add_measured_mass(*g, position, accepted_count, layout);
            }
            const bool is_prefill_step = !req.has_ended_prefill();
            req.record_step(step_duration);
            if (is_prefill_step && e.end_position >= req.prompt_token_count()) {
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
                .drafted_token_count = verifies ? b.verify_rows - 1 : 0,
                .accepted_draft_count = verifies ? accepted_count - 1 : 0,
                .step_duration = is_prefill_step ? std::chrono::steady_clock::duration::zero() : step_duration,
            };
            bool reaches_end = e.ends_request;
            if (verifies) {
                for (std::int64_t token : emitted) {
                    callbacks.on_decode(token, token_statistics);
                    token_statistics.drafted_token_count = 0;
                    token_statistics.accepted_draft_count = 0;
                    token_statistics.step_duration = std::chrono::steady_clock::duration::zero();
                }
                if (std::ranges::contains(stop_ids, emitted.back()) || !req.may_sample()) {
                    reaches_end = true;
                    if (next_launched) {
                        req.withdraw_latest_verify_step();
                    }
                }
            } else if (!samples.empty()) {
                const std::int64_t token = samples.front();
                req.apply_sample(e.sampled_position, token);
                callbacks.on_decode(token, token_statistics);
                if (std::ranges::contains(stop_ids, token)) {
                    reaches_end = true;
                    if (next_launched) {
                        req.withdraw_latest_step();
                    }
                }
            }
            if (reaches_end) {
                req.reach_end();
                req.set_drafts({});
            } else if (auto found = drafted.find(&req); found != drafted.end() && !next_launched) {
                req.set_drafts(found->second.subspan(1));
            }
            const bool reaches_state = e.ends_cycle || reaches_end;
            if (reaches_state) {
                writes.state_reachers.push_back(&req);
                if (req.has_pending_state()) {
                    writes.state_writers.push_back(&req);
                    e.stores_head_state = true;
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

task<void> executor::write_pages_and_states(batch& b, batch_writes&& writes) {
    std::size_t n_layers = cache.layer_count();
    // The copies wait for this step only, its draft pass included, not for the step launched after it, which writes none
    // of the copied lines.
    for (std::size_t shard = 0; shard < shards.size(); shard++) {
        at::Device device = shards[shard].lines.front().k.device();
        b.ends[shard].block(c10::cuda::getStreamFromExternal(shards[shard].copy_stream, device.index()));
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
                streams_of(&shard::copy_stream)
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
        streams_of(&shard::copy_stream)
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
            // The synced line keeps the stored state as a cached page; a new line takes its place.
            std::vector<std::size_t> counts(n_layers, 0);
            for (std::size_t layer = 0; layer < n_layers; layer++) {
                counts[layer] = layout.layers[layer].is_recurrent ? 1 : 0;
            }
            std::expected<std::vector<std::vector<std::uint32_t>>, page_error> replacements = cache.allocate(counts);
            if (!replacements) {
                req->run_out_of_lines();
                continue;
            }
            for (std::size_t layer = 0; layer < n_layers; layer++) {
                if (layout.layers[layer].is_recurrent) {
                    const std::uint32_t stored_line = req->replace_head_state_line(
                        layer,
                        (*replacements)[layer].front()
                    );
                    cache.release(layer, std::span(&stored_line, 1));
                }
            }
        }
    }
}

task<void> executor::refresh_views(batch& b) {
    std::size_t n_layers = cache.layer_count();
    for (batch::group* g : b.groups()) {
        for (const batch::entry& e : g->entries) {
            request_state& req = *e.req;
            req.finish_step();
            if (req.has_exit_cause()) {
                continue;
            }
            bool views_changed = release_passed_pages(req);
            if (e.ends_cycle) {
                req.sample_pages();
                views_changed = evict(req) || views_changed;
            }
            if (views_changed) {
                update_views(req);
            }
            if (e.ends_cycle) {
                co_await write_cycle_end(req, *requests[e.req.key()].callbacks);
                req.start_next_cycle();
            }
            std::optional<std::size_t> first_stale_page;
            if (views_changed || e.stores_head_state) {
                first_stale_page = 0;
            }
            std::vector<std::size_t> missing_counts = req.missing_lines();
            if (table_width_after(req, missing_counts) > view_page_capacity) {
                req.exceed_context();
                continue;
            }
            if (std::ranges::any_of(missing_counts, [](std::size_t count) { return count > 0; })) {
                std::expected<std::vector<std::vector<std::uint32_t>>, page_error> allocation = cache.allocate(
                    missing_counts
                );
                if (!allocation) {
                    req.run_out_of_lines();
                    continue;
                }
                for (std::size_t layer = 0; layer < n_layers; layer++) {
                    const std::size_t line_count = req.views()[layer].lines.size();
                    if (missing_counts[layer] > 0) {
                        first_stale_page = std::min(first_stale_page.value_or(line_count), line_count);
                    }
                    req.add_lines(layer, (*allocation)[layer]);
                }
                record_device_memory(req);
            }
            if (first_stale_page) {
                mark_stale_tables(e.req.key(), *first_stale_page);
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
}

// The step that last used the evicted lines has completed, so they are released right away.
bool executor::evict(request_state& req) {
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

task<void> executor::write_cycle_end(request_state& req, request_callbacks& callbacks) {
    auto cycle_end_state = std::make_shared<const state>(req.to_state());
    co_await store.write_state(req.lease_id(), std::shared_ptr<const state>(cycle_end_state));
    callbacks.on_cycle(cycle_end{.state = std::move(cycle_end_state)});
}

bool executor::release_passed_pages(request_state& req) {
    bool changed = false;
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        std::vector<std::size_t> passed_indices(
            req.passed_pages(layer, req.views()[layer].pages.size() * layout.page_size)
        );
        if (passed_indices.empty()) {
            continue;
        }
        std::iota(passed_indices.begin(), passed_indices.end(), 0);
        cache.release(layer, req.remove_pages(layer, passed_indices));
        changed = true;
    }
    return changed;
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
