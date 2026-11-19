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

// A sequence ends with oom when the sync of its new pages fails, as a state names only pages the store has. The step
// that last read the lines of pages that left a view has completed, so they are released right after the sync. A
// recurrent layer's state at head reaches the store at the resume points, cycle ends and the end, that find it pending.
task<std::vector<executor::sequence_id>> executor::write_back() {
    assert(!launched_batches.empty());
    std::unique_ptr<batch> written = std::move(launched_batches.front());
    launched_batches.pop_front();
    batch& b = *written;
    co_await join(b);
    batch_writes writes = apply_entries(b, std::chrono::steady_clock::now() - b.launch_time);
    co_await write_pages_and_states(b, std::move(writes));
    co_await refresh_views(b);
    std::vector<sequence_id> written_sequences;
    for (const batch::group* g : std::as_const(b).groups()) {
        for (const batch::entry& e : g->entries) {
            written_sequences.push_back(e.seq.key());
        }
    }
    co_return written_sequences;
}

task<void> executor::join(batch& b) {
    for (external_task<cudaError_t>& completion : b.completions) {
        cudaError_t status = co_await std::move(completion);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("write_back: ") + cudaGetErrorString(status));
        }
    }
    b.join_mass(shard_mass_heads, layout);
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
    // Each drafting sequence's next token then its drafts.
    std::unordered_map<const sequence*, std::span<const std::int64_t>> drafted;
    const std::vector<batch::entry> drafting_entries = b.drafting_entries();
    for (std::size_t d = 0; d < drafting_entries.size(); d++) {
        const auto row = static_cast<std::size_t>(b.next_tokens.size(1));
        drafted.emplace(
            &*drafting_entries[d].seq,
            std::span(b.next_tokens.const_data_ptr<std::int64_t>() + d * row, row)
        );
    }
    for (batch::group* g : b.groups()) {
        const bool verifies = g == &b.verify_group;
        for (std::size_t position = 0; position < g->entries.size(); position++) {
            batch::entry& e = g->entries[position];
            sequence& seq = *e.seq;
            request_callbacks& callbacks = *requests[e.seq.key()].callbacks;
            std::span<const std::int64_t> samples;
            if (e.is_sampling) {
                samples = std::span(next_sampled, verifies ? b.verify_rows : 1);
                next_sampled += samples.size();
            }
            if (seq.has_exited()) {
                continue;
            }
            if (seq.pending_page_writes()->any_failed()) {
                seq.exit(sequence::exit_reason::oom);
                continue;
            }
            // The tokens the step accepts, and of its samples those it emits. A step launched after this one took its next
            // tokens on the GPU and lags its accepted count - 1.
            const bool next_launched = seq.launched_step_count() > 1;
            std::size_t accepted_count = batch::forwarded(e);
            std::span<const std::int64_t> emitted = samples;
            if (verifies) {
                accepted_count = static_cast<std::size_t>(
                    b.accepted_token_counts.const_data_ptr<std::int64_t>()[position]
                );
                emitted = samples.first(accepted_count);
                const std::size_t start = e.start_position + seq.unapplied_position_shift();
                seq.apply_accepted_tokens(start, emitted, b.verify_rows);
                drafted_tokens.fetch_add(b.verify_rows - 1, std::memory_order_relaxed);
                accepted_drafts.fetch_add(accepted_count - 1, std::memory_order_relaxed);
                e.start_position = start;
                e.end_position = start + accepted_count;
            }
            std::size_t completed_count = seq.complete_pages(e.end_position);
            if (e.is_measured) {
                b.add_measured_mass(*g, position, accepted_count, layout);
            }
            const bool is_prefill_step = !seq.has_ended_prefill();
            seq.record_step(step_duration);
            if (is_prefill_step && e.end_position >= seq.prompt_token_count()) {
                seq.end_prefill();
                callbacks.on_prefill(
                    prefill_statistics{
                        .cached_token_count = seq.cached_token_count(),
                        .device_cached_token_count = seq.device_cached_token_count(),
                        .computed_token_count = seq.prompt_token_count() - seq.cached_token_count(),
                        .step_count = seq.prefill_step_count(),
                        .step_duration = seq.prefill_step_duration(),
                        .device_memory = device_memory(seq),
                    }
                );
            }
            // The first token a step delivers carries the step's counts and its duration; the prefill statistics hold
            // the duration of a prefill step.
            decode_statistics token_statistics{
                .device_memory = device_memory(seq),
                .drafted_token_count = verifies ? b.verify_rows - 1 : 0,
                .accepted_draft_count = verifies ? accepted_count - 1 : 0,
                .step_duration = is_prefill_step ? std::chrono::steady_clock::duration::zero() : step_duration,
            };
            if (verifies) {
                for (std::int64_t token : emitted) {
                    callbacks.on_decode(token, token_statistics);
                    token_statistics.drafted_token_count = 0;
                    token_statistics.accepted_draft_count = 0;
                    token_statistics.step_duration = std::chrono::steady_clock::duration::zero();
                }
                if (std::ranges::contains(stop_ids, emitted.back()) || !seq.may_sample()) {
                    seq.exit(sequence::exit_reason::finished);
                    if (next_launched) {
                        seq.withdraw_latest_verify_step();
                    }
                }
            } else if (!samples.empty()) {
                const std::int64_t token = samples.front();
                seq.apply_sample(e.sampled_position, token);
                callbacks.on_decode(token, token_statistics);
                if (std::ranges::contains(stop_ids, token)) {
                    seq.exit(sequence::exit_reason::finished);
                    if (seq.launched_step_count() > 1) {
                        seq.withdraw_latest_step();
                    }
                }
            }
            if (e.ends_sequence) {
                seq.exit(sequence::exit_reason::finished);
            }
            if (seq.has_exited()) {
                seq.set_drafts({});
            } else if (auto found = drafted.find(&seq); found != drafted.end() && !next_launched) {
                seq.set_drafts(found->second.subspan(1));
            }
            const bool reaches_state = e.ends_cycle || seq.reason_of_exit() == sequence::exit_reason::finished;
            if (reaches_state) {
                writes.state_reachers.push_back(&seq);
                if (seq.has_pending_state()) {
                    writes.state_writers.push_back(&seq);
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
                    const layer_view& view = seq.views()[layer];
                    std::size_t first_new_page = view.pages.size() - completed_count;
                    hashes[layer].insert(hashes[layer].end(), view.pages.begin() + first_new_page, view.pages.end());
                    owners[layer].resize(hashes[layer].size(), seq.lease_id());
                    line_ids[layer].insert(
                        line_ids[layer].end(),
                        view.lines.begin() + first_new_page,
                        view.lines.begin() + view.pages.size()
                    );
                }
                if (reaches_state) {
                    writes.page_writers.push_back(&seq);
                } else {
                    writes.spawned_writers.push_back(seq.pending_page_writes());
                }
            }
        }
    }
    for (sequence* seq : writes.state_writers) {
        for (std::size_t layer = 0; layer < n_layers; layer++) {
            if (layout.layers[layer].is_recurrent) {
                writes.new_hashes[layer].push_back(*seq->head_digest());
                writes.new_owners[layer].push_back(seq->lease_id());
                writes.new_lines[layer].push_back(seq->head_state_line(layer));
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
        at::Device device = lines[shard].front().k.device();
        b.ends[shard].block(c10::cuda::getStreamFromExternal(copy_streams[shard], device.index()));
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
            cache.sync(writes.spawned_hashes, writes.spawned_owners, writes.spawned_lines, copy_streams),
            std::move(writes.spawned_writers)
        ));
    }
    // A state names only pages the store has.
    for (sequence* seq : writes.state_reachers) {
        co_await seq->pending_page_writes()->wait();
        if (seq->pending_page_writes()->any_failed()) {
            seq->exit(sequence::exit_reason::oom);
        }
    }
    std::expected<void, page_error>
        sync_result = co_await cache.sync(writes.new_hashes, writes.new_owners, writes.new_lines, copy_streams);
    if (!sync_result) {
        for (sequence* seq : writes.page_writers) {
            seq->exit(sequence::exit_reason::oom);
        }
        for (sequence* seq : writes.state_writers) {
            seq->exit(sequence::exit_reason::oom);
        }
    } else {
        for (sequence* seq : writes.state_writers) {
            if (seq->reason_of_exit() == sequence::exit_reason::oom) {
                continue;
            }
            seq->store_head_state();
            // The synced line keeps the stored state as a cached page; a new line takes its place.
            std::vector<std::size_t> counts(n_layers, 0);
            for (std::size_t layer = 0; layer < n_layers; layer++) {
                counts[layer] = layout.layers[layer].is_recurrent ? 1 : 0;
            }
            std::expected<std::vector<std::vector<std::uint32_t>>, page_error> replacements = cache.allocate(counts);
            if (!replacements) {
                seq->exit(sequence::exit_reason::oom);
                continue;
            }
            for (std::size_t layer = 0; layer < n_layers; layer++) {
                if (layout.layers[layer].is_recurrent) {
                    const std::uint32_t stored_line = seq->replace_head_state_line(
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
            sequence& seq = *e.seq;
            seq.finish_step();
            if (seq.has_exited()) {
                continue;
            }
            bool views_changed = release_passed_pages(seq);
            if (e.ends_cycle) {
                seq.sample_pages();
                views_changed = evict(seq) || views_changed;
            }
            if (views_changed) {
                update_views(seq);
            }
            if (e.ends_cycle) {
                co_await write_cycle_end(seq, *requests[e.seq.key()].callbacks);
                seq.start_next_cycle();
            }
            std::optional<std::size_t> first_stale_page;
            if (views_changed || e.stores_head_state) {
                first_stale_page = 0;
            }
            std::vector<std::size_t> missing_counts = seq.missing_lines();
            if (std::ranges::any_of(missing_counts, [](std::size_t count) { return count > 0; })) {
                std::expected<std::vector<std::vector<std::uint32_t>>, page_error> allocation = cache.allocate(
                    missing_counts
                );
                if (!allocation) {
                    seq.exit(sequence::exit_reason::oom);
                    continue;
                }
                for (std::size_t layer = 0; layer < n_layers; layer++) {
                    const std::size_t line_count = seq.views()[layer].lines.size();
                    if (missing_counts[layer] > 0) {
                        first_stale_page = std::min(first_stale_page.value_or(line_count), line_count);
                    }
                    seq.add_lines(layer, (*allocation)[layer]);
                }
                record_device_memory(seq);
            }
            if (first_stale_page) {
                seq.upload_table(*first_stale_page, streams);
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
bool executor::evict(sequence& seq) {
    if (!seq.eviction_policy_in_pages().can_evict()) {
        return false;
    }
    bool evicted = false;
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        if (layout.layers[layer].mass_heads == 0) {
            continue;
        }
        std::vector<std::size_t> pages = seq.pages_to_evict(layer);
        if (!pages.empty()) {
            cache.release(layer, seq.remove_pages(layer, pages));
            evicted = true;
        }
    }
    return evicted;
}

task<void> executor::write_cycle_end(sequence& seq, request_callbacks& callbacks) {
    auto cycle_end_state = std::make_shared<const state>(seq.to_state());
    co_await store.write_state(seq.lease_id(), std::shared_ptr<const state>(cycle_end_state));
    callbacks.on_cycle(cycle_end{.state = std::move(cycle_end_state)});
}

bool executor::release_passed_pages(sequence& seq) {
    bool changed = false;
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        std::vector<std::size_t> passed_indices(
            seq.passed_pages(layer, seq.views()[layer].pages.size() * layout.page_size)
        );
        if (passed_indices.empty()) {
            continue;
        }
        std::iota(passed_indices.begin(), passed_indices.end(), 0);
        cache.release(layer, seq.remove_pages(layer, passed_indices));
        changed = true;
    }
    return changed;
}

void executor::update_views(sequence& seq) {
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        if (!layout.layers[layer].mirrors_held_pages) {
            continue;
        }
        const layer_view& view = seq.views()[layer];
        std::vector<std::size_t> unheld;
        for (std::size_t page = 0; page < view.pages.size(); page++) {
            if (!std::ranges::binary_search(seq.held_page_indices(), view.indices[page])) {
                unheld.push_back(page);
            }
        }
        if (!unheld.empty()) {
            cache.release(layer, seq.remove_pages(layer, unheld));
        }
    }
}

task<void> executor::synchronize_shards() {
    for (cudaStream_t stream : streams) {
        co_await synchronize(stream);
    }
}

}  // namespace pulsarlm
