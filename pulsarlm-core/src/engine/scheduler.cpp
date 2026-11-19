#include "pulsarlm/engine/scheduler.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pulsarlm {

scheduler::scheduler(executor& exec, std::size_t prefill_tokens_per_step)
: exec(exec),
  prefill_tokens_per_step(prefill_tokens_per_step) {
    if (prefill_tokens_per_step == 0) {
        throw std::invalid_argument("scheduler: prefill_tokens_per_step must be at least 1");
    }
}

event& scheduler::emplace_event(std::optional<event>& waited) {
    if (!waited) {
        waited.emplace();
    }
    return *waited;
}

// The awaiters a set event scheduled no longer refer to it, so it is destroyed right away.
void scheduler::wake(std::optional<event>& waited) noexcept {
    if (waited) {
        waited->set();
        waited.reset();
    }
}

task<void> scheduler::submit(request&& r, std::unique_ptr<request_callbacks>&& callbacks) {
    return serve(std::move(r), unstarted_callbacks(std::move(callbacks), end_with_loop_exited));
}

task<void> scheduler::serve(request r, unstarted_callbacks callbacks) {
    const std::optional<executor::sequence_id> id = co_await exec.start(std::move(r), callbacks.release());
    if (!id) {
        co_return;
    }
    event& exit_event = exit_events.try_emplace(*id).first->second;
    running_sequences.push_back(*id);
    wake(join_event);
    co_await exit_event;
    exit_events.erase(*id);
    co_await exec.stop(*id);
}

task<void> scheduler::run() {
    for (;;) {
        auto [entries, next_scan_start] = next_entries();
        if (exec.launched_batch_count() > 0 && (entries.empty() || !exec.allows_overlap(entries))) {
            co_await retire();
            continue;
        }
        if (entries.empty()) {
            co_await emplace_event(join_event);
            continue;
        }
        co_await exec.launch(entries);
        sampled_row_scan_start = next_scan_start;
        if (exec.launched_batch_count() > 1) {
            co_await retire();
        }
    }
}

task<void> scheduler::retire() {
    const std::vector<executor::sequence_id> stepped = co_await exec.write_back();
    for (executor::sequence_id id : stepped) {
        if (!exec.has_exited(id) && exec.is_cancelled(id)) {
            exec.cancel(id);
        }
    }
    std::erase_if(running_sequences, [&](executor::sequence_id id) { return exec.has_exited(id); });
    for (executor::sequence_id id : stepped) {
        if (exec.has_exited(id) && !exec.is_in_launched_batch(id)) {
            exit_events.at(id).set();
        }
    }
}

std::pair<std::vector<executor::batch_tokens>, std::size_t> scheduler::next_entries() const {
    const std::size_t sequence_count = running_sequences.size();
    std::vector<std::size_t> token_counts(sequence_count, 0);
    std::vector<bool> is_decoding(sequence_count, false);
    std::size_t budget = prefill_tokens_per_step;
    std::size_t row_total = 0;
    std::optional<std::size_t> first_index_left_out;
    for (std::size_t offset = 0; offset < sequence_count; offset++) {
        const std::size_t index = (sampled_row_scan_start + offset) % sequence_count;
        const executor::sequence_id id = running_sequences[index];
        if (!exec.is_steppable(id)) {
            continue;
        }
        const std::size_t pending_count = exec.pending_token_count(id);
        is_decoding[index] = pending_count == 1;
        std::size_t token_count = 0;
        if (is_decoding[index]) {
            token_count = 1;
        } else if (budget > 0) {
            token_count = std::min({pending_count, budget, exec.cycle_tokens_left(id)});
        }
        if (token_count == 0) {
            continue;
        }
        const std::size_t row_count = exec.sampled_row_count(id, token_count);
        if (row_total + row_count > exec.sampled_row_limit()) {
            if (!first_index_left_out) {
                first_index_left_out = index;
            }
            continue;
        }
        row_total += row_count;
        if (!is_decoding[index]) {
            budget -= token_count;
        }
        token_counts[index] = token_count;
    }

    std::vector<executor::batch_tokens> entries;
    for (const bool decodes : {true, false}) {
        for (std::size_t index = 0; index < sequence_count; index++) {
            if (token_counts[index] > 0 && is_decoding[index] == decodes) {
                entries.emplace_back(running_sequences[index], token_counts[index]);
            }
        }
    }
    return {std::move(entries), first_index_left_out.value_or(sampled_row_scan_start)};
}

}  // namespace pulsarlm
