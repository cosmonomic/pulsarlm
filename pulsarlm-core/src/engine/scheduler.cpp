#include "pulsarlm/engine/scheduler.hpp"

#include <algorithm>
#include <cstddef>
#include <expected>
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

task<void> scheduler::submit(request r, result_writer writer) {
    if (writer.cancelled()) {
        writer.end(std::unexpected(request_error::cancelled));
        co_return;
    }
    std::expected<executor::sequence_id, request_error> started = co_await exec.start(std::move(r), writer);
    if (!started) {
        writer.end(std::unexpected(started.error()));
        co_return;
    }
    const executor::sequence_id id = *started;
    event& exit_event = exit_events.try_emplace(id).first->second;
    running_sequences.push_back(id);
    wake(join_event);
    co_await exit_event;
    exit_events.erase(id);
    writer.end(co_await exec.stop(id));
}

task<void> scheduler::run() {
    for (;;) {
        std::vector<executor::batch_tokens> entries = next_entries();
        if (!in_flight_sequences.empty() &&
            (entries.empty() || !in_flight_allows_overlap || !exec.allows_overlap(entries))) {
            co_await retire();
            continue;
        }
        if (entries.empty()) {
            co_await emplace_event(join_event);
            continue;
        }
        std::vector<executor::sequence_id> stepped;
        for (const executor::batch_tokens& e : entries) {
            stepped.push_back(e.first);
        }
        const bool allows_overlap = exec.allows_overlap(entries);
        co_await exec.launch(entries);
        if (!in_flight_sequences.empty()) {
            co_await retire();
        }
        in_flight_sequences = std::move(stepped);
        in_flight_allows_overlap = allows_overlap;
    }
}

task<void> scheduler::retire() {
    std::vector<executor::sequence_id> stepped = std::move(in_flight_sequences);
    in_flight_sequences.clear();
    co_await exec.write_back();
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

std::vector<executor::batch_tokens> scheduler::next_entries() const {
    std::vector<executor::batch_tokens> entries;
    for (executor::sequence_id id : running_sequences) {
        if (exec.is_steppable(id) && exec.pending_token_count(id) == 1) {
            entries.emplace_back(id, 1);
        }
    }
    std::size_t budget = prefill_tokens_per_step;
    for (executor::sequence_id id : running_sequences) {
        if (budget == 0) {
            break;
        }
        if (!exec.is_steppable(id)) {
            continue;
        }
        const std::size_t pending_count = exec.pending_token_count(id);
        if (pending_count > 1) {
            std::size_t count = std::min({pending_count, budget, exec.cycle_tokens_left(id)});
            budget -= count;
            entries.emplace_back(id, count);
        }
    }
    return entries;
}

}  // namespace pulsarlm
