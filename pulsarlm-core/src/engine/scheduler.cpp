#include "pulsarlm/engine/scheduler.hpp"
#include "pulsarlm/engine/sequence.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <expected>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace pulsarlm {

scheduler::scheduler(executor& exec, std::size_t prefill_tokens_per_step)
: exec(exec),
  prefill_tokens_per_step(prefill_tokens_per_step) {
    assert(prefill_tokens_per_step > 0);
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
    std::expected<std::unique_ptr<sequence>, request_error> sequence_result = co_await exec.start(std::move(r), writer);
    if (!sequence_result) {
        writer.end(std::unexpected(sequence_result.error()));
        co_return;
    }
    sequence& seq = **sequence_result;
    seq.exit_event.emplace();
    running_sequences.push_back(&seq);
    wake(join_event);
    co_await *seq.exit_event;
    writer.end(co_await exec.finish(seq));
}

task<void> scheduler::run() {
    for (;;) {
        std::vector<batch_entry> entries = next_entries();
        if (!in_flight.empty() && (entries.empty() || !in_flight_allows_overlap || !exec.allows_overlap(entries))) {
            co_await retire();
            continue;
        }
        if (entries.empty()) {
            co_await emplace_event(join_event);
            continue;
        }
        std::vector<sequence*> stepped;
        for (const batch_entry& e : entries) {
            stepped.push_back(e.seq);
        }
        const bool allows_overlap = exec.allows_overlap(entries);
        co_await exec.launch(std::move(entries));
        if (!in_flight.empty()) {
            co_await retire();
        }
        in_flight = std::move(stepped);
        in_flight_allows_overlap = allows_overlap;
    }
}

task<void> scheduler::retire() {
    std::vector<sequence*> stepped = std::move(in_flight);
    in_flight.clear();
    co_await exec.write_back();
    for (sequence* seq : stepped) {
        if (!seq->end_reason && seq->writer.cancelled()) {
            seq->end_reason = sequence::exit_reason::cancelled;
        }
    }
    std::erase_if(running_sequences, [](const sequence* seq) { return seq->end_reason.has_value(); });
    for (sequence* seq : stepped) {
        if (seq->end_reason && seq->steps_in_flight == 0) {
            seq->exit_event->set();
        }
    }
}

std::vector<batch_entry> scheduler::next_entries() const {
    std::vector<batch_entry> entries;
    for (sequence* seq : running_sequences) {
        if (seq->last_step_launched || seq->end_reason) {
            continue;
        }
        assert(!seq->pending_tokens.empty());
        if (seq->pending_tokens.size() == 1) {
            entries.push_back(
                batch_entry{
                    .seq = seq,
                    .token_count = 1,
                    .samples = seq->may_sample(),
                    .ends_cycle = seq->ends_cycle(1),
                }
            );
        }
    }
    std::size_t budget = prefill_tokens_per_step;
    for (sequence* seq : running_sequences) {
        if (budget == 0) {
            break;
        }
        if (seq->last_step_launched || seq->end_reason) {
            continue;
        }
        if (seq->pending_tokens.size() > 1) {
            std::size_t count = std::min({seq->pending_tokens.size(), budget, seq->cycle_tokens_left()});
            budget -= count;
            entries.push_back(
                batch_entry{
                    .seq = seq,
                    .token_count = count,
                    .samples = count == seq->pending_tokens.size() && seq->may_sample(),
                    .ends_cycle = seq->ends_cycle(count),
                }
            );
        }
    }
    return entries;
}

}  // namespace pulsarlm
