#include "pulsarlm/engine/scheduler.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace pulsarlm {

scheduler::scheduler(executor& exec, std::size_t prefill_tokens_per_step)
: exec(exec),
  prefill_tokens_per_step(prefill_tokens_per_step) {
    assert(prefill_tokens_per_step >= 1);
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
    return serve(std::move(r), executor::unstarted_callbacks(std::move(callbacks), end_with_loop_exited));
}

// A request waiting when the loop exits is destroyed with its task, and its event leaves waiting_requests unread.
task<void> scheduler::admit() {
    if (admitted_count < exec.max_running_request_count()) {
        admitted_count++;
        co_return;
    }
    event admission;
    waiting_requests.push_back(&admission);
    co_await admission;
}

// The first waiting request takes the place, so a request arriving between the hand-off and its resumption waits
// behind it.
void scheduler::pass_admission() noexcept {
    if (waiting_requests.empty()) {
        admitted_count--;
        return;
    }
    event* first = waiting_requests.front();
    waiting_requests.pop_front();
    first->set();
}

task<void> scheduler::serve(request r, executor::unstarted_callbacks callbacks) {
    co_await admit();
    const std::optional<executor::request_id_type> id = co_await exec.start(std::move(r), std::move(callbacks));
    if (!id) {
        pass_admission();
        co_return;
    }
    event& exit_event = exit_events.try_emplace(*id).first->second;
    running_requests.push_back(*id);
    wake(join_event);
    co_await exit_event;
    exit_events.erase(*id);
    co_await exec.stop(*id);
    pass_admission();
}

task<void> scheduler::run() {
    for (;;) {
        auto [scheduled, next_scan_start] = next_scheduled_requests();
        if (exec.launched_batch_count() > 0 && (scheduled.empty() || !exec.allows_overlap(scheduled))) {
            co_await retire();
            continue;
        }
        if (scheduled.empty()) {
            co_await emplace_event(join_event);
            continue;
        }
        co_await exec.launch(scheduled);
        std::vector<executor::request_id_type> scheduled_ids;
        for (const executor::scheduled_request& s : scheduled) {
            scheduled_ids.push_back(s.request_id);
        }
        release_exited(scheduled_ids);
        sampled_row_scan_start = next_scan_start;
        if (exec.launched_batch_count() > 1) {
            co_await retire();
        }
    }
}

task<void> scheduler::retire() {
    const std::vector<executor::request_id_type> stepped = co_await exec.write_back();
    for (executor::request_id_type id : stepped) {
        if (!exec.has_exited(id) && exec.is_cancelled(id)) {
            exec.cancel(id);
        }
    }
    release_exited(stepped);
}

void scheduler::release_exited(std::span<const executor::request_id_type> ids) {
    std::erase_if(running_requests, [&](executor::request_id_type id) { return exec.has_exited(id); });
    for (executor::request_id_type id : ids) {
        if (exec.has_exited(id) && !exec.is_in_launched_batch(id)) {
            exit_events.at(id).set();
        }
    }
}

std::pair<std::vector<executor::scheduled_request>, std::size_t> scheduler::next_scheduled_requests() const {
    const std::size_t request_count = running_requests.size();
    std::vector<std::size_t> token_counts(request_count, 0);
    std::vector<bool> is_decoding(request_count, false);
    std::size_t budget = prefill_tokens_per_step;
    std::size_t row_total = 0;
    std::optional<std::size_t> first_index_left_out;
    for (std::size_t offset = 0; offset < request_count; offset++) {
        const std::size_t index = (sampled_row_scan_start + offset) % request_count;
        const executor::request_id_type id = running_requests[index];
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

    std::vector<executor::scheduled_request> scheduled;
    for (const bool decodes : {true, false}) {
        for (std::size_t index = 0; index < request_count; index++) {
            if (token_counts[index] > 0 && is_decoding[index] == decodes) {
                scheduled.emplace_back(running_requests[index], token_counts[index]);
            }
        }
    }
    return {std::move(scheduled), first_index_left_out.value_or(sampled_row_scan_start)};
}

}  // namespace pulsarlm
