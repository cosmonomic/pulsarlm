#include "pulsarlm/engine/scheduler.hpp"

#include "executor/step.hpp"

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
  queue(exec, prefill_tokens_per_step) {}

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
    queue.push(*id);
    co_await exit_event;
    exit_events.erase(*id);
    co_await exec.stop(*id);
    pass_admission();
}

task<void> scheduler::run() {
    // The last launched step, written back after the next step's launch, or once nothing is ready to launch.
    std::optional<std::pair<executor::step, executor::launch_result>> unwritten;
    for (;;) {
        co_await yield();
        std::vector<executor::scheduled_request> scheduled = queue.next();
        if (!scheduled.empty()) {
            executor::step prepared = exec.prepare(scheduled);
            release_exited();
            executor::launch_result launched = co_await exec.launch(prepared);
            if (unwritten) {
                co_await exec.write_back(unwritten->first, unwritten->second);
                release_exited();
            }
            unwritten.emplace(std::move(prepared), std::move(launched));
        } else if (unwritten) {
            co_await exec.write_back(unwritten->first, unwritten->second);
            release_exited();
            unwritten.reset();
        } else {
            co_await queue.wait_for_push();
        }
    }
}

void scheduler::release_exited() {
    const std::vector<executor::request_id_type> exited = exec.take_exited_requests();
    queue.erase(exited);
    for (executor::request_id_type id : exited) {
        exit_events.at(id).set();
    }
}

}  // namespace pulsarlm
