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

namespace {

// The awaiters a set event scheduled no longer refer to it, so it is destroyed right away.
void set_once(std::optional<event>& waited) noexcept {
    if (waited) {
        waited->set();
        waited.reset();
    }
}

}  // namespace

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
    unloaded_requests.push_back(*id);
    set_once(unloaded_request_added);
    co_await exit_event;
    exit_events.erase(*id);
    co_await exec.stop(*id);
    count_line_release();
    pass_admission();
}

// A request whose allocation fails without an exit stays first and retries after the next release of lines, or right
// away when lines were released while it allocated; its own failed allocation does not count.
task<void> scheduler::load_requests() {
    for (;;) {
        if (unloaded_requests.empty()) {
            co_await unloaded_request_added.emplace();
            continue;
        }
        const executor::request_id_type id = unloaded_requests.front();
        const std::size_t releases_before = line_release_count;
        // Requests preempted while the lines are taken go before id.
        if (co_await exec.allocate_lines(id)) {
            std::erase(unloaded_requests, id);
            queue.push(id);
            set_once(run_woken);
        } else if (exec.has_exited(id)) {
            std::erase(unloaded_requests, id);
            release_exited();
        } else {
            const bool released_meanwhile = line_release_count > releases_before;
            count_line_release();
            if (!released_meanwhile) {
                co_await lines_released.emplace();
            }
        }
    }
}

task<void> scheduler::run() {
    event_loop::current()->spawn(load_requests());
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
                co_await exec.evict(unwritten->first, unwritten->second);
                release_exited();
                count_line_release();
            }
            unwritten.emplace(std::move(prepared), std::move(launched));
        } else if (unwritten) {
            co_await exec.write_back(unwritten->first, unwritten->second);
            co_await exec.evict(unwritten->first, unwritten->second);
            release_exited();
            count_line_release();
            unwritten.reset();
        } else if (exec.has_page_writes()) {
            co_await exec.wait_for_page_writes();
            count_line_release();
        } else {
            co_await run_woken.emplace();
        }
    }
}

void scheduler::release_exited() {
    const std::vector<executor::request_id_type> exited = exec.take_exited_requests();
    queue.erase(exited);
    for (executor::request_id_type id : exited) {
        exit_events.at(id).set();
    }
    const std::vector<executor::request_id_type> preempted = exec.take_preempted_requests();
    if (preempted.empty()) {
        return;
    }
    queue.erase(preempted);
    count_line_release();
    unloaded_requests.insert(unloaded_requests.begin(), preempted.begin(), preempted.end());
    set_once(unloaded_request_added);
}

void scheduler::count_line_release() {
    line_release_count++;
    set_once(lines_released);
    set_once(run_woken);
}

}  // namespace pulsarlm
