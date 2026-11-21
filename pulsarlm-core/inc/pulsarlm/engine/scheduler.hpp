#pragma once

#include <cstddef>
#include <deque>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>

#include "pulsarlm/engine/event_loop.hpp"
#include "pulsarlm/engine/executor.hpp"
#include "pulsarlm/engine/request.hpp"
#include "pulsarlm/engine/request_callbacks.hpp"
#include "pulsarlm/engine/request_queue.hpp"

namespace pulsarlm {

// Decides what the executor runs: which requests join, which requests each step forwards and how many tokens, and when
// a request leaves.
class scheduler {
    executor& exec;
    request_queue queue;
    // Per running request, set once it exited; its request's task awaits it.
    std::unordered_map<executor::request_id_type, event> exit_events;
    // The requests past their wait for a record, from then until their record is erased or their start fails; at most
    // executor::max_running_request_count().
    std::size_t admitted_count = 0;
    // The events of the requests waiting for a record, in arrival order; each in its request's task.
    std::deque<event*> waiting_requests;
    // The started and the preempted requests that hold no lines, the preempted ones first, in the order they get lines.
    std::deque<executor::request_id_type> unloaded_requests;
    // The releases of lines so far: write-backs, stops, preemptions, failed allocations, and page writes that ended
    // while run had no step to launch.
    std::size_t line_release_count = 0;
    // Awaited by load_requests while no unloaded request is left, and after the first did not fit; set when one is
    // added, and when lines are released.
    std::optional<event> unloaded_request_added;
    std::optional<event> lines_released;
    // Awaited by run while it has no step to launch or write back; set when a request is pushed to the queue and when
    // lines are released.
    std::optional<event> run_woken;

    // Drops the requests that exited since the last call from the queue, and sets the exit event of each; moves the
    // requests preempted since the last call from the queue to the front of unloaded_requests.
    void release_exited();
    // Counts a release of lines, and wakes load_requests and run.
    void count_line_release();
    // Gives the unloaded requests their lines in order (executor::allocate_lines) and pushes each to the queue; ends
    // never.
    task<void> load_requests();
    // Waits, behind the requests that arrived before, until fewer than executor::max_running_request_count() requests
    // are admitted, and admits the request.
    task<void> admit();
    // Passes an admitted request's place to the first waiting request, or frees it.
    void pass_admission() noexcept;
    task<void> serve(request r, executor::unstarted_callbacks callbacks);

  public:
    scheduler(executor& exec, std::size_t prefill_tokens_per_step);

    // Serves r from the newest state the store holds for a prefix of its tokens; the executor takes callbacks and ends
    // them with the request's result or an error. The callbacks are ended with loop_exited when the returned task is
    // destroyed before the executor has taken them. The executor writes the state reached at the end of each eviction
    // cycle to the store, then passes it to the callbacks, and likewise the end state, and releases the state's lease
    // before it ends them. Waits, in arrival order, while executor::max_running_request_count() requests run, and ends
    // with cancelled when the callbacks are cancelled once it stops waiting. Ends with missing_state when the store's
    // lookup fails. Ends with oom, never waiting, when the free lines do not cover the state's pages and recurrent
    // states at the start, or the keys a step writes at its preparation, after the tokens of the steps before; the
    // caller resubmits.
    task<void> submit(request&& r, std::unique_ptr<request_callbacks>&& callbacks);
    // Steps the running requests until the loop stops: prepares a step of the schedulable requests
    // (executor::is_schedulable) and launches it, then writes back the step launched before it, so that a step runs
    // while the one before it is written back. A request whose callbacks are cancelled leaves after the steps it is in.
    task<void> run();
};

}  // namespace pulsarlm
