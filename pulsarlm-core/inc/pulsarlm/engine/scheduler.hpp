#pragma once

#include <cstddef>
#include <deque>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pulsarlm/engine/event_loop.hpp"
#include "pulsarlm/engine/executor.hpp"
#include "pulsarlm/engine/request.hpp"
#include "pulsarlm/engine/request_callbacks.hpp"

namespace pulsarlm {

// Decides what the executor runs: which requests join, which requests each step forwards and how many tokens, and when
// a request leaves.
class scheduler {
    executor& exec;
    std::size_t prefill_tokens_per_step;
    std::vector<executor::request_id_type> running_requests;
    // Set when a request joins the running set.
    std::optional<event> join_event;
    // Per running request, set once it exited and no launched step holds it; its request's task awaits it.
    std::unordered_map<executor::request_id_type, event> exit_events;
    // The position in running_requests from which the next step grants rows of sampled tokens.
    std::size_t sampled_row_scan_start = 0;
    // The requests past their wait for a record, from then until their record is erased or their start fails; at most
    // executor::max_running_request_count().
    std::size_t admitted_count = 0;
    // The events of the requests waiting for a record, in arrival order; each in its request's task.
    std::deque<event*> waiting_requests;

    static event& emplace_event(std::optional<event>& waited);
    static void wake(std::optional<event>& waited) noexcept;
    // The next step's scheduled requests: every decoding request, then the others' tokens up to prefill_tokens_per_step, never
    // past the end of a request's cycle. Leaves out the requests whose last step is launched. Rows of sampled tokens
    // are granted from sampled_row_scan_start on, and a request whose rows would raise the step's above
    // executor::sampled_row_limit is left out. Also returns the position of the first request left out, or
    // sampled_row_scan_start when none was.
    std::pair<std::vector<executor::scheduled_request>, std::size_t> next_scheduled_requests() const;
    // Writes the oldest launched step back, ends the requests in it whose callbacks are cancelled, and lets each request
    // that exited leave once no launched step holds it.
    task<void> retire();
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
    // lookup fails. Ends with oom, never waiting, when the free
    // lines do not cover the state's pages and one cycle's tokens at the start, or the next cycle's tokens while
    // running; the caller resubmits.
    task<void> submit(request&& r, std::unique_ptr<request_callbacks>&& callbacks);
    // Steps the running requests until the loop stops, launching each step before the previous one is written back
    // when executor::allows_overlap. A request whose callbacks are cancelled leaves after the
    // steps it is in.
    task<void> run();
};

}  // namespace pulsarlm
