#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pulsarlm/engine/event_loop.hpp"
#include "pulsarlm/engine/executor.hpp"
#include "pulsarlm/engine/request.hpp"
#include "pulsarlm/engine/request_callbacks.hpp"
#include "pulsarlm/resource.hpp"

namespace pulsarlm {

// Decides what the executor runs: which requests join, which sequences each step forwards and how many tokens, and when
// a sequence leaves.
class scheduler {
    // The callbacks of a request whose task has not passed them to the executor. Destroyed while it still holds
    // them, it ends them with loop_exited.
    using unstarted_callbacks = resource<
        std::unique_ptr<request_callbacks>,
        void (*)(std::unique_ptr<request_callbacks>&) noexcept>;

    executor& exec;
    std::size_t prefill_tokens_per_step;
    std::vector<executor::sequence_id> running_sequences;
    // Set when a sequence joins the running set.
    std::optional<event> join_event;
    // Per running sequence, set once it exited and no launched step holds it; its request's task awaits it.
    std::unordered_map<executor::sequence_id, event> exit_events;
    // The position in running_sequences from which the next step grants rows of sampled tokens.
    std::size_t sampled_row_scan_start = 0;

    static event& emplace_event(std::optional<event>& waited);
    static void wake(std::optional<event>& waited) noexcept;
    // The next step's entries: every decoding sequence, then the others' tokens up to prefill_tokens_per_step, never
    // past the end of a sequence's cycle. Leaves out the sequences whose last step is launched. Rows of sampled tokens
    // are granted from sampled_row_scan_start on, and a sequence whose rows would raise the step's above
    // executor::sampled_row_limit is left out. Also returns the position of the first sequence left out, or
    // sampled_row_scan_start when none was.
    std::pair<std::vector<executor::batch_tokens>, std::size_t> next_entries() const;
    // Writes the oldest launched step back, ends the sequences in it whose callbacks are cancelled, and lets each sequence
    // that exited leave once no launched step holds it.
    task<void> retire();
    task<void> serve(request r, unstarted_callbacks callbacks);

  public:
    scheduler(executor& exec, std::size_t prefill_tokens_per_step);

    // Serves r from the newest state the store holds for a prefix of its tokens; the executor takes callbacks and ends
    // them with the request's result or an error. The callbacks are ended with loop_exited when the returned task is
    // destroyed before the executor has taken them. The executor writes the state reached at the end of each eviction
    // cycle to the store, then passes it to the callbacks, and likewise the end state, and releases the state's lease
    // before it ends them. Ends with miss when the store's lookup fails. Ends with oom, never waiting, when the free
    // lines do not cover the state's pages and one cycle's tokens at the start, or the next cycle's tokens while
    // running; the caller resubmits.
    task<void> submit(request&& r, std::unique_ptr<request_callbacks>&& callbacks);
    // Steps the running sequences until the loop stops, launching each step before the previous one is written back
    // when executor::allows_overlap. A sequence whose callbacks are cancelled leaves after the
    // steps it is in.
    task<void> run();
};

}  // namespace pulsarlm
