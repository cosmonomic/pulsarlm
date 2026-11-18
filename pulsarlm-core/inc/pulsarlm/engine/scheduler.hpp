#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "pulsarlm/engine/event_loop.hpp"
#include "pulsarlm/engine/executor.hpp"
#include "pulsarlm/engine/request.hpp"
#include "pulsarlm/engine/request_callbacks.hpp"

namespace pulsarlm {

class sequence;
struct batch_entry;

// Decides what the executor runs: which requests join, which sequences each step forwards and how many tokens, and when
// a sequence leaves.
class scheduler {
    executor& exec;
    std::size_t prefill_tokens_per_step;
    std::vector<sequence*> running_sequences;
    // Set when a sequence joins the running set.
    std::optional<event> join_event;
    // The sequences of the launched step not written back yet, none when no step is, and whether that step is quiet
    // (executor::quiet).
    std::vector<sequence*> in_flight;
    bool in_flight_quiet = false;

    static event& emplace_event(std::optional<event>& waited);
    static void wake(std::optional<event>& waited) noexcept;
    // The next step's entries: every decoding sequence, then the others' tokens up to prefill_tokens_per_step, never
    // past the end of a sequence's cycle. Leaves out the sequences whose last step is launched.
    std::vector<batch_entry> next_entries() const;
    // Writes the step in flight back, ends the sequences in it whose callbacks are cancelled, and lets each sequence
    // that exited leave once no launched step holds it.
    task<void> retire();

  public:
    scheduler(executor& exec, std::size_t prefill_tokens_per_step);

    // Serves r from the newest state the store holds for a prefix of its tokens, then ends writer with its result or an
    // error. The executor writes the state reached at the end of each eviction cycle to the store, then appends it to
    // writer, and likewise the end state, and releases the state's lease before writer ends. Ends with miss when the
    // store's lookup fails. Ends with oom, never waiting, when the free lines do not cover the state's pages and one
    // cycle's tokens at the start, or the next cycle's tokens while running; the caller resubmits.
    task<void> submit(request r, result_writer writer);
    // Steps the running sequences until the loop stops, launching each step before the previous one is written back
    // when both are quiet (executor::quiet). A sequence whose callbacks are cancelled leaves after the steps it is in.
    task<void> run();
};

}  // namespace pulsarlm
