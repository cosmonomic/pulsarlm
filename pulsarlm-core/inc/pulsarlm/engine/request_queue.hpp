#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include "pulsarlm/engine/event_loop.hpp"
#include "pulsarlm/engine/executor.hpp"

namespace pulsarlm {

// The running requests of a scheduler, from which it takes the requests of each step.
class request_queue {
    executor& exec;
    std::size_t prefill_tokens_per_step;
    std::vector<executor::request_id_type> running_requests;
    // The position in running_requests from which the next step grants rows of sampled tokens.
    std::size_t sampled_row_scan_start = 0;
    // Awaited by wait_for_push.
    std::optional<event> changed;

  public:
    request_queue(executor& exec, std::size_t prefill_tokens_per_step);

    // Adds the request at id to the running requests and resumes a waiting wait_for_push.
    void push(executor::request_id_type id);
    // Drops the requests at ids from the running requests.
    void erase(std::span<const executor::request_id_type> ids);
    // The next step's scheduled requests, empty when no request is schedulable (executor::is_schedulable): every
    // decoding request, then the others' tokens up to prefill_tokens_per_step, never past the end of a request's cycle.
    // Rows of sampled tokens are granted from sampled_row_scan_start on, and a request whose rows would raise the
    // step's above executor::sampled_row_limit is left out; the scan of the next step starts at the first request left
    // out.
    std::vector<executor::scheduled_request> next();
    // Waits until the next push.
    task<void> wait_for_push();
};

}  // namespace pulsarlm
