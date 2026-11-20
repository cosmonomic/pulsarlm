#include "pulsarlm/engine/request_queue.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace pulsarlm {

request_queue::request_queue(executor& exec, std::size_t prefill_tokens_per_step)
: exec(exec),
  prefill_tokens_per_step(prefill_tokens_per_step) {
    assert(prefill_tokens_per_step >= 1);
}

// The awaiters a set event scheduled no longer refer to it, so it is destroyed right away.
void request_queue::push(executor::request_id_type id) {
    running_requests.push_back(id);
    if (changed) {
        changed->set();
        changed.reset();
    }
}

void request_queue::erase(std::span<const executor::request_id_type> ids) {
    std::erase_if(running_requests, [&](executor::request_id_type id) { return std::ranges::contains(ids, id); });
}

std::vector<executor::scheduled_request> request_queue::next() {
    const std::size_t request_count = running_requests.size();
    std::vector<std::size_t> token_counts(request_count, 0);
    std::vector<bool> is_decoding(request_count, false);
    std::size_t budget = prefill_tokens_per_step;
    std::size_t row_total = 0;
    std::optional<std::size_t> first_index_left_out;
    for (std::size_t offset = 0; offset < request_count; offset++) {
        const std::size_t index = (sampled_row_scan_start + offset) % request_count;
        const executor::request_id_type id = running_requests[index];
        if (!exec.is_schedulable(id)) {
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
    sampled_row_scan_start = first_index_left_out.value_or(sampled_row_scan_start);
    return scheduled;
}

task<void> request_queue::wait_for_push() {
    assert(!changed);
    changed.emplace();
    co_await *changed;
}

}  // namespace pulsarlm
