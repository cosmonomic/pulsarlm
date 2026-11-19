#pragma once

#include <atomic>
#include <cstdint>
#include <expected>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "pulsarlm/engine.hpp"
#include "wake_fd.hpp"

namespace pulsarlm {

// The results appended since the last take. prefill is set by the take that first finds the prefill ended, and end,
// the request's final state or error with its statistics, by the take that first finds the request ended.
struct taken_results {
    std::optional<prefill_statistics> prefill;
    std::vector<std::pair<std::int64_t, decode_statistics>> tokens;
    std::vector<cycle_end> cycle_ends;
    std::optional<std::pair<std::expected<std::shared_ptr<const state>, request_error>, request_statistics>> end;
};

// A request's results, appended by its callbacks and taken by Python.
class request_buffer {
    std::mutex mutex;
    taken_results results;
    std::atomic<bool> cancel_flag = false;

  public:
    void append(const prefill_statistics& statistics);
    void append(std::int64_t token, const decode_statistics& statistics);
    void append(cycle_end&& ended);
    void end(std::expected<std::shared_ptr<const state>, request_error>&& result, const request_statistics& statistics);
    taken_results take();
    void cancel() noexcept;
    bool cancelled() const noexcept;
};

// Appends to a buffer and notifies a wake_fd after each append.
class buffered_callbacks final : public request_callbacks {
    std::shared_ptr<request_buffer> buffer;
    std::shared_ptr<wake_fd> wake_signal;

  public:
    buffered_callbacks(std::shared_ptr<request_buffer> buffer, std::shared_ptr<wake_fd> wake) noexcept;

    void on_prefill(const prefill_statistics& statistics) noexcept override;
    void on_decode(std::int64_t token, const decode_statistics& statistics) noexcept override;
    void on_cycle(cycle_end&& ended) noexcept override;
    void on_end(
        std::expected<std::shared_ptr<const state>, request_error>&& result,
        const request_statistics& statistics
    ) noexcept override;
    bool cancelled() const noexcept override;
};

}  // namespace pulsarlm
