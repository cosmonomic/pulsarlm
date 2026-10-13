#pragma once

#include <atomic>
#include <cstdint>
#include <expected>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "pulsar/request_callbacks.hpp"
#include "wake_fd.hpp"

namespace pulsar {

// The results appended since the last take. end is set by the take that first finds the request ended.
struct taken_results {
    std::vector<std::int64_t> tokens;
    std::vector<cycle_end> cycle_ends;
    std::optional<std::expected<state, request_error>> end;
};

// A request's results, appended by its callbacks and taken by Python.
class request_buffer {
    std::mutex mutex;
    taken_results appended;
    std::atomic<bool> cancel_flag = false;

  public:
    void append(std::int64_t token);
    void append(cycle_end&& ended);
    void end(std::expected<state, request_error>&& result);
    taken_results take();
    void cancel() noexcept;
    bool cancelled() const noexcept;
};

// Appends to a buffer and notifies a wake_fd after each append.
class buffered_callbacks final : public request_callbacks {
    std::shared_ptr<request_buffer> buffer;
    std::shared_ptr<wake_fd> wake;

  public:
    buffered_callbacks(std::shared_ptr<request_buffer> buffer, std::shared_ptr<wake_fd> wake) noexcept;

    void on_token(std::int64_t token) noexcept override;
    void on_cycle_end(cycle_end&& ended) noexcept override;
    void on_end(std::expected<state, request_error>&& result) noexcept override;
    bool cancelled() const noexcept override;
};

}  // namespace pulsar
