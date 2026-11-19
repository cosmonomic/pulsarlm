#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <vector>

#include "pulsarlm/digest.hpp"
#include "pulsarlm/engine/state.hpp"

namespace pulsarlm {

// The state a request reached when an eviction cycle ended.
struct cycle_end {
    std::shared_ptr<const pulsarlm::state> state;
};

struct prefill_statistics {
    // The prompt tokens taken from stored pages, and those of them on pages hot on the device.
    std::size_t cached_token_count;
    std::size_t device_cached_token_count;
    // The prompt tokens the model computed, and the steps that computed them.
    std::size_t computed_token_count;
    std::size_t step_count;
    // Launch to write-back of each of those steps, summed.
    std::chrono::steady_clock::duration step_duration;
    // The bytes of each device's KV cache pool that the request's lines take when the prefill ends.
    std::size_t device_memory;
};

struct decode_statistics {
    // The bytes of each device's KV cache pool that the request's lines take when the token is delivered.
    std::size_t device_memory;
    // The drafts verified by the step that delivered the token, and those of them the step accepted.
    std::size_t drafted_token_count;
    std::size_t accepted_draft_count;
    // Launch to write-back of the step that delivered the token.
    std::chrono::steady_clock::duration step_duration;
};

struct request_statistics {
    // The request's tokens, then the tokens it sampled.
    std::size_t total_token_count;
    // The tokens of the full pages of the state the request resumed, which it did not compute; that state's tail is
    // computed again.
    std::size_t cached_token_count;
    // Those of the cached tokens on pages that every KV layer reading the page had hot in the KV cache when the request
    // started, not read from the store.
    std::size_t device_cached_token_count;
    // The most bytes of each device's KV cache pool that the request's lines took at once, lines it shared with other
    // requests included.
    std::size_t peak_device_memory;
    // The drafts the request's verify steps verified, and those of them they accepted.
    std::size_t drafted_token_count;
    std::size_t accepted_draft_count;
    // Launch to write-back of every step of the request, summed.
    std::chrono::steady_clock::duration step_duration;
};

enum class request_error { miss, oom, cancelled, loop_exited };

// One request's prefill statistics, sampled tokens and cycle ends, then its end: its final state or an error. Calls come
// one at a time and must not block; all but an on_end with loop_exited come from the loop thread.
class request_callbacks {
  public:
    virtual ~request_callbacks() = default;

    // Called once, before any on_decode, when the step that computed the last prompt token is written back; not called
    // for a request that fails before then.
    virtual void on_prefill(const prefill_statistics& statistics) noexcept = 0;
    virtual void on_decode(std::int64_t token, const decode_statistics& statistics) noexcept = 0;
    virtual void on_cycle(cycle_end&& ended) noexcept = 0;
    // Called once, last, with the request's final state, which leaves out the last sampled token, or its error, and its
    // statistics up to its end, all zero for a request that never started.
    virtual void on_end(
        std::expected<std::shared_ptr<const pulsarlm::state>, request_error>&& result,
        const request_statistics& statistics
    ) noexcept = 0;
    // Checked after each step; true ends the request with cancelled.
    virtual bool cancelled() const noexcept = 0;
};

// The engine's side of a request's callbacks. Destroyed before ending them, it ends them with loop_exited.
class result_writer {
    std::unique_ptr<request_callbacks> callbacks;

  public:
    explicit result_writer(std::unique_ptr<request_callbacks> callbacks) noexcept;
    result_writer(const result_writer&) = delete;
    result_writer& operator=(const result_writer&) = delete;
    result_writer(result_writer&& other) noexcept = default;
    result_writer& operator=(result_writer&&) = delete;
    ~result_writer();

    void append(const prefill_statistics& statistics) noexcept;
    void append(std::int64_t token, const decode_statistics& statistics) noexcept;
    void append(cycle_end&& ended) noexcept;
    // Called once; the callbacks are destroyed once it returns.
    void
    end(std::expected<std::shared_ptr<const pulsarlm::state>, request_error>&& result,
        const request_statistics& statistics) noexcept;
    bool cancelled() const noexcept;
};

}  // namespace pulsarlm
