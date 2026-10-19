#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <vector>

#include "pulsarlm/digest.hpp"

namespace pulsarlm {

// A page of a sequence: its hash, its index among the history's full pages, and its eviction density in each query head
// its layer measures mass in, 0 for a page never measured. A state the scheduler reaches holds one per head; an initial
// state may hold none, which reads as 0.
struct active_page {
    digest hash;
    std::size_t index;
    std::vector<float> density;

    bool operator==(const active_page&) const = default;
};

// One KV layer's active pages, in view order.
struct layer_sequence {
    std::vector<active_page> pages;

    bool operator==(const layer_sequence&) const = default;
};

// head is the hash of the newest full page of the history, which new pages chain from and every non-empty sequence
// ends with; absent before the first full page. layers holds one sequence per KV layer, and may be empty when head is
// absent. tail holds the tokens after head. A sequence's hashes are distinct: they come from one digest chain. A state
// that breaks this is undefined behaviour.
struct state {
    std::optional<digest> head;
    std::vector<layer_sequence> layers;
    std::vector<std::int64_t> tail;
};

// The state a request reached when an eviction cycle ended. It covers the request's first token_count tokens: the
// initial state's tail, then the request's tokens, then the tokens it sampled.
struct cycle_end {
    state reached;
    std::size_t token_count;
};

enum class request_error { miss, oom, cancelled, loop_exited };

// One request's sampled tokens and cycle ends, then its end: its final state or an error. Calls come one at a time and
// must not block; all but an on_end with loop_exited come from the loop thread.
class request_callbacks {
  public:
    virtual ~request_callbacks() = default;

    virtual void on_token(std::int64_t token) noexcept = 0;
    virtual void on_cycle_end(cycle_end&& ended) noexcept = 0;
    // Called once, last.
    virtual void on_end(std::expected<state, request_error>&& result) noexcept = 0;
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

    void append(std::int64_t token) noexcept;
    void append(cycle_end&& ended) noexcept;
    // Called once; the callbacks are destroyed once it returns.
    void end(std::expected<state, request_error>&& result) noexcept;
    bool cancelled() const noexcept;
};

}  // namespace pulsarlm
