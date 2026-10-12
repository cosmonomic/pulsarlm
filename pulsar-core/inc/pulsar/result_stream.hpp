#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "pulsar/digest.hpp"

namespace pulsar {

// The ordered page hashes of a view, then the tokens of its partial last page.
struct state {
    std::vector<digest> page_hashes;
    std::vector<std::int64_t> tail;
};

// The state a request reached when an eviction cycle ended. It covers the request's first token_count tokens: the
// initial state's tail, then the request's tokens, then the tokens it sampled.
struct cycle_end {
    state reached;
    std::size_t token_count;
};

enum class request_error { miss, oom, cancelled, loop_exited };

class result_writer;

// A request's sampled tokens and cycle ends, then its end: its final state or an error. Safe to use from any thread.
class result_stream {
    std::mutex mutex;
    std::vector<std::int64_t> tokens;
    std::vector<cycle_end> cycle_ends;
    std::optional<std::expected<state, request_error>> end;
    bool cancelled = false;

    friend class result_writer;

  public:
    struct taken {
        std::vector<std::int64_t> tokens;
        std::vector<cycle_end> cycle_ends;
        std::optional<std::expected<state, request_error>> end;
    };

    static std::pair<std::shared_ptr<result_stream>, result_writer> make();

    // The tokens and cycle ends appended since the last take, and the end once the stream ended. The end is taken
    // once.
    taken take();
    // The request ends with cancelled after its next step.
    void cancel();
};

// The loop thread's side of a result_stream. Destroyed before ending it, it ends the stream with loop_exited.
class result_writer {
    std::shared_ptr<result_stream> stream;

    explicit result_writer(std::shared_ptr<result_stream> stream) noexcept;

    friend class result_stream;

  public:
    result_writer(const result_writer&) = delete;
    result_writer& operator=(const result_writer&) = delete;
    result_writer(result_writer&& other) noexcept;
    result_writer& operator=(result_writer&&) = delete;
    ~result_writer();

    void append(std::int64_t token);
    void append(cycle_end ended);
    // Called once; the writer holds no stream after.
    void end(std::expected<state, request_error> result);
    bool cancelled() const;
};

}  // namespace pulsar
