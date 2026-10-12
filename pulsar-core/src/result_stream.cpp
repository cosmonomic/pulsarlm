#include "pulsar/result_stream.hpp"

#include <cassert>
#include <cstdint>
#include <expected>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>

namespace pulsar {

std::pair<std::shared_ptr<result_stream>, result_writer> result_stream::make() {
    std::shared_ptr<result_stream> stream = std::make_shared<result_stream>();
    return {stream, result_writer(stream)};
}

result_stream::taken result_stream::take() {
    std::lock_guard lock(mutex);
    return taken{
        .tokens = std::exchange(tokens, {}),
        .cycle_ends = std::exchange(cycle_ends, {}),
        .end = std::exchange(end, std::nullopt),
    };
}

void result_stream::cancel() {
    std::lock_guard lock(mutex);
    cancelled = true;
}

result_writer::result_writer(std::shared_ptr<result_stream> stream) noexcept
: stream(std::move(stream)) {}

result_writer::result_writer(result_writer&& other) noexcept
: stream(std::move(other.stream)) {}

result_writer::~result_writer() {
    if (stream) {
        end(std::unexpected(request_error::loop_exited));
    }
}

void result_writer::append(std::int64_t token) {
    assert(stream);
    std::lock_guard lock(stream->mutex);
    stream->tokens.push_back(token);
}

void result_writer::append(cycle_end ended) {
    assert(stream);
    std::lock_guard lock(stream->mutex);
    stream->cycle_ends.push_back(std::move(ended));
}

void result_writer::end(std::expected<state, request_error> result) {
    assert(stream);
    {
        std::lock_guard lock(stream->mutex);
        stream->end.emplace(std::move(result));
    }
    stream.reset();
}

bool result_writer::cancelled() const {
    assert(stream);
    std::lock_guard lock(stream->mutex);
    return stream->cancelled;
}

}  // namespace pulsar
