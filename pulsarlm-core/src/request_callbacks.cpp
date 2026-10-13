#include "pulsarlm/request_callbacks.hpp"

#include <cassert>
#include <cstdint>
#include <expected>
#include <memory>
#include <utility>

namespace pulsarlm {

result_writer::result_writer(std::unique_ptr<request_callbacks> callbacks) noexcept
: callbacks(std::move(callbacks)) {}

result_writer::~result_writer() {
    if (callbacks) {
        end(std::unexpected(request_error::loop_exited));
    }
}

void result_writer::append(std::int64_t token) noexcept {
    assert(callbacks);
    callbacks->on_token(token);
}

void result_writer::append(cycle_end&& ended) noexcept {
    assert(callbacks);
    callbacks->on_cycle_end(std::move(ended));
}

void result_writer::end(std::expected<state, request_error>&& result) noexcept {
    assert(callbacks);
    std::unique_ptr<request_callbacks> ended = std::move(callbacks);
    ended->on_end(std::move(result));
}

bool result_writer::cancelled() const noexcept {
    assert(callbacks);
    return callbacks->cancelled();
}

}  // namespace pulsarlm
