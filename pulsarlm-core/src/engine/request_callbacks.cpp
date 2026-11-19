#include "pulsarlm/engine/request_callbacks.hpp"

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
        end(std::unexpected(request_error::loop_exited), request_statistics{});
    }
}

void result_writer::append(const prefill_statistics& statistics) noexcept {
    assert(callbacks);
    callbacks->on_prefill(statistics);
}

void result_writer::append(std::int64_t token, const decode_statistics& statistics) noexcept {
    assert(callbacks);
    callbacks->on_decode(token, statistics);
}

void result_writer::append(cycle_end&& ended) noexcept {
    assert(callbacks);
    callbacks->on_cycle(std::move(ended));
}

void result_writer::end(
    std::expected<std::shared_ptr<const state>, request_error>&& result,
    const request_statistics& statistics
) noexcept {
    assert(callbacks);
    std::unique_ptr<request_callbacks> ended = std::move(callbacks);
    ended->on_end(std::move(result), statistics);
}

bool result_writer::cancelled() const noexcept {
    assert(callbacks);
    return callbacks->cancelled();
}

}  // namespace pulsarlm
