#include "request_buffer.hpp"

#include <utility>

namespace pulsarlm {

void request_buffer::append(const prefill_statistics& statistics) {
    std::lock_guard lock(mutex);
    results.prefill = statistics;
}

void request_buffer::append(std::int64_t token, const decode_statistics& statistics) {
    std::lock_guard lock(mutex);
    results.tokens.emplace_back(token, statistics);
}

void request_buffer::append(cycle_end&& ended) {
    std::lock_guard lock(mutex);
    results.cycle_ends.push_back(std::move(ended));
}

void request_buffer::end(
    std::expected<std::shared_ptr<const state>, request_error>&& result,
    const request_statistics& statistics
) {
    std::lock_guard lock(mutex);
    results.end.emplace(std::move(result), statistics);
}

taken_results request_buffer::take() {
    std::lock_guard lock(mutex);
    return std::exchange(results, {});
}

void request_buffer::cancel() noexcept {
    cancel_flag.store(true);
}

bool request_buffer::cancelled() const noexcept {
    return cancel_flag.load();
}

buffered_callbacks::buffered_callbacks(std::shared_ptr<request_buffer> buffer, std::shared_ptr<wake_fd> wake) noexcept
: buffer(std::move(buffer)),
  wake_signal(std::move(wake)) {}

void buffered_callbacks::on_prefill(const prefill_statistics& statistics) noexcept {
    buffer->append(statistics);
    wake_signal->notify();
}

void buffered_callbacks::on_decode(std::int64_t token, const decode_statistics& statistics) noexcept {
    buffer->append(token, statistics);
    wake_signal->notify();
}

void buffered_callbacks::on_cycle(cycle_end&& ended) noexcept {
    buffer->append(std::move(ended));
    wake_signal->notify();
}

void buffered_callbacks::on_end(
    std::expected<std::shared_ptr<const state>, request_error>&& result,
    const request_statistics& statistics
) noexcept {
    buffer->end(std::move(result), statistics);
    wake_signal->notify();
}

bool buffered_callbacks::cancelled() const noexcept {
    return buffer->cancelled();
}

}  // namespace pulsarlm
