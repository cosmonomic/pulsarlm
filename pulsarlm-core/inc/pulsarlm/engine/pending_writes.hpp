#pragma once

#include <cstddef>
#include <optional>

#include "pulsarlm/engine/event_loop.hpp"

namespace pulsarlm {

// Writes to the store that a write-back started without waiting for them.
class pending_writes {
    std::size_t count = 0;
    bool failed = false;
    std::optional<event> drained;

  public:
    void start() noexcept;
    // Ends a write, which failed unless written.
    void finish(bool written) noexcept;
    std::size_t size() const noexcept;
    bool any_failed() const noexcept;
    // Completes once no write is pending.
    task<void> wait();
};

}  // namespace pulsarlm
