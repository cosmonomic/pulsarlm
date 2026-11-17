#include "pulsarlm/engine/pending_writes.hpp"

#include <cassert>

namespace pulsarlm {

void pending_writes::start() noexcept {
    count++;
}

// The awaiters a set event scheduled no longer refer to it, so it is destroyed right away.
void pending_writes::finish(bool written) noexcept {
    assert(count > 0);
    count--;
    failed = failed || !written;
    if (count == 0 && drained) {
        drained->set();
        drained.reset();
    }
}

std::size_t pending_writes::size() const noexcept {
    return count;
}

bool pending_writes::any_failed() const noexcept {
    return failed;
}

task<void> pending_writes::wait() {
    while (count > 0) {
        if (!drained) {
            drained.emplace();
        }
        co_await *drained;
    }
}

}  // namespace pulsarlm
