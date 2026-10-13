#include "wake_fd.hpp"

#include <sys/eventfd.h>
#include <unistd.h>

#include <cassert>
#include <cerrno>
#include <cstdint>
#include <system_error>

namespace pulsarlm {

wake_fd::wake_fd()
: fd(eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC)) {
    if (fd < 0) {
        throw std::system_error(errno, std::system_category(), "eventfd");
    }
}

wake_fd::~wake_fd() {
    close(fd);
}

int wake_fd::fileno() const noexcept {
    return fd;
}

void wake_fd::notify() noexcept {
    if (pending.exchange(true)) {
        return;
    }
    std::uint64_t one = 1;
    [[maybe_unused]] ssize_t written = write(fd, &one, sizeof one);
    assert(written == sizeof one);
}

// The counter is read before pending is cleared: a notify that skipped its write while pending was set appended its
// results before the clear, so the take that follows the clear sees them.
void wake_fd::clear() noexcept {
    std::uint64_t count;
    [[maybe_unused]] ssize_t read_size = read(fd, &count, sizeof count);
    assert(read_size == sizeof count || errno == EAGAIN);
    pending.store(false);
}

}  // namespace pulsarlm
