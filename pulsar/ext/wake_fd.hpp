#pragma once

#include <atomic>

namespace pulsar {

// An eventfd (Linux-only) that request callbacks write after appending results. A notify while one is pending since
// the last clear skips the write, so a slow reader gets one wake for several appends.
class wake_fd {
    int fd;
    std::atomic<bool> pending = false;

  public:
    wake_fd();
    wake_fd(const wake_fd&) = delete;
    wake_fd& operator=(const wake_fd&) = delete;
    ~wake_fd();

    int fileno() const noexcept;
    // Safe from any thread.
    void notify() noexcept;
    // Called before taking the results that caused the wake.
    void clear() noexcept;
};

}  // namespace pulsar
