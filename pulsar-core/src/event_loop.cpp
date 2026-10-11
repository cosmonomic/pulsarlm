#include "pulsar/event_loop.hpp"

#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <coroutine>
#include <exception>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace pulsar {

namespace {

thread_local event_loop* running = nullptr;

using cuda_resolver = external_task<cudaError_t>::resolver;

void CUDART_CB resolve_synchronized(cudaStream_t, cudaError_t status, void* resolver) {
    std::unique_ptr<cuda_resolver> owned(static_cast<cuda_resolver*>(resolver));
    owned->resolve(status);
}

}  // namespace

// Outside run, the loop is being destroyed and frames are destroyed in any order, so an event can die before its
// awaiters; it detaches them.
event::~event() {
    assert(waiters == nullptr || event_loop::current() == nullptr);
    for (awaiter* waiter = waiters; waiter != nullptr; waiter = waiter->next) {
        waiter->handle = {};
    }
}

void event::set() noexcept {
    event_loop* loop = event_loop::current();
    assert(loop != nullptr);
    is_set = true;
    for (awaiter* waiter = std::exchange(waiters, nullptr); waiter != nullptr; waiter = waiter->next) {
        loop->schedule(std::exchange(waiter->handle, {}));
    }
}

// A frame destroyed while awaiting, after a panic or with its loop, unlinks its awaiter.
event::awaiter::~awaiter() {
    if (!handle) {
        return;
    }
    awaiter** link = &target.waiters;
    while (*link != this) {
        link = &(*link)->next;
    }
    *link = next;
}

void event::awaiter::await_suspend(std::coroutine_handle<> awaiting) noexcept {
    handle = awaiting;
    next = std::exchange(target.waiters, this);
}

event_loop* event_loop::current() noexcept {
    return running;
}

// Root tasks are destroyed after unlocking: destroying a frame can lock an external_task's shared state, and resolve
// locks the shared state before the loop.
event_loop::~event_loop() {
    decltype(root_tasks) remaining;
    {
        std::lock_guard lock(mutex);
        ready.clear();
        remaining.swap(root_tasks);
    }
}

void event_loop::panic(std::exception_ptr thrown) noexcept {
    if (exception == nullptr) {
        exception = std::move(thrown);
    }
}

// The root task is destroyed after unlocking, as in ~event_loop.
void event_loop::destroy_completed_root_task() noexcept {
    decltype(root_tasks)::node_type completed;
    {
        std::lock_guard lock(mutex);
        completed = root_tasks.extract(std::exchange(completed_root_task, {}));
    }
}

void event_loop::schedule(std::coroutine_handle<> handle) {
    std::lock_guard lock(mutex);
    ready.push_back(handle);
    wake.notify_one();
}

void event_loop::spawn(task<void> root_task) {
    assert(root_task.handle);
    std::coroutine_handle<> handle = root_task.handle;
    std::lock_guard lock(mutex);
    root_tasks.emplace(handle, std::move(root_task));
    ready.push_back(handle);
    wake.notify_one();
}

std::exception_ptr event_loop::run() {
    assert(running == nullptr);
    running = this;
    std::vector<std::coroutine_handle<>> resuming;
    while (exception == nullptr) {
        {
            std::unique_lock lock(mutex);
            wake.wait(lock, [this] { return stopping || !ready.empty(); });
            if (stopping) {
                break;
            }
            resuming.swap(ready);
        }
        for (std::coroutine_handle<> handle : resuming) {
            handle.resume();
            if (completed_root_task) {
                destroy_completed_root_task();
            }
            if (exception != nullptr) {
                break;
            }
        }
        resuming.clear();
    }
    running = nullptr;
    return exception;
}

void event_loop::stop() noexcept {
    std::lock_guard lock(mutex);
    stopping = true;
    wake.notify_one();
}

task<void> synchronize(cudaStream_t stream) {
    auto [done, resolver] = external_task<cudaError_t>::make();
    auto owned = std::make_unique<cuda_resolver>(std::move(resolver));
    cudaError_t enqueued = cudaStreamAddCallback(stream, resolve_synchronized, owned.get(), 0);
    TORCH_CHECK(enqueued == cudaSuccess, "synchronize: ", cudaGetErrorString(enqueued));
    owned.release();
    cudaError_t status = co_await std::move(done);
    TORCH_CHECK(status == cudaSuccess, "synchronize: ", cudaGetErrorString(status));
}

}  // namespace pulsar
