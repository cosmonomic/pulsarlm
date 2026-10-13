#include "pulsarlm/event_loop.hpp"

#include <cuda_runtime_api.h>

#include <cassert>
#include <coroutine>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace pulsarlm {

namespace {

thread_local event_loop* running = nullptr;

using cuda_resolver = external_task<cudaError_t>::resolver;

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("synchronize: ") + cudaGetErrorString(status));
    }
}

void CUDART_CB resolve_synchronized(cudaStream_t, cudaError_t status, void* resolver) {
    std::unique_ptr<cuda_resolver> owned(static_cast<cuda_resolver*>(resolver));
    owned->resolve(std::move(status));
}

}  // namespace

// Outside run, the loop is being destroyed and frames are destroyed in any order, so an event can die before its
// awaiters; it detaches them.
event::~event() {
    assert(waiters == nullptr || event_loop::current() == nullptr);
    for (awaiter* waiter = waiters; waiter != nullptr; waiter = waiter->next) {
        waiter->linked = false;
    }
}

void event::set() noexcept {
    event_loop* loop = event_loop::current();
    assert(loop != nullptr);
    is_set = true;
    awaiter* waiter = std::exchange(waiters, nullptr);
    while (waiter != nullptr) {
        awaiter* following = waiter->next;
        waiter->linked = false;
        loop->schedule(waiter->scheduled);
        waiter = following;
    }
}

// A frame destroyed while awaiting, after a panic or with its loop, unlinks its awaiter.
event::awaiter::~awaiter() {
    if (!linked) {
        return;
    }
    awaiter** link = &target.waiters;
    while (*link != this) {
        link = &(*link)->next;
    }
    *link = next;
}

void event::awaiter::await_suspend(std::coroutine_handle<> awaiting) noexcept {
    scheduled.handle = awaiting;
    linked = true;
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
        ready_front = nullptr;
        ready_back = nullptr;
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

void event_loop::push_ready(ready_node& node) noexcept {
    node.next = nullptr;
    if (ready_back == nullptr) {
        ready_front = &node;
    } else {
        ready_back->next = &node;
    }
    ready_back = &node;
}

void event_loop::schedule(ready_node& node) noexcept {
    std::lock_guard lock(mutex);
    push_ready(node);
    wake.notify_one();
}

void event_loop::spawn(task<void> root_task) {
    assert(root_task.handle);
    std::coroutine_handle<> handle = root_task.handle;
    std::lock_guard lock(mutex);
    auto [it, inserted] = root_tasks.try_emplace(handle, spawned_task{std::move(root_task), {.handle = handle}});
    push_ready(it->second.scheduled);
    wake.notify_one();
}

std::exception_ptr event_loop::run() {
    assert(running == nullptr);
    running = this;
    while (exception == nullptr) {
        ready_node* resuming;
        {
            std::unique_lock lock(mutex);
            wake.wait(lock, [this] { return stopping || ready_front != nullptr; });
            if (stopping) {
                break;
            }
            resuming = std::exchange(ready_front, nullptr);
            ready_back = nullptr;
        }
        // A node lives in its coroutine's frame, which resuming can destroy, so next is read first.
        while (resuming != nullptr && exception == nullptr) {
            std::coroutine_handle<> handle = resuming->handle;
            resuming = resuming->next;
            handle.resume();
            if (completed_root_task) {
                destroy_completed_root_task();
            }
        }
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
    check_cuda(cudaStreamAddCallback(stream, resolve_synchronized, owned.get(), 0));
    owned.release();
    check_cuda(co_await std::move(done));
}

}  // namespace pulsarlm
