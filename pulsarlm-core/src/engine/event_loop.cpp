#include "pulsarlm/engine/event_loop.hpp"

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
        loop->schedule(waiter->queue_node);
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
    queue_node.handle = awaiting;
    linked = true;
    next = std::exchange(target.waiters, this);
}

semaphore::semaphore(std::size_t count) noexcept
: count(count) {}

// Outside run, the loop is being destroyed and frames are destroyed in any order, so a semaphore can die before its
// awaiters; it detaches them.
semaphore::~semaphore() {
    assert(head == nullptr || event_loop::current() == nullptr);
    for (awaiter* waiter = head; waiter != nullptr; waiter = waiter->next) {
        waiter->linked = false;
    }
}

// The permit passes to the first waiter without going through count.
void semaphore::release() noexcept {
    awaiter* waiter = head;
    if (waiter == nullptr) {
        count++;
        return;
    }
    event_loop* loop = event_loop::current();
    assert(loop != nullptr);
    head = waiter->next;
    if (head == nullptr) {
        tail = nullptr;
    }
    waiter->linked = false;
    loop->schedule(waiter->queue_node);
}

bool semaphore::awaiter::await_suspend(std::coroutine_handle<> awaiting) noexcept {
    if (target.count > 0) {
        target.count--;
        return false;
    }
    queue_node.handle = awaiting;
    linked = true;
    if (target.tail == nullptr) {
        target.head = this;
    } else {
        target.tail->next = this;
    }
    target.tail = this;
    return true;
}

// A frame destroyed while awaiting, after a panic or with its loop, unlinks its awaiter.
semaphore::awaiter::~awaiter() {
    if (!linked) {
        return;
    }
    awaiter* previous = nullptr;
    awaiter** link = &target.head;
    while (*link != this) {
        previous = *link;
        link = &(*link)->next;
    }
    *link = next;
    if (target.tail == this) {
        target.tail = previous;
    }
}

semaphore_guard::semaphore_guard(semaphore& target) noexcept
: target(&target) {}

semaphore_guard::semaphore_guard(semaphore_guard&& other) noexcept
: target(std::exchange(other.target, nullptr)) {}

semaphore_guard& semaphore_guard::operator=(semaphore_guard&& other) noexcept {
    if (this != &other) {
        if (target != nullptr) {
            target->release();
        }
        target = std::exchange(other.target, nullptr);
    }
    return *this;
}

semaphore_guard::~semaphore_guard() {
    if (target != nullptr) {
        target->release();
    }
}

void shared_task_state::remove_reference() noexcept {
    if (reference_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::coroutine_handle<> frame = handle;
        frame.destroy();
    }
}

void shared_task_state::end() noexcept {
    ended = true;
    waiter* awaiting = std::exchange(waiters, nullptr);
    assert(awaiting == nullptr || event_loop::current() != nullptr);
    while (awaiting != nullptr) {
        waiter* following = awaiting->next;
        awaiting->linked = false;
        event_loop::current()->schedule(awaiting->queue_node);
        awaiting = following;
    }
}

// A frame destroyed while awaiting, after a panic or with its loop, unlinks its waiter.
shared_task_state::waiter::~waiter() {
    if (!linked) {
        return;
    }
    waiter** link = &target.waiters;
    while (*link != this) {
        link = &(*link)->next;
    }
    *link = next;
}

void shared_task_state::waiter::link(std::coroutine_handle<> awaiting) noexcept {
    queue_node.handle = awaiting;
    linked = true;
    next = std::exchange(target.waiters, this);
}

shared_task<void> make_ready_task() {
    shared_task<void> ready = []() -> shared_task<void> { co_return; }();
    ready.state->started = true;
    ready.state->handle.resume();
    return ready;
}

event_loop* event_loop::current() noexcept {
    return running;
}

// Root tasks are dropped after unlocking: destroying a frame can lock an external_task's shared state, and resolve
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

// The root task is dropped after unlocking, as in ~event_loop.
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
    wake_condition.notify_one();
}

void event_loop::schedule_root_task(shared_task<void> root_task) {
    std::coroutine_handle<> handle = root_task.state->handle;
    std::lock_guard lock(mutex);
    auto [it, inserted] = root_tasks.try_emplace(handle, spawned_task{std::move(root_task), {.handle = handle}});
    assert(inserted);
    push_ready(it->second.queue_node);
    wake_condition.notify_one();
}

std::exception_ptr event_loop::run() {
    assert(running == nullptr);
    running = this;
    while (exception == nullptr) {
        ready_node* resuming;
        {
            std::unique_lock lock(mutex);
            wake_condition.wait(lock, [this] { return stopping || ready_front != nullptr; });
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
    wake_condition.notify_one();
}

task<void> yield() {
    struct awaiter {
        event_loop::ready_node queue_node;

        bool await_ready() const noexcept {
            return false;
        }

        void await_suspend(std::coroutine_handle<> awaiting) noexcept {
            queue_node.handle = awaiting;
            event_loop::current()->schedule(queue_node);
        }

        void await_resume() const noexcept {}
    };
    co_await awaiter{};
}

task<void> synchronize(cudaStream_t stream) {
    check_cuda(co_await await_stream(stream));
}

external_task<cudaError_t> await_stream(cudaStream_t stream) {
    auto [done, resolver] = external_task<cudaError_t>::make();
    auto owned = std::make_unique<cuda_resolver>(std::move(resolver));
    check_cuda(cudaStreamAddCallback(stream, resolve_synchronized, owned.get(), 0));
    owned.release();
    return std::move(done);
}

}  // namespace pulsarlm
