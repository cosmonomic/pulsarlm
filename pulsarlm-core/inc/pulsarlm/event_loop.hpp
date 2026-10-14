#pragma once

#ifdef __CUDACC__
#error "event_loop.hpp holds coroutine types; keep it out of .cu files"
#endif

#include <cuda_runtime_api.h>

#include <cassert>
#include <condition_variable>
#include <coroutine>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>

namespace pulsarlm {

class event_loop;

// Starts when awaited and resumes its awaiter when it completes. Awaiting consumes the task: await a prvalue or
// std::move(t). An exception escaping the coroutine panics event_loop::current().
template <typename t> class [[nodiscard]] task {
  public:
    class promise_type;

  private:
    class value_result;
    class void_result;
    class awaiter;

    std::coroutine_handle<promise_type> handle;

    explicit task(std::coroutine_handle<promise_type> handle) noexcept;

    friend class event_loop;

  public:
    task(const task&) = delete;
    task& operator=(const task&) = delete;
    task(task&& other) noexcept;
    task& operator=(task&& other) noexcept;
    ~task();

    awaiter operator co_await() && noexcept;
    awaiter operator co_await() & = delete;
};

// Once set, schedules every coroutine awaiting it on event_loop::current(); awaiting it after does not suspend. Set and
// await it on the loop thread. Awaiters link into the event, so it does not move, and it must not be destroyed while
// awaited during run.
class event {
    class awaiter;

    awaiter* waiters = nullptr;
    bool is_set = false;

  public:
    event() noexcept = default;
    event(const event&) = delete;
    event& operator=(const event&) = delete;
    ~event();

    void set() noexcept;
    awaiter operator co_await() noexcept;
};

// task with a resolve callback that can set the result outside the event loop.
template <typename value_type = std::monostate> class [[nodiscard]] external_task {
    struct state;
    using shared_state = std::shared_ptr<state>;
    class awaiter;

    shared_state shared;

    explicit external_task(shared_state shared) noexcept;

    void abandon() noexcept;

  public:
    class resolver;

    static std::pair<external_task, resolver> make();

    external_task(const external_task&) = delete;
    external_task& operator=(const external_task&) = delete;
    external_task(external_task&& other) noexcept;
    external_task& operator=(external_task&& other) noexcept;
    ~external_task();

    awaiter operator co_await() && noexcept;
};

// Safe to call from any thread, once. Does nothing once its external_task was destroyed, so the value is dropped.
template <typename value_type> class external_task<value_type>::resolver {
    shared_state shared;

    explicit resolver(shared_state shared) noexcept;

    friend class external_task;

  public:
    void resolve(value_type&& value = {}) const;
};

class event_loop {
    // A coroutine linked into the ready queue, so scheduling it does not allocate.
    struct ready_node {
        std::coroutine_handle<> handle;
        ready_node* next = nullptr;
    };

    struct spawned_task {
        task<void> owned;
        ready_node scheduled;
    };

    std::mutex mutex;
    std::condition_variable wake;
    ready_node* ready_front = nullptr;
    ready_node* ready_back = nullptr;
    bool stopping = false;
    std::unordered_map<std::coroutine_handle<>, spawned_task> root_tasks;
    // Written and read on the loop thread only.
    std::exception_ptr exception;
    // A root task's final suspend returns to run, so one resume completes at most one root task.
    std::coroutine_handle<> completed_root_task;

    void panic(std::exception_ptr thrown) noexcept;
    void destroy_completed_root_task() noexcept;
    void push_ready(ready_node& node) noexcept;
    // Safe to call from any thread. node stays linked until its coroutine is resumed or the loop is destroyed.
    void schedule(ready_node& node) noexcept;

    template <typename> friend class task;
    template <typename> friend class external_task;
    friend class event;

  public:
    // The loop running on the calling thread, null outside run.
    static event_loop* current() noexcept;

    // Destroys the remaining root tasks without resuming them.
    ~event_loop();

    // Safe to call from any thread.
    void spawn(task<void> root_task);
    // Resumes scheduled coroutines on the calling thread until stop, or until a task panics. Returns the panic's
    // exception, or null after stop.
    std::exception_ptr run();
    // Safe to call from any thread.
    void stop() noexcept;
};

// Completes once the work queued on stream so far has finished. Uses cudaStreamAddCallback, which unlike
// cudaLaunchHostFunc also runs after a CUDA error; an error panics.
task<void> synchronize(cudaStream_t stream);

template <typename t> class task<t>::value_result {
    std::optional<t> value;

  public:
    void return_value(t&& returned) {
        value.emplace(std::move(returned));
    }

    t take() {
        return std::move(*value);
    }
};

template <typename t> class task<t>::void_result {
  public:
    void return_void() noexcept {}

    void take() noexcept {}
};

template <typename t>
class task<t>::promise_type : public std::conditional_t<std::is_void_v<t>, void_result, value_result> {
    struct final_awaiter;

    std::coroutine_handle<> continuation;
    bool panicked = false;

    friend class task::awaiter;

  public:
    task get_return_object() noexcept {
        return task(std::coroutine_handle<promise_type>::from_promise(*this));
    }

    std::suspend_always initial_suspend() noexcept {
        return {};
    }

    final_awaiter final_suspend() noexcept {
        return {};
    }

    void unhandled_exception() noexcept {
        panicked = true;
        event_loop::current()->panic(std::current_exception());
    }
};

// A panicked task resumes nothing. A task without continuation is a root task.
template <typename t> struct task<t>::promise_type::final_awaiter {
    bool await_ready() const noexcept {
        return false;
    }

    std::coroutine_handle<> await_suspend(std::coroutine_handle<promise_type> handle) const noexcept {
        promise_type& promise = handle.promise();
        if (promise.panicked) {
            return std::noop_coroutine();
        }
        if (promise.continuation) {
            return promise.continuation;
        }
        event_loop::current()->completed_root_task = handle;
        return std::noop_coroutine();
    }

    void await_resume() const noexcept {}
};

// Owns the awaited frame, which it destroys when the co_await expression ends.
template <typename t> class task<t>::awaiter {
    std::coroutine_handle<promise_type> handle;

  public:
    explicit awaiter(std::coroutine_handle<promise_type> handle) noexcept
    : handle(handle) {}

    awaiter(const awaiter&) = delete;
    awaiter& operator=(const awaiter&) = delete;

    ~awaiter() {
        if (handle) {
            handle.destroy();
        }
    }

    bool await_ready() const noexcept {
        assert(handle);
        return false;
    }

    std::coroutine_handle<> await_suspend(std::coroutine_handle<> awaiting) noexcept {
        handle.promise().continuation = awaiting;
        return handle;
    }

    t await_resume() {
        return handle.promise().take();
    }
};

template <typename t>
task<t>::task(std::coroutine_handle<promise_type> handle) noexcept
: handle(handle) {}

template <typename t>
task<t>::task(task&& other) noexcept
: handle(std::exchange(other.handle, {})) {}

template <typename t> task<t>& task<t>::operator=(task&& other) noexcept {
    if (this != &other) {
        if (handle) {
            handle.destroy();
        }
        handle = std::exchange(other.handle, {});
    }
    return *this;
}

template <typename t> task<t>::~task() {
    if (handle) {
        handle.destroy();
    }
}

template <typename t> typename task<t>::awaiter task<t>::operator co_await() && noexcept {
    return awaiter(std::exchange(handle, {}));
}

class event::awaiter {
    event& target;
    event_loop::ready_node scheduled;
    awaiter* next = nullptr;
    bool linked = false;

    friend class event;

  public:
    explicit awaiter(event& target) noexcept
    : target(target) {}

    awaiter(const awaiter&) = delete;
    awaiter& operator=(const awaiter&) = delete;
    ~awaiter();

    bool await_ready() const noexcept {
        return target.is_set;
    }

    void await_suspend(std::coroutine_handle<> awaiting) noexcept;

    void await_resume() const noexcept {}
};

inline event::awaiter event::operator co_await() noexcept {
    return awaiter(*this);
}

template <typename value_type> struct external_task<value_type>::state {
    std::mutex mutex;
    std::optional<value_type> value;
    // Its handle is null until awaited.
    event_loop::ready_node waiting;
    event_loop* loop = nullptr;
    bool alive = true;

    // A dropped value is destroyed by the caller after the lock is released, so its destructor never runs under the
    // lock.
    void set_value(value_type&& resolved) {
        std::lock_guard lock(mutex);
        if (!alive) {
            return;
        }
        assert(!value);
        value.emplace(std::move(resolved));
        if (waiting.handle) {
            loop->schedule(waiting);
        }
    }
};

template <typename value_type> class external_task<value_type>::awaiter {
    external_task owned;

  public:
    explicit awaiter(external_task&& owned) noexcept
    : owned(std::move(owned)) {}

    awaiter(const awaiter&) = delete;
    awaiter& operator=(const awaiter&) = delete;

    bool await_ready() const noexcept {
        assert(owned.shared);
        return false;
    }

    bool await_suspend(std::coroutine_handle<> awaiting) {
        state& awaited = *owned.shared;
        std::lock_guard lock(awaited.mutex);
        if (awaited.value) {
            return false;
        }
        awaited.waiting.handle = awaiting;
        awaited.loop = event_loop::current();
        return true;
    }

    // The value was written under the state's lock, then the handle scheduled under the loop's lock, so it is visible
    // here without locking.
    value_type await_resume() {
        return std::move(*owned.shared->value);
    }
};

template <typename value_type>
external_task<value_type>::external_task(shared_state shared) noexcept
: shared(std::move(shared)) {}

template <typename value_type> void external_task<value_type>::abandon() noexcept {
    if (shared) {
        std::lock_guard lock(shared->mutex);
        shared->alive = false;
    }
}

template <typename value_type>
std::pair<external_task<value_type>, typename external_task<value_type>::resolver> external_task<value_type>::make() {
    shared_state shared = std::make_shared<state>();
    return {external_task(shared), resolver(shared)};
}

template <typename value_type>
external_task<value_type>::external_task(external_task&& other) noexcept
: shared(std::move(other.shared)) {}

template <typename value_type>
external_task<value_type>& external_task<value_type>::operator=(external_task&& other) noexcept {
    if (this != &other) {
        abandon();
        shared = std::move(other.shared);
    }
    return *this;
}

template <typename value_type> external_task<value_type>::~external_task() {
    abandon();
}

template <typename value_type>
typename external_task<value_type>::awaiter external_task<value_type>::operator co_await() && noexcept {
    return awaiter(std::move(*this));
}

template <typename value_type>
external_task<value_type>::resolver::resolver(shared_state shared) noexcept
: shared(std::move(shared)) {}

template <typename value_type> void external_task<value_type>::resolver::resolve(value_type&& value) const {
    shared->set_value(value_type(std::move(value)));
}

}  // namespace pulsarlm
