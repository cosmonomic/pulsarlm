#include "pulsarlm/engine/event_loop.hpp"

#include <cuda_runtime_api.h>

#undef CHECK
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstddef>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using pulsarlm::event;
using pulsarlm::event_loop;
using pulsarlm::external_task;
using pulsarlm::shared_task;
using pulsarlm::task;

constexpr std::size_t root_count = 1000;
constexpr std::size_t chain_depth = 1000;
constexpr std::size_t event_waiter_count = 4;
constexpr std::size_t shared_waiter_count = 4;
constexpr std::chrono::milliseconds other_thread_delay(50);

void stop_current() {
    event_loop::current()->stop();
}

task<std::size_t> doubled(std::size_t value) {
    co_return value * 2;
}

task<std::string> quadrupled_text(std::size_t value) {
    std::size_t once = co_await doubled(value);
    std::size_t twice = co_await doubled(once);
    co_return std::to_string(twice);
}

task<void> nothing() {
    co_return;
}

task<void> chain_root(std::size_t value, std::size_t* matched, std::size_t* finished) {
    co_await nothing();
    if (co_await quadrupled_text(value) == std::to_string(value * 4)) {
        ++*matched;
    }
    if (++*finished == root_count) {
        stop_current();
    }
}

task<std::size_t> count_down(std::size_t depth) {
    if (depth == 0) {
        co_return 0;
    }
    co_return 1 + co_await count_down(depth - 1);
}

task<void> count_down_root(std::size_t depth, std::size_t* total) {
    *total = co_await count_down(depth);
    stop_current();
}

task<void> wait_and_count(event* target, std::size_t* woken) {
    co_await *target;
    if (++*woken == event_waiter_count) {
        stop_current();
    }
}

task<void> set_then_wait(event* target, std::size_t* woken) {
    target->set();
    co_await wait_and_count(target, woken);
}

task<void> external_root(external_task<int> value, external_task<void> signal, int* received) {
    co_await std::move(signal);
    *received = co_await std::move(value);
    stop_current();
}

task<void> await_event(event* target, [[maybe_unused]] std::shared_ptr<int> frame_alive) {
    co_await *target;
}

task<void> await_external(external_task<int> value, [[maybe_unused]] std::shared_ptr<int> frame_alive) {
    co_await std::move(value);
}

task<void> resolve_from_thread(external_task<int>::resolver resolver, std::thread* resolving) {
    *resolving = std::thread([resolver] { resolver.resolve(7); });
    co_return;
}

task<void> own_event(event** published) {
    event owned;
    *published = &owned;
    auto [never, unused_resolver] = external_task<void>::make();
    co_await std::move(never);
}

task<void> await_published_event(event** published) {
    co_await **published;
}

task<void> stopper() {
    stop_current();
    co_return;
}

task<void> thrower() {
    co_await nothing();
    throw std::runtime_error("boom");
}

task<void> record_current(event_loop** seen) {
    *seen = event_loop::current();
    stop_current();
    co_return;
}

task<std::thread::id> worker_thread_id() {
    co_return std::this_thread::get_id();
}

task<void> record_worker_thread(std::thread::id* seen) {
    *seen = std::this_thread::get_id();
    co_return;
}

task<void> await_on_worker(event_loop* worker, std::thread::id* valued_on, std::thread::id* void_on) {
    *valued_on = co_await pulsarlm::run_on(*worker, worker_thread_id());
    co_await pulsarlm::run_on(*worker, record_worker_thread(void_on));
    worker->stop();
    stop_current();
}

task<void> synchronize_twice(cudaStream_t stream, bool* synchronized) {
    co_await pulsarlm::synchronize(stream);
    co_await pulsarlm::synchronize(stream);
    *synchronized = true;
    stop_current();
}

shared_task<std::string> run_counted_after(event* gate, std::size_t* runs) {
    ++*runs;
    co_await *gate;
    co_return std::string("shared");
}

task<void> record_result(shared_task<std::string> shared, std::vector<const std::string*>* seen) {
    const std::string& result = co_await shared;
    seen->push_back(&result);
    if (seen->size() == shared_waiter_count) {
        stop_current();
    }
}

task<void> open_then_record_twice(event* gate, shared_task<std::string> shared, std::vector<const std::string*>* seen) {
    gate->set();
    co_await shared;
    co_await record_result(shared, seen);
}

task<int> forty_two() {
    co_return 42;
}

shared_task<int> counted_forty_two(std::size_t* runs) {
    ++*runs;
    co_return 42;
}

task<void> add_spawned(shared_task<int> from_task, shared_task<int> from_shared, int* sum) {
    int first = co_await from_task;
    int second = co_await from_shared;
    *sum = first + second;
    stop_current();
}

shared_task<void> hold_until(event* gate, [[maybe_unused]] std::shared_ptr<int> frame_alive) {
    co_await *gate;
}

shared_task<void> keep_frame_alive([[maybe_unused]] std::shared_ptr<int> frame_alive) {
    co_return;
}

task<void> observe_then_open(event* gate, std::weak_ptr<int> observed, bool* alive_before_open) {
    *alive_before_open = !observed.expired();
    gate->set();
    event_loop::current()->spawn(stopper());
    co_return;
}

task<void> await_then_stop(shared_task<void> awaited) {
    co_await awaited;
    stop_current();
}

task<void> count_and_stop_last(std::size_t* finished, std::thread::id* last_thread) {
    *last_thread = std::this_thread::get_id();
    if (++*finished == root_count) {
        stop_current();
    }
    co_return;
}

task<void> await_erased_then_typed(shared_task<void> erased, shared_task<int> typed, int* value) {
    co_await erased;
    *value = co_await typed;
    stop_current();
}

shared_task<void> shared_thrower() {
    co_await nothing();
    throw std::runtime_error("shared boom");
}

task<void> await_ready_tasks(shared_task<void> made_outside, bool* resumed) {
    co_await made_outside;
    co_await pulsarlm::make_ready_task();
    *resumed = true;
    stop_current();
}

}  // namespace

TEST_CASE("values chain through task<int>, task<std::string> and task<void> across many roots") {
    event_loop loop;
    std::size_t matched = 0;
    std::size_t finished = 0;
    for (std::size_t i = 0; i < root_count; i++) {
        loop.spawn(chain_root(i, &matched, &finished));
    }
    CHECK(loop.run() == nullptr);
    CHECK(finished == root_count);
    CHECK(matched == root_count);
}

TEST_CASE("a synchronous await chain returns through every level") {
    event_loop loop;
    std::size_t total = 0;
    loop.spawn(count_down_root(chain_depth, &total));
    CHECK(loop.run() == nullptr);
    CHECK(total == chain_depth);
}

TEST_CASE("an event wakes awaiters from before the set and does not suspend awaiters after it") {
    event target;
    event_loop loop;
    std::size_t woken = 0;
    for (std::size_t i = 0; i + 1 < event_waiter_count; i++) {
        loop.spawn(wait_and_count(&target, &woken));
    }
    loop.spawn(set_then_wait(&target, &woken));
    CHECK(loop.run() == nullptr);
    CHECK(woken == event_waiter_count);
}

TEST_CASE("external tasks resolve from another thread after awaiting and before awaiting") {
    event_loop loop;
    int received = 0;
    auto [value, value_resolver] = external_task<int>::make();
    auto [signal, signal_resolver] = external_task<void>::make();
    signal_resolver.resolve();
    std::thread resolving;
    loop.spawn(external_root(std::move(value), std::move(signal), &received));
    loop.spawn(resolve_from_thread(value_resolver, &resolving));
    CHECK(loop.run() == nullptr);
    resolving.join();
    CHECK(received == 7);
}

TEST_CASE("an external task destroyed before resolve drops the value") {
    auto value = std::make_shared<int>(7);
    std::weak_ptr<int> observed = value;
    std::optional<external_task<std::shared_ptr<int>>::resolver> resolver;
    {
        auto [pending, made_resolver] = external_task<std::shared_ptr<int>>::make();
        resolver = made_resolver;
    }
    resolver->resolve(std::move(value));
    CHECK(observed.expired());
}

TEST_CASE("a panic ends run with the thrown exception") {
    event_loop loop;
    loop.spawn(thrower());
    std::exception_ptr thrown = loop.run();
    REQUIRE(thrown != nullptr);
    try {
        std::rethrow_exception(thrown);
    } catch (const std::runtime_error& error) {
        CHECK(std::string(error.what()) == "boom");
    }
}

TEST_CASE("destroying the loop destroys suspended frames and a late resolve does nothing") {
    event target;
    std::optional<external_task<int>::resolver> late;
    auto frame_alive = std::make_shared<int>(0);
    std::weak_ptr<int> observed = frame_alive;
    {
        event_loop loop;
        auto [value, value_resolver] = external_task<int>::make();
        late = value_resolver;
        loop.spawn(await_event(&target, frame_alive));
        loop.spawn(await_external(std::move(value), std::move(frame_alive)));
        loop.spawn(stopper());
        CHECK(loop.run() == nullptr);
        CHECK(!observed.expired());
    }
    CHECK(observed.expired());
    late->resolve(1);
}

TEST_CASE("destroying the loop destroys an event in a frame and its awaiter in either order") {
    event* published = nullptr;
    event_loop loop;
    loop.spawn(own_event(&published));
    loop.spawn(await_published_event(&published));
    loop.spawn(stopper());
    CHECK(loop.run() == nullptr);
}

TEST_CASE("stop from another thread ends an idle run") {
    event_loop loop;
    std::thread stopping([&loop] {
        std::this_thread::sleep_for(other_thread_delay);
        loop.stop();
    });
    CHECK(loop.run() == nullptr);
    stopping.join();
}

TEST_CASE("current is null outside run and the running loop inside a task") {
    CHECK(event_loop::current() == nullptr);
    event_loop loop;
    event_loop* seen = nullptr;
    loop.spawn(record_current(&seen));
    CHECK(loop.run() == nullptr);
    CHECK(seen == &loop);
    CHECK(event_loop::current() == nullptr);
}

TEST_CASE("synchronize completes after the work queued on a CUDA stream") {
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    bool synchronized = false;
    {
        event_loop loop;
        loop.spawn(synchronize_twice(stream, &synchronized));
        CHECK(loop.run() == nullptr);
    }
    CHECK(synchronized);
    CHECK(cudaStreamDestroy(stream) == cudaSuccess);
}

TEST_CASE("run_on runs a task on another loop's thread and resumes the awaiter with its result") {
    event_loop worker;
    std::thread::id worker_id;
    std::thread worker_thread([&] {
        worker_id = std::this_thread::get_id();
        CHECK(worker.run() == nullptr);
    });
    event_loop loop;
    std::thread::id valued_on;
    std::thread::id void_on;
    loop.spawn(await_on_worker(&worker, &valued_on, &void_on));
    CHECK(loop.run() == nullptr);
    worker_thread.join();
    CHECK(valued_on == worker_id);
    CHECK(void_on == worker_id);
    CHECK(valued_on != std::this_thread::get_id());
}

TEST_CASE("a shared task runs once and gives every awaiter, before and after its end, a reference to one result") {
    event gate;
    event_loop loop;
    std::size_t runs = 0;
    std::vector<const std::string*> seen;
    shared_task<std::string> shared = run_counted_after(&gate, &runs);
    CHECK(runs == 0);
    for (std::size_t i = 0; i + 1 < shared_waiter_count; i++) {
        loop.spawn(record_result(shared, &seen));
    }
    loop.spawn(open_then_record_twice(&gate, shared, &seen));
    CHECK(loop.run() == nullptr);
    CHECK(runs == 1);
    REQUIRE(seen.size() == shared_waiter_count);
    for (const std::string* result : seen) {
        CHECK(result == seen.front());
        CHECK(*result == "shared");
    }
}

TEST_CASE("spawn runs a task or a shared task on the loop once and returns a shared task with its result") {
    event_loop loop;
    std::size_t runs = 0;
    int sum = 0;
    shared_task<int> from_task = loop.spawn(forty_two());
    shared_task<int> from_shared = loop.spawn(counted_forty_two(&runs));
    shared_task<int> spawned_again = loop.spawn(from_shared);
    loop.spawn(add_spawned(from_task, spawned_again, &sum));
    CHECK(loop.run() == nullptr);
    CHECK(sum == 84);
    CHECK(runs == 1);
}

TEST_CASE("a spawned shared task runs to its end after its handles are dropped, and then its frame is destroyed") {
    event gate;
    event_loop loop;
    auto frame_alive = std::make_shared<int>(0);
    std::weak_ptr<int> observed = frame_alive;
    bool alive_before_open = false;
    loop.spawn(hold_until(&gate, std::move(frame_alive)));
    loop.spawn(observe_then_open(&gate, observed, &alive_before_open));
    CHECK(loop.run() == nullptr);
    CHECK(alive_before_open);
    CHECK(observed.expired());
}

TEST_CASE("an ended shared task that was not spawned keeps its frame until the last copy is destroyed") {
    auto frame_alive = std::make_shared<int>(0);
    std::weak_ptr<int> observed = frame_alive;
    std::optional<shared_task<void>> last_copy;
    {
        event_loop loop;
        shared_task<void> held = keep_frame_alive(std::move(frame_alive));
        last_copy = held;
        loop.spawn(await_then_stop(held));
        CHECK(loop.run() == nullptr);
    }
    CHECK(!observed.expired());
    last_copy.reset();
    CHECK(observed.expired());
}

TEST_CASE("tasks spawned from another thread, which drops their handles, run on the loop's thread") {
    event_loop loop;
    std::size_t finished = 0;
    std::thread::id last_thread;
    std::thread spawning([&] {
        for (std::size_t i = 0; i < root_count; i++) {
            loop.spawn(count_and_stop_last(&finished, &last_thread));
        }
    });
    CHECK(loop.run() == nullptr);
    spawning.join();
    CHECK(finished == root_count);
    CHECK(last_thread == std::this_thread::get_id());
}

TEST_CASE("a shared_task<void> converted from a shared task of a value type awaits the same run") {
    event_loop loop;
    std::size_t runs = 0;
    int value = 0;
    shared_task<int> typed = counted_forty_two(&runs);
    shared_task<void> erased = typed;
    loop.spawn(await_erased_then_typed(erased, typed, &value));
    CHECK(loop.run() == nullptr);
    CHECK(runs == 1);
    CHECK(value == 42);
}

TEST_CASE("ready tasks made outside and inside the loop resume their awaiter") {
    shared_task<void> made_outside = pulsarlm::make_ready_task();
    event_loop loop;
    bool resumed = false;
    loop.spawn(await_ready_tasks(made_outside, &resumed));
    CHECK(loop.run() == nullptr);
    CHECK(resumed);
}

TEST_CASE("a panic in a shared task ends run with the thrown exception") {
    event_loop loop;
    loop.spawn(shared_thrower());
    std::exception_ptr thrown = loop.run();
    REQUIRE(thrown != nullptr);
    try {
        std::rethrow_exception(thrown);
    } catch (const std::runtime_error& error) {
        CHECK(std::string(error.what()) == "shared boom");
    }
}
