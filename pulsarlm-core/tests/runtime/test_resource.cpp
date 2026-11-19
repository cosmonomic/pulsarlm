#include "pulsarlm/resource.hpp"

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace {

using pulsarlm::resource;

// Appends each value it is called with to calls.
class recording_exit {
    std::vector<int>* calls;

  public:
    explicit recording_exit(std::vector<int>& calls)
    : calls(&calls) {}

    void operator()(int& value) const {
        calls->push_back(value);
    }

    void operator()(std::unique_ptr<int>& value) const {
        calls->push_back(*value);
    }
};

}  // namespace

TEST_CASE("a resource calls its exit function once with its value when destroyed") {
    std::vector<int> calls;
    {
        resource<int, recording_exit> guard(7, recording_exit(calls));
        REQUIRE(calls.empty());
    }

    REQUIRE(calls == std::vector<int>{7});
}

TEST_CASE("a moved-from resource calls nothing and the destination calls once") {
    std::vector<int> calls;
    {
        std::optional<resource<int, recording_exit>> source(std::in_place, 7, recording_exit(calls));
        resource<int, recording_exit> destination(std::move(*source));
        source.reset();
        REQUIRE(calls.empty());
    }

    REQUIRE(calls == std::vector<int>{7});
}

TEST_CASE("a released resource returns its value and calls nothing") {
    std::vector<int> calls;
    int released = 0;
    {
        resource<int, recording_exit> guard(7, recording_exit(calls));
        released = guard.release();
    }

    REQUIRE(released == 7);
    REQUIRE(calls.empty());
}

TEST_CASE("a resource holds a value that cannot be copied") {
    std::vector<int> calls;
    {
        resource<std::unique_ptr<int>, recording_exit> guard(std::make_unique<int>(7), recording_exit(calls));
        resource<std::unique_ptr<int>, recording_exit> destination(std::move(guard));
        REQUIRE(calls.empty());
    }
    REQUIRE(calls == std::vector<int>{7});

    std::unique_ptr<int> released;
    {
        resource<std::unique_ptr<int>, recording_exit> guard(std::make_unique<int>(9), recording_exit(calls));
        released = guard.release();
    }
    REQUIRE(*released == 9);
    REQUIRE(calls == std::vector<int>{7});
}
