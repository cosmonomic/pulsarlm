#pragma once

#include <cassert>
#include <optional>
#include <utility>

namespace pulsarlm {

// Holds a value and calls exit_function with it when destroyed, unless the value was released. An exit function that
// throws terminates the program.
template <typename value_type, typename exit_function_type> class resource {
    std::optional<value_type> held_value;
    exit_function_type exit_function;

  public:
    resource(value_type&& value, exit_function_type exit_function)
    : held_value(std::move(value)),
      exit_function(std::move(exit_function)) {}

    // Takes other's value; other holds none afterwards.
    resource(resource&& other)
    : held_value(std::move(other.held_value)),
      exit_function(std::move(other.exit_function)) {
        other.held_value.reset();
    }

    resource(const resource&) = delete;
    resource& operator=(const resource&) = delete;
    resource& operator=(resource&&) = delete;

    ~resource() {
        if (held_value) {
            exit_function(*held_value);
        }
    }

    // The value, which was not released.
    const value_type& value() const noexcept {
        assert(held_value);
        return *held_value;
    }

    // The value; the guard holds none afterwards and calls nothing when destroyed.
    value_type release() {
        value_type released = std::move(*held_value);
        held_value.reset();
        return released;
    }
};

}  // namespace pulsarlm
