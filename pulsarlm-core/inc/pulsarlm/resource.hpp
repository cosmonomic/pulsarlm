#pragma once

#include <optional>
#include <utility>

namespace pulsarlm {

// Holds a value and calls exit_function with it when destroyed, unless the value was released. An exit function that
// throws terminates the program.
template <typename value_type, typename exit_function_type> class resource {
    std::optional<value_type> value;
    exit_function_type exit_function;

  public:
    resource(value_type&& value, exit_function_type exit_function)
    : value(std::move(value)),
      exit_function(std::move(exit_function)) {}

    // Takes other's value; other holds none afterwards.
    resource(resource&& other)
    : value(std::move(other.value)),
      exit_function(std::move(other.exit_function)) {
        other.value.reset();
    }

    resource(const resource&) = delete;
    resource& operator=(const resource&) = delete;
    resource& operator=(resource&&) = delete;

    ~resource() {
        if (value) {
            exit_function(*value);
        }
    }

    // The value; the guard holds none afterwards and calls nothing when destroyed.
    value_type release() {
        value_type released = std::move(*value);
        value.reset();
        return released;
    }
};

}  // namespace pulsarlm
