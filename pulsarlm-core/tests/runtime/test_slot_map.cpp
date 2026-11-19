#include "pulsarlm/slot_map.hpp"

#include <catch2/catch_test_macros.hpp>

#include <iterator>
#include <stdexcept>
#include <vector>

namespace {

using pulsarlm::slot_map;

struct pinned_value {
    int number;

    explicit pinned_value(int number)
    : number(number) {}
    pinned_value(const pinned_value&) = delete;
    pinned_value& operator=(const pinned_value&) = delete;
};

static_assert(std::forward_iterator<slot_map<pinned_value>::iterator>);
static_assert(std::forward_iterator<slot_map<pinned_value>::const_iterator>);

}  // namespace

TEST_CASE("a slot_map constructs a value that cannot move and finds it by its key") {
    slot_map<pinned_value> values;
    slot_map<pinned_value>::key_type first = values.emplace(1);
    slot_map<pinned_value>::key_type second = values.emplace(2);

    CHECK(values.size() == 2);
    CHECK(values.at(first).number == 1);
    CHECK(values.at(second).number == 2);
    CHECK(values.contains(first));
}

TEST_CASE("a slot_map's values keep their address while others are added") {
    slot_map<pinned_value> values;
    slot_map<pinned_value>::key_type first = values.emplace(1);
    const pinned_value* address = &values.at(first);

    for (int number = 2; number < 2000; number++) {
        values.emplace(number);
    }

    CHECK(&values.at(first) == address);
    CHECK(address->number == 1);
}

TEST_CASE("an erased key stops matching, also after its slot holds another value") {
    slot_map<pinned_value> values;
    slot_map<pinned_value>::key_type erased = values.emplace(1);
    slot_map<pinned_value>::key_type other = values.emplace(2);

    CHECK(values.erase(erased));
    CHECK(values.size() == 1);
    CHECK_FALSE(values.contains(erased));
    CHECK_THROWS_AS(values.at(erased), std::out_of_range);
    CHECK_FALSE(values.erase(erased));
    CHECK(values.find(erased) == nullptr);

    slot_map<pinned_value>::key_type reusing = values.emplace(3);
    CHECK(reusing != erased);
    CHECK(static_cast<std::uint32_t>(reusing) == static_cast<std::uint32_t>(erased));
    CHECK_FALSE(values.contains(erased));
    CHECK_THROWS_AS(values.at(erased), std::out_of_range);
    CHECK(values.at(reusing).number == 3);
    CHECK(values.at(other).number == 2);
}

TEST_CASE("a slot_map's iterators visit its values in slot order and pass over erased ones") {
    slot_map<pinned_value> values;
    CHECK(values.begin() == values.end());

    values.emplace(1);
    slot_map<pinned_value>::key_type erased = values.emplace(2);
    values.emplace(3);
    values.erase(erased);

    std::vector<int> numbers;
    for (pinned_value& value : values) {
        numbers.push_back(value.number);
    }
    CHECK((numbers == std::vector<int>{1, 3}));

    const slot_map<pinned_value>& read_only = values;
    CHECK(std::distance(read_only.begin(), read_only.end()) == 2);
    CHECK(read_only.begin()->number == 1);
}
