#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pulsarlm {

// Values in slots that keep their address, each named by a key that stops matching once its value is erased, also
// after the slot holds another value. A value is constructed in its slot and need not be movable.
template <typename value_type> class slot_map {
    using generation_type = std::uint32_t;
    // The generation, incremented when the value is erased, and the value.
    using slot_type = std::pair<generation_type, std::optional<value_type>>;

    std::deque<slot_type> slots;
    std::vector<std::uint32_t> free_slots;

  public:
    // The slot in the low 32 bits, its generation in the high 32.
    using key_type = std::uint64_t;

  private:
    static constexpr unsigned generation_shift = 32;

    const slot_type* find(key_type key) const noexcept {
        const auto slot = static_cast<std::uint32_t>(key);
        if (slot >= slots.size() || slots[slot].first != static_cast<generation_type>(key >> generation_shift) ||
            !slots[slot].second) {
            return nullptr;
        }
        return &slots[slot];
    }

  public:
    // Constructs a value from arguments in a free slot, or in a new one, and returns its key.
    template <typename... argument_types> key_type emplace(argument_types&&... arguments) {
        if (free_slots.empty()) {
            slots.emplace_back();
            free_slots.push_back(static_cast<std::uint32_t>(slots.size() - 1));
        }
        const std::uint32_t slot = free_slots.back();
        slots[slot].second.emplace(std::forward<argument_types>(arguments)...);
        free_slots.pop_back();
        return (static_cast<key_type>(slots[slot].first) << generation_shift) | slot;
    }

    // Throws std::out_of_range when key's value was erased.
    void erase(key_type key) {
        at(key);
        slot_type& slot = slots[static_cast<std::uint32_t>(key)];
        slot.second.reset();
        slot.first++;
        free_slots.push_back(static_cast<std::uint32_t>(key));
    }

    // Throws std::out_of_range when key's value was erased.
    value_type& at(key_type key) {
        return const_cast<value_type&>(std::as_const(*this).at(key));
    }

    const value_type& at(key_type key) const {
        const slot_type* slot = find(key);
        if (slot == nullptr) {
            throw std::out_of_range("slot_map: the key's value was erased");
        }
        return *slot->second;
    }

    bool contains(key_type key) const noexcept {
        return find(key) != nullptr;
    }

    std::size_t size() const noexcept {
        return slots.size() - free_slots.size();
    }
};

}  // namespace pulsarlm
