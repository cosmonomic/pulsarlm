#pragma once

#include <cstdint>

namespace pulsarlm {

// The weights one fp8 scale covers: n output rows by k input columns.
struct fp8_block {
    std::int64_t n;
    std::int64_t k;
};

}  // namespace pulsarlm
