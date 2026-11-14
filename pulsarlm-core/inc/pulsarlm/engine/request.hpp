#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "pulsarlm/digest.hpp"
#include "pulsarlm/engine/eviction_policy.hpp"
#include "pulsarlm/engine/sampling_params.hpp"

namespace pulsarlm {

// tokens is a history chained from initial_digest, or from the engine's root without it, and holds at least one token.
// Without max_tokens, the request samples until one of the engine's stop tokens.
struct request {
    std::optional<digest> initial_digest;
    std::vector<std::int64_t> tokens;
    sampling_params sampling;
    std::optional<std::size_t> max_tokens;
    eviction_policy eviction;
};

}  // namespace pulsarlm
