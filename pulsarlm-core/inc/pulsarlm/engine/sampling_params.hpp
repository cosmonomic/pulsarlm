#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

namespace pulsarlm {

// temperature <= 0 samples the argmax. top_k 0 and top_p outside (0, 1) keep every token; with both set, a token must
// pass both. top_p keeps a token while the probability of the tokens ranked above it is at most top_p.
struct sampling_params {
    double temperature = 0.0;
    double top_p = 1.0;
    std::size_t top_k = 0;
    std::optional<std::uint64_t> seed;
};

}  // namespace pulsarlm
