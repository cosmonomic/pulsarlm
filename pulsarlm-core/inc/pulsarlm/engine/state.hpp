#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "pulsarlm/digest.hpp"

namespace pulsarlm {

// The running count, mean and sum of squared deviations from the mean (Welford) of the samples added.
struct moments {
    std::uint64_t count = 0;
    double mean = 0.0;
    double m2 = 0.0;

    bool operator==(const moments&) const = default;
};

// A page of a sequence: its hash, its index among the history's full pages, and the sums its eviction densities come
// from (add_step_mass): mass_sum in each query head its layer measures mass in, and weight_sum. A state the scheduler
// reaches holds a mass_sum per head; an initial state may hold none, which reads as 0, with weight_sum 0.
struct active_page {
    digest hash;
    std::size_t index;
    std::vector<float> mass_sum;
    float weight_sum = 0.0f;

    bool operator==(const active_page&) const = default;
};

// One KV layer's active pages, in view order, and the moments of the pages it sampled (add_sampled_page): of their
// densities, one per query head the layer measures mass in, or none for a layer that
// measures none or an initial state without them, and of their scores.
struct layer_sequence {
    std::vector<active_page> pages;
    std::vector<moments> density_moments;
    moments score_moments;

    bool operator==(const layer_sequence&) const = default;
};

// head is the hash of the newest full page of the history, which new pages chain from and every non-empty sequence
// ends with; absent before the first full page. layers holds one sequence per KV layer, and may be empty when head is
// absent. tail holds the tokens after head. A sequence's hashes are distinct: they come from one digest chain. A state
// that breaks this is undefined behaviour.
struct state {
    using lease_id_type = std::uint64_t;

    std::optional<digest> head;
    std::vector<layer_sequence> layers;
    std::vector<std::int64_t> tail;
};

}  // namespace pulsarlm
