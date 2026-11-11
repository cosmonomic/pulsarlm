#pragma once

#include <cstddef>
#include <optional>
#include <vector>

namespace pulsarlm {

// One KV layer's scores: threshold is the score_threshold of its score moments, and page_scores[i] the page_score of
// its pages[i], nullopt without one.
struct layer_scores {
    std::size_t layer;
    double threshold;
    std::vector<std::optional<double>> page_scores;
};

// The scores of a state's KV layers that measure attention mass, and cap_tokens, the most full-page tokens a view keeps
// at a cycle's end, nullopt without a max_context_window.
struct eviction_report {
    std::optional<std::size_t> cap_tokens;
    std::vector<layer_scores> layers;
};

}  // namespace pulsarlm
