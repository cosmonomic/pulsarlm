#pragma once

#include <cstddef>
#include <optional>

namespace pulsarlm {

// A request's eviction policy, its counts in tokens. A cycle ends once cycle_tokens tokens were forwarded since it
// began. Its end evicts, in each KV layer, the full pages outside those holding the first n_sink tokens and the last
// n_working tokens of the full pages whose page_score is below score_threshold at eviction_threshold, then the oldest
// ones until the view holds at most max_context_window tokens through the next cycle. eviction_threshold is in [0, 1],
// and 0 evicts nothing by score. A layer samples each full page outside the sinks once cycle_tokens tokens followed its
// last one (add_sampled_page). eviction_warmup counts cycles: a layer evicts by score once it sampled the pages of that
// many cycles' tokens twice over, as its density moments need that many samples before it samples scores, and its score
// moments as many before it evicts. attention_mass_decay is the density EMA's rate per token. The defaults evict
// nothing.
struct eviction_policy {
    std::size_t n_sink = 1024;
    std::size_t n_working = 8192;
    std::size_t cycle_tokens = 512;
    std::optional<std::size_t> max_context_window;
    double eviction_threshold = 0.0;
    std::size_t eviction_warmup = 1;
    double attention_mass_decay = 1.0 / 256;
};

}  // namespace pulsarlm
