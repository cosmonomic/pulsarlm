#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

namespace pulsarlm {

// A request's eviction policy, its counts in tokens. A cycle ends once cycle_tokens tokens were forwarded since it
// began. Its end evicts, in each KV layer, the full pages outside those holding the first n_sink tokens and the last
// n_working tokens of the full pages whose page_score is below score_threshold at eviction_threshold, then the oldest
// ones until the view holds at most max_context_capacity tokens, rounded up to whole pages, at the next cycle's end.
// eviction_threshold is in [0, 1], and 0 evicts nothing by score. Cycles end at the history's multiples of cycle_tokens. Page masses accumulate from the
// queries of the last mass_tokens tokens of each cycle, of all of them without mass_tokens, which is in [1,
// cycle_tokens]. At a cycle's end, a layer samples each full page outside the sinks whose last token is at least
// cycle_tokens and less than twice cycle_tokens before that end (add_sampled_page). eviction_warmup counts cycles: a layer evicts by score once it sampled the pages of that
// many cycles' tokens twice over, as its density moments need that many samples before it samples scores, and its score
// moments as many before it evicts. attention_mass_decay is the density EMA's rate per token. The defaults evict
// nothing.
struct eviction_policy {
    std::size_t n_sink = 1024;
    std::size_t n_working = 8192;
    std::size_t cycle_tokens = 512;
    std::optional<std::size_t> mass_tokens;
    // The view's cap, in tokens: 0 evicts nothing by capacity, a negative value uses the engine's max_context_window.
    std::int64_t max_context_capacity = 0;
    double eviction_threshold = 0.0;
    std::size_t eviction_warmup = 1;
    double attention_mass_decay = 1.0 / 256;
};

}  // namespace pulsarlm
