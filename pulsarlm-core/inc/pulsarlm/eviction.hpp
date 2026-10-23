#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace pulsarlm {

// A request's eviction policy, its counts in tokens. A cycle ends once cycle_tokens tokens were forwarded since it
// began. Its end evicts, in each KV layer, the full pages outside those holding the first n_sink tokens and the last
// n_working tokens of the full pages whose page_score is below score_threshold at eviction_threshold, then the oldest
// ones until the view holds at most max_context_window tokens through the next cycle. eviction_threshold is in [0, 1],
// and 0 evicts nothing by score. eviction_warmup counts cycles: a layer evicts by score once the pages of that many
// cycles' tokens have left its working set twice over, as its density moments need that many samples before it samples
// scores, and its score moments as many before it evicts. attention_mass_decay is the density EMA's rate per token. The defaults evict nothing.
struct eviction_policy {
    std::size_t n_sink = 1024;
    std::size_t n_working = 8192;
    std::size_t cycle_tokens = 512;
    std::optional<std::size_t> max_context_window;
    double eviction_threshold = 0.0;
    std::size_t eviction_warmup = 1;
    double attention_mass_decay = 1.0 / 256;
};

// An eviction_policy with its sink, working set, warm-up and context window in pages: n_sink and n_working round up,
// eviction_warmup is its cycles times the pages a cycle's tokens fill, rounded up, and max_pages rounds down. A cycle's end keeps the full pages and the pages a partial tail page and the next cycle's
// tokens fill within max_pages.
struct page_eviction_policy {
    std::size_t n_sink;
    std::size_t n_working;
    std::size_t cycle_tokens;
    std::optional<std::size_t> max_pages;
    double eviction_threshold;
    std::size_t eviction_warmup;
    double attention_mass_decay;

    bool can_evict() const noexcept;
};

page_eviction_policy to_pages(const eviction_policy& policy, std::size_t page_size) noexcept;

// The running count, mean and sum of squared deviations from the mean (Welford) of the samples added.
struct moments {
    std::uint64_t count = 0;
    double mean = 0.0;
    double m2 = 0.0;

    bool operator==(const moments&) const = default;
};

void add_sample(moments& m, double x) noexcept;

// Fills weights, one per token of a step of weights.size() tokens, with token t's query weight
// attention_mass_decay * (1 - attention_mass_decay)^(weights.size() - 1 - t).
void query_weights(std::span<float> weights, double attention_mass_decay) noexcept;

// Ages a view's mass_sum, n_q_heads per page, and weight_sum, one per page, by (1 - attention_mass_decay)^token_count,
// then adds the step's, laid out the same. A page joins at 0. A page's density in head h is mass_sum[h] / weight_sum:
// the mean of its masses over the query tokens that attended it, each weighted by its age in tokens.
void add_step_mass(
    std::span<float> mass_sum,
    std::span<float> weight_sum,
    std::span<const float> step_mass_sum,
    std::span<const float> step_weight_sum,
    std::size_t token_count,
    double attention_mass_decay
) noexcept;

// A page's score from its mass_sum, one per query head, and weight_sum: the max over the heads of its density
// standardized by the head's density_moments. A head with fewer than max(2, min_samples) samples or no spread is left
// out. -inf when no query token attended the page (weight_sum 0); nullopt when no head is left.
std::optional<double> page_score(
    std::span<const float> mass_sum,
    float weight_sum,
    std::span<const moments> density_moments,
    std::size_t min_samples
) noexcept;

// Adds a page leaving the working set: its density in each head to density_moments, then, once every head holds at
// least max(2, min_samples) samples, its page_score, when it has one, to score_moments. A page no query token attended
// adds nothing.
void add_leaving_page(
    std::span<moments> density_moments,
    moments& score_moments,
    std::span<const float> mass_sum,
    float weight_sum,
    std::size_t min_samples
) noexcept;

// The score at quantile threshold of a normal distribution with score_moments' mean and sample variance: -inf at
// threshold 0 or below max(2, min_samples) samples, +inf at threshold 1.
double score_threshold(const moments& score_moments, double threshold, std::size_t min_samples) noexcept;

// The view indices of the full pages a cycle's end evicts from a KV layer, ascending. mass_sum holds n_q_heads values
// per full page and weight_sum one; density_moments holds one per head. A page without a score is kept. The policy's
// eviction_warmup is the min_samples of page_score and score_threshold.
std::vector<std::size_t> pages_to_evict(
    std::span<const float> mass_sum,
    std::span<const float> weight_sum,
    std::size_t n_q_heads,
    std::span<const moments> density_moments,
    const moments& score_moments,
    std::size_t page_size,
    const page_eviction_policy& policy
);

}  // namespace pulsarlm
