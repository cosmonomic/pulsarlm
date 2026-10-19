#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace pulsarlm {

// A request's eviction policy, its counts in tokens. A cycle ends once cycle_tokens tokens were forwarded since it
// began. Its end evicts, in each KV layer, the full pages outside those holding the first n_sink tokens and the last
// n_working tokens of the full pages whose density is below the layer's threshold (density_threshold) in every query head, then
// the lowest-density ones until the view holds at most max_context_window tokens through the next cycle.
// eviction_target is in (0, 1], and 1 sets no threshold. attention_mass_decay is the density EMA's rate per token. The
// defaults evict nothing.
struct eviction_policy {
    std::size_t n_sink = 1024;
    std::size_t n_working = 8192;
    std::size_t cycle_tokens = 512;
    std::optional<std::size_t> max_context_window;
    double eviction_target = 1.0;
    double attention_mass_decay = 1.0 / 256;
};

// An eviction_policy with its sink, working set and context window in pages: n_sink and n_working round up and
// max_pages rounds down. attention_mass_decay is the rate per page that retains as much over a page as the rate per
// token does over page_size tokens. A cycle's end keeps the full pages and the pages a partial tail page and the next
// cycle's tokens fill within max_pages.
struct page_eviction_policy {
    std::size_t n_sink;
    std::size_t n_working;
    std::size_t cycle_tokens;
    std::optional<std::size_t> max_pages;
    double eviction_target;
    double attention_mass_decay;

    bool can_evict() const noexcept;
};

page_eviction_policy to_pages(const eviction_policy& policy, std::size_t page_size) noexcept;

// The density threshold of a KV layer read by n_q_heads query heads, in a model whose context holds max_context_pages pages:
// n_q_heads / (eviction_target * max_context_pages), 0 at eviction_target 1. A head's shares of one query sum to at
// most 1, as do their EMAs, so at most eviction_target * max_context_pages pages reach the threshold in some head.
double density_threshold(double eviction_target, std::size_t n_q_heads, std::size_t max_context_pages) noexcept;

// How many pages a step moves a sequence's latest page, the one holding its newest token, when it forwards token_count
// tokens and the next token sits at offset first_offset of its page: the pages whose first token the step forwards. The
// density EMA ages by one page per page of advance.
std::size_t latest_page_advance(std::size_t first_offset, std::size_t token_count, std::size_t page_size) noexcept;

// Fills weights, one per token of such a step, with each token's mass weight: page_mass_decay / page_size times
// (1 - page_mass_decay) per page between the token's page and the step's last. With add_step_mass, a page's
// density is then an EMA by page age of the shares its keys received.
void mass_weights(
    std::span<float> weights,
    std::size_t first_offset,
    std::size_t page_size,
    double page_mass_decay
) noexcept;

// Ages a view's densities, n_q_heads per page, by page_advance pages (latest_page_advance) at page_mass_decay per
// page, then adds step_mass, the step's mass in the same layout. A page joins at 0.
void add_step_mass(
    std::span<float> density,
    std::span<const float> step_mass,
    std::size_t page_advance,
    double page_mass_decay
) noexcept;

// The view indices of the full pages a cycle's end evicts from a KV layer, ascending. density holds n_q_heads values
// per full page, and threshold is the layer's density_threshold. A page ranks by its highest density among the heads.
std::vector<std::size_t> pages_to_evict(
    std::span<const float> density,
    std::size_t n_q_heads,
    double threshold,
    std::size_t page_size,
    const page_eviction_policy& policy
);

}  // namespace pulsarlm
