#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace pulsarlm {

// Page counts are full pages of a state. A cycle ends once cycle_tokens tokens were forwarded since it began. Its end
// evicts, in each KV layer, the full pages outside the first n_sink and the last n_working whose density is below the
// layer's bar (density_bar) in every query head, then the lowest-density ones until at most max_pages remain.
// eviction_target is in (0, 1], and 1 bars nothing. attention_mass_decay is the density EMA's rate per page. The
// defaults evict nothing.
struct eviction_policy {
    std::size_t n_sink = 64;
    std::size_t n_working = 512;
    std::size_t cycle_tokens = 512;
    std::optional<std::size_t> max_pages;
    double eviction_target = 1.0;
    double attention_mass_decay = 1.0 / 16;

    bool can_evict() const noexcept;
};

// The density bar of a KV layer read by n_q_heads query heads, in a model whose context holds max_context_pages pages:
// n_q_heads / (eviction_target * max_context_pages), 0 at eviction_target 1. A head's shares of one query sum to at
// most 1, as do their EMAs, so at most eviction_target * max_context_pages pages clear the bar in some head.
double density_bar(double eviction_target, std::size_t n_q_heads, std::size_t max_context_pages) noexcept;

// How many pages a step moves a sequence's latest page, the one holding its newest token, when it forwards token_count
// tokens and the next token sits at offset first_offset of its page: the pages whose first token the step forwards. The
// density EMA ages by one page per page of advance.
std::size_t latest_page_advance(std::size_t first_offset, std::size_t token_count, std::size_t page_size) noexcept;

// Fills weights, one per token of such a step, with each token's mass weight: attention_mass_decay / page_size times
// (1 - attention_mass_decay) per page between the token's page and the step's last. With add_step_mass, a page's
// density is then an EMA by page age of the shares its keys received.
void mass_weights(
    std::span<float> weights,
    std::size_t first_offset,
    std::size_t page_size,
    double attention_mass_decay
) noexcept;

// Ages a view's densities, n_q_heads per page, by page_advance pages (latest_page_advance), then adds step_mass, the
// step's mass in the same layout. A page joins at 0.
void add_step_mass(
    std::span<float> density,
    std::span<const float> step_mass,
    std::size_t page_advance,
    double attention_mass_decay
) noexcept;

// The view indices of the full pages a cycle's end evicts from a KV layer, ascending. density holds n_q_heads values
// per full page, and bar is the layer's density_bar. A page ranks by its highest density among the heads.
std::vector<std::size_t>
pages_to_evict(std::span<const float> density, std::size_t n_q_heads, double bar, const eviction_policy& policy);

}  // namespace pulsarlm
