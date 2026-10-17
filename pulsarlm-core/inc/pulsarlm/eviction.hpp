#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace pulsarlm {

// Page counts are full pages of a state. A cycle ends once cycle_tokens tokens were forwarded since it began. Its end
// evicts the full pages outside the first n_sink and the last n_working whose density is below the bar evict_tau sets,
// then the lowest-density ones until at most max_pages remain. attention_mass_decay is the density EMA's rate per
// token. The defaults evict nothing.
struct eviction_policy {
    std::size_t n_sink = 64;
    std::size_t n_working = 512;
    std::size_t cycle_tokens = 512;
    std::optional<std::size_t> max_pages;
    double evict_tau = 0.0;
    double attention_mass_decay = 1.0 / 512;

    bool can_evict() const noexcept;
};

// The density bar tau / (1 - tau). tau <= 0 bars nothing and tau >= 1 bars everything.
double evict_bar_from_tau(double tau) noexcept;

// Ages a view's page densities, each an EMA of the attention mass its keys received, by token_count tokens, then adds
// step_mass, each page's mass over those tokens. A page joins at 0.
void add_step_mass(
    std::span<float> density,
    std::span<const float> step_mass,
    std::size_t token_count,
    double attention_mass_decay
) noexcept;

// The view indices of the full pages a cycle's end evicts, ascending. density holds the density of each full page.
std::vector<std::size_t> pages_to_evict(std::span<const float> density, const eviction_policy& policy);

}  // namespace pulsarlm
