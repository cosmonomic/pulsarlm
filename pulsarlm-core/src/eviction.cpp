#include "pulsarlm/eviction.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <span>
#include <tuple>
#include <vector>

namespace pulsarlm {

namespace {

std::size_t ceil_div(std::size_t n, std::size_t d) {
    return (n + d - 1) / d;
}

}  // namespace

page_eviction_policy to_pages(const eviction_policy& policy, std::size_t page_size) noexcept {
    std::optional<std::size_t> max_pages;
    if (policy.max_context_window) {
        max_pages = *policy.max_context_window / page_size;
    }
    return page_eviction_policy{
        .n_sink = ceil_div(policy.n_sink, page_size),
        .n_working = ceil_div(policy.n_working, page_size),
        .cycle_tokens = policy.cycle_tokens,
        .max_pages = max_pages,
        .eviction_target = policy.eviction_target,
        .attention_mass_decay = 1.0 - std::pow(1.0 - policy.attention_mass_decay, static_cast<double>(page_size)),
    };
}

bool page_eviction_policy::can_evict() const noexcept {
    return eviction_target < 1.0 || max_pages.has_value();
}

double density_bar(double eviction_target, std::size_t n_q_heads, std::size_t max_context_pages) noexcept {
    if (eviction_target >= 1.0) {
        return 0.0;
    }
    return static_cast<double>(n_q_heads) / (eviction_target * static_cast<double>(max_context_pages));
}

std::size_t latest_page_advance(std::size_t first_offset, std::size_t token_count, std::size_t page_size) noexcept {
    return ceil_div(first_offset + token_count, page_size) - ceil_div(first_offset, page_size);
}

void mass_weights(
    std::span<float> weights,
    std::size_t first_offset,
    std::size_t page_size,
    double page_mass_decay
) noexcept {
    std::size_t last_page = (first_offset + weights.size() - 1) / page_size;
    for (std::size_t i = 0; i < weights.size(); i++) {
        std::size_t age = last_page - (first_offset + i) / page_size;
        weights[i] = static_cast<float>(
            page_mass_decay / static_cast<double>(page_size) * std::pow(1.0 - page_mass_decay, static_cast<double>(age))
        );
    }
}

void add_step_mass(
    std::span<float> density,
    std::span<const float> step_mass,
    std::size_t page_advance,
    double page_mass_decay
) noexcept {
    assert(density.size() == step_mass.size());
    auto retained = static_cast<float>(std::pow(1.0 - page_mass_decay, static_cast<double>(page_advance)));
    for (std::size_t i = 0; i < density.size(); i++) {
        density[i] = density[i] * retained + step_mass[i];
    }
}

std::vector<std::size_t> pages_to_evict(
    std::span<const float> density,
    std::size_t n_q_heads,
    double bar,
    std::size_t page_size,
    const page_eviction_policy& policy
) {
    assert(policy.n_working >= 1 && n_q_heads >= 1 && density.size() % n_q_heads == 0);
    std::size_t page_count = density.size() / n_q_heads;
    if (page_count <= policy.n_sink + policy.n_working) {
        return {};
    }

    struct candidate {
        std::size_t page;
        double density;
    };
    std::vector<candidate> kept;
    std::vector<std::size_t> evicted;
    for (std::size_t page = policy.n_sink; page < page_count - policy.n_working; page++) {
        std::span<const float> heads = density.subspan(page * n_q_heads, n_q_heads);
        double highest = *std::ranges::max_element(heads);
        if (highest < bar) {
            evicted.push_back(page);
        } else {
            kept.push_back(candidate{.page = page, .density = highest});
        }
    }

    std::size_t next_cycle_pages = ceil_div(policy.cycle_tokens + page_size - 1, page_size);
    std::size_t full_page_cap = policy.max_pages && *policy.max_pages > next_cycle_pages
        ? *policy.max_pages - next_cycle_pages
        : 0;
    if (policy.max_pages && page_count - evicted.size() > full_page_cap) {
        std::size_t excess = std::min(page_count - evicted.size() - full_page_cap, kept.size());
        std::sort(kept.begin(), kept.end(), [](const candidate& a, const candidate& b) {
            return std::tie(a.density, a.page) < std::tie(b.density, b.page);
        });
        for (std::size_t i = 0; i < excess; i++) {
            evicted.push_back(kept[i].page);
        }
    }
    std::sort(evicted.begin(), evicted.end());
    return evicted;
}

}  // namespace pulsarlm
