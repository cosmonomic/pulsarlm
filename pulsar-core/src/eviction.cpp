#include "pulsar/eviction.hpp"

#include <ATen/ATen.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <tuple>
#include <vector>

namespace pulsar {

bool eviction_policy::can_evict() const noexcept {
    return evict_tau > 0.0 || max_pages.has_value();
}

double evict_bar_from_tau(double tau) noexcept {
    if (tau <= 0.0) {
        return 0.0;
    }
    if (tau >= 1.0) {
        return std::numeric_limits<double>::infinity();
    }
    return tau / (1.0 - tau);
}

void cycle_density::add(const at::Tensor& step_mass, std::size_t token_count, double attention_mass_decay) {
    at::Tensor added = step_mass.clone();
    if (values.defined()) {
        assert(values.size(0) <= added.size(0));
        double retained = std::pow(1.0 - attention_mass_decay, static_cast<double>(token_count));
        added.narrow(0, 0, values.size(0)).add_(values, retained);
    }
    values = added;
}

void cycle_density::reset() noexcept {
    values.reset();
}

const at::Tensor& cycle_density::by_page() const noexcept {
    return values;
}

std::vector<std::size_t> pages_to_evict(
    std::span<const float> density,
    std::size_t page_size,
    std::size_t cycle_start,
    std::size_t forwarded_count,
    const eviction_policy& policy
) {
    assert(policy.n_working >= 1);
    std::size_t page_count = density.size();
    if (page_count <= policy.n_sink + policy.n_working) {
        return {};
    }

    struct candidate {
        std::size_t page;
        double density;
    };
    double bar = evict_bar_from_tau(policy.evict_tau);
    std::vector<candidate> kept;
    std::vector<std::size_t> evicted;
    for (std::size_t page = policy.n_sink; page < page_count - policy.n_working; page++) {
        std::size_t first_query = std::max(cycle_start, page * page_size);
        assert(first_query < forwarded_count);
        double weight =
            1.0 - std::pow(1.0 - policy.attention_mass_decay, static_cast<double>(forwarded_count - first_query));
        double corrected = density[page] / weight;
        if (corrected < bar) {
            evicted.push_back(page);
        } else {
            kept.push_back(candidate{.page = page, .density = corrected});
        }
    }

    if (policy.max_pages && page_count - evicted.size() > *policy.max_pages) {
        std::size_t excess = std::min(page_count - evicted.size() - *policy.max_pages, kept.size());
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

}  // namespace pulsar
