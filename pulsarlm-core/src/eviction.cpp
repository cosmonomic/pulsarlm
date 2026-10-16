#include "pulsarlm/eviction.hpp"

#include <ATen/ATen.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

namespace pulsarlm {

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

page_density::page_density(at::Tensor initial) noexcept
: values(std::move(initial)) {}

void page_density::add(const at::Tensor& step_mass, std::size_t token_count, double attention_mass_decay) {
    at::Tensor added = step_mass.clone();
    if (values.defined()) {
        assert(values.size(0) <= added.size(0));
        double retained = std::pow(1.0 - attention_mass_decay, static_cast<double>(token_count));
        added.narrow(0, 0, values.size(0)).add_(values, retained);
    }
    values = added;
}

void page_density::remove(std::span<const std::size_t> pages) {
    if (!values.defined() || pages.empty()) {
        return;
    }
    auto page_count = static_cast<std::size_t>(values.size(0));
    at::Tensor kept = at::empty(
        {static_cast<std::int64_t>(page_count - pages.size())},
        at::TensorOptions().dtype(at::kLong).pinned_memory(true)
    );
    auto* next_kept = kept.mutable_data_ptr<std::int64_t>();
    std::size_t next_removed = 0;
    for (std::size_t page = 0; page < page_count; page++) {
        if (next_removed < pages.size() && pages[next_removed] == page) {
            next_removed++;
        } else {
            *next_kept++ = static_cast<std::int64_t>(page);
        }
    }
    assert(next_removed == pages.size());
    values = values.index_select(0, kept.to(values.device(), /*non_blocking=*/true));
}

const at::Tensor& page_density::by_page() const noexcept {
    return values;
}

std::vector<std::size_t> pages_to_evict(std::span<const float> density, const eviction_policy& policy) {
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
        if (density[page] < bar) {
            evicted.push_back(page);
        } else {
            kept.push_back(candidate{.page = page, .density = density[page]});
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

}  // namespace pulsarlm
