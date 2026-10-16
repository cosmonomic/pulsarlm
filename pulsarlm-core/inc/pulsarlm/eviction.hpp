#pragma once

#include <ATen/core/Tensor.h>

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

// The density bar, a multiple of uniform attention: tau / (1 - tau), so tau 0.5 is parity with uniform. tau <= 0 bars
// nothing and tau >= 1 bars everything.
double evict_bar_from_tau(double tau) noexcept;

// The density of a view's pages: per page, an EMA of the attention mass its keys received. A page joins at 0.
class page_density {
    // fp32 [pages]; undefined while it holds no page.
    at::Tensor values;

  public:
    page_density() = default;
    explicit page_density(at::Tensor initial) noexcept;

    // Ages the densities by token_count tokens, then adds step_mass, each page's mass over those tokens. step_mass may
    // cover pages that joined the view since the last add.
    void add(const at::Tensor& step_mass, std::size_t token_count, double attention_mass_decay);
    // Drops the pages at the ascending view indices pages.
    void remove(std::span<const std::size_t> pages);
    const at::Tensor& by_page() const noexcept;
};

// The view indices of the full pages a cycle's end evicts, ascending. density holds the density of each full page.
std::vector<std::size_t> pages_to_evict(std::span<const float> density, const eviction_policy& policy);

}  // namespace pulsarlm
