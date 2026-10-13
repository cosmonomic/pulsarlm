#pragma once

#include <ATen/core/Tensor.h>

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace pulsar {

// Page counts are full pages of a state. A cycle ends once cycle_tokens tokens were forwarded since it began. Its end
// evicts the full pages outside the first n_sink and the last n_working whose density is below the bar evict_tau sets,
// then the lowest-density ones until at most max_pages remain. attention_mass_decay is the density EMA's rate per
// token. The defaults evict nothing.
struct eviction_policy {
    std::size_t n_sink = 0;
    std::size_t n_working = 1;
    std::size_t cycle_tokens = 512;
    std::optional<std::size_t> max_pages;
    double evict_tau = 0.0;
    double attention_mass_decay = 1.0 / 512;

    bool can_evict() const noexcept;
};

// The density bar, a multiple of uniform attention: tau / (1 - tau), so tau 0.5 is parity with uniform. tau <= 0 bars
// nothing and tau >= 1 bars everything.
double evict_bar_from_tau(double tau) noexcept;

// The density of a view's pages over one eviction cycle: per page, an EMA of the attention mass its keys received,
// starting from zero.
class cycle_density {
    // fp32 [pages]; undefined before the first add.
    at::Tensor values;

  public:
    // Ages the densities by token_count tokens, then adds step_mass, each page's mass over those tokens. step_mass may
    // cover pages that joined the view since the last add.
    void add(const at::Tensor& step_mass, std::size_t token_count, double attention_mass_decay);
    void reset() noexcept;
    const at::Tensor& by_page() const noexcept;
};

// The view indices of the full pages a cycle's end evicts, ascending. density holds the cycle density of each full
// page. The cycle's queries sat at view indices [cycle_start, forwarded_count); a page's density is divided by the EMA
// weight of the queries at or after its first key, so a page that joined during the cycle compares with older ones.
std::vector<std::size_t> pages_to_evict(
    std::span<const float> density,
    std::size_t page_size,
    std::size_t cycle_start,
    std::size_t forwarded_count,
    const eviction_policy& policy
);

}  // namespace pulsar
