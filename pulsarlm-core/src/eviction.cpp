#include "pulsarlm/eviction.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <vector>

namespace pulsarlm {

namespace {

std::size_t ceil_div(std::size_t n, std::size_t d) {
    return (n + d - 1) / d;
}

bool warmed_up(const moments& m, std::size_t min_samples) noexcept {
    return m.count >= std::max<std::size_t>(2, min_samples);
}

double standard_deviation(const moments& m) noexcept {
    return std::sqrt(m.m2 / static_cast<double>(m.count - 1));
}

// The standard normal quantile of p in (0, 1): Acklam's rational approximation, relative error below 1.15e-9, then
// one Halley step on std::erfc.
double normal_quantile(double p) noexcept {
    constexpr double a[] = {
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00
    };
    constexpr double b[] = {
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01
    };
    constexpr double c[] = {
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00
    };
    constexpr double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00};
    constexpr double p_low = 0.02425;
    double x;
    if (p < p_low || p > 1.0 - p_low) {
        double q = std::sqrt(-2.0 * std::log(p < p_low ? p : 1.0 - p));
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
        if (p > 1.0 - p_low) {
            x = -x;
        }
    } else {
        double q = p - 0.5;
        double r = q * q;
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    }
    double e = 0.5 * std::erfc(-x / std::numbers::sqrt2) - p;
    double u = e * std::sqrt(2.0 * std::numbers::pi) * std::exp(x * x / 2.0);
    return x - u / (1.0 + x * u / 2.0);
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
        .eviction_threshold = policy.eviction_threshold,
        .eviction_warmup = policy.eviction_warmup * ceil_div(policy.cycle_tokens, page_size),
        .attention_mass_decay = policy.attention_mass_decay,
    };
}

bool page_eviction_policy::can_evict() const noexcept {
    return eviction_threshold > 0.0 || max_pages.has_value();
}

void add_sample(moments& m, double x) noexcept {
    m.count++;
    double delta = x - m.mean;
    m.mean += delta / static_cast<double>(m.count);
    m.m2 += delta * (x - m.mean);
}

void query_weights(std::span<float> weights, double attention_mass_decay) noexcept {
    for (std::size_t t = 0; t < weights.size(); t++) {
        auto age = static_cast<double>(weights.size() - 1 - t);
        weights[t] = static_cast<float>(attention_mass_decay * std::pow(1.0 - attention_mass_decay, age));
    }
}

void add_step_mass(
    std::span<float> mass_sum,
    std::span<float> weight_sum,
    std::span<const float> step_mass_sum,
    std::span<const float> step_weight_sum,
    std::size_t token_count,
    double attention_mass_decay
) noexcept {
    assert(mass_sum.size() == step_mass_sum.size() && weight_sum.size() == step_weight_sum.size());
    auto retained = static_cast<float>(std::pow(1.0 - attention_mass_decay, static_cast<double>(token_count)));
    for (std::size_t i = 0; i < mass_sum.size(); i++) {
        mass_sum[i] = mass_sum[i] * retained + step_mass_sum[i];
    }
    for (std::size_t i = 0; i < weight_sum.size(); i++) {
        weight_sum[i] = weight_sum[i] * retained + step_weight_sum[i];
    }
}

std::optional<double> page_score(
    std::span<const float> mass_sum,
    float weight_sum,
    std::span<const moments> density_moments,
    std::size_t min_samples
) noexcept {
    if (weight_sum == 0.0f) {
        return -std::numeric_limits<double>::infinity();
    }
    std::optional<double> score;
    for (std::size_t h = 0; h < density_moments.size(); h++) {
        const moments& m = density_moments[h];
        if (!warmed_up(m, min_samples) || m.m2 == 0.0) {
            continue;
        }
        double density = static_cast<double>(mass_sum[h]) / static_cast<double>(weight_sum);
        double z = (density - m.mean) / standard_deviation(m);
        score = std::max(score.value_or(z), z);
    }
    return score;
}

void add_sampled_page(
    std::span<moments> density_moments,
    moments& score_moments,
    std::span<const float> mass_sum,
    float weight_sum,
    std::size_t min_samples
) noexcept {
    assert(density_moments.size() == mass_sum.size());
    if (weight_sum == 0.0f) {
        return;
    }
    for (std::size_t h = 0; h < density_moments.size(); h++) {
        add_sample(density_moments[h], static_cast<double>(mass_sum[h]) / static_cast<double>(weight_sum));
    }
    if (!std::ranges::all_of(density_moments, [&](const moments& m) { return warmed_up(m, min_samples); })) {
        return;
    }
    if (std::optional<double> score = page_score(mass_sum, weight_sum, density_moments, min_samples)) {
        add_sample(score_moments, *score);
    }
}

double score_threshold(const moments& score_moments, double threshold, std::size_t min_samples) noexcept {
    if (threshold <= 0.0 || !warmed_up(score_moments, min_samples)) {
        return -std::numeric_limits<double>::infinity();
    }
    if (threshold >= 1.0) {
        return std::numeric_limits<double>::infinity();
    }
    return score_moments.mean + standard_deviation(score_moments) * normal_quantile(threshold);
}

std::vector<std::size_t> pages_to_evict(
    std::span<const float> mass_sum,
    std::span<const float> weight_sum,
    std::size_t n_q_heads,
    std::span<const moments> density_moments,
    const moments& score_moments,
    std::size_t page_size,
    const page_eviction_policy& policy
) {
    assert(policy.n_working >= 1 && n_q_heads >= 1 && mass_sum.size() == weight_sum.size() * n_q_heads);
    std::size_t page_count = weight_sum.size();
    if (page_count <= policy.n_sink + policy.n_working) {
        return {};
    }
    std::size_t first_protected = page_count - policy.n_working;

    std::vector<bool> evicted(page_count, false);
    std::size_t evicted_count = 0;
    double threshold = score_threshold(score_moments, policy.eviction_threshold, policy.eviction_warmup);
    for (std::size_t page = policy.n_sink; page < first_protected; page++) {
        std::optional<double> score = page_score(
            mass_sum.subspan(page * n_q_heads, n_q_heads),
            weight_sum[page],
            density_moments,
            policy.eviction_warmup
        );
        if (score && *score < threshold) {
            evicted[page] = true;
            evicted_count++;
        }
    }

    if (policy.max_pages) {
        std::size_t next_cycle_pages = ceil_div(policy.cycle_tokens + page_size - 1, page_size);
        std::size_t full_page_cap = *policy.max_pages > next_cycle_pages ? *policy.max_pages - next_cycle_pages : 0;
        for (std::size_t page = policy.n_sink; page < first_protected && page_count - evicted_count > full_page_cap;
             page++) {
            if (!evicted[page]) {
                evicted[page] = true;
                evicted_count++;
            }
        }
    }

    std::vector<std::size_t> pages;
    for (std::size_t page = 0; page < page_count; page++) {
        if (evicted[page]) {
            pages.push_back(page);
        }
    }
    return pages;
}

}  // namespace pulsarlm
