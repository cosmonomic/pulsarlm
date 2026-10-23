#include "pulsarlm/eviction.hpp"

#undef CHECK
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

namespace {

using Catch::Matchers::WithinAbs;
using pulsarlm::eviction_policy;
using pulsarlm::moments;
using pulsarlm::page_eviction_policy;
using pulsarlm::pages_to_evict;

using page_list = std::vector<std::size_t>;

constexpr std::size_t page_size = 4;
constexpr double infinity = std::numeric_limits<double>::infinity();

// Two samples with this mean and standard deviation.
moments two_samples(double mean, double deviation) {
    return moments{.count = 2, .mean = mean, .m2 = deviation * deviation};
}

// One query head with standard normal densities and scores, and each page's weight sum 1, so a page's score is its
// mass sum.
page_list evicted(const std::vector<float>& mass_sum, const page_eviction_policy& policy) {
    std::vector<float> weight_sum(mass_sum.size(), 1.0f);
    std::vector<moments> density_moments{two_samples(0.0, 1.0)};
    return pages_to_evict(mass_sum, weight_sum, 1, density_moments, two_samples(0.0, 1.0), page_size, policy);
}

TEST_CASE("the defaults evict nothing") {
    page_eviction_policy defaults = pulsarlm::to_pages(eviction_policy{}, page_size);
    REQUIRE(!defaults.can_evict());
    REQUIRE(evicted(std::vector<float>(8, -5.0f), defaults).empty());
}

TEST_CASE("the sink and working tokens round up to pages and the context window down") {
    page_eviction_policy policy = pulsarlm::to_pages(
        eviction_policy{.n_sink = 5, .n_working = 8, .max_context_window = 23},
        page_size
    );
    REQUIRE(policy.n_sink == 2);
    REQUIRE(policy.n_working == 2);
    REQUIRE(policy.max_pages == 5);
    REQUIRE(pulsarlm::to_pages(eviction_policy{.attention_mass_decay = 0.5}, page_size).attention_mass_decay == 0.5);
    REQUIRE(!pulsarlm::to_pages(eviction_policy{}, page_size).max_pages);
    REQUIRE(pulsarlm::to_pages(eviction_policy{.eviction_threshold = 0.2}, page_size).can_evict());
    REQUIRE(
        pulsarlm::to_pages(eviction_policy{.cycle_tokens = 9, .eviction_warmup = 2}, page_size).eviction_warmup == 6
    );
}

TEST_CASE("samples add to their count, mean and sum of squared deviations") {
    moments m;
    for (double x : {1.0, 2.0, 4.0}) {
        pulsarlm::add_sample(m, x);
    }
    REQUIRE(m.count == 3);
    REQUIRE_THAT(m.mean, WithinAbs(7.0 / 3.0, 1e-12));
    REQUIRE_THAT(m.m2, WithinAbs(42.0 / 9.0, 1e-12));
}

TEST_CASE("a token's query weight decays with its age in tokens at the step's end") {
    std::vector<float> weights(3);
    pulsarlm::query_weights(weights, 0.5);
    REQUIRE(weights == std::vector<float>{0.125f, 0.25f, 0.5f});
}

TEST_CASE("adding step mass ages the sums by the step's tokens, then adds the step's") {
    std::vector<float> mass_sum{4.0f, 8.0f};
    std::vector<float> weight_sum{2.0f};
    pulsarlm::add_step_mass(mass_sum, weight_sum, std::vector<float>{1.0f, -1.0f}, std::vector<float>{0.5f}, 2, 0.5);
    REQUIRE(mass_sum == std::vector<float>{2.0f, 1.0f});
    REQUIRE(weight_sum == std::vector<float>{1.0f});
}

TEST_CASE("a page's score is its highest density standardized by its head's moments") {
    std::vector<moments> density_moments{two_samples(1.0, 2.0), two_samples(0.0, 0.5)};
    REQUIRE(pulsarlm::page_score(std::vector<float>{6.0f, 1.0f}, 2.0f, density_moments, 2) == 1.0);
    REQUIRE(pulsarlm::page_score(std::vector<float>{2.0f, -2.0f}, 2.0f, density_moments, 2) == 0.0);
}

TEST_CASE("a head without two samples or spread does not score a page") {
    std::vector<moments> density_moments{
        moments{.count = 1, .mean = 9.0},
        two_samples(0.0, 0.0),
        two_samples(0.0, 1.0)
    };
    REQUIRE(pulsarlm::page_score(std::vector<float>{50.0f, 50.0f, -1.0f}, 1.0f, density_moments, 2) == -1.0);
    REQUIRE(!pulsarlm::page_score(std::vector<float>{1.0f, 1.0f}, 1.0f, std::vector<moments>(2), 2));
    REQUIRE(!pulsarlm::page_score(std::vector<float>{1.0f}, 1.0f, std::vector<moments>{}, 2));
}

TEST_CASE("a page no query token attended scores lowest") {
    REQUIRE(
        pulsarlm::page_score(std::vector<float>{0.0f}, 0.0f, std::vector<moments>{two_samples(0.0, 1.0)}, 2) ==
        -infinity
    );
}

TEST_CASE("a sampled page adds its densities, then its score once the heads can standardize it") {
    std::vector<moments> density_moments(1);
    moments score_moments;
    pulsarlm::add_sampled_page(density_moments, score_moments, std::vector<float>{2.0f}, 2.0f, 2);
    REQUIRE(density_moments[0] == moments{.count = 1, .mean = 1.0, .m2 = 0.0});
    REQUIRE(score_moments.count == 0);

    pulsarlm::add_sampled_page(density_moments, score_moments, std::vector<float>{3.0f}, 1.0f, 2);
    REQUIRE(density_moments[0].count == 2);
    REQUIRE(score_moments.count == 1);
    REQUIRE_THAT(score_moments.mean, WithinAbs(1.0 / std::sqrt(2.0), 1e-12));

    pulsarlm::add_sampled_page(density_moments, score_moments, std::vector<float>{7.0f}, 0.0f, 2);
    REQUIRE(density_moments[0].count == 2);
    REQUIRE(score_moments.count == 1);
}

TEST_CASE("heads and scores count once they hold the warm-up's samples") {
    std::vector<moments> density_moments{moments{.count = 3, .mean = 0.0, .m2 = 2.0}};
    REQUIRE(!pulsarlm::page_score(std::vector<float>{1.0f}, 1.0f, density_moments, 4));
    REQUIRE(pulsarlm::page_score(std::vector<float>{1.0f}, 1.0f, density_moments, 3) == 1.0);

    moments score_moments;
    pulsarlm::add_sampled_page(density_moments, score_moments, std::vector<float>{0.0f}, 1.0f, 5);
    REQUIRE(density_moments[0].count == 4);
    REQUIRE(score_moments.count == 0);
    pulsarlm::add_sampled_page(density_moments, score_moments, std::vector<float>{0.0f}, 1.0f, 5);
    REQUIRE(score_moments.count == 1);

    moments scores = two_samples(1.0, 2.0);
    REQUIRE(pulsarlm::score_threshold(scores, 0.5, 3) == -infinity);
    REQUIRE(pulsarlm::score_threshold(scores, 0.5, 0) == 1.0);
}

TEST_CASE("the score threshold is the normal quantile of the scores' moments") {
    moments scores = two_samples(1.0, 2.0);
    REQUIRE(pulsarlm::score_threshold(scores, 0.5, 2) == 1.0);
    REQUIRE_THAT(pulsarlm::score_threshold(scores, 0.975, 2), WithinAbs(1.0 + 2.0 * 1.959963984540054, 1e-9));
    REQUIRE_THAT(pulsarlm::score_threshold(scores, 0.01, 2), WithinAbs(1.0 - 2.0 * 2.326347874040841, 1e-9));
    REQUIRE(pulsarlm::score_threshold(scores, 0.0, 2) == -infinity);
    REQUIRE(pulsarlm::score_threshold(scores, 1.0, 2) == infinity);
    REQUIRE(pulsarlm::score_threshold(moments{.count = 1, .mean = 1.0}, 0.5, 2) == -infinity);
}

TEST_CASE("pages scoring below the threshold are evicted outside the sink and working pages") {
    page_eviction_policy policy{.n_sink = 2, .n_working = 3, .eviction_threshold = 0.5};
    REQUIRE(evicted({-9.0f, -9.0f, -1.0f, 0.5f, -0.2f, -9.0f, -9.0f, -9.0f}, policy) == page_list{2, 4});
    REQUIRE(evicted({-9.0f, -9.0f, 0.0f, 0.0f, 0.0f, -9.0f, -9.0f, -9.0f}, policy).empty());
}

TEST_CASE("a page without a score stays, and one no query token attended goes") {
    page_eviction_policy policy{.n_sink = 0, .n_working = 1, .eviction_threshold = 0.5};
    std::vector<float> mass_sum{-5.0f, -5.0f, 0.0f};
    std::vector<float> weight_sum{1.0f, 0.0f, 1.0f};
    std::vector<moments> unsampled_head{moments{}};
    REQUIRE(
        pages_to_evict(mass_sum, weight_sum, 1, unsampled_head, two_samples(0.0, 1.0), page_size, policy) ==
        page_list{1}
    );
}

TEST_CASE("fewer than two scores evict nothing by score") {
    page_eviction_policy policy{.n_sink = 0, .n_working = 1, .eviction_threshold = 0.9};
    std::vector<float> mass_sum{-5.0f, -5.0f};
    std::vector<float> weight_sum{1.0f, 1.0f};
    std::vector<moments> density_moments{two_samples(0.0, 1.0)};
    moments one_score{.count = 1};
    REQUIRE(pages_to_evict(mass_sum, weight_sum, 1, density_moments, one_score, page_size, policy).empty());
}

TEST_CASE("a layer evicts nothing by score before its score moments hold the warm-up's samples") {
    page_eviction_policy policy{.n_sink = 0, .n_working = 1, .eviction_threshold = 0.5, .eviction_warmup = 3};
    REQUIRE(evicted({-5.0f, -5.0f, 0.0f}, policy).empty());
}

TEST_CASE("a view no longer than the protected pages evicts nothing") {
    page_eviction_policy policy{.n_sink = 2, .n_working = 3, .eviction_threshold = 1.0};
    REQUIRE(evicted(std::vector<float>(5, -9.0f), policy).empty());
}

TEST_CASE("the cap evicts the oldest unprotected pages, leaving room for a partial tail page and the next cycle") {
    page_eviction_policy policy{.n_sink = 1, .n_working = 1, .cycle_tokens = 5, .max_pages = 6};
    REQUIRE(evicted({0.3f, 0.1f, 0.2f, 0.1f, 0.5f, 0.4f, 0.0f, 0.0f}, policy) == page_list{1, 2, 3, 4});
}

TEST_CASE("a cap within the next cycle's pages evicts every unprotected page") {
    page_eviction_policy policy{.n_sink = 1, .n_working = 1, .cycle_tokens = 1, .max_pages = 1};
    REQUIRE(evicted(std::vector<float>(4, 0.5f), policy) == page_list{1, 2});
}

TEST_CASE("the cap counts the pages the threshold evicted") {
    page_eviction_policy policy{
        .n_sink = 0,
        .n_working = 1,
        .cycle_tokens = 1,
        .max_pages = 6,
        .eviction_threshold = 0.5,
    };
    REQUIRE(evicted({0.3f, 0.1f, -0.2f, -0.1f, 0.5f, 0.4f, 0.25f, 0.0f}, policy) == page_list{0, 2, 3});
}

}  // namespace
