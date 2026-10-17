#include "pulsarlm/eviction.hpp"

#undef CHECK
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace {

using pulsarlm::eviction_policy;
using pulsarlm::pages_to_evict;

using page_list = std::vector<std::size_t>;

// One query head per page.
page_list evicted(const std::vector<float>& density, double bar, const eviction_policy& policy) {
    return pages_to_evict(density, 1, bar, policy);
}

TEST_CASE("the defaults evict nothing") {
    REQUIRE(!eviction_policy{}.can_evict());
    REQUIRE(pulsarlm::density_bar(eviction_policy{}.eviction_target, 8, 1024) == 0.0);
    REQUIRE(evicted(std::vector<float>(8, 0.0f), 0.0, eviction_policy{}).empty());
}

TEST_CASE("the bar keeps at most the target's share of the context over the heads") {
    REQUIRE(pulsarlm::density_bar(0.5, 8, 1024) == 8.0 / 512.0);
    REQUIRE(pulsarlm::density_bar(1.0, 8, 1024) == 0.0);
    REQUIRE(eviction_policy{.eviction_target = 0.8}.can_evict());
}

TEST_CASE("pages below the bar are evicted outside the sink and working pages") {
    eviction_policy policy{.n_sink = 2, .n_working = 3, .eviction_target = 0.5};
    REQUIRE(evicted(std::vector<float>(8, 0.0f), 0.4, policy) == page_list{2, 3, 4});
    REQUIRE(evicted({0.0f, 0.0f, 0.0f, 0.7f, 0.2f, 0.0f, 0.0f, 0.0f}, 0.4, policy) == page_list{2, 4});
}

TEST_CASE("a page any head holds at the bar stays") {
    eviction_policy policy{.n_sink = 0, .n_working = 1, .eviction_target = 0.5};
    std::vector<float> density{0.1f, 0.5f, 0.2f, 0.3f, 0.0f, 0.0f};
    REQUIRE(pages_to_evict(density, 2, 0.4, policy) == page_list{1});
}

TEST_CASE("a view no longer than the protected pages evicts nothing") {
    eviction_policy policy{.n_sink = 2, .n_working = 3, .eviction_target = 0.5};
    REQUIRE(evicted(std::vector<float>(5, 0.0f), 0.4, policy).empty());
}

TEST_CASE("the cap evicts the lowest-density pages, the oldest first on ties") {
    eviction_policy policy{.n_sink = 0, .n_working = 1, .max_pages = 4};
    REQUIRE(evicted({0.3f, 0.1f, 0.2f, 0.1f, 0.5f, 0.4f, 0.0f, 0.0f}, 0.0, policy) == page_list{1, 2, 3, 6});
}

TEST_CASE("the cap ranks a page by its highest head") {
    eviction_policy policy{.n_sink = 0, .n_working = 1, .max_pages = 2};
    std::vector<float> density{0.3f, 0.0f, 0.0f, 0.2f, 0.1f, 0.1f, 0.0f, 0.0f};
    REQUIRE(pages_to_evict(density, 2, 0.0, policy) == page_list{1, 2});
}

TEST_CASE("the cap counts the pages the bar evicted") {
    eviction_policy policy{.n_sink = 0, .n_working = 1, .max_pages = 5, .eviction_target = 0.5};
    REQUIRE(evicted({0.3f, 0.1f, 0.2f, 0.1f, 0.5f, 0.4f, 0.25f, 0.0f}, 0.125, policy) == page_list{1, 2, 3});
}

TEST_CASE("a step advances the latest page once per page whose first token it forwards") {
    REQUIRE(pulsarlm::latest_page_advance(0, 1, 4) == 1);
    REQUIRE(pulsarlm::latest_page_advance(1, 3, 4) == 0);
    REQUIRE(pulsarlm::latest_page_advance(3, 2, 4) == 1);
    REQUIRE(pulsarlm::latest_page_advance(2, 11, 4) == 3);
}

TEST_CASE("a token's mass weight decays with its page's age in the step") {
    std::vector<float> weights(6);
    pulsarlm::mass_weights(weights, 2, 4, 0.5);
    REQUIRE(weights == std::vector<float>{0.0625f, 0.0625f, 0.125f, 0.125f, 0.125f, 0.125f});
}

TEST_CASE("adding step mass ages the densities, then adds the mass") {
    std::vector<float> density{4.0f, 8.0f, 0.0f};
    std::vector<float> step_mass{1.0f, 0.0f, 3.0f};
    pulsarlm::add_step_mass(density, step_mass, 1, 0.5);
    REQUIRE(density == std::vector<float>{3.0f, 4.0f, 3.0f});

    pulsarlm::add_step_mass(density, std::vector<float>(3, 0.0f), 2, 0.5);
    REQUIRE(density == std::vector<float>{0.75f, 1.0f, 0.75f});
}

}  // namespace
