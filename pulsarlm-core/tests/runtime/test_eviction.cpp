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

page_list evicted(const std::vector<float>& density, const eviction_policy& policy) {
    return pages_to_evict(density, policy);
}

TEST_CASE("the defaults evict nothing") {
    REQUIRE(!eviction_policy{}.can_evict());
    REQUIRE(evicted(std::vector<float>(8, 0.0f), eviction_policy{}).empty());
}

TEST_CASE("the bar is tau / (1 - tau), from nothing at 0 to everything at 1") {
    REQUIRE(pulsarlm::evict_bar_from_tau(0.0) == 0.0);
    REQUIRE(pulsarlm::evict_bar_from_tau(0.5) == 1.0);
    REQUIRE(pulsarlm::evict_bar_from_tau(0.75) == 3.0);
    REQUIRE(std::isinf(pulsarlm::evict_bar_from_tau(1.0)));

    eviction_policy everything{.n_sink = 0, .n_working = 1, .evict_tau = 1.0, .attention_mass_decay = 1.0};
    REQUIRE(evicted(std::vector<float>(4, 1e9f), everything) == page_list{0, 1, 2});
}

TEST_CASE("pages below the bar are evicted outside the sink and working pages") {
    eviction_policy policy{.n_sink = 2, .n_working = 3, .evict_tau = 0.4, .attention_mass_decay = 1.0};
    REQUIRE(evicted(std::vector<float>(8, 0.0f), policy) == page_list{2, 3, 4});
    REQUIRE(evicted({0.0f, 0.0f, 0.0f, 0.7f, 0.2f, 0.0f, 0.0f, 0.0f}, policy) == page_list{2, 4});
}

TEST_CASE("a view no longer than the protected pages evicts nothing") {
    eviction_policy policy{.n_sink = 2, .n_working = 3, .evict_tau = 0.4, .attention_mass_decay = 1.0};
    REQUIRE(evicted(std::vector<float>(5, 0.0f), policy).empty());
}

TEST_CASE("the cap evicts the lowest-density pages, the oldest first on ties") {
    eviction_policy policy{.n_sink = 0, .n_working = 1, .max_pages = 4, .attention_mass_decay = 1.0};
    REQUIRE(evicted({0.3f, 0.1f, 0.2f, 0.1f, 0.5f, 0.4f, 0.0f, 0.0f}, policy) == page_list{1, 2, 3, 6});
}

TEST_CASE("the cap counts the pages the bar evicted") {
    eviction_policy
        policy{.n_sink = 0, .n_working = 1, .max_pages = 5, .evict_tau = 0.125, .attention_mass_decay = 1.0};
    REQUIRE(evicted({0.3f, 0.1f, 0.2f, 0.1f, 0.5f, 0.4f, 0.25f, 0.0f}, policy) == page_list{1, 2, 3});
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
