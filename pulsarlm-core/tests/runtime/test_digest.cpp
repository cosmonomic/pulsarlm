#include "pulsarlm/digest.hpp"
#include "pulsarlm/engine/kv_cache.hpp"

#include <ATen/ATen.h>

#undef CHECK
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace {

using pulsarlm::digest;

constexpr pulsarlm::page_shape shape{.n_layers = 2, .page_size = 4, .n_kv_heads = 2, .head_dim = 8, .dtype = at::kHalf};

const digest parent{.hi = 1, .lo = 2};

}  // namespace

TEST_CASE("page digests chain each full page from the one before it") {
    std::vector<std::int64_t> tokens{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::span<const std::int64_t> all(tokens);

    std::vector<digest> digests = pulsarlm::make_page_digests(parent, all, 4);

    digest first = pulsarlm::make_digest(parent, all.subspan(0, 4));
    digest second = pulsarlm::make_digest(first, all.subspan(4, 4));
    REQUIRE(digests.size() == 2);
    REQUIRE(digests[0] == first);
    REQUIRE(digests[1] == second);
}

TEST_CASE("page digests of fewer tokens than a page are empty") {
    std::vector<std::int64_t> tokens{1, 2, 3};
    REQUIRE(pulsarlm::make_page_digests(parent, tokens, 4).empty());
    REQUIRE(pulsarlm::make_page_digests(parent, {}, 4).empty());
}

digest root_of(std::string_view model_name, std::string_view model_hash, std::vector<pulsarlm::page_shape> shapes) {
    return pulsarlm::make_root_digest(model_name, model_hash, shapes);
}

TEST_CASE("the root digest changes with the model name, the model hash and every field of every page shape") {
    digest root = root_of("qwen3-8b", "1", {shape, shape});
    REQUIRE(root == root_of("qwen3-8b", "1", {shape, shape}));

    pulsarlm::page_shape other_page_size = shape;
    other_page_size.page_size = 8;
    pulsarlm::page_shape other_dtype = shape;
    other_dtype.dtype = at::kBFloat16;
    pulsarlm::page_shape other_layers = shape;
    other_layers.n_layers = 3;
    pulsarlm::page_shape other_heads = shape;
    other_heads.n_kv_heads = 4;
    pulsarlm::page_shape other_head_dim = shape;
    other_head_dim.head_dim = 16;
    pulsarlm::page_shape values_only = shape;
    values_only.values_only = true;

    std::vector<digest> others{
        root_of("qwen3-9b", "1", {shape, shape}),
        root_of("qwen3-8b", "2", {shape, shape}),
        root_of("qwen3-8b1", "", {shape, shape}),
        root_of("qwen3-8b", "1", {shape}),
        root_of("qwen3-8b", "1", {shape, other_page_size}),
        root_of("qwen3-8b", "1", {shape, other_dtype}),
        root_of("qwen3-8b", "1", {shape, other_layers}),
        root_of("qwen3-8b", "1", {other_heads, shape}),
        root_of("qwen3-8b", "1", {shape, other_head_dim}),
        root_of("qwen3-8b", "1", {shape, values_only}),
    };
    for (const digest& other : others) {
        REQUIRE(other != root);
    }
}

TEST_CASE("a request's root digest is the engine's under a policy that cannot evict") {
    pulsarlm::page_eviction_policy keeps_every_page{.n_sink = 64, .n_working = 512, .cycle_tokens = 512};
    REQUIRE(pulsarlm::make_request_root_digest(parent, keeps_every_page) == parent);
    keeps_every_page.cycle_tokens = 1024;
    REQUIRE(pulsarlm::make_request_root_digest(parent, keeps_every_page) == parent);
}

TEST_CASE("a request's root digest changes with the engine's root and every field of a policy that can evict") {
    const pulsarlm::page_eviction_policy policy{
        .n_sink = 64,
        .n_working = 512,
        .cycle_tokens = 512,
        .max_pages = std::nullopt,
        .eviction_threshold = 0.5,
        .eviction_warmup = 32,
        .attention_mass_decay = 1.0 / 256,
    };
    digest root = pulsarlm::make_request_root_digest(parent, policy);
    REQUIRE(root != parent);
    REQUIRE(root == pulsarlm::make_request_root_digest(parent, policy));

    std::vector<pulsarlm::page_eviction_policy> others(8, policy);
    others[0].n_sink = 65;
    others[1].n_working = 513;
    others[2].cycle_tokens = 256;
    others[3].max_pages = 4096;
    others[4].max_pages = 0;
    others[5].eviction_threshold = 0.25;
    others[6].eviction_warmup = 64;
    others[7].attention_mass_decay = 1.0 / 128;
    for (const pulsarlm::page_eviction_policy& other : others) {
        REQUIRE(pulsarlm::make_request_root_digest(parent, other) != root);
    }
    REQUIRE(pulsarlm::make_request_root_digest(digest{.hi = 1, .lo = 3}, policy) != root);
}
