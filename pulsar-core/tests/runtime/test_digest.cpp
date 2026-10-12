#include "pulsar/digest.hpp"
#include "pulsar/kv_cache.hpp"

#include <ATen/ATen.h>

#undef CHECK
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <span>
#include <vector>

namespace {

using pulsar::digest;

constexpr pulsar::page_shape shape{.n_layers = 2, .page_size = 4, .n_kv_heads = 2, .head_dim = 8, .dtype = at::kHalf};

const digest parent{.hi = 1, .lo = 2};

}  // namespace

TEST_CASE("page digests chain each full page from the one before it") {
    std::vector<std::int64_t> tokens{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::span<const std::int64_t> all(tokens);

    std::vector<digest> digests = pulsar::make_page_digests(parent, all, 4);

    digest first = pulsar::make_digest(parent, all.subspan(0, 4));
    digest second = pulsar::make_digest(first, all.subspan(4, 4));
    REQUIRE(digests.size() == 2);
    REQUIRE(digests[0] == first);
    REQUIRE(digests[1] == second);
}

TEST_CASE("page digests of fewer tokens than a page are empty") {
    std::vector<std::int64_t> tokens{1, 2, 3};
    REQUIRE(pulsar::make_page_digests(parent, tokens, 4).empty());
    REQUIRE(pulsar::make_page_digests(parent, {}, 4).empty());
}

TEST_CASE("the root digest changes with the model and every field of the page shape") {
    digest root = pulsar::make_root_digest("qwen3-8b@1", shape);
    REQUIRE(root == pulsar::make_root_digest("qwen3-8b@1", shape));

    pulsar::page_shape other_page_size = shape;
    other_page_size.page_size = 8;
    pulsar::page_shape other_dtype = shape;
    other_dtype.dtype = at::kBFloat16;
    pulsar::page_shape other_layers = shape;
    other_layers.n_layers = 3;
    pulsar::page_shape other_heads = shape;
    other_heads.n_kv_heads = 4;
    pulsar::page_shape other_head_dim = shape;
    other_head_dim.head_dim = 16;

    std::vector<digest> others{
        pulsar::make_root_digest("qwen3-8b@2", shape),
        pulsar::make_root_digest("qwen3-8b@1", other_page_size),
        pulsar::make_root_digest("qwen3-8b@1", other_dtype),
        pulsar::make_root_digest("qwen3-8b@1", other_layers),
        pulsar::make_root_digest("qwen3-8b@1", other_heads),
        pulsar::make_root_digest("qwen3-8b@1", other_head_dim),
    };
    for (const digest& other : others) {
        REQUIRE(other != root);
    }
}
