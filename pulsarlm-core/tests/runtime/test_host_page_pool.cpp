#include "pulsarlm/host_page_pool.hpp"

#include <ATen/ATen.h>
#include <cuda_runtime_api.h>

#undef CHECK
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>

namespace {

constexpr std::size_t block_bytes = 4096;

std::int64_t floats(std::size_t bytes) {
    return static_cast<std::int64_t>(bytes / sizeof(float));
}

bool page_locked(const void* data) {
    cudaPointerAttributes attributes{};
    return cudaPointerGetAttributes(&attributes, data) == cudaSuccess && attributes.type == cudaMemoryTypeHost;
}

TEST_CASE("pool pages are pinned and inside the buffer") {
    pulsarlm::host_page_pool pool(4 * block_bytes, block_bytes);
    at::Tensor page = pool.allocate(floats(block_bytes), at::kFloat).value();
    REQUIRE(page_locked(page.const_data_ptr()));
    REQUIRE(pool.offset_of(page.const_data_ptr(), block_bytes).has_value());
    REQUIRE(page_locked(pool.buffer().const_data_ptr()));
}

TEST_CASE("a freed block goes to the next page that fits it") {
    pulsarlm::host_page_pool pool(2 * block_bytes, block_bytes);
    std::optional<at::Tensor> first = pool.allocate(floats(block_bytes), at::kFloat);
    const void* freed = first->const_data_ptr();
    first.reset();
    REQUIRE(pool.allocate(floats(block_bytes), at::kFloat)->const_data_ptr() == freed);
}

TEST_CASE("a full pool gives no page") {
    pulsarlm::host_page_pool pool(2 * block_bytes, block_bytes);
    at::Tensor first = pool.allocate(floats(block_bytes), at::kFloat).value();
    at::Tensor second = pool.allocate(floats(block_bytes), at::kFloat).value();
    REQUIRE(!pool.allocate(floats(block_bytes), at::kFloat).has_value());
}

TEST_CASE("small pages split a large block and merge back into it") {
    pulsarlm::host_page_pool pool(4 * block_bytes, 4 * block_bytes);
    std::optional<at::Tensor> small = pool.allocate(floats(block_bytes), at::kFloat);
    REQUIRE(!pool.allocate(floats(4 * block_bytes), at::kFloat).has_value());
    std::optional<at::Tensor> others = pool.allocate(floats(2 * block_bytes), at::kFloat);
    REQUIRE(others.has_value());
    small.reset();
    others.reset();
    REQUIRE(pool.allocate(floats(4 * block_bytes), at::kFloat).has_value());
}

TEST_CASE("a page takes the block its size rounds up to") {
    pulsarlm::host_page_pool pool(4 * block_bytes, 4 * block_bytes);
    at::Tensor odd = pool.allocate(floats(3 * block_bytes), at::kFloat).value();
    REQUIRE(!pool.allocate(floats(block_bytes), at::kFloat).has_value());
}

TEST_CASE("a pool smaller than its largest block has no block") {
    pulsarlm::host_page_pool pool(block_bytes, 2 * block_bytes);
    REQUIRE(!pool.allocate(floats(block_bytes), at::kFloat).has_value());
    REQUIRE(!pool.offset_of(nullptr, 0).has_value());
}

TEST_CASE("a page outlives its pool") {
    std::optional<pulsarlm::host_page_pool> pool(std::in_place, 2 * block_bytes, block_bytes);
    at::Tensor page = pool->allocate(floats(block_bytes), at::kFloat).value();
    pool.reset();
    page.fill_(1.0f);
    REQUIRE(page.eq(1.0f).all().item<bool>());
}

}  // namespace
