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

bool page_locked(const void* data) {
    cudaPointerAttributes attributes{};
    return cudaPointerGetAttributes(&attributes, data) == cudaSuccess && attributes.type == cudaMemoryTypeHost;
}

TEST_CASE("pool pages are pinned and inside the buffer") {
    pulsarlm::host_page_pool pool(4 * block_bytes, block_bytes);
    at::Tensor page = pool.allocate(block_bytes).value();
    REQUIRE(page_locked(page.const_data_ptr()));
    REQUIRE(pool.offset_of(page.const_data_ptr(), block_bytes).has_value());
    REQUIRE(page_locked(pool.buffer().const_data_ptr()));
}

TEST_CASE("a freed block goes to the next page that fits it") {
    pulsarlm::host_page_pool pool(2 * block_bytes, block_bytes);
    std::optional<at::Tensor> first = pool.allocate(block_bytes);
    const void* freed = first->const_data_ptr();
    first.reset();
    REQUIRE(pool.allocate(block_bytes)->const_data_ptr() == freed);
}

TEST_CASE("a full pool gives no page") {
    pulsarlm::host_page_pool pool(2 * block_bytes, block_bytes);
    at::Tensor first = pool.allocate(block_bytes).value();
    at::Tensor second = pool.allocate(block_bytes).value();
    REQUIRE(!pool.allocate(block_bytes).has_value());
}

TEST_CASE("small pages split a large block and merge back into it") {
    pulsarlm::host_page_pool pool(4 * block_bytes, 4 * block_bytes);
    std::optional<at::Tensor> small = pool.allocate(block_bytes);
    REQUIRE(!pool.allocate(4 * block_bytes).has_value());
    std::optional<at::Tensor> others = pool.allocate(2 * block_bytes);
    REQUIRE(others.has_value());
    small.reset();
    others.reset();
    REQUIRE(pool.allocate(4 * block_bytes).has_value());
}

TEST_CASE("a page takes the block its size rounds up to") {
    pulsarlm::host_page_pool pool(4 * block_bytes, 4 * block_bytes);
    at::Tensor odd = pool.allocate(3 * block_bytes).value();
    REQUIRE(!pool.allocate(block_bytes).has_value());
}

TEST_CASE("a pool smaller than its largest block has no block") {
    pulsarlm::host_page_pool pool(block_bytes, 2 * block_bytes);
    REQUIRE(!pool.allocate(block_bytes).has_value());
    REQUIRE(!pool.offset_of(nullptr, 0).has_value());
}

TEST_CASE("a page outlives its pool") {
    std::optional<pulsarlm::host_page_pool> pool(std::in_place, 2 * block_bytes, block_bytes);
    at::Tensor page = pool->allocate(block_bytes).value();
    pool.reset();
    page.fill_(1.0f);
    REQUIRE(page.eq(1.0f).all().item<bool>());
}

}  // namespace
