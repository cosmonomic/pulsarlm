#include "pulsarlm/line_allocator.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace {

using pulsarlm::line_allocator;

constexpr std::size_t small = 0;
constexpr std::size_t large = 1;
// Two small lines fill a region, the size of a large line.
const std::vector<std::size_t> blocks{16, 32};

std::uint32_t take(line_allocator& lines, std::size_t block_class) {
    line_allocator::transaction taking(lines);
    std::optional<std::uint32_t> line_id = taking.take(block_class);
    REQUIRE(line_id.has_value());
    REQUIRE(taking.commit().empty());
    return *line_id;
}

std::optional<std::uint32_t> try_take(line_allocator& lines, std::size_t block_class) {
    line_allocator::transaction taking(lines);
    return taking.take(block_class);
}

std::uint32_t region_of(const line_allocator& lines, std::size_t block_class, std::uint32_t line_id) {
    return static_cast<std::uint32_t>(line_id * lines.block_bytes(block_class) / lines.region_bytes());
}

}  // namespace

TEST_CASE("a pool holds the whole regions of its largest block") {
    line_allocator lines(blocks, 3 * 32 + 16);
    REQUIRE(lines.region_bytes() == 32);
    REQUIRE(lines.line_count(small) == 6);
    REQUIRE(lines.line_count(large) == 3);
    REQUIRE(pulsarlm::min_pool_bytes(std::vector<std::size_t>{16, 32, 16, 16}) == 96);
}

TEST_CASE("small lines fill a region before taking another") {
    line_allocator lines(blocks, 2 * 32);
    std::uint32_t first = take(lines, small);
    std::uint32_t second = take(lines, small);
    REQUIRE(region_of(lines, small, first) == region_of(lines, small, second));
    REQUIRE(first != second);
    std::uint32_t large_line = take(lines, large);
    REQUIRE(region_of(lines, large, large_line) != region_of(lines, small, first));
    REQUIRE(!try_take(lines, small));
}

TEST_CASE("no line joins a region caching a page") {
    line_allocator lines(blocks, 2 * 32);
    std::uint32_t cached = take(lines, small);
    lines.release(small, cached, true);
    std::uint32_t next = take(lines, small);
    REQUIRE(region_of(lines, small, next) != region_of(lines, small, cached));
}

TEST_CASE("a line released without a page frees its block for the region's next line") {
    line_allocator lines(blocks, 32);
    std::uint32_t held = take(lines, small);
    std::uint32_t freed = take(lines, small);
    lines.release(small, freed, false);
    REQUIRE(take(lines, small) == freed);
    lines.release(small, held, false);
    lines.release(small, freed, false);
    take(lines, large);
}

TEST_CASE("a take reclaims the region whose newest cached line is the oldest") {
    line_allocator lines(blocks, 2 * 32);
    std::uint32_t a = take(lines, small);
    std::uint32_t a_other = take(lines, small);
    std::uint32_t b = take(lines, large);
    lines.release(small, a, true);
    lines.release(large, b, true);
    lines.release(small, a_other, true);

    line_allocator::transaction taking(lines);
    std::optional<std::uint32_t> line_id = taking.take(large);
    REQUIRE(line_id == b);
    REQUIRE(taking.commit() == std::vector<std::uint32_t>{region_of(lines, large, b)});
}

TEST_CASE("a region holding a held line is not reclaimed") {
    line_allocator lines(blocks, 32);
    std::uint32_t held = take(lines, small);
    std::uint32_t cached = take(lines, small);
    lines.release(small, cached, true);
    REQUIRE(!try_take(lines, small));
    REQUIRE(!try_take(lines, large));
    lines.release(small, held, false);
    REQUIRE(try_take(lines, large).has_value());
}

TEST_CASE("an uncommitted transaction undoes its takes, holds and reclaims") {
    line_allocator lines(blocks, 2 * 32);
    std::uint32_t cached = take(lines, large);
    lines.release(large, cached, true);
    {
        line_allocator::transaction taking(lines);
        REQUIRE(taking.take(large).has_value());
        REQUIRE(taking.take(large) == cached);
        REQUIRE(!taking.take(small));
    }
    {
        line_allocator::transaction taking(lines);
        taking.hold(large, cached);
        REQUIRE(taking.take(large).has_value());
        REQUIRE(!taking.take(large));
    }
    line_allocator::transaction taking(lines);
    REQUIRE(taking.take(large).has_value());
    REQUIRE(taking.take(large) == cached);
    REQUIRE(taking.commit().size() == 1);
}

TEST_CASE("a held cached line keeps its region from reclaim until released") {
    line_allocator lines(blocks, 32);
    std::uint32_t cached = take(lines, large);
    lines.release(large, cached, true);
    {
        line_allocator::transaction holding(lines);
        holding.hold(large, cached);
        holding.commit();
    }
    REQUIRE(!try_take(lines, small));
    lines.release(large, cached, true);
    REQUIRE(try_take(lines, small).has_value());
}
