#include "pulsarlm/engine/line_allocator.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
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

TEST_CASE("a take matches a brute-force search over regions of many words") {
    // Regions of 4096 units of 16 bytes, 64 words; the classes span orders below, at and above a word.
    const std::vector<std::size_t> sizes{16, 64, 512, 2048, 16384, 65536};
    const std::size_t region_bytes = 65536;
    const std::size_t region_count = 3;
    line_allocator lines(sizes, region_count * region_bytes);
    std::vector<std::vector<bool>> taken_units(region_count, std::vector<bool>(region_bytes / 16, false));
    std::vector<std::vector<std::pair<std::size_t, std::uint32_t>>> held(region_count);

    auto first_free = [&](std::size_t region, std::size_t units) -> std::optional<std::size_t> {
        for (std::size_t first = 0; first < taken_units[region].size(); first += units) {
            bool free = true;
            for (std::size_t unit = first; unit < first + units && free; unit++) {
                free = !taken_units[region][unit];
            }
            if (free) {
                return first;
            }
        }
        return std::nullopt;
    };
    auto largest_free_units = [&](std::size_t region) {
        for (std::size_t units = region_bytes / 16; units > 0; units /= 2) {
            if (first_free(region, units)) {
                return units;
            }
        }
        return std::size_t{0};
    };

    std::uint64_t state = 7;
    auto next = [&](std::uint64_t bound) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return (state >> 33) % bound;
    };
    for (std::size_t step = 0; step < 4000; step++) {
        bool releases = next(3) == 0;
        std::size_t loaded = 0;
        for (const auto& region_lines : held) {
            loaded += region_lines.size();
        }
        if (releases && loaded > 0) {
            std::size_t region = next(region_count);
            while (held[region].empty()) {
                region = (region + 1) % region_count;
            }
            std::size_t index = next(held[region].size());
            auto [block_class, line_id] = held[region][index];
            held[region].erase(held[region].begin() + static_cast<std::ptrdiff_t>(index));
            lines.release(block_class, line_id, false);
            std::size_t first = line_id * sizes[block_class] % region_bytes / 16;
            for (std::size_t unit = first; unit < first + sizes[block_class] / 16; unit++) {
                taken_units[region][unit] = false;
            }
            continue;
        }
        std::size_t block_class = next(sizes.size() - 1);
        std::size_t units = sizes[block_class] / 16;
        std::optional<std::size_t> expected_region;
        std::size_t expected_room = 0;
        for (std::size_t region = 0; region < region_count; region++) {
            std::size_t room = largest_free_units(region);
            bool open = !held[region].empty() && room >= units;
            bool better = !expected_region || held[*expected_region].empty() || room < expected_room;
            if (open && better) {
                expected_region = region;
                expected_room = room;
            }
        }
        if (!expected_region) {
            for (std::size_t region = 0; region < region_count && !expected_region; region++) {
                if (held[region].empty()) {
                    expected_region = region;
                }
            }
        }
        std::optional<std::uint32_t> line_id = [&] {
            line_allocator::transaction taking(lines);
            std::optional<std::uint32_t> id = taking.take(block_class);
            taking.commit();
            return id;
        }();
        REQUIRE(line_id.has_value() == expected_region.has_value());
        if (!line_id) {
            continue;
        }
        std::size_t region = *line_id * sizes[block_class] / region_bytes;
        std::size_t first = *line_id * sizes[block_class] % region_bytes / 16;
        INFO("step " << step << " class " << block_class);
        REQUIRE(region == *expected_region);
        REQUIRE(first_free(region, units) == first);
        for (std::size_t unit = first; unit < first + units; unit++) {
            taken_units[region][unit] = true;
        }
        held[region].emplace_back(block_class, *line_id);
    }
}
