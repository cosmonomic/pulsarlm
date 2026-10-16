#include "pulsarlm/line_allocator.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace pulsarlm {

namespace {

std::size_t ceil_div(std::size_t n, std::size_t d) {
    return (n + d - 1) / d;
}

std::size_t largest(std::span<const std::size_t> sizes) {
    return *std::ranges::max_element(sizes);
}

}  // namespace

line_allocator::line_allocator(std::span<const std::size_t> block_bytes, std::size_t pool_bytes)
: unit_bytes(*std::ranges::min_element(block_bytes)),
  region_units(largest(block_bytes) / unit_bytes),
  region_words(ceil_div(region_units, 64)),
  regions(pool_bytes / largest(block_bytes)),
  taken_units(regions.size() * region_words, 0) {
    for (std::size_t bytes : block_bytes) {
        assert(std::has_single_bit(bytes));
        block_orders.push_back(static_cast<std::size_t>(std::countr_zero(bytes / unit_bytes)));
    }
    for (std::uint32_t r = 0; r < regions.size(); r++) {
        free_regions.insert(r);
    }
}

std::size_t line_allocator::block_bytes(std::size_t block_class) const noexcept {
    return unit_bytes << block_orders[block_class];
}

std::size_t line_allocator::line_count(std::size_t block_class) const noexcept {
    return regions.size() * region_bytes() / block_bytes(block_class);
}

std::size_t line_allocator::region_bytes() const noexcept {
    return region_units * unit_bytes;
}

std::pair<std::uint32_t, std::uint32_t>
line_allocator::lines_of_region(std::size_t block_class, std::uint32_t r) const noexcept {
    auto per_region = static_cast<std::uint32_t>(region_bytes() / block_bytes(block_class));
    return {r * per_region, (r + 1) * per_region};
}

std::span<std::uint64_t> line_allocator::units_of(std::uint32_t r) noexcept {
    return std::span(taken_units).subspan(r * region_words, region_words);
}

std::span<const std::uint64_t> line_allocator::units_of(std::uint32_t r) const noexcept {
    return std::span(taken_units).subspan(r * region_words, region_words);
}

bool line_allocator::units_free(std::uint32_t r, std::size_t first, std::size_t count) const noexcept {
    std::span<const std::uint64_t> units = units_of(r);
    for (std::size_t unit = first; unit < first + count; unit++) {
        if ((units[unit / 64] >> (unit % 64)) & 1) {
            return false;
        }
    }
    return true;
}

void line_allocator::set_units(std::uint32_t r, std::size_t first, std::size_t count, bool taken) noexcept {
    std::span<std::uint64_t> units = units_of(r);
    for (std::size_t unit = first; unit < first + count; unit++) {
        std::uint64_t bit = std::uint64_t{1} << (unit % 64);
        units[unit / 64] = taken ? units[unit / 64] | bit : units[unit / 64] & ~bit;
    }
}

std::optional<std::size_t> line_allocator::free_block(std::uint32_t r, std::size_t order) const noexcept {
    std::size_t count = std::size_t{1} << order;
    for (std::size_t first = 0; first < region_units; first += count) {
        if (units_free(r, first, count)) {
            return first;
        }
    }
    return std::nullopt;
}

std::optional<std::size_t> line_allocator::largest_free_order(std::uint32_t r) const noexcept {
    for (std::size_t order = std::countr_zero(region_units) + 1; order > 0; order--) {
        if (free_block(r, order - 1)) {
            return order - 1;
        }
    }
    return std::nullopt;
}

void line_allocator::index(std::uint32_t r) {
    const region& indexed = regions[r];
    if (indexed.held == 0 && indexed.cached == 0) {
        assert(std::ranges::all_of(units_of(r), [](std::uint64_t word) { return word == 0; }));
        free_regions.insert(r);
    } else if (indexed.held == 0) {
        reclaimable_regions.emplace(indexed.newest_cache, r);
    } else if (indexed.cached == 0) {
        if (std::optional<std::size_t> order = largest_free_order(r)) {
            open_regions.emplace(*order, r);
        }
    }
}

void line_allocator::unindex(std::uint32_t r) noexcept {
    const region& indexed = regions[r];
    if (indexed.held == 0 && indexed.cached == 0) {
        free_regions.erase(r);
    } else if (indexed.held == 0) {
        reclaimable_regions.erase({indexed.newest_cache, r});
    } else if (indexed.cached == 0) {
        if (std::optional<std::size_t> order = largest_free_order(r)) {
            open_regions.erase({*order, r});
        }
    }
}

std::pair<std::uint32_t, std::size_t>
line_allocator::region_unit(std::size_t block_class, std::uint32_t line_id) const noexcept {
    std::size_t offset = line_id * block_bytes(block_class);
    return {static_cast<std::uint32_t>(offset / region_bytes()), offset % region_bytes() / unit_bytes};
}

void line_allocator::release(std::size_t block_class, std::uint32_t line_id, bool keeps_page) {
    auto [r, unit] = region_unit(block_class, line_id);
    unindex(r);
    region& released = regions[r];
    assert(released.held > 0);
    released.held--;
    if (keeps_page) {
        released.cached++;
        released.newest_cache = ++cache_count;
    } else {
        set_units(r, unit, std::size_t{1} << block_orders[block_class], false);
    }
    index(r);
}

line_allocator::transaction::transaction(line_allocator& lines) noexcept
: lines(lines) {}

line_allocator::transaction::~transaction() {
    if (committed) {
        return;
    }
    for (auto& [r, before] : saved) {
        lines.unindex(r);
        lines.regions[r] = before.first;
        std::ranges::copy(before.second, lines.units_of(r).begin());
        lines.index(r);
    }
}

void line_allocator::transaction::save(std::uint32_t r) {
    std::span<const std::uint64_t> units = lines.units_of(r);
    saved.try_emplace(r, lines.regions[r], std::vector<std::uint64_t>(units.begin(), units.end()));
}

std::optional<std::uint32_t> line_allocator::transaction::take(std::size_t block_class) {
    std::size_t order = lines.block_orders[block_class];
    std::uint32_t r = 0;
    if (auto open = lines.open_regions.lower_bound({order, 0}); open != lines.open_regions.end()) {
        r = open->second;
    } else if (!lines.free_regions.empty()) {
        r = *lines.free_regions.begin();
    } else if (!lines.reclaimable_regions.empty()) {
        r = lines.reclaimable_regions.begin()->second;
        save(r);
        lines.unindex(r);
        lines.regions[r].cached = 0;
        std::ranges::fill(lines.units_of(r), 0);
        lines.index(r);
        reclaimed.push_back(r);
    } else {
        return std::nullopt;
    }
    save(r);
    lines.unindex(r);
    std::optional<std::size_t> first = lines.free_block(r, order);
    assert(first);
    lines.set_units(r, *first, std::size_t{1} << order, true);
    lines.regions[r].held++;
    lines.index(r);
    return static_cast<std::uint32_t>(
        (r * lines.region_bytes() + *first * lines.unit_bytes) / lines.block_bytes(block_class)
    );
}

void line_allocator::transaction::hold(std::size_t block_class, std::uint32_t line_id) {
    std::uint32_t r = lines.region_unit(block_class, line_id).first;
    save(r);
    lines.unindex(r);
    region& held = lines.regions[r];
    assert(held.cached > 0);
    held.cached--;
    held.held++;
    lines.index(r);
}

std::vector<std::uint32_t> line_allocator::transaction::commit() noexcept {
    committed = true;
    return std::move(reclaimed);
}

std::size_t min_pool_bytes(std::span<const std::size_t> block_bytes) {
    std::size_t region = largest(block_bytes);
    std::size_t total = std::accumulate(block_bytes.begin(), block_bytes.end(), std::size_t{0});
    return ceil_div(total, region) * region;
}

}  // namespace pulsarlm
