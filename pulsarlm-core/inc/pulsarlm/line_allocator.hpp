#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pulsarlm {

// Lines of several block sizes in one pool of bytes. A block class's line takes an aligned block of its size, a power
// of two; its id is its offset divided by that size. A line is held, cached (not held, keeping a page) or free.
//
// Regions are the aligned blocks of the largest block size; the pool bytes past the last whole one are unused. A new
// line goes into the region with no cached line whose largest free block is the smallest that fits it, else into a free
// region, else into the reclaimable region (no held line) whose newest cached line is the oldest, freeing its cached
// lines. So a region's cached lines are freed together, and no new line joins a region caching older pages.
class line_allocator {
    struct region {
        std::uint32_t held = 0;
        std::uint32_t cached = 0;
        // The release that last cached one of its lines, by release order.
        std::uint64_t newest_cache = 0;
    };

    std::vector<std::size_t> block_orders;
    std::size_t unit_bytes;
    std::size_t region_units;
    std::size_t region_words;
    std::vector<region> regions;
    // region_words words per region, one bit per unit of unit_bytes, set where a held or cached line lies.
    std::vector<std::uint64_t> taken_units;
    // Regions with no cached line, a taken unit and a free one, by their largest free block's order.
    std::set<std::pair<std::size_t, std::uint32_t>> open_regions;
    std::set<std::uint32_t> free_regions;
    // Regions with no held line and a cached one, by newest_cache.
    std::set<std::pair<std::uint64_t, std::uint32_t>> reclaimable_regions;
    std::uint64_t cache_count = 0;

    std::span<std::uint64_t> units_of(std::uint32_t r) noexcept;
    std::span<const std::uint64_t> units_of(std::uint32_t r) const noexcept;
    bool units_free(std::uint32_t r, std::size_t first, std::size_t count) const noexcept;
    void set_units(std::uint32_t r, std::size_t first, std::size_t count, bool taken) noexcept;
    // The first unit of a free block of order in region r.
    std::optional<std::size_t> free_block(std::uint32_t r, std::size_t order) const noexcept;
    std::optional<std::size_t> largest_free_order(std::uint32_t r) const noexcept;
    void index(std::uint32_t r);
    void unindex(std::uint32_t r) noexcept;
    std::pair<std::uint32_t, std::size_t> region_unit(std::size_t block_class, std::uint32_t line_id) const noexcept;

  public:
    // block_bytes holds the block size of each block class, a power of two.
    line_allocator(std::span<const std::size_t> block_bytes, std::size_t pool_bytes);

    std::size_t block_bytes(std::size_t block_class) const noexcept;
    // The ids a block class's lines can have.
    std::size_t line_count(std::size_t block_class) const noexcept;
    std::size_t region_bytes() const noexcept;
    // The ids of the block class's lines that lie in region r.
    std::pair<std::uint32_t, std::uint32_t> lines_of_region(std::size_t block_class, std::uint32_t r) const noexcept;
    // Releases a held line: it is cached when it keeps a page, else free.
    void release(std::size_t block_class, std::uint32_t line_id, bool keeps_page);

    // Changes to the lines, kept by commit and undone otherwise, when the transaction is destroyed. Nothing else uses
    // the allocator while one is open.
    class transaction {
        line_allocator& lines;
        std::unordered_map<std::uint32_t, std::pair<region, std::vector<std::uint64_t>>> saved;
        std::vector<std::uint32_t> reclaimed;
        bool committed = false;

        void save(std::uint32_t r);

      public:
        explicit transaction(line_allocator& lines) noexcept;
        transaction(const transaction&) = delete;
        transaction& operator=(const transaction&) = delete;
        ~transaction();

        // A held line of the block class; nullopt when none fits.
        std::optional<std::uint32_t> take(std::size_t block_class);
        // A cached line becomes held.
        void hold(std::size_t block_class, std::uint32_t line_id);
        // Keeps the changes. Returns the regions whose cached lines a take freed; their pages are gone.
        std::vector<std::uint32_t> commit() noexcept;
    };
};

// The smallest pool whose regions hold one line of each block size of block_bytes.
std::size_t min_pool_bytes(std::span<const std::size_t> block_bytes);

}  // namespace pulsarlm
