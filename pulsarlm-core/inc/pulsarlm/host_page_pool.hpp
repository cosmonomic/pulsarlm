#pragma once

#include <ATen/ATen.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_set>
#include <vector>

namespace pulsarlm {

// Pinned host memory for the pages a kv_cache copies to and from its store: one buffer of capacity_bytes, rounded
// down to a multiple of largest_bytes rounded up to a power of two, managed as a buddy allocator over blocks of
// min_block_bytes doubling up to that size. A request takes the smallest free block that holds it, splitting a larger
// one; a freed block merges with its free buddy. A page may be freed on any thread and may outlive the pool.
class host_page_pool {
    static constexpr std::size_t min_block_bytes = 4096;

    struct shared_state {
        std::mutex mutex;
        std::byte* data = nullptr;
        std::size_t bytes = 0;
        // free_blocks[order] holds the offsets of the free blocks of min_block_bytes << order.
        std::vector<std::unordered_set<std::size_t>> free_blocks;

        ~shared_state();
        std::optional<std::size_t> take(std::size_t order);
        void give_back(std::size_t offset, std::size_t order);
    };

    std::shared_ptr<shared_state> state;
    at::Tensor buffer_bytes;

  public:
    host_page_pool(std::size_t capacity_bytes, std::size_t largest_bytes);

    // A pinned uint8 tensor of bytes in the pool; nullopt when no free block holds it.
    std::optional<at::Tensor> allocate(std::size_t bytes);
    // The byte offset of [data, data + bytes) in buffer(); nullopt when the range is not inside it.
    std::optional<std::size_t> offset_of(const void* data, std::size_t bytes) const noexcept;
    // The whole buffer as a pinned byte tensor, empty when the pool has no block.
    const at::Tensor& buffer() const noexcept;
};

}  // namespace pulsarlm
