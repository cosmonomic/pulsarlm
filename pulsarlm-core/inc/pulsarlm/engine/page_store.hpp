#pragma once

#include <ATen/core/Tensor.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

#include "pulsarlm/digest.hpp"
#include "pulsarlm/engine/event_loop.hpp"

namespace pulsarlm {

// One line of a layer of the whole model as uint8 [line_bytes(shape)], laid out as the line, in host memory.
struct page {
    at::Tensor bytes;
};

// The part of a page one GPU's line holds: a device view of those bytes of the line of kv_cache_shard cache_shard, and
// their offset in the page.
struct page_shard {
    at::Tensor bytes;
    std::size_t host_offset;
    std::size_t cache_shard;
};

// One page across the GPUs holding it; on one GPU holding whole lines, a single shard covering the line.
struct sharded_page {
    std::vector<page_shard> shards;
};

// A request whose step produced pages[page] of a put.
struct page_owner {
    std::uint64_t request;
    std::size_t page;
};

enum class page_error { miss, oom };

class page_store {
  public:
    virtual ~page_store() = default;

    // A shard's copies run on streams[shard.cache_shard].

    // Copies the page of hashes[i] into every shard of lines[i], and completes once the copies are done with true for
    // each page it had.
    virtual task<std::vector<bool>>
    fetch(std::span<const digest> hashes, std::vector<sharded_page>&& lines, std::span<const cudaStream_t> streams) = 0;
    // Takes the pages of hashes, matched by index, and copies the ones it keeps, completing once the copies are done.
    // The shards of a page cover it once. A page has an owner for every request that produced it. Fails with oom when
    // it has no host memory for a page it keeps.
    virtual task<std::expected<void, page_error>>
    put(std::vector<digest>&& hashes,
        std::vector<sharded_page>&& pages,
        std::vector<page_owner>&& owners,
        std::span<const cudaStream_t> streams) = 0;
};

}  // namespace pulsarlm
