#pragma once

#include <ATen/core/Tensor.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "pulsarlm/digest.hpp"
#include "pulsarlm/engine/event_loop.hpp"
#include "pulsarlm/engine/state.hpp"

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

enum class page_error { miss, oom };

// The pages of the KV layers and the states that reference them. A lease references the pages of the state it reached
// and the pages written for it, so the store keeps them until release.
class page_store {
  public:
    using page_index_type = std::size_t;

    virtual ~page_store() = default;

    // A shard's copies run on streams[shard.cache_shard].

    // Copies the page of hashes[i] into every shard of lines[i], and completes once the copies are done with true for
    // each page it had.
    virtual task<std::vector<bool>> read_pages(
        std::span<const digest> hashes,
        std::vector<sharded_page>&& lines,
        std::span<const cudaStream_t> streams
    ) = 0;
    // Takes the pages of hashes, matched by index, and copies the ones it keeps, completing once the copies are done.
    // The shards of a page cover it once. owners pairs the lease of every request whose step produced a page with the
    // page's index in hashes. Fails with oom when it has no host memory for a page it keeps.
    virtual task<std::expected<void, page_error>> write_pages(
        std::vector<digest>&& hashes,
        std::vector<sharded_page>&& pages,
        std::vector<std::pair<state::lease_id_type, page_index_type>>&& owners,
        std::span<const cudaStream_t> streams
    ) = 0;
    // The newest stored state reaching a proper prefix of tokens, whose full pages have the digests page_digests, or an
    // empty state when there is none, with a new lease on it. Every page of the state stays readable until release.
    // Fails with miss, holding no lease, when the lookup fails.
    virtual task<std::expected<std::pair<std::shared_ptr<const state>, state::lease_id_type>, page_error>>
    read_state(std::span<const std::int64_t> tokens, std::span<const digest> page_digests) = 0;
    // The lease's request reached s, whose pages were written: the lease references them in place of its previous
    // state's, releases the pages written for it that s covers, and s becomes findable.
    virtual task<void> write_state(state::lease_id_type lease_id, std::shared_ptr<const state>&& s) = 0;
    virtual void release(state::lease_id_type lease_id) noexcept = 0;
};

}  // namespace pulsarlm
