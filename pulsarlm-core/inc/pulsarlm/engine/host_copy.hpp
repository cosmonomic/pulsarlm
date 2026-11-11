#pragma once

#include <cuda_runtime_api.h>

#include <optional>
#include <span>
#include <vector>

#include "pulsarlm/engine/event_loop.hpp"
#include "pulsarlm/engine/host_page_pool.hpp"
#include "pulsarlm/engine/kv_cache.hpp"

namespace pulsarlm {

// A shard's copies run on streams[shard.cache_shard], and a call completes once every stream has finished them. The
// shards of one cache shard are views of one pool of lines; those whose host page is in pool copy in one launch per
// cache shard, the others with a cudaMemcpyAsync each.

// Host copies of sources, each shard into its bytes of the page, in blocks of pool or, past its room, pageable; nullopt
// when host memory runs out. A page takes the bytes its shards reach.
task<std::optional<std::vector<page>>>
copy_to_host(host_page_pool& pool, std::span<const sharded_page> sources, std::span<const cudaStream_t> streams);
// Copies each shard's bytes of pages[i] into the shards of targets[i]. A page outside pool is staged through one of its
// blocks when it has room.
task<void> copy_from_host(
    host_page_pool& pool,
    std::span<const sharded_page> targets,
    std::span<const page> pages,
    std::span<const cudaStream_t> streams
);

}  // namespace pulsarlm
