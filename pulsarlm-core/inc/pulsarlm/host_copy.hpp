#pragma once

#include <cuda_runtime_api.h>

#include <optional>
#include <span>
#include <vector>

#include "pulsarlm/event_loop.hpp"
#include "pulsarlm/host_page_pool.hpp"
#include "pulsarlm/kv_cache.hpp"

namespace pulsarlm {

// The copies run on stream and complete once the stream has finished them. Device pages are views of one pool of lines;
// the host pages in pool copy in one launch, the others with a cudaMemcpyAsync each.

// Host copies of sources, in blocks of pool or, past its room, pageable; nullopt when host memory runs out.
task<std::optional<std::vector<page>>>
copy_to_host(host_page_pool& pool, std::span<const page> sources, cudaStream_t stream);
// Copies pages[i] into targets[i]. A page outside pool is staged through one of its blocks when it has room.
task<void>
copy_from_host(host_page_pool& pool, std::span<const page> targets, std::span<const page> pages, cudaStream_t stream);

}  // namespace pulsarlm
