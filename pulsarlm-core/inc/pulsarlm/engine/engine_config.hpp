#pragma once

#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pulsarlm {

// model_name names the model and model_hash its files' version; with the page shapes of the whole model's KV layers
// they make the root digest. Shard s of the model runs on devices[s], its part of the KV layers' lines sharing one pool of
// kv_cache_pool_bytes there, which must hold a line of every KV layer. A step forwards every decoding request and at
// most prefill_tokens_per_step tokens of the others. A request's view holds at most max_context_window tokens, its last
// page partly filled, and its eviction_policy::max_context_capacity is at most max_context_window. With decode_graphs,
// the decoding requests' forward is replayed from CUDA graphs (step_graphs.hpp) over page tables as wide as such a view.
// A step drafts up to draft_tokens tokens per request with the model's drafter (model::drafter), which it needs when
// draft_tokens is above 0. At most max_running_requests requests run at once; a request submitted beyond them waits
// until one ends.
struct engine_config {
    std::string model_name;
    std::string model_hash;
    std::size_t page_size;
    at::ScalarType kv_dtype;
    std::size_t kv_cache_pool_bytes;
    std::vector<at::Device> devices;
    std::vector<std::int64_t> stop_ids;
    std::size_t prefill_tokens_per_step;
    bool decode_graphs = false;
    std::size_t max_context_window;
    std::size_t draft_tokens = 0;
    std::size_t max_running_requests = 64;
};

}  // namespace pulsarlm
