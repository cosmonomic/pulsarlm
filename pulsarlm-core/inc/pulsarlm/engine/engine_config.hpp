#pragma once

#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace pulsarlm {

// model_name names the model and model_hash its files' version; with the page shapes of the whole model's KV layers
// they make the root digest. Shard s of the model runs on devices[s], its part of the KV layers' lines sharing one pool of
// kv_cache_pool_bytes there, which must hold a line of every KV layer. A step forwards every decoding sequence and at
// most prefill_tokens_per_step tokens of the others. With decode_graphs and max_context_window, the decoding sequences'
// forward is replayed from CUDA graphs (decode_graphs.hpp) over page tables wide enough for a view of
// max_context_window tokens, the requests' eviction_policy::max_context_window; a step with a wider view runs without a
// graph. A step drafts up to draft_tokens tokens per sequence with the model's drafter (model::drafter), which it needs
// when draft_tokens is above 0.
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
    std::optional<std::size_t> max_context_window;
    std::size_t draft_tokens = 0;
};

}  // namespace pulsarlm
