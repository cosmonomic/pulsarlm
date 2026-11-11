#pragma once

#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pulsarlm {

// model_name names the model and its weights version; with the page shapes of the whole model's KV layers it makes
// the root digest. Shard s of the model runs on devices[s], its part of the KV layers' lines sharing one pool of
// kv_cache_pool_bytes there, which must hold a line of every KV layer. A step forwards every decoding sequence and at
// most prefill_tokens_per_step tokens of the others.
struct engine_config {
    std::string model_name;
    std::size_t page_size;
    at::ScalarType kv_dtype;
    std::size_t kv_cache_pool_bytes;
    std::vector<at::Device> devices;
    std::vector<std::int64_t> stop_ids;
    std::size_t vocab_size;
    std::size_t prefill_tokens_per_step;
};

}  // namespace pulsarlm
