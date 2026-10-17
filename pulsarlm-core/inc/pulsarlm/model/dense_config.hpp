#pragma once

#include "pulsarlm/rope.hpp"

#include <cstdint>

namespace pulsarlm {

// The Hugging Face config fields of a dense decoder with one attention layer and one MLP per layer.
struct dense_config {
    std::int64_t vocab_size;
    std::int64_t n_layers;
    std::int64_t n_heads;
    std::int64_t n_kv_heads;
    std::int64_t head_dim;
    rope_parameters rope;
    double rms_norm_eps;
};

}  // namespace pulsarlm
