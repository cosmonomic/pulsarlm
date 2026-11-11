#pragma once

#include <cstdint>
#include <vector>

#include "pulsarlm/engine/model/qwen3_5_config.hpp"

namespace pulsarlm {

// Hugging Face's text config. The first partial_rotary_factor * head_dim dims of a query or key head rotate, in
// rotate_half pairs among themselves at the frequencies of a head of that many dims; the rest pass through.
struct qwen3_5_moe_config {
    std::int64_t vocab_size;
    std::int64_t hidden_size;
    std::vector<qwen3_5_layer_type> layer_types;
    std::int64_t n_heads;
    std::int64_t n_kv_heads;
    std::int64_t head_dim;
    double rope_theta;
    double partial_rotary_factor;
    std::int64_t linear_n_key_heads;
    std::int64_t linear_n_value_heads;
    std::int64_t linear_key_head_dim;
    std::int64_t linear_value_head_dim;
    std::int64_t linear_conv_kernel;
    double rms_norm_eps;
    std::int64_t num_experts_per_tok;
};

}  // namespace pulsarlm
