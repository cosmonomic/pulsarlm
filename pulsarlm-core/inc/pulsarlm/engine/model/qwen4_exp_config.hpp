#pragma once

#include <cstdint>
#include <vector>

#include "pulsarlm/kernels/gate_activation.hpp"

namespace pulsarlm {

enum class qwen4_exp_layer_type { linear_attention, indexed_attention };

// Hugging Face's text config. ple_layers are zero-indexed.
struct qwen4_exp_config {
    std::int64_t vocab_size;
    std::int64_t hidden_size;
    std::vector<qwen4_exp_layer_type> layer_types;
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
    gate_activation linear_output_gate;
    double rms_norm_eps;
    std::int64_t num_experts_per_tok;
    std::int64_t hc_count;
    std::vector<std::int64_t> ple_layers;
    std::int64_t ngram_size;
    std::int64_t eos_token_id;
    std::int64_t indexer_n_heads;
    std::int64_t indexer_head_dim;
    std::int64_t indexer_budget;
    std::int64_t indexer_compress_ratio;
};

}  // namespace pulsarlm
