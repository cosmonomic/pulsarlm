#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "pulsarlm/kernels/rope_parameters.hpp"

namespace pulsarlm {

enum class gemma4_layer_type { sliding_attention, full_attention };

// Hugging Face's text config. global_n_kv_heads is the KV heads of a full-attention layer. The last
// num_kv_shared_layers layers keep no KV: each reads the KV of the last earlier layer of its type. A
// hidden_size_per_layer_input of 0 means no per-layer inputs. With enable_moe_block, every layer adds the output of
// its top_k_experts experts to its MLP's.
struct gemma4_config {
    std::int64_t vocab_size;
    std::int64_t hidden_size;
    std::int64_t n_heads;
    std::int64_t n_kv_heads;
    std::int64_t head_dim;
    std::int64_t global_n_kv_heads;
    std::int64_t global_head_dim;
    bool attention_k_eq_v;
    std::vector<gemma4_layer_type> layer_types;
    std::int64_t sliding_window;
    rope_parameters sliding_rope;
    rope_parameters full_rope;
    double rms_norm_eps;
    std::optional<double> final_logit_softcapping;
    std::int64_t num_kv_shared_layers;
    std::int64_t hidden_size_per_layer_input;
    bool enable_moe_block = false;
    std::int64_t top_k_experts = 0;
};

}  // namespace pulsarlm
