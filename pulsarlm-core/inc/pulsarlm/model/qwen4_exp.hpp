#pragma once

#include "pulsarlm/model/interface.hpp"
#include "pulsarlm/model/qwen3_5.hpp"
#include "pulsarlm/ops.hpp"
#include "pulsarlm/rope.hpp"

#include <ATen/core/Dict.h>
#include <ATen/core/Tensor.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

// Qwen4-Exp text decoder (qwen4_exp), over the paged CUDA ops: Qwen3.5's Gated DeltaNet and gated full-attention
// layers, each followed by qwen3_5_moe's sparse block, on hc_count residual streams mixed into each sublayer's input
// and injected back from its output, with hashed n-gram embeddings (PLE) added to the streams before the ple_layers.
// The full-attention layers attend every key, which is QSA's selection while a query sees at most its budget of
// complete blocks. Weights carry Hugging Face's names under model.language_model., the untied head lm_head.weight, and
// each PLE layer's n-gram table its shards ple.ple_embedding.ngram_embedding.shard_<i>.weight.

namespace pulsarlm {

// Hugging Face's text config, its indexed_attention layers full_attention. ple_layers are zero-indexed.
struct qwen4_exp_config {
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
    gate_activation linear_output_gate;
    double rms_norm_eps;
    std::int64_t num_experts_per_tok;
    std::int64_t hc_count;
    std::vector<std::int64_t> ple_layers;
    std::int64_t ngram_size;
    std::int64_t eos_token_id;
};

// A Gated DeltaNet layer keeps Qwen3.5's two recurrent KV layers. A PLE layer then keeps two more: its convolution's
// last (kernel - 1) * ngram_size inputs, [hc_count * hidden, (kernel - 1) * ngram_size] in the weights' dtype, and the
// last ngram_size - 1 tokens, int64, each XOR eos_token_id so that a zeroed line holds EOS.
class Qwen4ExpModel : public Model {
    // Query and key heads are stored with their dims permuted as Qwen3_5Model stores them.
    struct AttentionWeights {
        at::Tensor q_proj;  // [n_heads * 2 * head_dim, hidden], each head's query then gate
        at::Tensor k_proj;
        at::Tensor v_proj;
        at::Tensor o_proj;
        at::Tensor q_norm;  // fp32 1 + weight
        at::Tensor k_norm;
    };

    struct DeltaNetWeights {
        at::Tensor in_proj_qkv;
        at::Tensor in_proj_z;
        at::Tensor in_proj_b;
        at::Tensor in_proj_a;
        at::Tensor conv;  // [conv channels, 1, linear_conv_kernel]
        at::Tensor dt_bias;  // fp32
        at::Tensor neg_a;  // fp32 -exp(A_log)
        at::Tensor norm;
        at::Tensor out_proj;
    };

    struct MoeWeights {
        at::Tensor router;  // [num_experts, hidden]
        at::Tensor experts_gate_up_proj;  // [num_experts, 2 * moe_intermediate_size, hidden]
        at::Tensor experts_down_proj;  // [num_experts, hidden, moe_intermediate_size]
        at::Tensor shared_gate_proj;
        at::Tensor shared_up_proj;
        at::Tensor shared_down_proj;
        at::Tensor shared_expert_gate;  // [1, hidden]
    };

    struct MixerWeights {
        at::Tensor hc_norm;  // fp32 1 + weight, [hc_count * hidden], one RMSNorm per stream
        at::Tensor input_mix_weight_down;
        at::Tensor input_mix_weight_up;
    };

    struct HyperConnectionWeights {
        MixerWeights mixer;
        at::Tensor block_inject_weight;
    };

    struct PleWeights {
        at::Tensor key_proj;
        at::Tensor value_proj;
        at::Tensor norm_key;  // fp32 1 + weight, one RMSNorm per stream
        at::Tensor norm_query;
        at::Tensor norm_conv;
        at::Tensor conv;  // [hc_count * hidden, 1, kernel]
        at::Tensor layer_multipliers;  // int64 [ngram_size]
        at::Tensor head_vocab_sizes;  // int64 [n-gram heads]
        at::Tensor head_offsets;
        at::Tensor ngram_embedding;  // [rows, embedding dim / n-gram heads]
    };

    struct LayerWeights {
        HyperConnectionWeights attn_hyper_connection;
        HyperConnectionWeights mlp_hyper_connection;
        MoeWeights moe;
        std::optional<AttentionWeights> attention;
        std::optional<DeltaNetWeights> deltanet;
        std::optional<PleWeights> ple;
    };

    struct MixedInput {
        at::Tensor input;  // [ntok, hidden]
        at::Tensor injection_weights;  // [ntok, hc_count]
    };

    qwen4_exp_config config;
    std::int64_t n_layers;
    rope_parameters rope;
    // The first KV layer of each layer.
    std::vector<std::size_t> kv_layer_of;
    std::vector<kv_layer> kv_specs;
    at::Tensor embed_tokens;
    at::Tensor lm_head;
    MixerWeights hyper_connection_mixer;
    std::vector<LayerWeights> layers;

    bool full_attention(std::int64_t layer) const;
    at::Tensor stream_norm(const at::Tensor& x, const at::Tensor& weight) const;
    // The mix of the normed streams [ntok, hc_count * hidden] a sublayer or the head reads.
    at::Tensor mixed_streams(const at::Tensor& normed, const MixerWeights& w) const;
    MixedInput mix(const at::Tensor& x, const HyperConnectionWeights& w) const;
    at::Tensor final_mix(const at::Tensor& x) const;
    at::Tensor attention(const at::Tensor& h, std::int64_t layer, const GroupBatch& g, std::span<const kv> lines) const;
    at::Tensor deltanet(const at::Tensor& h, std::int64_t layer, const GroupBatch& g, std::span<const kv> lines) const;
    at::Tensor moe(const at::Tensor& h, std::int64_t layer) const;
    // Each token's n-gram ids into the layer's table, [ntok, n-gram heads], updating its token lines.
    at::Tensor ngram_ids(const PleWeights& w, std::size_t token_layer, const GroupBatch& g, const kv& lines) const;
    at::Tensor ple(const at::Tensor& x, std::int64_t layer, const GroupBatch& g, std::span<const kv> lines) const;
    at::Tensor group_forward(const GroupBatch& g, std::span<const kv> lines) const;

  public:
    // weights are on the lines' device, in the attention lines' dtype; the PLE buffers are int64.
    Qwen4ExpModel(qwen4_exp_config config, const c10::Dict<std::string, at::Tensor>& weights);

    std::vector<kv_layer> kv_layers() const override;
    StepLogits forward(const StepBatch& batch, std::span<const kv> lines) override;
};

}  // namespace pulsarlm
