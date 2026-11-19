#pragma once

#include "pulsarlm/engine/communicator.hpp"
#include "pulsarlm/engine/model/interface.hpp"
#include "pulsarlm/engine/model/qwen3_5_config.hpp"
#include "pulsarlm/engine/model/linear.hpp"
#include "pulsarlm/kernels/rope.hpp"

#include <ATen/core/Dict.h>
#include <ATen/core/Tensor.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

// Qwen3.5 text decoder (qwen3_5), over the paged CUDA ops: Gated DeltaNet layers and gated full-attention layers.
// Weights carry Hugging Face's names under model.language_model., the untied head lm_head.weight.

namespace pulsarlm {

// A Gated DeltaNet layer keeps two recurrent KV layers: its delta rule state S, fp32 [value heads, key head dim, value
// head dim], then its convolution's last linear_conv_kernel - 1 inputs, [conv channels, linear_conv_kernel - 1] in the
// weights' dtype. With the weights of its multi-token prediction layer (mtp.*, as vLLM v0.31.0 runs Qwen3.5's MTP), the
// model drafts: its hidden rows are the normed final hidden states, and the MTP layer, a full-attention layer with a
// dense MLP after fc over the normed token embedding and hidden row, keeps its own KV layer, the last.
class qwen3_5_model : public model, public draft_head {
    // Query and key heads are stored with their dims permuted so that the paged attention ops' partial RoPE, which
    // rotates pairs (j, j + head_dim / 2), rotates Hugging Face's pairs (j, j + rotary_dim / 2).
    struct attention_weights {
        linear q_proj;  // [n_heads * 2 * head_dim, hidden], each head's query then gate
        linear k_proj;
        linear v_proj;
        linear o_proj;
        at::Tensor q_norm;  // fp32 1 + weight
        at::Tensor k_norm;
    };

    struct deltanet_weights {
        linear in_proj_qkv;
        linear in_proj_z;
        linear in_proj_b;
        linear in_proj_a;
        at::Tensor conv;  // [conv channels, 1, linear_conv_kernel]
        at::Tensor dt_bias;  // fp32
        at::Tensor neg_a;  // fp32 -exp(A_log)
        at::Tensor norm;
        linear out_proj;
    };

    struct layer_weights {
        at::Tensor input_layernorm;  // fp32 1 + weight
        at::Tensor post_attention_layernorm;
        linear gate_proj;
        linear up_proj;
        linear down_proj;
        std::optional<attention_weights> attention;
        std::optional<deltanet_weights> deltanet;
    };

    // The MTP layer's own weights; its decoder layer is layers[n_layers].
    struct mtp_weights {
        at::Tensor pre_fc_norm_embedding;  // fp32 1 + weight
        at::Tensor pre_fc_norm_hidden;
        at::Tensor fc;  // [hidden, 2 * hidden], the embedding's columns first
        at::Tensor norm;
    };

    qwen3_5_config config;
    communicator comm;
    // This GPU's heads.
    std::int64_t n_heads;
    std::int64_t n_kv_heads;
    std::int64_t linear_n_key_heads;
    std::int64_t linear_n_value_heads;
    std::int64_t n_layers;
    rope_parameters rope;
    // The first KV layer of each layer.
    std::vector<std::size_t> kv_layer_of;
    std::vector<kv_layer_shard> kv_shards;
    at::Tensor embed_tokens;
    at::Tensor lm_head;
    at::Tensor norm;  // fp32 1 + weight
    // The decoder layers, then the MTP layer when the model drafts.
    std::vector<layer_weights> layers;
    std::optional<mtp_weights> mtp;

    bool full_attention(std::int64_t layer) const;
    at::Tensor attention(const at::Tensor& h, std::int64_t layer, const batch& g, std::span<const kv> lines) const;
    at::Tensor deltanet(const at::Tensor& h, std::int64_t layer, const batch& g, std::span<const kv> lines) const;
    at::Tensor embed(const at::Tensor& tokens) const;
    // The decoder layer layer over x, its attention or delta rule then its MLP, each with its residual.
    at::Tensor decoder_layer(const at::Tensor& x, std::int64_t layer, const batch& g, std::span<const kv> lines) const;
    // The logits [rows, vocab_size] of the normed hidden rows, from every rank's part of lm_head.
    at::Tensor logits(const at::Tensor& rows) const;

  public:
    // weights are on the lines' device, in the attention lines' dtype, except that fp8 linears and their scales may
    // be on the host. With comm of more than one rank, weights hold this rank's slices, cut as vLLM v0.31.0 cuts
    // Qwen3.5: embed_tokens and lm_head hold rows [rank * rows, (rank + 1) * rows) of the vocabulary padded to a
    // multiple of 64, rows its padded size / comm.size(), the padding rows zero.
    qwen3_5_model(qwen3_5_config config, const c10::Dict<std::string, at::Tensor>& weights, communicator comm = {});

    std::vector<kv_layer> kv_layers() const override;
    std::vector<kv_layer_shard> kv_layer_shards() const override;
    output forward(const batch& b, std::span<const kv> lines) override;
    std::int64_t arrival_counter_count(std::int64_t sequences, std::int64_t tokens) const override;
    draft_head* drafter() override;
    std::int64_t hidden_dim() const override;
    output draft(const batch& b, const at::Tensor& hidden, std::span<const kv> lines) override;
};

}  // namespace pulsarlm
