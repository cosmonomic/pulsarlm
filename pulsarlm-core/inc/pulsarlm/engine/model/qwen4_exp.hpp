#pragma once

#include "pulsarlm/engine/communicator.hpp"
#include "pulsarlm/engine/model/interface.hpp"
#include "pulsarlm/engine/model/qwen4_exp_config.hpp"
#include "pulsarlm/kernels/ops.hpp"
#include "pulsarlm/kernels/rope.hpp"

#include <ATen/core/Dict.h>
#include <ATen/core/Tensor.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

// Qwen4-Exp text decoder (qwen4_exp), over the paged CUDA ops: Qwen3.5's Gated DeltaNet layers and indexed attention
// layers, each followed by qwen3_5_moe's sparse block, on hc_count residual streams mixed into each sublayer's input
// and injected back from its output, with hashed n-gram embeddings (PLE) added to the streams before the ple_layers.
// An indexed attention layer is Qwen3.5's gated attention over the keys QSA selects: each query's top
// indexer_budget / indexer_compress_ratio complete blocks of indexer_compress_ratio view keys, scored by its indexer
// heads against the RMSNorm of each block's mean raw index key roped at the block's first key, and the incomplete
// block holding it. Weights carry Hugging Face's names under model.language_model., the untied head lm_head.weight, and
// each PLE layer's n-gram table its shards ple.ple_embedding.ngram_embedding.shard_<i>.weight.
//
// Across the N GPUs of a communicator, GPU g holds the g-th of N equal parts, as vLLM v0.31.0 splits them: of the query
// heads; of the KV heads, or KV head g / (N / KV heads) when N exceeds them; of the Gated DeltaNet key heads and value
// heads, in_proj_qkv's and conv1d's q, k and v channels each split apart; of every expert's and the shared expert's
// intermediate size, the gate and up halves of gate_up_proj each split apart; of the embedding's and lm_head's rows,
// the vocabulary padded to a multiple of 64; and of each PLE layer's n-gram table rows, its shards concatenated. Every
// other weight is whole on every GPU. The outputs of o_proj, out_proj, the sparse block and the embedding lookups are
// summed across the GPUs, and the logits gathered.

namespace pulsarlm {

// An indexed attention layer's KV layer also keeps each token's raw index key. A Gated DeltaNet layer keeps Qwen3.5's two
// recurrent KV layers. A PLE layer then keeps two more: its convolution's
// last (kernel - 1) * ngram_size inputs, [hc_count * hidden, (kernel - 1) * ngram_size] in the weights' dtype, and the
// last ngram_size - 1 tokens, int64, each XOR eos_token_id so that a zeroed line holds EOS.
// With the checkpoint's mtp.* weights, the model drafts with its MTP layer, vLLM v0.31.0's Qwen4ExpMultiTokenPredictor
// with each MTP row a position later: the row at position t reads the streams after the last layer at t - 1, normed
// over all streams and projected by fc_hidden on each, and the token at t, normed and projected by fc_embedding and
// injected into every stream at weight 1; then one indexed attention layer with its MoE block and hyper connections,
// without PLE, and hyper_connection_mixer before lm_head.
class qwen4_exp_model : public model, public draft_head {
    // Query and key heads are stored with their dims permuted as qwen3_5_model stores them.
    struct attention_weights {
        // The rows of q_proj [n_heads * 2 * head_dim, hidden], each head's query then gate, of k_proj, of v_proj, then
        // of the indexer's index_qk_proj [(indexer_n_heads + 1) * indexer_head_dim, hidden], its query heads then its
        // key.
        at::Tensor projections;
        at::Tensor o_proj;
        at::Tensor q_norm;  // fp32 1 + weight
        at::Tensor k_norm;
        at::Tensor index_q_norm;  // fp32 1 + weight
        at::Tensor index_k_norm;
    };

    struct deltanet_weights {
        at::Tensor in_proj;  // the rows of in_proj_qkv, in_proj_a, in_proj_b, then in_proj_z
        at::Tensor conv;  // [conv channels, 1, linear_conv_kernel]
        at::Tensor dt_bias;  // fp32
        at::Tensor neg_a;  // fp32 -exp(A_log)
        at::Tensor norm;
        at::Tensor out_proj;
    };

    struct moe_weights {
        at::Tensor router_and_gate;  // [num_experts + 1, hidden]: the router's rows, then shared_expert_gate's
        // The held experts [first_expert, first_expert + held): every expert's part of the intermediate rows, or
        // whole experts, num_experts / GPUs of them.
        at::Tensor experts_gate_up_proj;  // [held, 2 * intermediate rows, hidden]
        at::Tensor experts_down_proj;  // [held, hidden, intermediate rows]
        std::int64_t first_expert;
        at::Tensor shared_gate_up_proj;  // the rows of the shared expert's gate_proj, then its up_proj
        at::Tensor shared_down_proj;
    };

    struct mixer_weights {
        at::Tensor hc_norm;  // fp32 1 + weight, [hc_count * hidden], one RMSNorm per stream
        at::Tensor input_mix_weight_down;
        at::Tensor input_mix_weight_up;
    };

    struct hyper_connection_weights {
        at::Tensor hc_norm;  // fp32 1 + weight, [hc_count * hidden], one RMSNorm per stream
        at::Tensor down_and_inject;  // the rows of input_mix_weight_down, then block_inject_weight
        at::Tensor input_mix_weight_up;
    };

    struct ple_weights {
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
        std::int64_t first_row;  // of the whole table, the row ngram_embedding starts at
    };

    struct mtp_weights {
        at::Tensor pre_fc_norm_embedding;  // fp32 1 + weight
        at::Tensor pre_fc_norm_hidden;  // fp32 1 + weight, [hc_count * hidden], one RMSNorm over every stream
        at::Tensor fc_embedding;
        at::Tensor fc_hidden;  // [hidden, hidden], each stream's
        mixer_weights mixer;
    };

    struct layer_weights {
        hyper_connection_weights attn_hyper_connection;
        hyper_connection_weights mlp_hyper_connection;
        moe_weights moe;
        std::optional<attention_weights> attention;
        std::optional<deltanet_weights> deltanet;
        std::optional<ple_weights> ple;
    };

    // This GPU's head counts.
    struct shard_heads {
        std::int64_t n_heads;
        std::int64_t n_kv_heads;
        std::int64_t linear_n_key_heads;
        std::int64_t linear_n_value_heads;
    };

    struct mixed_input {
        at::Tensor input;  // [ntok, hidden]
        at::Tensor injection_weights;  // [ntok, hc_count]
    };

    qwen4_exp_config config;
    communicator comm;
    shard_heads heads;
    std::int64_t n_layers;
    rope_parameters rope;
    // fp32 [rotary_dim / 2]: Hugging Face's RoPE frequencies, which the indexer rotates its queries and blocks by.
    at::Tensor index_inv_freq;
    // The first KV layer of each layer.
    std::vector<std::size_t> kv_layer_of;
    std::vector<kv_layer> kv_specs;
    std::vector<kv_layer_shard> kv_shards;
    at::Tensor embed_tokens;
    // Of the padded vocabulary, the row embed_tokens and lm_head start at.
    std::int64_t first_vocab_row;
    at::Tensor lm_head;
    mixer_weights hyper_connection_mixer;
    // The decoder layers, then the MTP layer when the model drafts.
    std::vector<layer_weights> layers;
    std::optional<mtp_weights> mtp;

    bool indexed_attention(std::int64_t layer) const;
    // The rows of a table split across the GPUs at ids, [ids..., table columns], summed across the GPUs; this GPU's
    // part of the table starts at first_row.
    at::Tensor lookup(const at::Tensor& table, std::int64_t first_row, const at::Tensor& ids) const;
    // The logits [rows, vocab_size] of the final mix x [rows, hidden], from every GPU's part of lm_head.
    at::Tensor logits(const at::Tensor& x) const;
    // int32 [ntok, words]: each query's selected blocks, as attn_indexed_cuda reads them.
    at::Tensor selected_blocks(
        const attention_weights& w,
        const at::Tensor& queries,
        const kv& lines,
        const at::Tensor& page_tables,
        const std::optional<at::Tensor>& cu_seqlens_q,
        const at::Tensor& key_counts,
        const at::Tensor& page_positions
    ) const;
    at::Tensor stream_norm(const at::Tensor& x, const at::Tensor& weight) const;
    // The mix of the normed streams [ntok, hc_count * hidden] the head reads.
    at::Tensor mixed_streams(const at::Tensor& normed, const mixer_weights& w) const;
    mixed_input mix(const at::Tensor& x, const hyper_connection_weights& w) const;
    // The streams x plus the block output out injected by injection_weights, and their mix by w.
    std::pair<at::Tensor, mixed_input> injected_mix(
        const at::Tensor& x,
        const at::Tensor& out,
        const at::Tensor& injection_weights,
        const hyper_connection_weights& w
    ) const;
    mixed_input
    mix_normed(const at::Tensor& normed, const at::Tensor& projected, const hyper_connection_weights& w) const;
    at::Tensor final_mix(const at::Tensor& x) const;
    at::Tensor attention(const at::Tensor& h, std::int64_t layer, const batch& g, std::span<const kv> lines) const;
    at::Tensor deltanet(const at::Tensor& h, std::int64_t layer, const batch& g, std::span<const kv> lines) const;
    at::Tensor moe(const at::Tensor& h, std::int64_t layer, const at::Tensor& counters) const;
    // Each token's n-gram ids into the layer's table, [ntok, n-gram heads], updating its token lines.
    at::Tensor ngram_ids(const ple_weights& w, std::size_t token_layer, const batch& g, const kv& lines) const;
    at::Tensor ple(const at::Tensor& x, std::int64_t layer, const batch& g, std::span<const kv> lines) const;

  public:
    // weights are this GPU's parts, on the lines' device, in the attention lines' dtype; the PLE buffers are int64.
    // Throws std::invalid_argument when a split part is not a whole number of heads.
    qwen4_exp_model(qwen4_exp_config config, const c10::Dict<std::string, at::Tensor>& weights, communicator comm = {});

    std::vector<kv_layer> kv_layers() const override;
    std::vector<kv_layer_shard> kv_layer_shards() const override;
    output forward(const batch& b, std::span<const kv> lines) override;
    std::int64_t arrival_counter_count(std::int64_t sequences, std::int64_t tokens) const override;
    draft_head* drafter() override;
    std::int64_t hidden_dim() const override;
    output draft(const batch& b, const at::Tensor& hidden, std::span<const kv> lines) override;
};

}  // namespace pulsarlm
