#pragma once

#include <ATen/core/Tensor.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace pulsarlm {

// A recurrent layer's state: one tensor of sizes and dtype.
struct recurrent_state {
    std::vector<std::int64_t> sizes;
    at::ScalarType dtype;

    bool operator==(const recurrent_state&) const = default;
};

// With values_only, one tensor serves as both K and V. A layer with a sliding window attends only its last
// sliding_window keys. An indexed attention layer (index_head_dim above 0) also keeps each token's index key, which its
// queries select the keys they attend by. A recurrent layer (state set) keeps no keys: its page is the state after the
// page's last token, and a state reaches only its newest full page; head counts and values_only do not apply to it.
struct kv_layer {
    std::size_t n_kv_heads;
    std::size_t head_dim;
    // The query heads of each attention layer reading this KV layer.
    std::size_t n_q_heads = 0;
    bool values_only = false;
    std::optional<std::size_t> sliding_window;
    std::size_t index_head_dim = 0;
    std::optional<recurrent_state> state;
};

// One KV layer's lines, [n_lines, page_size, n_kv_heads, head_dim] each; k and v are one tensor for a values_only
// layer. K is stored unrotated, as are an indexed attention layer's index keys [n_lines, page_size, index_head_dim],
// undefined for other layers. A recurrent layer's lines are one tensor [n_lines, state sizes...] as both k and v.
struct kv {
    at::Tensor k;
    at::Tensor v;
    at::Tensor index_keys;
};

// One scheduler group's query tokens + KV placement + attention descriptor.
// tokens are the group's tokens, concatenated across the group's sequences. Every KV
// layer has its own view of each sequence: row l of pos holds each token's view index
// in KV layer l's view, and each KV layer's page_tables row s lists the lines of
// sequence s's view pages in that layer; a new token is written into its view page's
// line, at its view index. A prefill group carries cu_seqlens_q; a decode group, whose
// sequences forward one token each, leaves it undefined.
struct GroupBatch {
    at::Tensor tokens;
    at::Tensor pos;  // int64 [n_kv_layers, ntok]
    // int32 [n_kv_layers, num_seqs, max_pages]. A recurrent layer's row holds the line whose state the forward starts
    // from and updates in place, then the line it writes the state after page_end_tokens[s] of the sequence's tokens
    // into.
    at::Tensor page_tables;
    // int32 [num_seqs], defined when a KV layer is recurrent: the number of sequence s's tokens in the group whose last
    // ends its newest full page, 0 when none of them ends a page.
    at::Tensor page_end_tokens;
    at::Tensor cu_seqlens_q;  // int32 [num_seqs + 1], defined iff prefill
    // int32 [n_kv_layers, num_seqs]: the keys in sequence s's view in KV layer l, the group's tokens included.
    at::Tensor key_counts;
    // int32 [num_seqs, 3], the RoPE position of each key by its view index (see attn_decode_cuda). pos holds each
    // query's position under the same layout.
    at::Tensor rope_layout;
    // int32 [n_kv_layers, num_seqs + 1], defined when the forward measures attention mass: sequence s's view pages in
    // KV layer l are the rows [cu_view_pages[l, s], cu_view_pages[l, s + 1]) of page_mass[l], at least as many as its
    // keys fill.
    at::Tensor cu_view_pages;
    // Per KV layer, fp32 [total_view_pages of the layer], zeroed by the caller, or undefined for a layer that measures
    // no mass. The forward sets each view page's mass: the weight the group's queries gave its keys, as the attention
    // ops scale it, averaged over query heads, then maxed over the page's slots and over the attention layers reading
    // the KV layer.
    std::vector<at::Tensor> page_mass;
    // fp32 [num_seqs] in the group's row order: each sequence's attention-mass EMA
    // gain alpha, which the attention kernels index by sequence. Per request, so one
    // batched forward may carry several rates.
    at::Tensor attention_mass_decay;
    // TOKENS: the length the accumulated mass is stated against. <= 0 states it
    // against each query's own attended key count, which only the attention op can
    // read. Engine-level, so one value serves every sequence in the group.
    double mass_length_gain = 0.0;
    // int64 [n]: the rows whose logits the forward returns, in order. Undefined returns every row's.
    at::Tensor logit_rows;

    bool is_prefill() const {
        return this->cu_seqlens_q.defined();
    }
};

// A scheduler step's two groups; either may be absent.
struct StepBatch {
    std::optional<GroupBatch> prefill;
    std::optional<GroupBatch> decode;
};

// Per-group logits [ntok, vocab] (in the model dtype) for the present groups.
struct StepLogits {
    std::optional<at::Tensor> prefill;
    std::optional<at::Tensor> decode;
};

// tokens + KV + batch -> logits.
struct Model {
    // The layers that keep KV, in the order forward takes their lines. A layer reading another layer's KV is not one.
    virtual std::vector<kv_layer> kv_layers() const = 0;

    // lines holds one kv per entry of kv_layers().
    virtual StepLogits forward(const StepBatch& batch, std::span<const kv> lines) = 0;

    // Pre-RoPE query capture for QK-attention recall. set_capture_layers names the
    // attention layers whose pre-RoPE Q the next forward should stash; after a
    // forward, captured_q(is_prefill, layer) returns that group's pre-RoPE Q
    // [ntok, n_heads, head_dim] in the group's token order (an undefined tensor
    // when the layer was not captured or the group was absent this step). Default:
    // no capture (Q discarded).
    virtual bool supports_capture() const {
        return false;
    }
    virtual void set_capture_layers(const std::vector<int64_t>& layers) {}
    virtual at::Tensor captured_q(bool is_prefill, int64_t layer) const {
        return {};
    }
    // That group's per-(query row, query head) softmax denominator for the same
    // layer, fp32 [ntok, n_heads]; undefined when the layer was not captured.
    virtual at::Tensor captured_lse(bool is_prefill, int64_t layer) const {
        return {};
    }

    virtual ~Model() = default;
};

}  // namespace pulsarlm
