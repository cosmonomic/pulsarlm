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

// The tokens of one group of a step, concatenated across the group's sequences, and where they sit. Every KV layer has
// its own view of each sequence: a new token is written into its view page's line, at its view index. A token's RoPE
// position is the same in every KV layer. A prefill group carries cu_seqlens_q; a decode group, whose sequences forward
// one token each, leaves it undefined.
struct GroupBatch {
    at::Tensor tokens;
    // int64 [ntok]: each token's RoPE position.
    at::Tensor pos;
    // int32 [n_kv_layers, num_seqs, max_pages]: the lines of sequence s's view pages in KV layer l. A recurrent layer's
    // row holds the line whose state the forward starts from and updates in place, then the line it writes the state
    // after page_end_tokens[s] of the sequence's tokens into.
    at::Tensor page_tables;
    // int32 [num_seqs]: the number of sequence s's tokens in the group whose last ends its newest full page, 0 when none
    // of them ends a page.
    at::Tensor page_end_tokens;
    // int32 [num_seqs + 1]; prefill groups only.
    at::Tensor cu_seqlens_q;
    // int32 [n_kv_layers, num_seqs]: the keys in sequence s's view in KV layer l, the group's tokens included.
    at::Tensor key_counts;
    // int32 [n_kv_layers, num_seqs, max_pages], laid out as page_tables: the RoPE position of the first key of sequence
    // s's view pages in KV layer l (see attn_decode_cuda).
    at::Tensor page_positions;
    // int32 [n_kv_layers, num_seqs + 1], defined when the forward measures attention mass: sequence s's view pages in
    // KV layer l are the rows [cu_view_pages[l, s], cu_view_pages[l, s + 1]) of page_mass[l], at least as many as its
    // keys fill.
    at::Tensor cu_view_pages;
    // Per KV layer, fp32 [total_view_pages of the layer, n_q_heads of the layer], zeroed by the caller, or undefined
    // for a layer that measures no mass. The forward sets each view page's mass in each query head, as
    // attn_prefill_cuda measures it with mass_weights, maxed over the attention layers reading the KV layer.
    std::vector<at::Tensor> page_mass;
    // fp32 [ntok], defined with cu_view_pages: each token's mass weight.
    at::Tensor mass_weights;
    // int64 [n]: the rows whose logits the forward returns, in order. Undefined returns every row's.
    at::Tensor logit_rows;

    bool is_prefill() const {
        return cu_seqlens_q.defined();
    }
};

// A step's two groups; either may be absent.
struct StepBatch {
    std::optional<GroupBatch> prefill;
    std::optional<GroupBatch> decode;
};

// Each present group's logits [rows, vocab] in the model's dtype.
struct StepLogits {
    std::optional<at::Tensor> prefill;
    std::optional<at::Tensor> decode;
};

class Model {
  public:
    virtual ~Model() = default;

    // The layers that keep KV, in the order forward takes their lines. A layer reading another layer's KV is not one.
    virtual std::vector<kv_layer> kv_layers() const = 0;
    // lines holds one kv per entry of kv_layers().
    virtual StepLogits forward(const StepBatch& batch, std::span<const kv> lines) = 0;
};

}  // namespace pulsarlm
