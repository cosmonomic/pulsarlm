#pragma once

#include "pulsarlm/engine/graph_segments.hpp"

#include <ATen/core/Tensor.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace pulsarlm {

// A recurrent layer's state: one tensor of sizes and dtype.
struct recurrent_state {
    std::vector<std::int64_t> sizes;
    at::ScalarType dtype;
    // Whether a line holds a window of the last sizes.back() inputs before a sequence's next token rather than a
    // state. The engine drafting k tokens a step widens a window by k inputs and keeps a state in k + 1 slots.
    bool input_window = false;

    bool operator==(const recurrent_state&) const = default;
};

// With values_only, one tensor serves as both K and V. A layer with a sliding window attends only its last
// sliding_window keys. An indexed attention layer (index_head_dim above 0) also keeps each token's index key, which its
// queries select the keys they attend by. A recurrent layer (state set) keeps no keys: its page is the state after the
// page's last token, and a state reaches only its newest full page; head counts and values_only do not apply to it.
struct kv_layer {
    std::size_t n_kv_heads;
    std::size_t head_dim;
    // The query heads of each attention layer reading this KV layer, and the number of those layers.
    std::size_t n_q_heads = 0;
    std::size_t n_readers = 1;
    bool values_only = false;
    std::optional<std::size_t> sliding_window;
    std::size_t index_head_dim = 0;
    std::optional<recurrent_state> state;
    // The elements of the row an attention line keeps besides its keys: the model's hidden state at the latest position
    // written into the line, which the first position of the next page reads.
    std::size_t last_hidden_dim = 0;
    // Whether the layer measures no mass and holds exactly the full pages the other attention layers hold, its pages
    // left out of the ones positions are ranked among.
    bool mirrors_held_pages = false;
};

// One KV layer's lines, [n_lines, page_size, n_kv_heads, head_dim] each; k and v are one tensor for a values_only
// layer. K is stored unrotated, as are an indexed attention layer's index keys [n_lines, page_size, index_head_dim],
// undefined for other layers. A recurrent layer's lines are one tensor [n_lines, state sizes...] as both k and v.
struct kv {
    at::Tensor k;
    at::Tensor v;
    at::Tensor index_keys;
    // [n_lines, last_hidden_dim], for a layer that keeps it.
    at::Tensor last_hidden;
};

// A KV layer of the whole model and the ranges [first, last) along its line's first dimension, KV heads or a state's
// first size, that one GPU's line holds, in order. An indexed attention layer's index keys are always whole.
struct kv_layer_shard {
    kv_layer full;
    std::vector<std::pair<std::size_t, std::size_t>> rows;
};

class draft_head;

class model {
  public:
    // The query tokens a model computes in one call, concatenated across the requests, and where they sit. The requests
    // come in three runs: requests with one query token, then requests that keep a recurrent state and a mass view per
    // query token, then prefill chunks. Every KV layer has its own view of each request: a new token is written into
    // its view page's line, at its view index. A token's RoPE position is the same in every KV layer.
    struct batch {
        // Requests [0, single_query_request_count) have one query token.
        std::int64_t single_query_request_count = 0;
        // The next per_query_state_request_count requests keep a recurrent state and a mass view per query token, and
        // each has state_slots query tokens. Only a model with a draft head receives such requests.
        std::int64_t per_query_state_request_count = 0;
        // int64 [ntok], each query token's token, then, in a batch padded to a step bucket, padding tokens of no
        // request, which the forward computes and nothing reads.
        at::Tensor tokens;
        // int64, laid out as tokens: each token's RoPE position.
        at::Tensor pos;
        // int32 [n_kv_layers, num_seqs, max_pages]: the lines of sequence s's view pages in KV layer l. A recurrent
        // layer's row holds the sequence's state lines, its slots then its boundary lines (state_rows).
        at::Tensor page_tables;
        // int64 [num_seqs]: the history position of each sequence's first token in the batch.
        at::Tensor history_positions;
        // int32 [num_seqs]: the tokens each sequence's previous step forwarded and did not accept.
        at::Tensor rejected_tokens;
        std::int64_t page_size = 0;
        // The slots of a recurrent layer keeping a state; a window layer keeps one.
        std::int64_t state_slots = 1;
        // int32 [num_seqs + 1]: sequence s's query tokens are [cu_seqlens_q[s], cu_seqlens_q[s + 1]).
        at::Tensor cu_seqlens_q;
        // int32 [num_seqs - single_query_request_count + 1]: cu_seqlens_q of the sequences from the second run on, less
        // the first run's query tokens.
        at::Tensor cu_seqlens_q_from_second_run;
        // int32 [num_seqs - single_query_request_count - per_query_state_request_count + 1]: cu_seqlens_q of the third
        // run's sequences, less the query tokens of the first two runs.
        at::Tensor cu_seqlens_q_from_third_run;
        // int32 [single_query_request_count + per_query_state_request_count]: the trailing query tokens whose recurrent
        // state each request of the first two runs keeps (ops::cuda::gated_delta_rule_step's kept_state_counts), 1 in
        // the first run and state_slots in the second. Undefined in a draft head's batch, which reaches no recurrent
        // layer.
        at::Tensor kept_state_counts;
        // int32 [n_kv_layers, num_seqs]: the keys in sequence s's view in KV layer l, the batch's tokens included.
        at::Tensor key_counts;
        // int32 [n_kv_layers, num_seqs, max_pages], laid out as page_tables: the RoPE position of the first key of
        // sequence s's view pages in KV layer l (see ops::cuda::attn_decode).
        at::Tensor page_positions;
        // int32 [n_kv_layers, mass views + 1], defined when the forward measures attention mass: mass view v's pages
        // in KV layer l are the rows [cu_view_pages[l, v], cu_view_pages[l, v + 1]) of page_mass_sum[l] and
        // page_weight_sum[l], at least as many as its sequence's keys fill. A sequence of the second run has a mass
        // view per query token, any other sequence one, in the order of the query tokens.
        at::Tensor cu_view_pages;
        // Per KV layer, fp32 [total_view_pages of the layer, n_readers * n_q_heads of the layer] and [total_view_pages
        // of the layer], zeroed by the caller, or undefined for a layer that measures no mass. The forward sets each
        // view page's mass_sum in each query head of each attention layer reading the KV layer, and its weight_sum, as
        // ops::cuda::attn_prefill measures them with query_weights: reader r's heads are the columns [r * n_q_heads,
        // (r + 1) * n_q_heads). Every reader attends the same pages, so weight_sum is the same for all of them.
        std::vector<at::Tensor> page_mass_sum;
        std::vector<at::Tensor> page_weight_sum;
        // fp32 [query tokens], defined with cu_view_pages: each query token's query weight.
        at::Tensor query_weights;
        // int64 [logit tokens]: the query tokens whose logits the forward returns, in order.
        at::Tensor logit_tokens;
        // int32 zeros, at least the model's arrival_counter_count for the batch: the arrival counters of its ops
        // (ops/cuda/arrival_counters.cuh). The batches of one stream's steps may share them.
        at::Tensor arrival_counters;
    };

    // The logits [logit tokens, vocab] in the model's dtype of the batch's logit tokens.
    struct output {
        at::Tensor logits;
        // The hidden rows of every entry of the batch's tokens; undefined for a model without a draft head.
        at::Tensor hidden;
    };

    virtual ~model() = default;

    // This GPU's layers that keep KV, in the order forward takes their lines. A layer reading another layer's KV is
    // not one.
    virtual std::vector<kv_layer> kv_layers() const = 0;
    // Per entry of kv_layers(), the layer of the whole model and the part of its line this GPU holds.
    virtual std::vector<kv_layer_shard> kv_layer_shards() const {
        std::vector<kv_layer_shard> shards;
        for (const kv_layer& layer : kv_layers()) {
            std::size_t rows = layer.state ? static_cast<std::size_t>(layer.state->sizes.front()) : layer.n_kv_heads;
            shards.push_back(kv_layer_shard{.full = layer, .rows = {{0, rows}}});
        }
        return shards;
    }
    // lines holds one kv per entry of kv_layers(). Every call that depends on the batch's runs or its requests goes
    // through segments.run_between_segments, which receives the query tokens' rows of its tensors. hidden holds a row
    // per entry of b.tokens, padding included.
    virtual output forward(const batch& b, std::span<const kv> lines, graph_segments& segments) = 0;
    // Whether the collectives of a forward of token_count entries of tokens can be captured in a CUDA graph outside
    // its calls between segments.
    virtual bool allows_segment_capture(std::int64_t token_count) const {
        return true;
    }
    // The model's draft head, null for a model without one.
    virtual draft_head* drafter() {
        return nullptr;
    }
    // The arrival counters a batch of sequences forwarding tokens tokens needs.
    virtual std::int64_t arrival_counter_count(std::int64_t sequences, std::int64_t tokens) const {
        return 0;
    }
};

// A model's multi-token prediction head, one decoder layer with its own KV layer (kv_layer::mirrors_held_pages and
// last_hidden_dim set) that drafts the token after the next. Its row at position t reads the hidden row of position
// t - 1 and the token at t, and predicts the token at t + 1.
class draft_head {
  public:
    virtual ~draft_head() = default;

    // The elements of a hidden row.
    virtual std::int64_t hidden_dim() const = 0;
    // Drafts for the rows of b: row i at position b.pos[i] with token b.tokens[i] and the hidden row hidden[i] of the
    // position before it, zeros before a sequence's first position. Writes the rows' KV into the head's KV layer of
    // lines. The model's forward keeps the hidden row at the latest position written into each of that layer's lines
    // in its last_hidden, which a row reads when the position before it was forwarded by an earlier step.
    // Its calls that depend on the batch go through segments, as model::forward's.
    virtual model::output
    draft(const model::batch& b, const at::Tensor& hidden, std::span<const kv> lines, graph_segments& segments) = 0;
};

// b's runs lie within its sequences, and its query starts are defined.
void check_batch(const model::batch& b);

}  // namespace pulsarlm
