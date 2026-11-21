#pragma once

#include "pulsarlm/ops/rope_parameters.hpp"

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <optional>
#include <string>

namespace pulsarlm::ops {

// The paged attention ops attend a batch of sequences, each a view: an ordered list of lines (pages of page_size keys)
// of the pools, laid out as ops::cuda::write_kv's. Key j of sequence s lives at slot j % page_size of line
// page_tables[s][j / page_size], and a line may appear in several views.
//
// The pools hold unrotated keys. The ops rotate key j of sequence s to the RoPE position
// page_positions[s][j / page_size] + j % page_size, in the rotate_half convention with the frequencies that the
// rope_parameters of fields theta, partial_rotary_factor and llama3_scaling states. Queries arrive rotated to their own
// positions, which the caller gives under the same page positions.
//
// The ops take these arguments, in this order where they take them:
// - keys, [lines, page_size, n_kv_heads, head_dim], each line packed and 16-byte aligned, lines any number of elements
//   apart, and values, shaped as keys and of one stride with them: the key and value pools of one KV layer, which may
//   be the same tensor;
// - page_tables, page_positions and key_counts, a page_views' fields;
// - theta, partial_rotary_factor and llama3_scaling;
// - scale;
// - sliding_window, which when given limits the query at view index p to the keys [max(0, p - sliding_window + 1), p];
//   the ops read no page table entry whose page lies wholly before the window of the sequence's first query, so those
//   entries may hold any value;
// - k_norm_weight, which when given is fp32 [head_dim] and multiplies each stored key before rotation;
// - mass_sum, weight_sum, cu_view_pages and query_weights, a page_mass' fields, all given or none.

// One KV layer's rows of a batch's page tables.
struct page_views {
    // int32 [num_seqs, max_pages].
    at::Tensor page_tables;
    // int32 [num_seqs, max_pages], laid out as page_tables.
    at::Tensor page_positions;
    // int32 [num_seqs], the keys in each view.
    at::Tensor key_counts;
};

// The page mass a step measures; the ops add to mass_sum and weight_sum, which the caller zeroed. Without its fields
// the ops measure no mass. Query token i attends view page b of its sequence when it attends one of b's key tokens; its
// mass on b in query head h is the max over b's key tokens it attends of the scaled score. The query tokens of a
// sequence measure their mass into one mass view v of the sequence's view pages, v = s unless an op states otherwise.
// Views never share mass, whatever lines they share.
struct page_mass {
    // fp32 [total_view_pages, n_q_heads], zeroed by the caller. Over the query tokens attending b,
    // mass_sum[cu_view_pages[v] + b, h] is set to the sum of query_weights[i] times the mass; it stays 0 for a page no
    // query token attends.
    at::Tensor mass_sum;
    // fp32 [total_view_pages], zeroed by the caller. weight_sum[cu_view_pages[v] + b] is set to the sum of
    // query_weights[i] over the query tokens attending b; it stays 0 for a page no query token attends.
    at::Tensor weight_sum;
    // int32 [mass views + 1]; mass view v owns view pages [cu_view_pages[v], cu_view_pages[v + 1]), at least its
    // sequence's keys'.
    at::Tensor cu_view_pages;
    // fp32 [total_q], non-negative.
    at::Tensor query_weights;
};

}  // namespace pulsarlm::ops

namespace pulsarlm::ops::cuda {

// Why the attention ops cannot run layers of this shape in dtype, or nullopt when they can: float16 and bfloat16 run
// on the tensor-core kernels only, float32 on the scalar kernel.
std::optional<std::string> attention_unsupported_reason(
    at::ScalarType dtype,
    int64_t page_size,
    int64_t head_dim,
    int64_t n_q_heads,
    int64_t n_kv_heads
);

// Writes a batch's new keys and values into the lines of paged pools. k_new and v_new are [new tokens, n_kv_heads,
// head_dim]. The tokens sit at the view indices of ops::cuda::attn_prefill's queries (with cu_seqlens_q) or
// ops::cuda::attn_decode's (without): the token at view index j of sequence s goes to line
// page_tables[s][j / page_size], slot j % page_size. Keys are stored as given; the attention ops read them unrotated.
void write_kv(
    const at::Tensor& keys,
    const at::Tensor& values,
    const at::Tensor& k_new,
    const at::Tensor& v_new,
    const at::Tensor& page_tables,
    const at::Tensor& key_counts,
    const std::optional<at::Tensor>& cu_seqlens_q
);

// Decode: one query per sequence, q [num_seqs, n_q_heads, head_dim], at view index key_counts[s] - 1, attending the
// keys [0, key_counts[s]) or its window of them. num_splits above 0 forces the split (flash-decode) path with that
// many splits, 1 included; 0 picks the split count from occupancy. Returns o [num_seqs, n_q_heads, head_dim], written
// into out when given, which must then be contiguous, of that shape and of q's dtype and device; out may be a slice of
// a larger tensor.
at::Tensor attn_decode(
    const at::Tensor& q,
    const at::Tensor& keys,
    const at::Tensor& values,
    const at::Tensor& page_tables,
    const at::Tensor& page_positions,
    const at::Tensor& key_counts,
    double theta,
    double partial_rotary_factor,
    const std::optional<llama3_rope_scaling>& llama3_scaling,
    double scale,
    const std::optional<std::int64_t>& sliding_window,
    const std::optional<at::Tensor>& k_norm_weight,
    const std::optional<at::Tensor>& mass_sum = std::nullopt,
    const std::optional<at::Tensor>& weight_sum = std::nullopt,
    const std::optional<at::Tensor>& cu_view_pages = std::nullopt,
    const std::optional<at::Tensor>& query_weights = std::nullopt,
    int64_t num_splits = 0,
    const std::optional<at::Tensor>& out = std::nullopt
);

// Prefill: a ragged batch of query chunks, causal over each view. Sequence i's queries are
// q[cu_seqlens_q[i] : cu_seqlens_q[i + 1]] (q [total_q, n_q_heads, head_dim]) at view indices
// [key_counts[i] - seq_q_i, key_counts[i]), and the query at view index p attends the keys [0, p] or its window of
// them. Each query i of the first per_query_mass_request_count sequences measures its mass into mass view i of its own,
// and each later sequence s into mass view cu_seqlens_q[per_query_mass_request_count] + s -
// per_query_mass_request_count; a positive count needs a mass. Returns o [total_q, n_q_heads, head_dim], written into
// out as ops::cuda::attn_decode writes it.
at::Tensor attn_prefill(
    const at::Tensor& q,
    const at::Tensor& keys,
    const at::Tensor& values,
    const at::Tensor& page_tables,
    const at::Tensor& page_positions,
    const at::Tensor& key_counts,
    const at::Tensor& cu_seqlens_q,
    double theta,
    double partial_rotary_factor,
    const std::optional<llama3_rope_scaling>& llama3_scaling,
    double scale,
    const std::optional<std::int64_t>& sliding_window,
    const std::optional<at::Tensor>& k_norm_weight,
    const std::optional<at::Tensor>& mass_sum = std::nullopt,
    const std::optional<at::Tensor>& weight_sum = std::nullopt,
    const std::optional<at::Tensor>& cu_view_pages = std::nullopt,
    const std::optional<at::Tensor>& query_weights = std::nullopt,
    std::int64_t per_query_mass_request_count = 0,
    const std::optional<at::Tensor>& out = std::nullopt
);

}  // namespace pulsarlm::ops::cuda
