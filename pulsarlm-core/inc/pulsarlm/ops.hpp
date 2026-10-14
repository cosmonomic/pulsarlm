#pragma once

#include <ATen/core/Tensor.h>

#include "pulsarlm/rope.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <tuple>

namespace pulsarlm {

// RMSNorm. Stateless.
at::Tensor rmsnorm_cuda(const at::Tensor& x, const at::Tensor& weight, double eps);

// Scatter a batch's new K/V rows into paged pools. k_pool/v_pool are [num_pages,
// page_size, n_kv_heads, head_dim] of one stride, each page packed and 16-byte aligned
// with pages any number of elements apart, and may be one tensor; k_new/v_new are
// [num_new_tokens, n_kv_heads, head_dim]. The tokens sit at view indices as the queries
// of attn_prefill (with cu_seqlens_q) or attn_decode (without), and the token at view
// index j of sequence s goes to line page_tables[s][j / page_size], offset
// j % page_size. Keys are stored as given; the attention ops take them UNROTATED.
void write_kv_cuda(
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& k_new,
    const at::Tensor& v_new,
    const at::Tensor& page_tables,
    const std::optional<at::Tensor>& cu_seqlens_q,
    const at::Tensor& seqlens_k
);

// int32 [n_layers, tables.size(), width]; row s of each layer holds the first width
// entries of that layer's row of tables[s], a contiguous int32 [n_layers, capacity]
// device tensor, and zeros past capacity. The tables must stay alive until the stream
// has run the gather.
at::Tensor gather_page_tables_cuda(std::span<const at::Tensor> tables, int64_t n_layers, int64_t width);

// Copies segment i, int64 {device byte offset, host byte offset, bytes}, between the
// device byte tensor and the pinned host byte tensor, toward the host when to_host. The
// kernel reads the pinned segments, so they stay alive until the stream has run it.
void copy_segments_cuda(const at::Tensor& device, const at::Tensor& host, const at::Tensor& segments, bool to_host);

// int64 [rows]: per row of scores [rows, vocab], argmax(x / T + g) with each g drawn from Gumbel(0, 1), which is
// index i with probability softmax(x / T)_i; a row with T <= 0 takes argmax(x). temperatures is fp32 [rows]; draws is
// int64 [2, rows], each row's seed and draw count, of which the noise is a function.
at::Tensor gumbel_race_cuda(const at::Tensor& scores, const at::Tensor& temperatures, const at::Tensor& draws);

// Why the attention ops cannot run layers of this shape in dtype, or nullopt when they can: float16 and bfloat16 run
// on the tensor-core kernels only, float32 on the scalar kernel.
std::optional<std::string> attention_unsupported_reason(
    at::ScalarType dtype,
    int64_t page_size,
    int64_t head_dim,
    int64_t n_q_heads,
    int64_t n_kv_heads
);

// Paged attention over a batch of sequences, each a VIEW: an ordered list of lines
// (page_size-key pages) of the shared pool. Key index j of sequence s lives at
// (page_tables[s][j / page_size], j % page_size). A line may appear in several views.
// k_pool and v_pool are laid out as write_kv_cuda's.
//
// The pool holds UNROTATED keys. The op rotates key j of sequence s to the RoPE
// position rope_layout[s] gives index j, in the HF/Qwen2 rotate_half convention with
// the frequencies rope states. k_norm_weight, when given, is fp32 [head_dim] and
// multiplies each stored key before rotation. rope_layout is int32 [num_seqs, 3] holding
// {n_sink, working_lo, short_offset}:
//   j < n_sink                 -> j
//   n_sink <= j < working_lo   -> short_offset
//   j >= working_lo            -> short_offset + 1 + j - working_lo
// so {0, 0, -1} is the identity and a compacted layout collapses the distant region
// [n_sink, working_lo) onto short_offset. Queries arrive ALREADY rotated, by the
// caller, to the same layout's position of their own view index.
//
// sliding_window, when given, limits the query at view index p to the keys
// [max(0, p - sliding_window + 1), p]. The op reads no page table entry whose page lies
// wholly before the window of the sequence's first query, so those entries may hold any
// value.
//
// mass, when given, receives each attended key's normalized softmax weight, added into
// the QUERY head's own column (no sum over the group), at the key's VIEW row:
// sequence s's key j accumulates into mass[cu_view_pages[s] + j / page_size,
// j % page_size, h]. Views therefore never share mass, whatever lines they share.
//   mass                 fp32 [total_view_pages, page_size, n_q_heads], mutated in place
//   cu_view_pages        int32 [num_seqs + 1]; sequence s owns view pages
//                        [cu_view_pages[s], cu_view_pages[s+1]), at least its context's
//   attention_mass_decay fp32 [num_seqs]; sequence s's EMA gain alpha, the factor its
//                        received weight is scaled by (1.0 accumulates the raw weight)
//   mass_length_gain     TOKENS. Each query's weight is rescaled by the length its
//                        share is stated against, so uniform attention over exactly
//                        that length stores 1. <= 0 takes the keys that query attended
//                        (its window's keys under sliding_window),
//                        which only the op can read. A caller wanting another CONSTANT
//                        length passes 1 and multiplies once afterwards: the mass sums
//                        over query tokens, so a constant commutes with it.
// The three tensors are passed together or not at all; without them the op skips its
// mass pass entirely.
//
// lse_capture, when given, receives each (query row, query head)'s logsumexp of
// scale * q.k over the keys that row attended, fp32 contiguous [total_q, n_q_heads]:
// the softmax denominator the attention used, so exp(logit - lse) is a key's weight.

// Decode: one query per sequence (q [num_seqs, n_q_heads, head_dim]) at view index
// context_lens[s] - 1, attending keys [0, context_lens[s]) or its window of them.
// num_splits > 0 forces the split (flash-decode) path with that many splits, 1
// included; 0 picks the split count from occupancy.
// Returns o [num_seqs, n_q_heads, head_dim].
at::Tensor attn_decode_cuda(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& page_tables,
    const at::Tensor& context_lens,
    const at::Tensor& rope_layout,
    const rope_parameters& rope,
    double scale,
    std::optional<int64_t> sliding_window,
    const std::optional<at::Tensor>& k_norm_weight,
    const std::optional<at::Tensor>& mass = std::nullopt,
    const std::optional<at::Tensor>& cu_view_pages = std::nullopt,
    const std::optional<at::Tensor>& attention_mass_decay = std::nullopt,
    double mass_length_gain = 0.0,
    const std::optional<at::Tensor>& lse_capture = std::nullopt,
    int64_t num_splits = 0
);

// Prefill: a ragged batch of query chunks, causal over each view. Sequence i's queries
// are q[cu_seqlens_q[i] : cu_seqlens_q[i+1]] (q [total_q, n_q_heads, head_dim]) at view
// indices [ctx_start_i, seqlens_k[i]) with ctx_start_i = seqlens_k[i] - seq_q_i, and
// the query at view index p attends keys [0, p] or its window of them. The mass is the within-chunk EMA: the
// query at offset o from its chunk's end contributes alpha * (1 - alpha)^o times its
// weight. Returns o [total_q, n_q_heads, head_dim].
at::Tensor attn_prefill_cuda(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& page_tables,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& seqlens_k,
    const at::Tensor& rope_layout,
    const rope_parameters& rope,
    double scale,
    std::optional<int64_t> sliding_window,
    const std::optional<at::Tensor>& k_norm_weight,
    const std::optional<at::Tensor>& mass = std::nullopt,
    const std::optional<at::Tensor>& cu_view_pages = std::nullopt,
    const std::optional<at::Tensor>& attention_mass_decay = std::nullopt,
    double mass_length_gain = 0.0,
    const std::optional<at::Tensor>& lse_capture = std::nullopt
);

// Fused w4a16 dequant-GEMM: y = x @ dequant(W)^T for a Linear with group-wise
// symmetric int4 weights. Computes y[m,n] = sum_k x[m,k] * w_deq[n,k] with
// w_deq[n,k] = int4(weight_packed[n,k]) * weight_scale[n, k/group_size].
//   x             [M, K]    fp16/bf16, contiguous
//   weight_packed [N, K/8]  int32, 8 signed int4 per int32 along K (LSB-first)
//   weight_scale  [N, K/group_size]  per-group scale (upcast to fp32 internally)
//   group_size    quant group along K (multiple of 16; K a multiple of it)
// Returns y [M, N], same dtype as x.
at::Tensor gemm_w4a16_cuda(
    const at::Tensor& x,
    const at::Tensor& weight_packed,
    const at::Tensor& weight_scale,
    int64_t group_size
);

// Split-K variant of gemm_w4a16 with the K-split count forced to num_splits
// (>= 1). Same semantics and result; always routes through the split+combine path
// (even at num_splits == 1; num_splits may exceed the K-chunk count, the extra
// splits being empty and contributing 0).
at::Tensor gemm_w4a16_split_cuda(
    const at::Tensor& x,
    const at::Tensor& weight_packed,
    const at::Tensor& weight_scale,
    int64_t group_size,
    int64_t num_splits
);

// Marlin-style w4a16 dequant-GEMM: one-time weight repack into a fragment-matched
// int4 layout + an online GEMM consuming it. Same math as gemm_w4a16.
//   repack_w4a16(weight_packed [N,K/8] int32, weight_scale [N,K/G], group_size)
//     -> (weight_marlin int32 [Npad*Kpad/8], scale_marlin fp32 [K/G, N])
//        (opaque layout; Npad/Kpad are N/K rounded up to the Marlin tiling).
//   gemm_w4a16_marlin(x [M,K], weight_marlin, scale_marlin, group_size) -> y [M,N]
std::tuple<at::Tensor, at::Tensor>
repack_w4a16_cuda(const at::Tensor& weight_packed, const at::Tensor& weight_scale, int64_t group_size);

at::Tensor gemm_w4a16_marlin_cuda(
    const at::Tensor& x,
    const at::Tensor& weight_marlin,
    const at::Tensor& scale_marlin,
    int64_t group_size
);

// Rotary position embedding (RoPE), HF/Qwen2 "rotate_half" convention with the
// frequencies rope states. Stateless.
//   x         [n_heads, seq, head_dim]  (head_dim even)
//   pos [seq]  int64; pos of each row (may be arbitrary / non-contiguous).
// Returns the rotated x, same shape and dtype.
at::Tensor rope_cuda(const at::Tensor& x, const at::Tensor& pos, const rope_parameters& rope);

}  // namespace pulsarlm
