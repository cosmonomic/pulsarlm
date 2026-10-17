#pragma once

#include <ATen/core/Tensor.h>

#include "pulsarlm/rope.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <tuple>

namespace pulsarlm {

// RMSNorm over x's last axis, in fp32. weight is in x's dtype or fp32.
at::Tensor rmsnorm_cuda(const at::Tensor& x, const at::Tensor& weight, double eps);

// The Gated DeltaNet ops run a ragged batch: sequence s holds tokens [cu_seqlens_q[s], cu_seqlens_q[s + 1]) of x, or
// token s alone without cu_seqlens_q. rows [num_seqs, >= 2] int32 holds each sequence's running line, whose state the
// sequence continues and which then holds the state after its last token, and its boundary line, which receives the
// state after its first page_end_tokens[s] tokens when page_end_tokens[s] > 0. A pool's lines lie any number of
// elements apart, a state pool's lines 16-byte aligned.

// silu of the causal depthwise convolution of x [tokens, channels] by weight [channels, 1, kernel] with its taps
// dilation tokens apart, in x's dtype. A line of conv_pool [lines, channels, (kernel - 1) * dilation] holds the
// (kernel - 1) * dilation inputs before a sequence's next token.
at::Tensor causal_conv_cuda(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_pool,
    const at::Tensor& rows,
    const std::optional<at::Tensor>& cu_seqlens_q,
    const at::Tensor& page_end_tokens,
    std::int64_t dilation
);

// The gated delta rule's outputs [tokens, value heads, value dim] in x's dtype. Each row of x [tokens, 2 * key_heads *
// key dim + value heads * value dim] holds the token's query heads, key heads and value heads; value head h reads key
// head h / (value heads / key_heads). Queries and keys are L2-normalized, with 1e-6 added to the sum of squares, and
// queries scaled by 1 / sqrt(key dim). a and b are [tokens, value heads] in x's dtype: the state decays by
// exp(neg_a * softplus(a + dt_bias)) and learns at sigmoid(b), rounded to x's dtype; dt_bias and neg_a are fp32
// [value heads]. A line of state_pool, fp32 [lines, value heads, key dim, value dim], holds a sequence's state.
at::Tensor gated_delta_rule_cuda(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& dt_bias,
    const at::Tensor& neg_a,
    const at::Tensor& state_pool,
    const at::Tensor& rows,
    const std::optional<at::Tensor>& cu_seqlens_q,
    const at::Tensor& page_end_tokens,
    int64_t key_heads
);

enum class gate_activation { silu, sigmoid };

// weight * RMSNorm(x) * activation(z) over the last axis, the normalized x and its product with weight each rounded to
// x's dtype. z and weight are in x's dtype.
at::Tensor gated_rmsnorm_cuda(
    const at::Tensor& x,
    const at::Tensor& z,
    const at::Tensor& weight,
    double eps,
    gate_activation activation = gate_activation::silu
);

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

// Indexed attention: attn_prefill_cuda's ragged batch, decode included as one query per sequence, in which the query
// at view index p attends only the keys j <= p of its selected blocks: bit b of selected's row for the query, word
// b / 32, bit b % 32, selects the keys [b * block_size, (b + 1) * block_size), block_size dividing the page size.
// selected is int32 [total_q, words] with
// words * 32 * block_size at least the page tables' keys. A query selects at least one key. The mass is
// attn_prefill_cuda's, the length gain, at mass_length_gain <= 0, stated against the keys each query attends.
// Returns o [total_q, n_q_heads, head_dim].
at::Tensor attn_indexed_cuda(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& page_tables,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& seqlens_k,
    const at::Tensor& rope_layout,
    const rope_parameters& rope,
    double scale,
    const at::Tensor& selected,
    int64_t block_size,
    const std::optional<at::Tensor>& mass,
    const std::optional<at::Tensor>& cu_view_pages,
    const std::optional<at::Tensor>& attention_mass_decay,
    double mass_length_gain
);

// QSA's blocks [sequences, max_blocks, dim] in index_keys' dtype: block b of sequence s is the mean, in fp32 and
// rounded, of the raw keys [b * ratio, (b + 1) * ratio) of its view, then its RMSNorm by norm_weight (fp32, 1 + weight)
// as rmsnorm_cuda computes it, then Hugging Face's partial RoPE of its first 2 * inv_freq.size(0) dims at the position
// rope_layout[s] gives key b * ratio (see attn_decode_cuda), cos and sin rounded to the dtype. index_keys is [lines,
// page_size, dim] with each line packed, page_tables int32 [sequences, max_pages] whose view keys are max_pages *
// page_size, and ratio divides the page size.
at::Tensor index_blocks_cuda(
    const at::Tensor& index_keys,
    const at::Tensor& page_tables,
    const at::Tensor& rope_layout,
    const at::Tensor& norm_weight,
    const at::Tensor& inv_freq,
    int64_t ratio,
    double eps
);

// QSA's selection, int32 [rows, words] as attn_indexed_cuda reads it: row r's top min(top, max_blocks) blocks by scores
// (index_scores_cuda's), the lower block first among equal scores and no block scoring -inf, then the block
// row_blocks[r] when it starts at or before view_index[r]. row_blocks and view_index are int64 [rows]; words * 32
// exceeds max_blocks.
at::Tensor select_blocks_cuda(
    const at::Tensor& scores,
    const at::Tensor& row_blocks,
    const at::Tensor& view_index,
    int64_t top,
    int64_t ratio,
    int64_t words
);

// QSA's block scores, fp32 [rows, max_blocks]: for query row r of sequence row_sequence[r], block b below row_blocks[r]
// scores scale times the sum over its heads of relu(queries[r, h] . blocks[row_sequence[r], b]), dot products
// accumulated in fp32, on the tensor cores for fp16 and bf16 with heads dividing 16 and a dim of 64, 128 or 256;
// the blocks past row_blocks[r] score -inf. queries [rows, heads, dim] and blocks [sequences, max_blocks, dim] share a
// dtype; row_sequence and row_blocks are int64 [rows].
at::Tensor index_scores_cuda(
    const at::Tensor& queries,
    const at::Tensor& blocks,
    const at::Tensor& row_sequence,
    const at::Tensor& row_blocks,
    double scale
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
