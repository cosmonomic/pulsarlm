#pragma once

#include <ATen/core/Tensor.h>

#include "pulsarlm/kernels/expert_activation.hpp"
#include "pulsarlm/kernels/gate_activation.hpp"
#include "pulsarlm/kernels/rope.hpp"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <tuple>

namespace pulsarlm {

// RMSNorm over x's last axis, in fp32. weight is in x's dtype or fp32.
at::Tensor rmsnorm_cuda(const at::Tensor& x, const at::Tensor& weight, double eps);

// The RMSNorm of each of x's n_streams streams over its hidden entries, in fp32, times weight, in x's dtype. x is
// contiguous [tokens, n_streams * hidden], weight fp32 [n_streams * hidden].
at::Tensor stream_rmsnorm_cuda(const at::Tensor& x, const at::Tensor& weight, int64_t n_streams, double eps);

// Each stream's RMSNorm of x as stream_rmsnorm_cuda computes it, and that normed x times weight^T [out,
// n_streams * hidden], in x's dtype: (normed, projected). Batches of up to 4 tokens run as one kernel.
std::tuple<at::Tensor, at::Tensor> stream_rmsnorm_linear_cuda(
    const at::Tensor& x,
    const at::Tensor& norm_weight,
    int64_t n_streams,
    double eps,
    const at::Tensor& weight
);

// x [tokens, hc * hidden] plus, in each stream, out [tokens, hidden] times its injection weight [tokens, hc], as
// hyper_connection_inject_cuda computes it, then stream_rmsnorm_linear_cuda of that sum: (sum, normed, projected).
// Batches of up to 4 tokens run as one kernel.
std::tuple<at::Tensor, at::Tensor, at::Tensor> injected_stream_rmsnorm_linear_cuda(
    const at::Tensor& x,
    const at::Tensor& out,
    const at::Tensor& injection_weights,
    const at::Tensor& norm_weight,
    int64_t n_streams,
    double eps,
    const at::Tensor& weight
);

// A hyper-connection's block input and injection weights from its normed streams [tokens, hc * hidden], its low-rank
// mix logits [tokens, rank], the up projection [hc * hidden, rank] and the injection logits [tokens, hc], whose rows
// may lie apart: with gate logits silu(mix_logits / hc) times up^T, the mean over streams of sigmoid(gate) times the
// stream [tokens, hidden], and 2 * sigmoid(inject_logits / hc) [tokens, hc], each operation rounded to the dtype as the
// ATen ops round it. Batches of up to 4 tokens run as one kernel.
std::tuple<at::Tensor, at::Tensor> hyper_connection_mix_cuda(
    const at::Tensor& normed,
    const at::Tensor& mix_logits,
    const at::Tensor& up_weight,
    const at::Tensor& inject_logits
);

// x [tokens, hc * hidden] plus, in each stream, the block output out [tokens, hidden] times that stream's injection
// weight [tokens, hc], the product rounded to the dtype.
at::Tensor
hyper_connection_inject_cuda(const at::Tensor& x, const at::Tensor& out, const at::Tensor& injection_weights);

// x + sigmoid(gate_logits) * y for x and y contiguous [rows, columns] and gate_logits [rows, 1], whose rows may lie
// apart, each operation rounded to the dtype as the ATen ops round it.
at::Tensor sigmoid_gated_add_cuda(const at::Tensor& x, const at::Tensor& gate_logits, const at::Tensor& y);

// y * sigmoid(gate_logits) for y contiguous [rows, columns] or [rows, middle, columns] and gate_logits of its sizes with
// a contiguous last axis, each operation rounded to the dtype as the ATen ops round it.
at::Tensor sigmoid_gated_cuda(const at::Tensor& y, const at::Tensor& gate_logits);

// silu(x[:, :n]) * x[:, n:] for contiguous x [tokens, 2 * n], silu rounded to the dtype before the product.
at::Tensor silu_mul_cuda(const at::Tensor& x);

// Replaces in place each negative token -(r + 1) of tokens (contiguous int64 [rows]) by samples[r], an earlier
// step's sampled tokens (int64) on tokens' device; the other tokens stay.
void fill_awaited_tokens_cuda(const at::Tensor& tokens, const at::Tensor& samples);

// x [tokens, in] times weight^T [out, in] in x's dtype, as at::linear; batches of 1 to 4 fp16 or bf16 tokens run as one
// matrix-vector kernel, others through cuBLAS.
at::Tensor small_batch_linear_cuda(const at::Tensor& x, const at::Tensor& weight);

// silu(x gate^T) * (x up^T) for gate_up [2 * out, in], the gate rows first, as silu_mul_cuda(at::linear(x, gate_up));
// batches of 1 to 4 fp16 or bf16 tokens run as one kernel, others through cuBLAS and silu_mul_cuda.
at::Tensor small_batch_gated_linear_cuda(const at::Tensor& x, const at::Tensor& gate_up);

// Each token's sum over its top_k experts of top_weights times down_proj(activation(gate) * up) of its row of h
// [tokens, hidden], with top_experts (int64) and top_weights (fp32) [tokens, top_k], gate_up_proj
// [experts, 2 * intermediate, hidden], gate rows first, and down_proj [experts, hidden, intermediate]. gate, up and each
// expert's output are rounded to the dtype, the activation before the product with up, and the weighted outputs are
// summed in fp32 in top_k order. Nothing is read back to the host.
at::Tensor routed_experts_cuda(
    const at::Tensor& h,
    const at::Tensor& top_experts,
    const at::Tensor& top_weights,
    const at::Tensor& gate_up_proj,
    const at::Tensor& down_proj,
    expert_activation activation,
    const at::Tensor& counters
);

// routed_experts_cuda with each token's top_k experts by the softmax in fp32 of its router_logits [tokens, experts],
// whose rows may lie apart, in descending order with the lower expert first among equal probabilities, weighted by
// their probabilities divided by their sum. gate_up_proj and down_proj hold the experts [first_expert, first_expert +
// gate_up_proj.size(0)); the assignments to the other experts are left out of the sums.
at::Tensor routed_experts_cuda(
    const at::Tensor& h,
    const at::Tensor& router_logits,
    int64_t top_k,
    int64_t first_expert,
    const at::Tensor& gate_up_proj,
    const at::Tensor& down_proj,
    expert_activation activation,
    const at::Tensor& counters
);

// The arrival counters routed_experts_cuda needs for tokens rows of h, each routed to top_k of the held experts, as
// both forms use them.
int64_t
routed_experts_counters(int64_t tokens, int64_t top_k, int64_t held_experts, int64_t intermediate, int64_t hidden);

// An op taking counters takes int32 zeros on its tensors' device, at least as many as its count function gives, and
// leaves them zero (arrival_counters.cuh); the ordered calls of one stream may share them.

// Where a ragged batch's sequences keep their recurrent state. Per sequence, lines [num_seqs, >= slots + 2] int32 holds
// slots lines holding states, then the boundary lines of its even and odd pages. history_positions (int64 [num_seqs])
// is the history position of each sequence's first token in the batch, and rejected_tokens (int32 [num_seqs]) the
// tokens its previous step forwarded and did not keep. A sequence continues from slot slots - 1 - min(rejected_tokens[s],
// slots - 1): a layer keeping a state per token has a slot per token a step may not keep and one more, a layer keeping
// a window of inputs one slot whose window it reads rejected_tokens[s] inputs earlier. A token at history position p
// ends a page when p + 1 is a multiple of page_size, and the state after it goes to the boundary line of page p /
// page_size, alternating between the two.
struct state_rows {
    at::Tensor lines;
    std::int64_t slots;
    at::Tensor history_positions;
    at::Tensor rejected_tokens;
    std::int64_t page_size;
};

// The Gated DeltaNet ops run a ragged batch: sequence s holds tokens [cu_seqlens_q[s], cu_seqlens_q[s + 1]) of x, or
// token s alone without cu_seqlens_q; a verify op's batch holds tokens_per_sequence tokens of each sequence in turn. The
// state after the batch's last token of a sequence goes to its last slot, the state after its newest page end to that
// page's boundary line; a verify op of n tokens per sequence also writes the state after each token t into slot slots - n
// + t. A pool's
// lines lie any number of elements apart, a state pool's lines 16-byte aligned.

// silu of the causal depthwise convolution of x [tokens, channels], whose rows may lie apart, by weight [channels, 1,
// kernel] with its taps dilation tokens apart, contiguous in x's dtype. A line of conv_pool [lines, channels, window]
// holds the last window inputs before a sequence's next token, window at least (kernel - 1) * dilation and the extra
// ones the inputs of up to slots - 1 tokens a step does not keep: the convolution reads those that end window -
// rejected_tokens[s] entries in. rows.slots is 1.
at::Tensor causal_conv_cuda(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_pool,
    const state_rows& rows,
    const std::optional<at::Tensor>& cu_seqlens_q,
    std::int64_t dilation
);

// causal_conv_cuda over tokens_per_sequence tokens of each sequence; the line keeps the inputs after the last.
at::Tensor causal_conv_verify_cuda(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_pool,
    const state_rows& rows,
    std::int64_t tokens_per_sequence,
    std::int64_t dilation
);

// The gated delta rule's gated outputs [tokens, value heads * value dim] in x's dtype. Each row of x [tokens, 2 *
// key_heads * key dim + value heads * value dim] holds the token's query heads, key heads and value heads; value head h
// reads key head h / (value heads / key_heads). Queries and keys are L2-normalized, with 1e-6 added to the sum of
// squares, and queries scaled by 1 / sqrt(key dim). x is contiguous. a and b are [tokens, value heads] in x's dtype,
// their rows equally far apart: the state decays by exp(neg_a * softplus(a + dt_bias)) and learns at sigmoid(b),
// rounded to x's dtype; dt_bias and neg_a are fp32 [value heads]. A line of state_pool, fp32 [lines, value heads, key
// dim, value dim], holds a sequence's state. Each value head's output o, rounded to x's dtype, leaves as norm_weight *
// RMSNorm(o) * activation(z), the normalized o and its product with norm_weight each rounded to x's dtype; z is
// [tokens, value heads * value dim] and norm_weight [value dim], both in x's dtype, each row of z contiguous.
at::Tensor gated_delta_rule_cuda(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& dt_bias,
    const at::Tensor& neg_a,
    const at::Tensor& state_pool,
    const state_rows& rows,
    const std::optional<at::Tensor>& cu_seqlens_q,
    int64_t key_heads,
    const at::Tensor& z,
    const at::Tensor& norm_weight,
    double eps,
    gate_activation activation,
    const at::Tensor& counters
);

// gated_delta_rule_cuda over tokens_per_sequence tokens of each sequence, at most rows.slots, writing the state after
// each token t into slot rows.slots - tokens_per_sequence + t.
at::Tensor gated_delta_rule_verify_cuda(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& dt_bias,
    const at::Tensor& neg_a,
    const at::Tensor& state_pool,
    const state_rows& rows,
    int64_t tokens_per_sequence,
    int64_t key_heads,
    const at::Tensor& z,
    const at::Tensor& norm_weight,
    double eps,
    gate_activation activation,
    const at::Tensor& counters
);

// The arrival counters gated_delta_rule_cuda and gated_delta_rule_verify_cuda need for num_seqs sequences.
int64_t gated_delta_rule_counters(int64_t num_seqs, int64_t value_heads);

// Writes a batch's new keys and values into the lines of paged pools. k_pool and v_pool are [lines, page_size,
// n_kv_heads, head_dim] of one stride, each line packed and 16-byte aligned, lines any number of elements apart, and
// may be one tensor; k_new and v_new are [new tokens, n_kv_heads, head_dim]. The tokens sit at the view indices of
// attn_prefill_cuda's queries (with cu_seqlens_q) or attn_decode_cuda's (without): the token at view index j of
// sequence s goes to line page_tables[s][j / page_size], slot j % page_size. Keys are stored as given; the attention
// ops read them unrotated.
void write_kv_cuda(
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& k_new,
    const at::Tensor& v_new,
    const at::Tensor& page_tables,
    const std::optional<at::Tensor>& cu_seqlens_q,
    const at::Tensor& seqlens_k
);

// int32 [n_layers, tables.size(), width]: row s of each layer holds the first page_counts[s] entries of that layer's
// row of tables[s], a contiguous int32 [n_layers, capacity] device tensor; the entries past them are left unwritten. The
// tables stay alive until the stream has run the gather.
at::Tensor gather_page_tables_cuda(
    std::span<const at::Tensor> tables,
    std::span<const std::size_t> page_counts,
    int64_t n_layers,
    int64_t width
);

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

// The paged attention ops attend a batch of sequences, each a view: an ordered list of lines (pages of page_size keys)
// of the pools, laid out as write_kv_cuda's. Key j of sequence s lives at slot j % page_size of line
// page_tables[s][j / page_size], and a line may appear in several views.
//
// The pools hold unrotated keys. The ops rotate key j of sequence s to the RoPE position
// page_positions[s][j / page_size] + j % page_size, in the rotate_half convention with the frequencies rope states.
// page_positions is int32 [num_seqs, max_pages], laid out as page_tables. k_norm_weight, when given, is fp32 [head_dim]
// and multiplies each stored key before rotation. Queries arrive rotated to their own positions, which the caller
// gives under the same page positions.
//
// sliding_window, when given, limits the query at view index p to the keys [max(0, p - sliding_window + 1), p]. The
// ops read no page table entry whose page lies wholly before the window of the sequence's first query, so those
// entries may hold any value.
//
// mass_sum, weight_sum, cu_view_pages and query_weights are passed together or not at all; without them the ops measure
// no mass. Query token i attends view page b of its sequence when it attends one of b's key tokens; its mass on b in
// query head h is the max over b's key tokens it attends of the scaled score. Over the query tokens attending b,
// mass_sum[cu_view_pages[s] + b, h] is set to the sum of query_weights[i] times the mass, and
// weight_sum[cu_view_pages[s] + b] to the sum of query_weights[i]; both stay 0 for a page no query token attends.
// Views never share mass, whatever lines they share.
//   mass_sum      fp32 [total_view_pages, n_q_heads], zeroed by the caller
//   weight_sum    fp32 [total_view_pages], zeroed by the caller
//   cu_view_pages int32 [num_seqs + 1]; sequence s owns view pages [cu_view_pages[s], cu_view_pages[s + 1]), at least
//                 its keys'
//   query_weights fp32 [total_q], non-negative

// Decode: one query per sequence, q [num_seqs, n_q_heads, head_dim], at view index context_lens[s] - 1, attending the
// keys [0, context_lens[s]) or its window of them. num_splits above 0 forces the split (flash-decode) path with that
// many splits, 1 included; 0 picks the split count from occupancy. Returns o [num_seqs, n_q_heads, head_dim].
at::Tensor attn_decode_cuda(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& page_tables,
    const at::Tensor& context_lens,
    const at::Tensor& page_positions,
    const rope_parameters& rope,
    double scale,
    std::optional<int64_t> sliding_window,
    const std::optional<at::Tensor>& k_norm_weight,
    const std::optional<at::Tensor>& mass_sum = std::nullopt,
    const std::optional<at::Tensor>& weight_sum = std::nullopt,
    const std::optional<at::Tensor>& cu_view_pages = std::nullopt,
    const std::optional<at::Tensor>& query_weights = std::nullopt,
    int64_t num_splits = 0
);

// Prefill: a ragged batch of query chunks, causal over each view. Sequence i's queries are
// q[cu_seqlens_q[i] : cu_seqlens_q[i + 1]] (q [total_q, n_q_heads, head_dim]) at view indices
// [seqlens_k[i] - seq_q_i, seqlens_k[i]), and the query at view index p attends the keys [0, p] or its window of them.
// Returns o [total_q, n_q_heads, head_dim].
at::Tensor attn_prefill_cuda(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& page_tables,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& seqlens_k,
    const at::Tensor& page_positions,
    const rope_parameters& rope,
    double scale,
    std::optional<int64_t> sliding_window,
    const std::optional<at::Tensor>& k_norm_weight,
    const std::optional<at::Tensor>& mass_sum = std::nullopt,
    const std::optional<at::Tensor>& weight_sum = std::nullopt,
    const std::optional<at::Tensor>& cu_view_pages = std::nullopt,
    const std::optional<at::Tensor>& query_weights = std::nullopt
);

// attn_prefill_cuda measuring each query's mass apart: query i owns the rows [cu_view_pages[i], cu_view_pages[i + 1])
// of mass_sum and weight_sum, its sequence's view pages; cu_view_pages is int32 [total_q + 1]. A verify group's
// accepted rows are then added on their own.
at::Tensor attn_prefill_verify_cuda(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& page_tables,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& seqlens_k,
    const at::Tensor& page_positions,
    const rope_parameters& rope,
    double scale,
    std::optional<int64_t> sliding_window,
    const std::optional<at::Tensor>& k_norm_weight,
    const at::Tensor& mass_sum,
    const at::Tensor& weight_sum,
    const at::Tensor& cu_view_pages,
    const at::Tensor& query_weights
);

// Indexed attention: attn_prefill_cuda's ragged batch, decode included as one query per sequence, in which the query
// at view index p attends only the keys j <= p of its selected blocks: bit b of selected's row for the query, word
// b / 32, bit b % 32, selects the keys [b * block_size, (b + 1) * block_size), block_size dividing the page size.
// selected is int32 [total_q, words] with words * 32 * block_size at least the page tables' keys. A query selects at
// least one key and at most max_selected_blocks blocks, and attends only the pages of its selected keys. Returns o
// [total_q, n_q_heads, head_dim].
at::Tensor attn_indexed_cuda(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& page_tables,
    const std::optional<at::Tensor>& cu_seqlens_q,
    const at::Tensor& seqlens_k,
    const at::Tensor& page_positions,
    const rope_parameters& rope,
    double scale,
    const at::Tensor& selected,
    int64_t block_size,
    int64_t max_selected_blocks,
    const std::optional<at::Tensor>& mass_sum,
    const std::optional<at::Tensor>& weight_sum,
    const std::optional<at::Tensor>& cu_view_pages,
    const std::optional<at::Tensor>& query_weights
);

// attn_indexed_cuda measuring each query's mass apart: query i owns the rows [cu_view_pages[i], cu_view_pages[i + 1])
// of mass_sum and weight_sum, its sequence's view pages; cu_view_pages is int32 [total_q + 1]. A verify group's
// accepted rows are then added on their own.
at::Tensor attn_indexed_verify_cuda(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& page_tables,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& seqlens_k,
    const at::Tensor& page_positions,
    const rope_parameters& rope,
    double scale,
    const at::Tensor& selected,
    int64_t block_size,
    int64_t max_selected_blocks,
    const at::Tensor& mass_sum,
    const at::Tensor& weight_sum,
    const at::Tensor& cu_view_pages,
    const at::Tensor& query_weights
);

// QSA's index queries [tokens, n_heads, dim] from the first n_heads * dim columns of index_qk [tokens, >= n_heads *
// dim], each row contiguous: each head's RMSNorm by norm_weight (fp32, 1 + weight) as rmsnorm_cuda computes it, then
// Hugging Face's partial RoPE of its first 2 * inv_freq.size(0) dims at the token's position (int64 [tokens]), as
// index_blocks_cuda rotates a block. dim is a power of two up to 1024.
at::Tensor index_queries_cuda(
    const at::Tensor& index_qk,
    const at::Tensor& norm_weight,
    const at::Tensor& inv_freq,
    const at::Tensor& positions,
    int64_t n_heads,
    double eps
);

// QSA's blocks [sequences, max_blocks, dim] in index_keys' dtype: block b of sequence s is the mean, in fp32 and
// rounded, of the raw keys [b * ratio, (b + 1) * ratio) of its view, then its RMSNorm by norm_weight (fp32, 1 + weight)
// as rmsnorm_cuda computes it, then Hugging Face's partial RoPE of its first 2 * inv_freq.size(0) dims at the position
// page_positions gives key b * ratio (see attn_decode_cuda), cos and sin rounded to the dtype. index_keys is [lines,
// page_size, dim] with each line packed, page_tables and page_positions int32 [sequences, max_pages], and ratio divides
// the page size. key_counts is int32 [sequences], the keys in each view; a block ending past its view's key count is
// left unwritten, its keys unread.
at::Tensor index_blocks_cuda(
    const at::Tensor& index_keys,
    const at::Tensor& page_tables,
    const at::Tensor& page_positions,
    const at::Tensor& key_counts,
    const at::Tensor& norm_weight,
    const at::Tensor& inv_freq,
    int64_t ratio,
    double eps
);

// QSA's selection, int32 [rows, words] as attn_indexed_cuda reads it: row r's top min(top, complete blocks) of its
// complete blocks by scores (index_scores_cuda's), the lower block first among equal scores, then the block after them
// when it starts at or before the row's view index. The rows are laid out as index_scores_cuda's; words * 32 exceeds
// max_blocks.
at::Tensor select_blocks_cuda(
    const at::Tensor& scores,
    const at::Tensor& key_counts,
    const std::optional<at::Tensor>& cu_seqlens_q,
    int64_t first_row,
    int64_t top,
    int64_t ratio,
    int64_t words
);

// QSA's block scores, fp32 [rows, max_blocks]: block b among row r's complete blocks of sequence s scores scale
// times the sum over its heads of relu(queries[r, h] . blocks[s, b]), dot products accumulated in fp32, on the tensor
// cores for fp16 and bf16 with heads dividing 16 and a dim of 64, 128 or 256; the other blocks are left unwritten.
// queries [rows, heads, dim] and blocks [sequences, max_blocks, dim] share a dtype. Row r is row first_row + r of a
// ragged batch in which sequence s holds rows [cu_seqlens_q[s], cu_seqlens_q[s + 1]), or row s alone without
// cu_seqlens_q (int32 [sequences + 1]), the last rows of its view of key_counts[s] keys (int32 [sequences]); its
// complete blocks are the blocks of ratio keys ending at or before its key.
at::Tensor index_scores_cuda(
    const at::Tensor& queries,
    const at::Tensor& blocks,
    const at::Tensor& key_counts,
    const std::optional<at::Tensor>& cu_seqlens_q,
    int64_t first_row,
    int64_t ratio,
    double scale
);

// Fused int4 dequant-GEMM: y = x @ dequant(W)^T for a linear with group-wise
// symmetric int4 weights. Computes y[m,n] = sum_k x[m,k] * w_deq[n,k] with
// w_deq[n,k] = int4(weight_packed[n,k]) * weight_scale[n, k/group_size].
//   x             [M, K]    fp16/bf16, contiguous
//   weight_packed [N, K/8]  int32, 8 signed int4 per int32 along K (LSB-first)
//   weight_scale  [N, K/group_size]  per-group scale (upcast to fp32 internally)
//   group_size    quant group along K (multiple of 16; K a multiple of it)
// Returns y [M, N], same dtype as x.
at::Tensor gemm_int4_cuda(
    const at::Tensor& x,
    const at::Tensor& weight_packed,
    const at::Tensor& weight_scale,
    int64_t group_size
);

// Split-K variant of gemm_int4 with the K-split count forced to num_splits
// (>= 1). Same semantics and result; always routes through the split+combine path
// (even at num_splits == 1; num_splits may exceed the K-chunk count, the extra
// splits being empty and contributing 0).
at::Tensor gemm_int4_split_cuda(
    const at::Tensor& x,
    const at::Tensor& weight_packed,
    const at::Tensor& weight_scale,
    int64_t group_size,
    int64_t num_splits
);

// Marlin dequant-GEMMs: y [M, N] = x [M, K] @ dequant(W)^T in x's dtype (fp16 or bf16), over weights repacked once
// into an opaque layout. w_deq[n, k] = w[n, k] * weight_scale[n / block_n, k / group_size]; group_size is a multiple
// of 16 dividing K. The repacks return (weight_marlin, scale_marlin), which the GEMM takes with the same N, block_n
// and group_size.
//   repack_int4_marlin: weight_packed int32 [N, K/8], 8 signed int4 per int32 along K, least significant first.
//   repack_fp8_marlin:  weight float8_e4m3fn [N, K]; the NaN codes are not supported.
//   weight_scale [ceil(N / block_n), K / group_size], upcast to fp32.
std::tuple<at::Tensor, at::Tensor> repack_int4_marlin_cuda(
    const at::Tensor& weight_packed,
    const at::Tensor& weight_scale,
    int64_t block_n,
    int64_t group_size
);

std::tuple<at::Tensor, at::Tensor>
repack_fp8_marlin_cuda(const at::Tensor& weight, const at::Tensor& weight_scale, int64_t block_n, int64_t group_size);

at::Tensor gemm_int4_marlin_cuda(
    const at::Tensor& x,
    const at::Tensor& weight_marlin,
    const at::Tensor& scale_marlin,
    int64_t n,
    int64_t block_n,
    int64_t group_size
);

at::Tensor gemm_fp8_marlin_cuda(
    const at::Tensor& x,
    const at::Tensor& weight_marlin,
    const at::Tensor& scale_marlin,
    int64_t n,
    int64_t block_n,
    int64_t group_size
);

// x [n_heads, seq, head_dim], head_dim even, with row i rotated to position pos[i] (int64 [seq]) in the rotate_half
// convention with the frequencies rope states.
at::Tensor rope_cuda(const at::Tensor& x, const at::Tensor& pos, const rope_parameters& rope);

}  // namespace pulsarlm
