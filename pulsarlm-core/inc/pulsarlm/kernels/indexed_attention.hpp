#pragma once

#include "pulsarlm/kernels/paged_attention.hpp"
#include "pulsarlm/kernels/rope_parameters.hpp"

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <optional>

namespace pulsarlm {

// The blocks each query of a batch selects.
struct block_selection {
    // int32 [total_q, words] with words * 32 * block_size at least the page tables' keys: bit b of the query's row,
    // word b / 32, bit b % 32, selects the keys [b * block_size, (b + 1) * block_size).
    at::Tensor selected;
    // Divides the page size.
    std::int64_t block_size;
    // The most blocks a query selects.
    std::int64_t max_selected_blocks;
};

// Indexed attention: attn_prefill_cuda's ragged batch, decode included as one query per sequence, in which the query
// at view index p attends only the keys j <= p of its selected blocks. A query selects at least one key and at most
// selection.max_selected_blocks blocks, and attends only the pages of its selected keys. Returns o [total_q, n_q_heads,
// head_dim], written into out as attn_decode_cuda writes it.
at::Tensor attn_indexed_cuda(
    const at::Tensor& q,
    const kv_pools& pools,
    const page_views& views,
    const std::optional<at::Tensor>& cu_seqlens_q,
    const rope_parameters& rope,
    double scale,
    const block_selection& selection,
    const std::optional<page_mass>& mass,
    const std::optional<at::Tensor>& out
);

// attn_indexed_cuda measuring each query's mass apart: query i owns the rows [cu_view_pages[i], cu_view_pages[i + 1])
// of mass_sum and weight_sum, its sequence's view pages; cu_view_pages is int32 [total_q + 1]. The accepted query
// tokens of a request verifying its drafts are then added on their own.
at::Tensor attn_indexed_verify_cuda(
    const at::Tensor& q,
    const kv_pools& pools,
    const page_views& views,
    const at::Tensor& cu_seqlens_q,
    const rope_parameters& rope,
    double scale,
    const block_selection& selection,
    const page_mass& mass,
    const std::optional<at::Tensor>& out
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
// views.page_positions gives key b * ratio (see page_views), cos and sin rounded to the dtype. index_keys is [lines,
// page_size, dim] with each line packed, and ratio divides the page size. A block ending past its view's key count is
// left unwritten, its keys unread.
at::Tensor index_blocks_cuda(
    const at::Tensor& index_keys,
    const page_views& views,
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

}  // namespace pulsarlm
