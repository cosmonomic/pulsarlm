#pragma once

#include <c10/macros/Macros.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

// The per-batch inputs every paged attention kernel reads, and the per-key arithmetic
// they share: a key's RoPE position in its sequence's view, its rotation, and its page's
// row in the mass output.
//
// The pool holds UNROTATED keys. A key's position is its view page's entry in
// page_positions plus its offset in the page, so a line shared by several views takes
// each view's own position, and the kernels rotate every key as they read it. Queries
// arrive already rotated.

namespace pulsarlm {
namespace attn {

constexpr size_t kSharedMemoryWithoutOptIn = 48 * 1024;

template <typename scalar_t> struct PagedAttnParams {
    const scalar_t* q;  // [total_q, n_q_heads, head_dim], rotated
    const scalar_t* k_pool;  // [num_pages, page_size, n_kv_heads, head_dim], unrotated
    const scalar_t* v_pool;  // [num_pages, page_size, n_kv_heads, head_dim]
    // Elements from one pool page, token and head to the next, in k_pool and v_pool alike.
    int64_t page_stride;
    int64_t token_stride;
    int64_t head_stride;
    const int32_t* page_tables;  // [num_seqs, max_pages]
    const int32_t* cu_seqlens_q;  // [num_seqs + 1]; null when each sequence has one query
    const int32_t* seqlens_k;  // [num_seqs]
    const int32_t* page_positions;  // [num_seqs, max_pages]: each view page's first RoPE position
    const uint64_t* rope_turns;  // [head_dim / 2], see rope_turns_per_position
    const float* k_norm_weight;  // [head_dim] or null
    // [total_view_pages, n_q_heads]; null measures no mass. The kernels leave each attended page's S there, the sum
    // over its queries of mass_weights times the log share, and the decode kernels its max score until the LSE.
    float* mass;
    const int32_t* cu_view_pages;  // [num_seqs + 1]
    const float* mass_weights;  // [total_q]
    float* attended_weight;  // [total_view_pages], each page's W: the sum of mass_weights over its queries

    scalar_t* o;  // [total_q, n_q_heads, head_dim]
    int num_seqs;
    int n_q_heads;
    int n_kv_heads;
    int head_dim;
    int max_pages;
    int group;
    int total_view_pages;
    int sliding_window;  // 0 when there is no window
    float scale;
};

template <typename scalar_t>
__device__ __forceinline__ int first_key(const PagedAttnParams<scalar_t>& p, int query_index) {
    return p.sliding_window > 0 ? max(0, query_index - p.sliding_window + 1) : 0;
}

// Key element d as stored, times its k_norm_weight entry when there is one.
template <typename scalar_t>
__device__ __forceinline__ float stored_key(const PagedAttnParams<scalar_t>& p, scalar_t value, int d) {
    const float key = static_cast<float>(value);
    return p.k_norm_weight ? __fmul_rn(key, p.k_norm_weight[d]) : key;
}

template <typename scalar_t>
__device__ __forceinline__ void
stored_key_pair(const PagedAttnParams<scalar_t>& p, const scalar_t* key, int pair, int half, float& lo, float& hi) {
    lo = stored_key(p, key[pair], pair);
    hi = stored_key(p, key[pair + half], pair + half);
}

// The RoPE positions of one sequence's keys by their view index.
struct KeyPositions {
    const int32_t* page_positions;
    int page_size;

    __device__ __forceinline__ int position(int j) const {
        return this->page_positions[j / this->page_size] + j % this->page_size;
    }
};

template <typename scalar_t>
__device__ __forceinline__ KeyPositions key_positions(const PagedAttnParams<scalar_t>& p, int s, int page_size) {
    return KeyPositions{p.page_positions + static_cast<int64_t>(s) * p.max_pages, page_size};
}

// cos/sin of the RoPE angle position * turns_per_position, rounded to fp32.
//
// The product wraps mod 2^64, which drops the whole turns exactly and leaves the
// angle's fraction of a turn to 2^-64. Its top 32 bits split into the nearest quarter
// turn and a remainder within an eighth of a turn of it, and only that remainder goes
// through the hardware __sincosf, whose absolute error on it is under 2^-21.
__device__ __forceinline__ void
rope_sincos_turns(int position, uint64_t turns_per_position, float& cos_out, float& sin_out) {
    constexpr float kRadiansPerUnit = 1.46291807926715968e-9f;  // 2 pi / 2^32
    const uint64_t turns = static_cast<uint64_t>(static_cast<int64_t>(position)) * turns_per_position;
    const uint32_t top = static_cast<uint32_t>(turns >> 32);
    const uint32_t quadrant = (top + (1u << 29)) >> 30;
    const int32_t remainder = static_cast<int32_t>(top - (quadrant << 30));

    float sine, cosine;
    __sincosf(static_cast<float>(remainder) * kRadiansPerUnit, &sine, &cosine);
    // Quadrants 1 and 3 swap cosine and sine; cos is negative in quadrants 1 and 2, sin in 2 and 3.
    const bool swapped = quadrant & 1;
    const uint32_t cos_sign = ((quadrant + 1) & 2) << 30;
    const uint32_t sin_sign = (quadrant & 2) << 30;
    cos_out = __uint_as_float(__float_as_uint(swapped ? sine : cosine) ^ cos_sign);
    sin_out = __uint_as_float(__float_as_uint(swapped ? cosine : sine) ^ sin_sign);
}

template <typename scalar_t> struct RotatedPair {
    scalar_t lo;
    scalar_t hi;
};

// The rotate_half rotation of one frequency pair (x[i], x[i + head_dim/2]), rounded to
// the pool dtype. The rounding modes are explicit so that every kernel holding the
// same key computes the same bits.
template <typename scalar_t>
__device__ __forceinline__ RotatedPair<scalar_t> rope_rotate(float lo, float hi, float cos_a, float sin_a) {
    return RotatedPair<scalar_t>{
        static_cast<scalar_t>(__fsub_rn(__fmul_rn(lo, cos_a), __fmul_rn(hi, sin_a))),
        static_cast<scalar_t>(__fadd_rn(__fmul_rn(hi, cos_a), __fmul_rn(lo, sin_a))),
    };
}

// The row of sequence s's view page b in mass and attended_weight, so two views sharing a line accumulate apart.
template <typename scalar_t>
__device__ __forceinline__ int64_t mass_row(const PagedAttnParams<scalar_t>& p, int s, int view_page) {
    const int row = p.cu_view_pages[s] + view_page;
    CUDA_KERNEL_ASSERT(row < p.cu_view_pages[s + 1] && row < p.total_view_pages);
    return row;
}

}  // namespace attn
}  // namespace pulsarlm
