#include "attn_common.cuh"

#include "pulsarlm/kernels/attn_tiles.cuh"
#include "pulsarlm/kernels/attn_tuning.hpp"
#include "pulsarlm/ops.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <math_constants.h>

#include <algorithm>
#include <cstdint>

// Paged-KV decode attention, and the scatter that writes new K/V into the pool. One
// query per sequence attends its whole view, or the window of it sliding_window
// leaves.
//
// The tensor-core path runs one warp group per (sequence s, kv head g): the `group`
// query heads sharing g occupy MMA rows 0..group-1 (rows group..15 are zero and
// discarded), and the view is streamed one page (PAGE_SIZE keys) at a time, each key
// rotated to its position as it is staged. Online softmax runs per query row in shared
// memory between the two MMAs, the running output stays in the registers of the
// warps that own its dim slabs, and pass 1 stashes every scaled score so the mass pass
// never re-reads K. The QKᵀ/PV tiles and their fragment layout live in attn_tiles.cuh.
//
// The split (flash-decode) path spreads one (s, g)'s view over nsplits CTAs:
//   split kernel   : grid num_seqs * n_kv_heads * nsplits; each CTA runs the online
//                    softmax over one slice and writes its RAW state.
//   combine kernel : grid num_seqs * n_kv_heads; reduces the partials per query row
//                    with the flash rescale into o and LSE.
//   mass kernel    : grid num_seqs * n_kv_heads; adds exp(score - LSE) per key.
// Partial buffers (fp32), for (seq s, kv head g, split i, row m):
//   o_partial [num_seqs, n_kv_heads, nsplits, group, head_dim]
//   m_partial, l_partial [num_seqs, n_kv_heads, nsplits, group]
//   lse_out   [num_seqs, n_kv_heads, group]

namespace pulsarlm {
namespace {

using attn::KeyLayout;
using attn::kThreads;
using attn::PagedAttnParams;

// One block per token; the n_kv_heads*head_dim elements are strided across the
// block's threads. The pools may be one tensor.
template <typename scalar_t, int PAGE_SIZE>
__global__ void write_kv_kernel(
    scalar_t* k_pool,
    scalar_t* v_pool,
    const scalar_t* __restrict__ k_new,
    const scalar_t* __restrict__ v_new,
    const int32_t* __restrict__ page_tables,
    const int32_t* __restrict__ cu_seqlens_q,
    const int32_t* __restrict__ seqlens_k,
    int num_seqs,
    int max_pages,
    int n_kv_heads,
    int head_dim
) {
    const int t = blockIdx.x;
    int s = t;
    int index = seqlens_k[t] - 1;
    if (cu_seqlens_q) {
        int lo = 0, hi = num_seqs;
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (cu_seqlens_q[mid] <= t) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        s = lo;
        index = seqlens_k[s] - (cu_seqlens_q[s + 1] - t);
    }
    const int line = page_tables[static_cast<int64_t>(s) * max_pages + index / PAGE_SIZE];

    const int64_t n = static_cast<int64_t>(n_kv_heads) * head_dim;
    const int64_t dst = (static_cast<int64_t>(line) * PAGE_SIZE + index % PAGE_SIZE) * n;
    const int64_t src = static_cast<int64_t>(t) * n;

    for (int64_t e = threadIdx.x; e < n; e += blockDim.x) {
        k_pool[dst + e] = k_new[src + e];
        v_pool[dst + e] = v_new[src + e];
    }
}

// Every tile size is a multiple of 16 bytes, which keeps each tile 16-byte aligned for
// ldmatrix.
template <typename scalar_t, int PAGE_SIZE, int HEAD_DIM> struct DecodeTiles {
    scalar_t* sQ;  // [16, HEAD_DIM]
    scalar_t* sK;  // [PAGE_SIZE, HEAD_DIM]
    scalar_t* sV;  // [PAGE_SIZE, HEAD_DIM]
    scalar_t* sP;  // [16, PAGE_SIZE]
    float* sS;  // [16, PAGE_SIZE]
    float* Mrow;  // [16]
    float* Lrow;  // [16]
    float* row_rescale;  // [16]
    float* LSE;  // [16]

    static constexpr size_t bytes() {
        return sizeof(scalar_t) * (16 * HEAD_DIM + 2 * PAGE_SIZE * HEAD_DIM + 16 * PAGE_SIZE) +
            sizeof(float) * (16 * PAGE_SIZE + 4 * 16);
    }

    __device__ static DecodeTiles carve(unsigned char* base) {
        DecodeTiles tiles;
        tiles.sQ = reinterpret_cast<scalar_t*>(base);
        tiles.sK = tiles.sQ + 16 * HEAD_DIM;
        tiles.sV = tiles.sK + PAGE_SIZE * HEAD_DIM;
        tiles.sP = tiles.sV + PAGE_SIZE * HEAD_DIM;
        tiles.sS = reinterpret_cast<float*>(tiles.sP + 16 * PAGE_SIZE);
        tiles.Mrow = tiles.sS + 16 * PAGE_SIZE;
        tiles.Lrow = tiles.Mrow + 16;
        tiles.row_rescale = tiles.Lrow + 16;
        tiles.LSE = tiles.row_rescale + 16;
        return tiles;
    }
};

int64_t window_page_span(int64_t sliding_window, int64_t page_size) {
    return (sliding_window - 1 + page_size - 1) / page_size + 1;
}

// Load the group's query heads into MMA rows 0..group-1 and zero the rest, and reset
// the softmax state.
template <typename scalar_t, int HEAD_DIM, int NWARPS>
__device__ __forceinline__ void
begin_group(const PagedAttnParams<scalar_t>& p, int s, int g, scalar_t* sQ, float* Mrow, float* Lrow) {
    const int tid = threadIdx.x;
    for (int i = tid; i < 16 * HEAD_DIM; i += NWARPS * 32) {
        const int row = i / HEAD_DIM, col = i % HEAD_DIM;
        scalar_t v = static_cast<scalar_t>(0);
        if (row < p.group) {
            const int h = g * p.group + row;
            v = p.q[(static_cast<int64_t>(s) * p.n_q_heads + h) * HEAD_DIM + col];
        }
        sQ[i] = v;
    }
    if (tid < 16) {
        Mrow[tid] = -CUDART_INF_F;
        Lrow[tid] = 0.0f;
    }
}

// Online-softmax attention over view tiles [t_start, t_end) for one (seq s, kv head
// g), keys from index first on. Tiles wholly before first are not read.
// Accumulates the RAW, UNNORMALIZED state: out (this lane's tile_pv_registers share of
// the sum of exp(scale*q.k - Mrow)*V per query row), Mrow (running row max), Lrow
// (running denominator); the caller owns normalization and LSE. When seq_scores is
// non-null, each scaled score is stashed to seq_scores[key * group + m] for the mass
// pass. A tile with no keys ends the loop, so a slice past the view leaves out = 0,
// Mrow = -inf, Lrow = 0.
template <typename scalar_t, int PAGE_SIZE, int HEAD_DIM, bool BF16, int NWARPS>
__device__ __forceinline__ void tc_attend_tiles(
    const PagedAttnParams<scalar_t>& p,
    int s,
    int g,
    int ctx_len,
    int first,
    const KeyLayout& layout,
    float* __restrict__ seq_scores,
    int t_start,
    int t_end,
    const DecodeTiles<scalar_t, PAGE_SIZE, HEAD_DIM>& tiles,
    float* out
) {
    scalar_t* sQ = tiles.sQ;
    scalar_t* sK = tiles.sK;
    scalar_t* sV = tiles.sV;
    scalar_t* sP = tiles.sP;
    float* sS = tiles.sS;
    float* Mrow = tiles.Mrow;
    float* Lrow = tiles.Lrow;
    float* row_rescale = tiles.row_rescale;
    constexpr int HALF = HEAD_DIM / 2;
    static_assert((NWARPS * 32) % HALF == 0, "every thread must rotate one fixed frequency pair");
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int pair = tid % HALF;
    const uint64_t turns = p.rope_turns[pair];
    const int32_t* seq_table = p.page_tables + static_cast<int64_t>(s) * p.max_pages;
    const int64_t stride_slot = static_cast<int64_t>(p.n_kv_heads) * HEAD_DIM;

    for (int t = t_start; t < t_end; ++t) {
        const int tile_base = t * PAGE_SIZE;
        const int valid = min(PAGE_SIZE, ctx_len - tile_base);
        if (valid <= 0) {
            break;
        }
        const int before_window = max(0, first - tile_base);
        if (before_window >= PAGE_SIZE) {
            continue;
        }
        const int64_t line_base = (static_cast<int64_t>(seq_table[t]) * PAGE_SIZE * p.n_kv_heads + g) * HEAD_DIM;
        for (int i = tid; i < PAGE_SIZE * HALF; i += NWARPS * 32) {
            const int off = i / HALF;
            scalar_t lo = static_cast<scalar_t>(0), hi = static_cast<scalar_t>(0);
            if (off >= before_window && off < valid) {
                float cos_a, sin_a, k_lo, k_hi;
                attn::rope_sincos_turns(layout.position(tile_base + off), turns, cos_a, sin_a);
                attn::stored_key_pair(p, p.k_pool + line_base + off * stride_slot, pair, HALF, k_lo, k_hi);
                const auto k = attn::rope_rotate<scalar_t>(k_lo, k_hi, cos_a, sin_a);
                lo = k.lo;
                hi = k.hi;
            }
            sK[off * HEAD_DIM + pair] = lo;
            sK[off * HEAD_DIM + pair + HALF] = hi;
        }
        for (int i = tid; i < PAGE_SIZE * HEAD_DIM; i += NWARPS * 32) {
            const int off = i / HEAD_DIM, d = i % HEAD_DIM;
            sV[i] = off >= before_window && off < valid ? p.v_pool[line_base + off * stride_slot + d]
                                                        : static_cast<scalar_t>(0);
        }
        __syncthreads();

        attn::tile_qkt<scalar_t, PAGE_SIZE, HEAD_DIM, BF16>(sQ, sK, sS, warp, NWARPS);
        __syncthreads();

        // Warp 0's first 16 lanes own the query rows.
        if (warp == 0 && lane < 16) {
            const int m = lane;
            float tmax = -CUDART_INF_F;
#pragma unroll
            for (int c = 0; c < PAGE_SIZE; ++c) {
                if (c >= before_window && c < valid) {
                    tmax = fmaxf(tmax, p.scale * sS[m * PAGE_SIZE + c]);
                }
            }
            const float newM = fmaxf(Mrow[m], tmax);
            const float corr = __expf(Mrow[m] - newM);  // 0 when Mrow == -inf
            float lsum = Lrow[m] * corr;
#pragma unroll
            for (int c = 0; c < PAGE_SIZE; ++c) {
                float pw = 0.0f;
                if (c >= before_window && c < valid) {
                    const float ssc = p.scale * sS[m * PAGE_SIZE + c];
                    pw = __expf(ssc - newM);
                    if (seq_scores && m < p.group) {
                        seq_scores[static_cast<int64_t>(tile_base + c) * p.group + m] = ssc;
                    }
                }
                sP[m * PAGE_SIZE + c] = static_cast<scalar_t>(pw);
                lsum += pw;
            }
            row_rescale[m] = corr;
            Mrow[m] = newM;
            Lrow[m] = lsum;
        }
        __syncthreads();

        attn::tile_pv_registers<scalar_t, PAGE_SIZE, HEAD_DIM, BF16, NWARPS>(sP, sV, row_rescale, out, warp);
        __syncthreads();
    }
}

// Add each attended key's normalized weight exp(score - lse[m]), times the sequence's
// mass gain, into query head g*group + m's column of the key's mass row. A sequence
// has one query, so no other thread of the launch writes the same slot.
template <typename scalar_t>
__device__ __forceinline__ void accumulate_group_mass(
    const PagedAttnParams<scalar_t>& p,
    int page_size,
    int s,
    int g,
    int ctx_len,
    const float* seq_scores,
    const float* lse
) {
    float* mass = attn::seq_mass(p, s, ctx_len, page_size);
    const int first = attn::first_key(p, ctx_len - 1);
    const float attended = static_cast<float>(ctx_len - first);
    const float gain = p.mass_decay[s] * (p.mass_length_gain > 0.0f ? p.mass_length_gain : attended);
    for (int j = first + static_cast<int>(threadIdx.x); j < ctx_len; j += blockDim.x) {
        const float* col = seq_scores + static_cast<int64_t>(j) * p.group;
        float* row = mass + static_cast<int64_t>(j) * p.n_q_heads + g * p.group;
        for (int m = 0; m < p.group; ++m) {
            row[m] += gain * __expf(col[m] - lse[m]);
        }
    }
}

// Single-CTA tensor-core decode: one CTA per (seq s, kv head g) attends the whole view,
// then adds the mass from the stashed scores.
template <typename scalar_t, int PAGE_SIZE, int HEAD_DIM, bool BF16, int NWARPS>
__global__ void __launch_bounds__(NWARPS * 32) attn_decode_tc_kernel(
    const PagedAttnParams<scalar_t> p,
    float* __restrict__ scores,
    int64_t scores_stride_seq,
    int64_t scores_stride_head
) {
    const int tid = threadIdx.x;
    const int s = blockIdx.x / p.n_kv_heads;
    const int g = blockIdx.x % p.n_kv_heads;
    const int ctx_len = p.seqlens_k[s];
    const int ntiles = (ctx_len + PAGE_SIZE - 1) / PAGE_SIZE;
    const int first = attn::first_key(p, ctx_len - 1);
    float* seq_scores = p.mass ? scores + s * scores_stride_seq + g * scores_stride_head : nullptr;

    extern __shared__ __align__(16) unsigned char shared_memory[];
    const auto tiles = DecodeTiles<scalar_t, PAGE_SIZE, HEAD_DIM>::carve(shared_memory);
    const float* Mrow = tiles.Mrow;
    const float* Lrow = tiles.Lrow;
    float* LSE = tiles.LSE;
    float out[attn::pv_slabs_per_warp<HEAD_DIM, NWARPS>() * attn::kPvSlabAccums] = {};

    begin_group<scalar_t, HEAD_DIM, NWARPS>(p, s, g, tiles.sQ, tiles.Mrow, tiles.Lrow);
    __syncthreads();
    tc_attend_tiles<scalar_t, PAGE_SIZE, HEAD_DIM, BF16, NWARPS>(
        p,
        s,
        g,
        ctx_len,
        first,
        attn::key_layout(p.rope_layout, s),
        seq_scores,
        first / PAGE_SIZE,
        ntiles,
        tiles,
        out
    );

    const int gid = (tid & 31) >> 2;
    const float inv_low = 1.0f / Lrow[gid], inv_high = 1.0f / Lrow[gid + 8];
    scalar_t* o_group = p.o + (static_cast<int64_t>(s) * p.n_q_heads + g * p.group) * HEAD_DIM;
    attn::for_each_pv_register<HEAD_DIM, NWARPS>(out, tid >> 5, [&](int row, int col, float value) {
        if (row < p.group) {
            o_group[row * HEAD_DIM + col] = static_cast<scalar_t>(value * (row < 8 ? inv_low : inv_high));
        }
    });
    if (tid < p.group) {
        const int m = tid;
        const int h = g * p.group + m;
        LSE[m] = Mrow[m] + logf(Lrow[m]);
        if (p.lse) {
            p.lse[static_cast<int64_t>(s) * p.n_q_heads + h] = LSE[m];
        }
    }
    __syncthreads();

    if (p.mass) {
        accumulate_group_mass(p, PAGE_SIZE, s, g, ctx_len, seq_scores, LSE);
    }
}

template <typename scalar_t, int PAGE_SIZE, int HEAD_DIM, bool BF16, int NWARPS>
__global__ void __launch_bounds__(NWARPS * 32) paged_decode_split_kernel(
    const PagedAttnParams<scalar_t> p,
    float* __restrict__ scores,
    int64_t scores_stride_seq,
    int64_t scores_stride_head,
    float* __restrict__ o_partial,
    float* __restrict__ m_partial,
    float* __restrict__ l_partial,
    int nsplits,
    int tiles_per_split
) {
    const int tid = threadIdx.x;
    const int split = blockIdx.x % nsplits;
    const int sg = blockIdx.x / nsplits;
    const int g = sg % p.n_kv_heads;
    const int s = sg / p.n_kv_heads;
    const int ctx_len = p.seqlens_k[s];
    const int first = attn::first_key(p, ctx_len - 1);
    float* seq_scores = p.mass ? scores + s * scores_stride_seq + g * scores_stride_head : nullptr;

    extern __shared__ __align__(16) unsigned char shared_memory[];
    const auto tiles = DecodeTiles<scalar_t, PAGE_SIZE, HEAD_DIM>::carve(shared_memory);
    const float* Mrow = tiles.Mrow;
    const float* Lrow = tiles.Lrow;
    float out[attn::pv_slabs_per_warp<HEAD_DIM, NWARPS>() * attn::kPvSlabAccums] = {};

    begin_group<scalar_t, HEAD_DIM, NWARPS>(p, s, g, tiles.sQ, tiles.Mrow, tiles.Lrow);
    __syncthreads();
    const int t_start = first / PAGE_SIZE + split * tiles_per_split;
    tc_attend_tiles<scalar_t, PAGE_SIZE, HEAD_DIM, BF16, NWARPS>(
        p,
        s,
        g,
        ctx_len,
        first,
        attn::key_layout(p.rope_layout, s),
        seq_scores,
        t_start,
        t_start + tiles_per_split,
        tiles,
        out
    );
    __syncthreads();

    // An empty slice keeps Mrow = -inf, Lrow = 0, out = 0, which the combine ignores.
    const int64_t base = (static_cast<int64_t>(sg) * nsplits + split) * p.group;
    if (tid < p.group) {
        m_partial[base + tid] = Mrow[tid];
        l_partial[base + tid] = Lrow[tid];
    }
    float* o_slice = o_partial + base * HEAD_DIM;
    attn::for_each_pv_register<HEAD_DIM, NWARPS>(out, tid >> 5, [&](int row, int col, float value) {
        if (row < p.group) {
            o_slice[row * HEAD_DIM + col] = value;
        }
    });
}

// Combine the nsplits partials per query row (flash rescale) into normalized o and
// LSE. One CTA per (seq, kv head).
template <typename scalar_t>
__global__ void paged_decode_combine_kernel(
    const PagedAttnParams<scalar_t> p,
    float* __restrict__ lse_out,
    const float* __restrict__ o_partial,
    const float* __restrict__ m_partial,
    const float* __restrict__ l_partial,
    int nsplits
) {
    const int tid = threadIdx.x;
    const int g = blockIdx.x % p.n_kv_heads;
    const int s = blockIdx.x / p.n_kv_heads;
    const int64_t sg = static_cast<int64_t>(s) * p.n_kv_heads + g;
    const int group = p.group;
    const int head_dim = p.head_dim;
    const float* mp = m_partial + sg * nsplits * group;
    const float* lp = l_partial + sg * nsplits * group;
    const float* op = o_partial + sg * nsplits * group * head_dim;

    for (int m = 0; m < group; ++m) {
        float M = -CUDART_INF_F;
        for (int i = 0; i < nsplits; ++i) {
            M = fmaxf(M, mp[i * group + m]);
        }
        float l = 0.0f;
        for (int i = 0; i < nsplits; ++i) {
            const float mi = mp[i * group + m];
            if (mi > -CUDART_INF_F) {
                l += lp[i * group + m] * __expf(mi - M);
            }
        }
        const bool bad = (l == 0.0f) || !isfinite(M);
        const float invl = bad ? 0.0f : 1.0f / l;

        const int h = g * group + m;
        scalar_t* orow = p.o + (static_cast<int64_t>(s) * p.n_q_heads + h) * head_dim;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            if (!bad) {
                for (int i = 0; i < nsplits; ++i) {
                    const float mi = mp[i * group + m];
                    if (mi > -CUDART_INF_F) {
                        acc += op[(static_cast<int64_t>(i) * group + m) * head_dim + d] * __expf(mi - M);
                    }
                }
            }
            orow[d] = static_cast<scalar_t>(acc * invl);
        }
        if (tid == 0) {
            lse_out[sg * group + m] = bad ? -CUDART_INF_F : (M + logf(l));
        }
    }
}

template <typename scalar_t>
__global__ void paged_decode_mass_kernel(
    const PagedAttnParams<scalar_t> p,
    int page_size,
    const float* __restrict__ scores,
    int64_t scores_stride_seq,
    int64_t scores_stride_head,
    const float* __restrict__ lse_out
) {
    const int g = blockIdx.x % p.n_kv_heads;
    const int s = blockIdx.x / p.n_kv_heads;
    const float* seq_scores = scores + s * scores_stride_seq + g * scores_stride_head;
    const float* lse = lse_out + (static_cast<int64_t>(s) * p.n_kv_heads + g) * p.group;
    accumulate_group_mass(p, page_size, s, g, p.seqlens_k[s], seq_scores, lse);
}

// Split count targeting ~2 CTAs per SM. A split must own at least two tiles, so the
// count is capped at ceil(max_pages / 2), and at 32 overall.
int auto_splits(int64_t num_seqs, int64_t n_kv_heads, int64_t max_pages) {
    const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int64_t base_ctas = num_seqs * n_kv_heads;
    int64_t splits = (2LL * sm_count + base_ctas - 1) / base_ctas;
    splits = std::min(splits, std::max<int64_t>((max_pages + 1) / 2, 1));
    return static_cast<int>(std::clamp<int64_t>(splits, 1, 32));
}

template <typename scalar_t, bool BF16>
void launch_tc_decode(const attn::PagedAttnBatch& b, int64_t forced_splits, cudaStream_t stream) {
    const PagedAttnParams<scalar_t> p = b.params<scalar_t>();
    const int64_t group = b.n_q_heads / b.n_kv_heads;
    const int64_t base_ctas = b.num_seqs * b.n_kv_heads;
    auto fopts = b.q.options().dtype(at::kFloat);

    // Pass-1 scaled scores, keyed by (seq, kv head, view key, query row); sized by the
    // batch's view capacity, not the pool.
    const int64_t kv_capacity = b.max_pages * b.page_size;
    const at::Tensor scores = b.mass.defined() ? at::empty({b.num_seqs, b.n_kv_heads, kv_capacity, group}, fopts)
                                               : at::Tensor{};
    float* scores_ptr = scores.defined() ? scores.data_ptr<float>() : nullptr;
    const int64_t stride_head = kv_capacity * group;
    const int64_t stride_seq = b.n_kv_heads * stride_head;

    const int64_t attended_pages = b.sliding_window > 0
        ? std::min(b.max_pages, window_page_span(b.sliding_window, b.page_size))
        : b.max_pages;
    const int nsplits = forced_splits > 0 ? static_cast<int>(forced_splits)
                                          : auto_splits(b.num_seqs, b.n_kv_heads, attended_pages);
    const bool split = forced_splits > 0 || nsplits > 1;

    if (!split) {
        attn::dispatch_page_head("attn_decode", b.page_size, b.head_dim, [&](auto bs_tag, auto hd_tag) {
            constexpr int BS = decltype(bs_tag)::value;
            constexpr int HD = decltype(hd_tag)::value;
            attn::dispatch_attn_tuning([&](auto row_tag) {
                constexpr int NWARPS = attn::kAttnTuningTable[decltype(row_tag)::value].decode_warps;
                constexpr size_t smem = DecodeTiles<scalar_t, BS, HD>::bytes();
                const auto kernel = attn_decode_tc_kernel<scalar_t, BS, HD, BF16, NWARPS>;
                attn::allow_shared_memory(kernel, smem);
                kernel<<<static_cast<int>(base_ctas), NWARPS * 32, smem, stream>>>(
                    p,
                    scores_ptr,
                    stride_seq,
                    stride_head
                );
            });
        });
        return;
    }

    const int tiles_per_split = static_cast<int>((attended_pages + nsplits - 1) / nsplits);
    auto o_partial = at::empty({b.num_seqs, b.n_kv_heads, nsplits, group, b.head_dim}, fopts);
    auto m_partial = at::empty({b.num_seqs, b.n_kv_heads, nsplits, group}, fopts);
    auto l_partial = at::empty({b.num_seqs, b.n_kv_heads, nsplits, group}, fopts);
    // The combine kernel's [num_seqs, n_kv_heads, group] layout flattens to
    // [num_seqs, n_q_heads], so a caller-provided capture is written directly.
    auto lse_out = b.lse.defined() ? b.lse.view({b.num_seqs, b.n_kv_heads, group})
                                   : at::empty({b.num_seqs, b.n_kv_heads, group}, fopts);

    attn::dispatch_page_head("attn_decode", b.page_size, b.head_dim, [&](auto bs_tag, auto hd_tag) {
        constexpr int BS = decltype(bs_tag)::value;
        constexpr int HD = decltype(hd_tag)::value;
        attn::dispatch_attn_tuning([&](auto row_tag) {
            constexpr int NWARPS = attn::kAttnTuningTable[decltype(row_tag)::value].decode_warps;
            constexpr size_t smem = DecodeTiles<scalar_t, BS, HD>::bytes();
            const auto kernel = paged_decode_split_kernel<scalar_t, BS, HD, BF16, NWARPS>;
            attn::allow_shared_memory(kernel, smem);
            kernel<<<static_cast<int>(base_ctas * nsplits), NWARPS * 32, smem, stream>>>(
                p,
                scores_ptr,
                stride_seq,
                stride_head,
                o_partial.data_ptr<float>(),
                m_partial.data_ptr<float>(),
                l_partial.data_ptr<float>(),
                nsplits,
                tiles_per_split
            );
        });
    });
    paged_decode_combine_kernel<scalar_t><<<static_cast<int>(base_ctas), kThreads, 0, stream>>>(
        p,
        lse_out.data_ptr<float>(),
        o_partial.data_ptr<float>(),
        m_partial.data_ptr<float>(),
        l_partial.data_ptr<float>(),
        nsplits
    );
    if (scores_ptr) {
        paged_decode_mass_kernel<scalar_t><<<static_cast<int>(base_ctas), kThreads, 0, stream>>>(
            p,
            static_cast<int>(b.page_size),
            scores_ptr,
            stride_seq,
            stride_head,
            lse_out.data_ptr<float>()
        );
    }
}

at::Tensor paged_decode(const attn::PagedAttnBatch& b, bool force_scalar, int64_t num_splits) {
    TORCH_CHECK(num_splits >= 0, "attn_decode: num_splits must be >= 0");
    if (b.num_seqs == 0) {
        return b.o;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    // The group's query heads share the 16 MMA rows of one tile.
    bool tensor_core = !force_scalar && b.tensor_core_eligible() && b.n_q_heads / b.n_kv_heads <= 16;
    if (tensor_core) {
        attn::dispatch_page_head("attn_decode", b.page_size, b.head_dim, [&](auto bs_tag, auto hd_tag) {
            constexpr int BS = decltype(bs_tag)::value;
            constexpr int HD = decltype(hd_tag)::value;
            tensor_core = attn::fits_shared_memory(DecodeTiles<at::Half, BS, HD>::bytes());
        });
    }
    if (!tensor_core) {
        attn::launch_attn_scalar(b, stream);
        return b.o;
    }
    attn::dispatch_tensor_core_dtype(b.q.scalar_type(), [&](auto type_tag, auto bf16_tag) {
        using scalar_t = typename decltype(type_tag)::type;
        launch_tc_decode<scalar_t, decltype(bf16_tag)::value>(b, num_splits, stream);
    });
    return b.o;
}

}  // namespace

void write_kv_cuda(
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& k_new,
    const at::Tensor& v_new,
    const at::Tensor& page_tables,
    const std::optional<at::Tensor>& cu_seqlens_q,
    const at::Tensor& seqlens_k
) {
    TORCH_CHECK(
        k_pool.is_cuda() && v_pool.is_cuda() && k_new.is_cuda() && v_new.is_cuda() && page_tables.is_cuda() &&
            seqlens_k.is_cuda() && (!cu_seqlens_q || cu_seqlens_q->is_cuda()),
        "write_kv: all inputs must be CUDA tensors"
    );
    TORCH_CHECK(
        k_pool.scalar_type() == v_pool.scalar_type() && k_pool.scalar_type() == k_new.scalar_type() &&
            k_new.scalar_type() == v_new.scalar_type(),
        "write_kv: k_pool, v_pool, k_new, v_new must share a dtype"
    );
    TORCH_CHECK(
        k_pool.dim() == 4 && k_pool.sizes() == v_pool.sizes(),
        "write_kv: pools must be 4-D [num_lines, page_size, n_kv_heads, head_dim] of one shape"
    );
    TORCH_CHECK(
        k_pool.is_contiguous() && v_pool.is_contiguous(),
        "write_kv: pools must be contiguous (mutated in place)"
    );
    const int64_t page_size = k_pool.size(1);
    const int64_t n_kv_heads = k_pool.size(2);
    const int64_t head_dim = k_pool.size(3);
    const int64_t num_new_tokens = k_new.size(0);
    TORCH_CHECK(
        k_new.dim() == 3 && k_new.sizes() == v_new.sizes() && k_new.size(1) == n_kv_heads && k_new.size(2) == head_dim,
        "write_kv: k_new, v_new must be [num_new_tokens, n_kv_heads, head_dim] of the pool's heads"
    );
    TORCH_CHECK(
        page_tables.scalar_type() == at::kInt && page_tables.dim() == 2,
        "write_kv: page_tables must be int32 [num_seqs, max_pages]"
    );
    const int64_t num_seqs = page_tables.size(0);
    TORCH_CHECK(
        seqlens_k.scalar_type() == at::kInt && seqlens_k.dim() == 1 && seqlens_k.size(0) == num_seqs,
        "write_kv: seqlens_k must be int32 [num_seqs]"
    );
    if (cu_seqlens_q) {
        TORCH_CHECK(
            cu_seqlens_q->scalar_type() == at::kInt && cu_seqlens_q->dim() == 1 &&
                cu_seqlens_q->size(0) == num_seqs + 1,
            "write_kv: cu_seqlens_q must be int32 [num_seqs + 1]"
        );
    } else {
        TORCH_CHECK(num_new_tokens == num_seqs, "write_kv: one new token per sequence without cu_seqlens_q");
    }
    if (num_new_tokens == 0) {
        return;
    }

    auto kn = k_new.contiguous();
    auto vn = v_new.contiguous();
    auto tables = page_tables.contiguous();
    auto seqlens = seqlens_k.contiguous();
    auto cu = cu_seqlens_q ? cu_seqlens_q->contiguous() : at::Tensor();

    const int threads = static_cast<int>(std::min<int64_t>(256, n_kv_heads * head_dim));
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, k_pool.scalar_type(), "write_kv_cuda", [&] {
        attn::dispatch_page_size("write_kv", page_size, [&](auto page_size_tag) {
            constexpr int BS = decltype(page_size_tag)::value;
            write_kv_kernel<scalar_t, BS><<<static_cast<int>(num_new_tokens), threads, 0, stream>>>(
                k_pool.data_ptr<scalar_t>(),
                v_pool.data_ptr<scalar_t>(),
                kn.data_ptr<scalar_t>(),
                vn.data_ptr<scalar_t>(),
                tables.data_ptr<int32_t>(),
                cu.defined() ? cu.data_ptr<int32_t>() : nullptr,
                seqlens.data_ptr<int32_t>(),
                static_cast<int>(num_seqs),
                static_cast<int>(tables.size(1)),
                static_cast<int>(n_kv_heads),
                static_cast<int>(head_dim)
            );
        });
    });
}

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
    const std::optional<at::Tensor>& mass,
    const std::optional<at::Tensor>& cu_view_pages,
    const std::optional<at::Tensor>& attention_mass_decay,
    double mass_length_gain,
    const std::optional<at::Tensor>& lse_capture,
    int64_t num_splits
) {
    const auto batch = attn::paged_attn_batch(
        "attn_decode",
        q,
        k_pool,
        v_pool,
        page_tables,
        std::nullopt,
        context_lens,
        rope_layout,
        rope,
        scale,
        sliding_window,
        k_norm_weight,
        mass,
        cu_view_pages,
        attention_mass_decay,
        mass_length_gain,
        lse_capture
    );
    return paged_decode(batch, /*force_scalar=*/false, num_splits);
}

at::Tensor attn_decode_scalar_cuda(
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
    const std::optional<at::Tensor>& mass,
    const std::optional<at::Tensor>& cu_view_pages,
    const std::optional<at::Tensor>& attention_mass_decay,
    double mass_length_gain,
    const std::optional<at::Tensor>& lse_capture
) {
    const auto batch = attn::paged_attn_batch(
        "attn_decode_scalar",
        q,
        k_pool,
        v_pool,
        page_tables,
        std::nullopt,
        context_lens,
        rope_layout,
        rope,
        scale,
        sliding_window,
        k_norm_weight,
        mass,
        cu_view_pages,
        attention_mass_decay,
        mass_length_gain,
        lse_capture
    );
    return paged_decode(batch, /*force_scalar=*/true, 0);
}

}  // namespace pulsarlm
