#include "attn_common.cuh"

#include "attn_tiles.cuh"
#include "attn_tuning.hpp"
#include "pulsarlm/ops.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <math_constants.h>

#include <algorithm>
#include <cstdint>
#include <limits>

// Paged-KV decode attention, and the scatter that writes new K/V into the pool. One
// query per sequence attends its whole view, or the window of it sliding_window
// leaves.
//
// The tensor-core path runs one warp group per (sequence s, kv head g): the `group`
// query heads sharing g occupy MMA rows 0..group-1 (rows group..15 are zero and
// discarded), and the view is streamed a key tile (decode_key_tile keys, independent of
// the page size) at a time, each key rotated to its position once staged. Online
// softmax runs per query row across the CTA between the two MMAs, the running output
// stays in the registers of the warps that own its dim slabs, and pass 1 stashes every
// scaled score so the mass pass never re-reads K. The QKᵀ/PV tiles and their fragment layout live in attn_tiles.cuh.
//
// The split (flash-decode) path spreads one (s, g)'s view over nsplits CTAs:
//   split kernel   : grid num_seqs * n_kv_heads * nsplits; each CTA runs the online
//                    softmax over one slice and writes its RAW state.
//   combine kernel : grid num_seqs * n_q_heads; reduces one query row's partials
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
    int head_dim,
    int64_t page_stride,
    int64_t token_stride,
    int64_t head_stride
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
    const int64_t token = static_cast<int64_t>(line) * page_stride + (index % PAGE_SIZE) * token_stride;
    const int64_t src = static_cast<int64_t>(t) * n;

    for (int64_t e = threadIdx.x; e < n; e += blockDim.x) {
        const int64_t dst = token + e / head_dim * head_stride + e % head_dim;
        k_pool[dst] = k_new[src + e];
        v_pool[dst] = v_new[src + e];
    }
}

// Keys one staged tile holds: 4096 / head_dim, at least one MMA's 16, which keeps a CTA
// within the 48 KiB of shared memory every device grants without opting in up to head
// dim 256.
constexpr int decode_key_tile(int head_dim) {
    return 4096 / head_dim > 16 ? 4096 / head_dim : 16;
}

// Every tile size is a multiple of 16 bytes, which keeps each tile 16-byte aligned for
// ldmatrix. sQ and the ring's tiles are staged through attn::swizzled_offset.
template <typename scalar_t, int HEAD_DIM, int NWARPS> struct DecodeTiles {
    static constexpr int KEY_TILE = decode_key_tile(HEAD_DIM);
    static constexpr int QKT_PARTS = attn::qkt_parts<KEY_TILE, NWARPS>();
    static constexpr int TILE_ELEMS = KEY_TILE * HEAD_DIM;

    scalar_t* sQ;  // [16, HEAD_DIM]
    scalar_t* ring;  // [2][K, V][KEY_TILE, HEAD_DIM]
    scalar_t* sP;  // [16, KEY_TILE]
    float* sS;  // [QKT_PARTS][16, KEY_TILE]
    float* Mrow;  // [16]
    float* Lrow;  // [16]
    float* row_rescale;  // [16]
    float* LSE;  // [16]

    static constexpr size_t bytes() {
        return sizeof(scalar_t) * (16 * HEAD_DIM + 2 * 2 * TILE_ELEMS + 16 * KEY_TILE) +
            sizeof(float) * (QKT_PARTS * 16 * KEY_TILE + 4 * 16);
    }

    __device__ static DecodeTiles carve(unsigned char* base) {
        DecodeTiles tiles;
        tiles.sQ = reinterpret_cast<scalar_t*>(base);
        tiles.ring = tiles.sQ + 16 * HEAD_DIM;
        tiles.sP = tiles.ring + 2 * 2 * TILE_ELEMS;
        tiles.sS = reinterpret_cast<float*>(tiles.sP + 16 * KEY_TILE);
        tiles.Mrow = tiles.sS + QKT_PARTS * 16 * KEY_TILE;
        tiles.Lrow = tiles.Mrow + 16;
        tiles.row_rescale = tiles.Lrow + 16;
        tiles.LSE = tiles.row_rescale + 16;
        return tiles;
    }

    __device__ scalar_t* keys(int slot) const {
        return ring + slot * 2 * TILE_ELEMS;
    }

    __device__ scalar_t* values(int slot) const {
        return ring + slot * 2 * TILE_ELEMS + TILE_ELEMS;
    }
};

// Tiles of tile_keys keys a sliding window can touch when it does not start on a tile boundary.
int64_t window_tile_span(int64_t sliding_window, int64_t tile_keys) {
    return (sliding_window - 1 + tile_keys - 1) / tile_keys + 1;
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
        sQ[attn::swizzled_offset<HEAD_DIM>(row, col)] = v;
    }
    if (tid < 16) {
        Mrow[tid] = -CUDART_INF_F;
        Lrow[tid] = 0.0f;
    }
}

// Online-softmax attention over view tiles [t_start, t_end) of KEY_TILE keys for one
// (seq s, kv head g), keys from index first on. Tiles wholly before first are not read.
// Accumulates the RAW, UNNORMALIZED state: out (this lane's tile_pv_registers share of
// the sum of exp(scale*q.k - Mrow)*V per query row), Mrow (running row max), Lrow
// (running denominator); the caller owns normalization and LSE. When seq_scores is
// non-null, each scaled score is stashed to seq_scores[key * group + m] for the mass
// pass. A slice past the view leaves out = 0, Mrow = -inf, Lrow = 0.
//
// Raw K and V tiles stream through two slots each by cp.async: a tile's keys are
// staged once QKᵀ frees their slot, two tiles ahead, and its values once PV frees
// theirs, one tile ahead. Each thread rotates the key pairs it copied itself once its
// own copies land, so the barrier that follows publishes rotated keys. Pools that are
// one tensor hold K = V, which is then read once: the rotation writes the key tile from
// the staged values.
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
    const DecodeTiles<scalar_t, HEAD_DIM, NWARPS>& tiles,
    float* out
) {
    using Tiles = DecodeTiles<scalar_t, HEAD_DIM, NWARPS>;
    constexpr int KEY_TILE = Tiles::KEY_TILE;
    constexpr int THREADS = NWARPS * 32;
    constexpr int HALF = HEAD_DIM / 2;
    static_assert(sizeof(scalar_t) == 2, "the tile staging moves 8 elements at a time");
    constexpr int kChunkElems = 16 / sizeof(scalar_t);
    constexpr int kPairChunks = KEY_TILE * HALF / kChunkElems;
    // The softmax gives each of the 16 query rows ROW_LANES consecutive threads.
    constexpr int ROW_LANES = THREADS / 16;
    static_assert(ROW_LANES <= 32 && KEY_TILE % ROW_LANES == 0, "a row's lanes must share a warp and split the tile");
    constexpr int kRowCols = KEY_TILE / ROW_LANES;

    scalar_t* sP = tiles.sP;
    float* sS = tiles.sS;
    float* Mrow = tiles.Mrow;
    float* Lrow = tiles.Lrow;
    float* row_rescale = tiles.row_rescale;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int32_t* seq_table = p.page_tables + static_cast<int64_t>(s) * p.max_pages;
    const bool keys_are_values = p.k_pool == p.v_pool;
    const int t_first = max(t_start, first / KEY_TILE);
    const int t_last = min(t_end, (ctx_len + KEY_TILE - 1) / KEY_TILE);

    auto attended = [&](int key) { return key >= first && key < ctx_len; };
    // Chunk c is the 16-byte chunks at elements d and d + HALF of one key, so the thread that copies a key's
    // halves is the one that rotates them. Always commits a group, so a wait count means the same tiles throughout.
    auto stage = [&](const scalar_t* pool, scalar_t* dst, int t) {
        if (t < t_last) {
            for (int c = tid; c < kPairChunks; c += THREADS) {
                const int off = c / (HALF / kChunkElems);
                const int d = (c % (HALF / kChunkElems)) * kChunkElems;
                const int key = t * KEY_TILE + off;
                const bool copied = attended(key);
                const int64_t base = copied ? static_cast<int64_t>(seq_table[key / PAGE_SIZE]) * p.page_stride +
                        g * p.head_stride + (key % PAGE_SIZE) * p.token_stride
                                            : 0;
                const int bytes = copied ? 16 : 0;
                attn::cp_async_16(dst + attn::swizzled_offset<HEAD_DIM>(off, d), pool + base + d, bytes);
                attn::cp_async_16(dst + attn::swizzled_offset<HEAD_DIM>(off, d + HALF), pool + base + d + HALF, bytes);
            }
        }
        attn::cp_async_commit();
    };
    auto stage_keys = [&](int t) { stage(p.k_pool, tiles.keys((t - t_first) % 2), keys_are_values ? t_last : t); };
    auto stage_values = [&](int t) { stage(p.v_pool, tiles.values((t - t_first) % 2), t); };
    // A thread's chunks all sit at the same pairs, so their RoPE turns and k_norm weights are read once.
    static_assert(THREADS % (HALF / kChunkElems) == 0, "every chunk of a thread must hold the same pairs");
    const int pair0 = (tid % (HALF / kChunkElems)) * kChunkElems;
    uint64_t turns[kChunkElems];
    float lo_weight[kChunkElems];
    float hi_weight[kChunkElems];
#pragma unroll
    for (int j = 0; j < kChunkElems; ++j) {
        turns[j] = p.rope_turns[pair0 + j];
        lo_weight[j] = p.k_norm_weight ? p.k_norm_weight[pair0 + j] : 1.0f;
        hi_weight[j] = p.k_norm_weight ? p.k_norm_weight[pair0 + j + HALF] : 1.0f;
    }
    auto rotate = [&](int t) {
        const int slot = (t - t_first) % 2;
        scalar_t* sK = tiles.keys(slot);
        const scalar_t* raw = keys_are_values ? tiles.values(slot) : sK;
        for (int c = tid; c < kPairChunks; c += THREADS) {
            const int off = c / (HALF / kChunkElems);
            const int key = t * KEY_TILE + off;
            const int lo_at = attn::swizzled_offset<HEAD_DIM>(off, pair0);
            const int hi_at = attn::swizzled_offset<HEAD_DIM>(off, pair0 + HALF);
            alignas(16) scalar_t lo[kChunkElems];
            alignas(16) scalar_t hi[kChunkElems];
            *reinterpret_cast<uint4*>(lo) = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(hi) = make_uint4(0, 0, 0, 0);
            if (attended(key)) {
                *reinterpret_cast<uint4*>(lo) = *reinterpret_cast<const uint4*>(raw + lo_at);
                *reinterpret_cast<uint4*>(hi) = *reinterpret_cast<const uint4*>(raw + hi_at);
                const int position = layout.position(key);
#pragma unroll
                for (int j = 0; j < kChunkElems; ++j) {
                    float cos_a, sin_a;
                    attn::rope_sincos_turns(position, turns[j], cos_a, sin_a);
                    const auto k = attn::rope_rotate<scalar_t>(
                        __fmul_rn(static_cast<float>(lo[j]), lo_weight[j]),
                        __fmul_rn(static_cast<float>(hi[j]), hi_weight[j]),
                        cos_a,
                        sin_a
                    );
                    lo[j] = k.lo;
                    hi[j] = k.hi;
                }
            }
            *reinterpret_cast<uint4*>(sK + lo_at) = *reinterpret_cast<const uint4*>(lo);
            *reinterpret_cast<uint4*>(sK + hi_at) = *reinterpret_cast<const uint4*>(hi);
        }
    };

    stage_keys(t_first);
    stage_values(t_first);
    stage_keys(t_first + 1);
    const int row = tid / ROW_LANES;
    const int row_lane = tid % ROW_LANES;
    for (int t = t_first; t < t_last; ++t) {
        const int slot = (t - t_first) % 2;
        const int tile_base = t * KEY_TILE;
        // Only the keys of tile t + 1 may still be in flight.
        attn::cp_async_wait<1>();
        rotate(t);
        // Also ends the previous tile's PV, which frees the value slot of tile t + 1.
        __syncthreads();
        stage_values(t + 1);

        attn::tile_qkt_parts<scalar_t, KEY_TILE, HEAD_DIM, BF16, NWARPS>(tiles.sQ, tiles.keys(slot), sS, warp);
        // Also frees this tile's key slot for tile t + 2.
        __syncthreads();
        stage_keys(t + 2);

        // A tile in [t_first, t_last) holds an attended key, so newM is finite.
        float scores[kRowCols];
        float tmax = -CUDART_INF_F;
#pragma unroll
        for (int j = 0; j < kRowCols; ++j) {
            const int c = j * ROW_LANES + row_lane;
            float raw_score = 0.0f;
#pragma unroll
            for (int part = 0; part < Tiles::QKT_PARTS; ++part) {
                raw_score += sS[(part * 16 + row) * KEY_TILE + c];
            }
            scores[j] = attended(tile_base + c) ? p.scale * raw_score : -CUDART_INF_F;
            tmax = fmaxf(tmax, scores[j]);
        }
#pragma unroll
        for (int lanes = ROW_LANES / 2; lanes > 0; lanes >>= 1) {
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, lanes));
        }
        const float oldM = Mrow[row];
        const float newM = fmaxf(oldM, tmax);
        float lsum = 0.0f;
#pragma unroll
        for (int j = 0; j < kRowCols; ++j) {
            const int c = j * ROW_LANES + row_lane;
            const bool kept = scores[j] > -CUDART_INF_F;
            const float pw = kept ? __expf(scores[j] - newM) : 0.0f;
            if (kept && seq_scores && row < p.group) {
                seq_scores[static_cast<int64_t>(tile_base + c) * p.group + row] = scores[j];
            }
            sP[row * KEY_TILE + c] = static_cast<scalar_t>(pw);
            lsum += pw;
        }
#pragma unroll
        for (int lanes = ROW_LANES / 2; lanes > 0; lanes >>= 1) {
            lsum += __shfl_xor_sync(0xffffffffu, lsum, lanes);
        }
        if (row_lane == 0) {
            const float corr = __expf(oldM - newM);  // 0 when oldM == -inf
            row_rescale[row] = corr;
            Mrow[row] = newM;
            Lrow[row] = Lrow[row] * corr + lsum;
        }
        __syncthreads();

        attn::tile_pv_registers<scalar_t, KEY_TILE, HEAD_DIM, BF16, NWARPS, true>(
            sP,
            tiles.values(slot),
            row_rescale,
            out,
            warp
        );
    }
}

// Add each attended key's normalized weight exp(score - lse[m]), times the sequence's EMA rate, into query head
// g*group + m's column of the key's mass row. A sequence has one query, so no other thread of the launch writes the
// same slot.
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
    const float rate = p.mass_decay[s];
    for (int j = first + static_cast<int>(threadIdx.x); j < ctx_len; j += blockDim.x) {
        const float* col = seq_scores + static_cast<int64_t>(j) * p.group;
        float* row = mass + static_cast<int64_t>(j) * p.n_q_heads + g * p.group;
        for (int m = 0; m < p.group; ++m) {
            row[m] += rate * __expf(col[m] - lse[m]);
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
    const int first = attn::first_key(p, ctx_len - 1);
    float* seq_scores = p.mass ? scores + s * scores_stride_seq + g * scores_stride_head : nullptr;

    extern __shared__ __align__(16) unsigned char shared_memory[];
    const auto tiles = DecodeTiles<scalar_t, HEAD_DIM, NWARPS>::carve(shared_memory);
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
        0,
        std::numeric_limits<int>::max(),
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
        LSE[tid] = Mrow[tid] + logf(Lrow[tid]);
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
    const auto tiles = DecodeTiles<scalar_t, HEAD_DIM, NWARPS>::carve(shared_memory);
    const float* Mrow = tiles.Mrow;
    const float* Lrow = tiles.Lrow;
    float out[attn::pv_slabs_per_warp<HEAD_DIM, NWARPS>() * attn::kPvSlabAccums] = {};

    begin_group<scalar_t, HEAD_DIM, NWARPS>(p, s, g, tiles.sQ, tiles.Mrow, tiles.Lrow);
    __syncthreads();
    const int t_start = first / DecodeTiles<scalar_t, HEAD_DIM, NWARPS>::KEY_TILE + split * tiles_per_split;
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

// One CTA per (seq, query head). Shared memory: kThreads reduction floats, then one
// weight per split.
template <typename scalar_t>
__global__ void paged_decode_combine_kernel(
    const PagedAttnParams<scalar_t> p,
    float* __restrict__ lse_out,
    const float* __restrict__ o_partial,
    const float* __restrict__ m_partial,
    const float* __restrict__ l_partial,
    int nsplits
) {
    extern __shared__ float combine_shared[];
    float* scratch = combine_shared;
    float* weight = combine_shared + kThreads;
    const int tid = threadIdx.x;
    const int h = blockIdx.x % p.n_q_heads;
    const int s = blockIdx.x / p.n_q_heads;
    const int group = p.group;
    const int m = h % group;
    const int64_t row = (static_cast<int64_t>(s) * p.n_kv_heads + h / group) * nsplits * group + m;
    const float* mp = m_partial + row;
    const float* lp = l_partial + row;
    const float* op = o_partial + row * p.head_dim;

    float local_max = -CUDART_INF_F;
    for (int i = tid; i < nsplits; i += kThreads) {
        local_max = fmaxf(local_max, mp[i * group]);
    }
    const float M = attn::block_reduce_max<kThreads>(scratch, tid, local_max);
    float local_l = 0.0f;
    for (int i = tid; i < nsplits; i += kThreads) {
        const float mi = mp[i * group];
        weight[i] = mi > -CUDART_INF_F ? __expf(mi - M) : 0.0f;
        local_l += lp[i * group] * weight[i];
    }
    const float l = attn::block_reduce_sum<kThreads>(scratch, tid, local_l);
    const bool bad = (l == 0.0f) || !isfinite(M);
    const float invl = bad ? 0.0f : 1.0f / l;

    scalar_t* orow = p.o + (static_cast<int64_t>(s) * p.n_q_heads + h) * p.head_dim;
    const int64_t split_stride = static_cast<int64_t>(group) * p.head_dim;
    for (int d = tid; d < p.head_dim; d += kThreads) {
        float acc = 0.0f;
        if (!bad) {
            for (int i = 0; i < nsplits; ++i) {
                acc += op[i * split_stride + d] * weight[i];
            }
        }
        orow[d] = static_cast<scalar_t>(acc * invl);
    }
    if (tid == 0) {
        lse_out[static_cast<int64_t>(s) * p.n_q_heads + h] = bad ? -CUDART_INF_F : (M + logf(l));
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

// The split count with the least estimated time: the waves of resident CTAs the launch
// takes, each as long as a split's key tiles plus one tile for a CTA's fixed work. At
// most 128 splits, at least two tiles per split, no empty split; ties go to fewer splits.
int auto_splits(int64_t base_ctas, int64_t max_tiles, int64_t resident_ctas) {
    int64_t best_splits = 1;
    int64_t best_cost = std::numeric_limits<int64_t>::max();
    for (int64_t splits = 1; splits <= std::clamp<int64_t>((max_tiles + 1) / 2, 1, 128); ++splits) {
        const int64_t tiles_per_split = (max_tiles + splits - 1) / splits;
        const int64_t used = (max_tiles + tiles_per_split - 1) / tiles_per_split;
        const int64_t cost = (base_ctas * used + resident_ctas - 1) / resident_ctas * (tiles_per_split + 1);
        if (cost < best_cost) {
            best_cost = cost;
            best_splits = used;
        }
    }
    return static_cast<int>(best_splits);
}

template <typename scalar_t, bool BF16> int64_t resident_split_ctas(const attn::PagedAttnBatch& b) {
    int64_t resident = 0;
    attn::dispatch_page_head("attn_decode", b.page_size, b.head_dim, [&](auto bs_tag, auto hd_tag) {
        constexpr int BS = decltype(bs_tag)::value;
        constexpr int HD = decltype(hd_tag)::value;
        attn::dispatch_attn_tuning([&](auto row_tag) {
            constexpr int NWARPS = attn::kAttnTuningTable[decltype(row_tag)::value].decode_warps;
            constexpr size_t smem = DecodeTiles<scalar_t, HD, NWARPS>::bytes();
            static const int64_t kernel_resident = [] {
                const auto kernel = paged_decode_split_kernel<scalar_t, BS, HD, BF16, NWARPS>;
                attn::allow_shared_memory(kernel, smem);
                int per_sm = 0;
                C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, NWARPS * 32, smem));
                return static_cast<int64_t>(per_sm) * at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
            }();
            resident = kernel_resident;
        });
    });
    return resident;
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

    const int64_t key_tile = decode_key_tile(static_cast<int>(b.head_dim));
    const int64_t view_tiles = (kv_capacity + key_tile - 1) / key_tile;
    const int64_t attended_tiles = b.sliding_window > 0
        ? std::min(view_tiles, window_tile_span(b.sliding_window, key_tile))
        : view_tiles;
    const int nsplits = forced_splits > 0
        ? static_cast<int>(forced_splits)
        : auto_splits(base_ctas, attended_tiles, resident_split_ctas<scalar_t, BF16>(b));
    const bool split = forced_splits > 0 || nsplits > 1;

    if (!split) {
        attn::dispatch_page_head("attn_decode", b.page_size, b.head_dim, [&](auto bs_tag, auto hd_tag) {
            constexpr int BS = decltype(bs_tag)::value;
            constexpr int HD = decltype(hd_tag)::value;
            attn::dispatch_attn_tuning([&](auto row_tag) {
                constexpr int NWARPS = attn::kAttnTuningTable[decltype(row_tag)::value].decode_warps;
                constexpr size_t smem = DecodeTiles<scalar_t, HD, NWARPS>::bytes();
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

    const int tiles_per_split = static_cast<int>((attended_tiles + nsplits - 1) / nsplits);
    auto o_partial = at::empty({b.num_seqs, b.n_kv_heads, nsplits, group, b.head_dim}, fopts);
    auto m_partial = at::empty({b.num_seqs, b.n_kv_heads, nsplits, group}, fopts);
    auto l_partial = at::empty({b.num_seqs, b.n_kv_heads, nsplits, group}, fopts);
    auto lse_out = at::empty({b.num_seqs, b.n_kv_heads, group}, fopts);

    attn::dispatch_page_head("attn_decode", b.page_size, b.head_dim, [&](auto bs_tag, auto hd_tag) {
        constexpr int BS = decltype(bs_tag)::value;
        constexpr int HD = decltype(hd_tag)::value;
        attn::dispatch_attn_tuning([&](auto row_tag) {
            constexpr int NWARPS = attn::kAttnTuningTable[decltype(row_tag)::value].decode_warps;
            constexpr size_t smem = DecodeTiles<scalar_t, HD, NWARPS>::bytes();
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
    const size_t combine_smem = sizeof(float) * (kThreads + nsplits);
    paged_decode_combine_kernel<scalar_t>
        <<<static_cast<int>(b.num_seqs * b.n_q_heads), kThreads, combine_smem, stream>>>(
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

}  // namespace

bool attn::decode_fits(int64_t page_size, int64_t head_dim) {
    bool fits = false;
    attn::dispatch_page_head("attn_decode", page_size, head_dim, [&](auto, auto hd_tag) {
        attn::dispatch_attn_tuning([&](auto row_tag) {
            constexpr int NWARPS = attn::kAttnTuningTable[decltype(row_tag)::value].decode_warps;
            fits = attn::fits_shared_memory(DecodeTiles<at::Half, decltype(hd_tag)::value, NWARPS>::bytes());
        });
    });
    return fits;
}

namespace {

at::Tensor paged_decode(const attn::PagedAttnBatch& b, int64_t num_splits) {
    TORCH_CHECK(num_splits >= 0, "attn_decode: num_splits must be >= 0");
    if (b.num_seqs == 0) {
        return b.o;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    if (b.q.scalar_type() == at::kFloat) {
        attn::launch_attn_scalar(b, stream);
        return b.o;
    }
    std::optional<std::string> unsupported = attention_unsupported_reason(
        b.q.scalar_type(),
        b.page_size,
        b.head_dim,
        b.n_q_heads,
        b.n_kv_heads
    );
    TORCH_CHECK(!unsupported, "attn_decode: ", *unsupported);
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
    attn::check_pools("write_kv", k_pool, v_pool);
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
                static_cast<int>(head_dim),
                k_pool.stride(0),
                k_pool.stride(1),
                k_pool.stride(2)
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
        attention_mass_decay
    );
    return paged_decode(batch, num_splits);
}

}  // namespace pulsarlm
