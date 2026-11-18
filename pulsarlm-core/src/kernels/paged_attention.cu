#include "pulsarlm/kernels/dependent_launch.cuh"
#include "pulsarlm/kernels/attn_common.cuh"

#include "pulsarlm/kernels/attn_tiles.cuh"
#include "pulsarlm/kernels/attn_tuning.hpp"
#include "pulsarlm/kernels/ops.hpp"

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
// stays in the registers of the warps that own its dim slabs. When the op measures mass,
// each query row's max score over each view page, times the query's weight, is written to
// the page's mass_sum entry. The QKᵀ/PV tiles and their fragment layout live in
// attn_tiles.cuh.
//
// The split (flash-decode) path spreads one (s, g)'s view over nsplits CTAs, each
// holding whole pages:
//   split kernel   : grid num_seqs * n_kv_heads * nsplits; each CTA runs the online
//                    softmax over one slice and writes its RAW state and its pages'
//                    mass_sum entries.
//   combine kernel : grid num_seqs * n_q_heads; reduces one query row's partials
//                    with the flash rescale into o and LSE.
// Partial buffers (fp32), for (seq s, kv head g, split i, row m):
//   o_partial [num_seqs, n_kv_heads, nsplits, group, head_dim]
//   m_partial, l_partial [num_seqs, n_kv_heads, nsplits, group]
//   lse_out   [num_seqs, n_kv_heads, group]

namespace pulsarlm {
namespace {

using attn::key_positions;
using attn::cta_threads;
using attn::paged_attn_params;

// One block per token; the n_kv_heads*head_dim elements are strided across the
// block's threads. The pools may be one tensor.
template <typename scalar_t, int page_size>
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
    wait_for_prior_kernels();
    allow_next_kernel();
    const int t = blockIdx.x;
    int s = t;
    int index = 0;
    if (!cu_seqlens_q) {
        index = seqlens_k[t] - 1;
    } else {
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
    const int line = page_tables[static_cast<int64_t>(s) * max_pages + index / page_size];

    const int64_t n = static_cast<int64_t>(n_kv_heads) * head_dim;
    const int64_t token = static_cast<int64_t>(line) * page_stride + (index % page_size) * token_stride;
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

// Key tiles of one page, 1 when a tile holds whole pages.
constexpr int decode_tiles_per_page(int page_size, int head_dim) {
    return page_size > decode_key_tile(head_dim) ? page_size / decode_key_tile(head_dim) : 1;
}

// Every tile size is a multiple of 16 bytes, which keeps each tile 16-byte aligned for
// ldmatrix. sQ and the ring's tiles are staged through attn::swizzled_offset.
template <typename scalar_t, int head_dim, int nwarps> struct decode_tiles {
    static constexpr int key_tile = decode_key_tile(head_dim);
    static constexpr int qkt_parts = attn::qkt_parts<key_tile, nwarps>();
    static constexpr int tile_elems = key_tile * head_dim;

    scalar_t* sQ;  // [16, head_dim]
    scalar_t* ring;  // [2][K, V][key_tile, head_dim]
    scalar_t* sP;  // [16, key_tile]
    float* sS;  // [qkt_parts][16, key_tile]
    float* Mrow;  // [16]
    float* Lrow;  // [16]
    float* row_rescale;  // [16]

    static constexpr size_t bytes() {
        return sizeof(scalar_t) * (16 * head_dim + 2 * 2 * tile_elems + 16 * key_tile) +
            sizeof(float) * (qkt_parts * 16 * key_tile + 3 * 16);
    }

    __device__ static decode_tiles carve(unsigned char* base) {
        decode_tiles tiles;
        tiles.sQ = reinterpret_cast<scalar_t*>(base);
        tiles.ring = tiles.sQ + 16 * head_dim;
        tiles.sP = tiles.ring + 2 * 2 * tile_elems;
        tiles.sS = reinterpret_cast<float*>(tiles.sP + 16 * key_tile);
        tiles.Mrow = tiles.sS + qkt_parts * 16 * key_tile;
        tiles.Lrow = tiles.Mrow + 16;
        tiles.row_rescale = tiles.Lrow + 16;
        return tiles;
    }

    __device__ scalar_t* keys(int slot) const {
        return ring + slot * 2 * tile_elems;
    }

    __device__ scalar_t* values(int slot) const {
        return ring + slot * 2 * tile_elems + tile_elems;
    }
};

// Tiles of tile_keys keys a sliding window can touch when it does not start on a tile boundary.
int64_t window_tile_span(int64_t sliding_window, int64_t tile_keys) {
    return (sliding_window - 1 + tile_keys - 1) / tile_keys + 1;
}

// Load the group's query heads into MMA rows 0..group-1 and zero the rest, and reset
// the softmax state.
template <typename scalar_t, int head_dim, int nwarps>
__device__ __forceinline__ void
begin_group(const paged_attn_params<scalar_t>& p, int s, int g, scalar_t* sQ, float* Mrow, float* Lrow) {
    const int tid = threadIdx.x;
    for (int i = tid; i < 16 * head_dim; i += nwarps * 32) {
        const int row = i / head_dim, col = i % head_dim;
        scalar_t v = static_cast<scalar_t>(0);
        if (row < p.group) {
            const int h = g * p.group + row;
            v = p.q[(static_cast<int64_t>(s) * p.n_q_heads + h) * head_dim + col];
        }
        sQ[attn::swizzled_offset<head_dim>(row, col)] = v;
    }
    if (tid < 16) {
        Mrow[tid] = -CUDART_INF_F;
        Lrow[tid] = 0.0f;
    }
}

// Online-softmax attention over view tiles [t_start, t_end) of key_tile keys for one
// (seq s, kv head g), keys from index first on. Tiles wholly before first are not read.
// Accumulates the RAW, UNNORMALIZED state: out (this lane's tile_pv_registers share of
// the sum of exp(scale*q.k - Mrow)*V per query row), Mrow (running row max), Lrow
// (running denominator); the caller owns normalization and LSE. When group_mass is
// non-null, each row m's max scaled score over each view page b the tiles attend, times
// the query's weight, is written to group_mass[b * n_q_heads + m], and the weight to
// page_weight[b] when page_weight is non-null; a page must not straddle two calls. A
// slice past the view leaves out = 0, Mrow = -inf, Lrow = 0.
//
// Raw K and V tiles stream through two slots each by cp.async: a tile's keys are
// staged once QKᵀ frees their slot, two tiles ahead, and its values once PV frees
// theirs, one tile ahead. Each thread rotates the key pairs it copied itself once its
// own copies land, so the barrier that follows publishes rotated keys. Pools that are
// one tensor hold K = V, which is then read once: the rotation writes the key tile from
// the staged values.
template <typename scalar_t, int page_size, int head_dim, bool bf16, int nwarps>
__device__ __forceinline__ void tc_attend_tiles(
    const paged_attn_params<scalar_t>& p,
    int s,
    int g,
    int ctx_len,
    int first,
    const key_positions& layout,
    float* __restrict__ group_mass,
    float* __restrict__ page_weight,
    int t_start,
    int t_end,
    const decode_tiles<scalar_t, head_dim, nwarps>& tiles,
    float* out
) {
    using tiles_type = decode_tiles<scalar_t, head_dim, nwarps>;
    constexpr int key_tile = tiles_type::key_tile;
    constexpr int threads = nwarps * 32;
    constexpr int half = head_dim / 2;
    static_assert(sizeof(scalar_t) == 2, "the tile staging moves 8 elements at a time");
    constexpr int chunk_elems = 16 / sizeof(scalar_t);
    constexpr int pair_chunks = key_tile * half / chunk_elems;
    // The softmax gives each of the 16 query rows row_lanes consecutive threads.
    constexpr int row_lanes = threads / 16;
    static_assert(row_lanes <= 32 && key_tile % row_lanes == 0, "a row's lanes must share a warp and split the tile");
    constexpr int row_cols = key_tile / row_lanes;
    // A tile holds tile_pages pages, or part of one page when it is smaller than a page.
    constexpr int page_keys = key_tile < page_size ? key_tile : page_size;
    constexpr int tile_pages = key_tile / page_keys;

    scalar_t* sP = tiles.sP;
    float* sS = tiles.sS;
    float* Mrow = tiles.Mrow;
    float* Lrow = tiles.Lrow;
    float* row_rescale = tiles.row_rescale;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int32_t* seq_table = p.page_tables + static_cast<int64_t>(s) * p.max_pages;
    const bool keys_are_values = p.k_pool == p.v_pool;
    const int t_first = max(t_start, first / key_tile);
    const int t_last = min(t_end, (ctx_len + key_tile - 1) / key_tile);

    auto attended = [&](int key) { return key >= first && key < ctx_len; };
    // Chunk c is the 16-byte chunks at elements d and d + HALF of one key, so the thread that copies a key's
    // halves is the one that rotates them. Always commits a group, so a wait count means the same tiles throughout.
    auto stage = [&](const scalar_t* pool, scalar_t* dst, int t) {
        if (t < t_last) {
            for (int c = tid; c < pair_chunks; c += threads) {
                const int off = c / (half / chunk_elems);
                const int d = (c % (half / chunk_elems)) * chunk_elems;
                const int key = t * key_tile + off;
                const bool copied = attended(key);
                const int64_t base = copied ? static_cast<int64_t>(seq_table[key / page_size]) * p.page_stride +
                        g * p.head_stride + (key % page_size) * p.token_stride
                                            : 0;
                const int bytes = copied ? 16 : 0;
                attn::cp_async_16(dst + attn::swizzled_offset<head_dim>(off, d), pool + base + d, bytes);
                attn::cp_async_16(dst + attn::swizzled_offset<head_dim>(off, d + half), pool + base + d + half, bytes);
            }
        }
        attn::cp_async_commit();
    };
    auto stage_keys = [&](int t) { stage(p.k_pool, tiles.keys((t - t_first) % 2), keys_are_values ? t_last : t); };
    auto stage_values = [&](int t) { stage(p.v_pool, tiles.values((t - t_first) % 2), t); };
    // A thread's chunks all sit at the same pairs, so their RoPE turns and k_norm weights are read once.
    static_assert(threads % (half / chunk_elems) == 0, "every chunk of a thread must hold the same pairs");
    const int pair0 = (tid % (half / chunk_elems)) * chunk_elems;
    uint64_t turns[chunk_elems];
    float lo_weight[chunk_elems];
    float hi_weight[chunk_elems];
#pragma unroll
    for (int j = 0; j < chunk_elems; ++j) {
        turns[j] = p.rope_turns[pair0 + j];
        lo_weight[j] = p.k_norm_weight ? p.k_norm_weight[pair0 + j] : 1.0f;
        hi_weight[j] = p.k_norm_weight ? p.k_norm_weight[pair0 + j + half] : 1.0f;
    }
    auto rotate = [&](int t) {
        const int slot = (t - t_first) % 2;
        scalar_t* sK = tiles.keys(slot);
        const scalar_t* raw = keys_are_values ? tiles.values(slot) : sK;
        for (int c = tid; c < pair_chunks; c += threads) {
            const int off = c / (half / chunk_elems);
            const int key = t * key_tile + off;
            const int lo_at = attn::swizzled_offset<head_dim>(off, pair0);
            const int hi_at = attn::swizzled_offset<head_dim>(off, pair0 + half);
            alignas(16) scalar_t lo[chunk_elems];
            alignas(16) scalar_t hi[chunk_elems];
            *reinterpret_cast<uint4*>(lo) = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(hi) = make_uint4(0, 0, 0, 0);
            if (attended(key)) {
                *reinterpret_cast<uint4*>(lo) = *reinterpret_cast<const uint4*>(raw + lo_at);
                *reinterpret_cast<uint4*>(hi) = *reinterpret_cast<const uint4*>(raw + hi_at);
                const int position = layout.position(key);
#pragma unroll
                for (int j = 0; j < chunk_elems; ++j) {
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
    const int row = tid / row_lanes;
    const int row_lane = tid % row_lanes;
    // The row's max score over the page the tiles are in, every lane of the row holding it.
    int run_page = -1;
    float run_max = -CUDART_INF_F;
    const float weight = group_mass ? p.query_weights[s] : 0.0f;
    auto flush_page = [&]() {
        if (run_page >= 0 && row_lane == 0 && row < p.group) {
            group_mass[static_cast<int64_t>(run_page) * p.n_q_heads + row] = weight * run_max;
            if (page_weight && row == 0) {
                page_weight[run_page] = weight;
            }
        }
    };
    for (int t = t_first; t < t_last; ++t) {
        const int slot = (t - t_first) % 2;
        const int tile_base = t * key_tile;
        // Only the keys of tile t + 1 may still be in flight.
        attn::cp_async_wait<1>();
        rotate(t);
        // Also ends the previous tile's PV, which frees the value slot of tile t + 1.
        __syncthreads();
        stage_values(t + 1);

        attn::tile_qkt_parts<scalar_t, key_tile, head_dim, bf16, nwarps>(tiles.sQ, tiles.keys(slot), sS, warp);
        // Also frees this tile's key slot for tile t + 2.
        __syncthreads();
        stage_keys(t + 2);

        // A tile in [t_first, t_last) holds an attended key, so newM is finite.
        float scores[row_cols];
        float tmax = -CUDART_INF_F;
#pragma unroll
        for (int j = 0; j < row_cols; ++j) {
            const int c = j * row_lanes + row_lane;
            float raw_score = 0.0f;
#pragma unroll
            for (int part = 0; part < tiles_type::qkt_parts; ++part) {
                raw_score += sS[(part * 16 + row) * key_tile + c];
            }
            scores[j] = attended(tile_base + c) ? p.scale * raw_score : -CUDART_INF_F;
            tmax = fmaxf(tmax, scores[j]);
        }
        if (group_mass) {
#pragma unroll
            for (int k = 0; k < tile_pages; ++k) {
                float page_max = -CUDART_INF_F;
#pragma unroll
                for (int j = 0; j < row_cols; ++j) {
                    if ((j * row_lanes + row_lane) / page_keys == k) {
                        page_max = fmaxf(page_max, scores[j]);
                    }
                }
#pragma unroll
                for (int lanes = row_lanes / 2; lanes > 0; lanes >>= 1) {
                    page_max = fmaxf(page_max, __shfl_xor_sync(0xffffffffu, page_max, lanes));
                }
                const int page = (tile_base + k * page_keys) / page_size;
                if (page_max > -CUDART_INF_F) {
                    if (page != run_page) {
                        flush_page();
                        run_page = page;
                        run_max = page_max;
                    } else {
                        run_max = fmaxf(run_max, page_max);
                    }
                }
            }
        }
#pragma unroll
        for (int lanes = row_lanes / 2; lanes > 0; lanes >>= 1) {
            tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, lanes));
        }
        const float oldM = Mrow[row];
        const float newM = fmaxf(oldM, tmax);
        float lsum = 0.0f;
#pragma unroll
        for (int j = 0; j < row_cols; ++j) {
            const int c = j * row_lanes + row_lane;
            const bool kept = scores[j] > -CUDART_INF_F;
            const float pw = kept ? __expf(scores[j] - newM) : 0.0f;
            sP[row * key_tile + c] = static_cast<scalar_t>(pw);
            lsum += pw;
        }
#pragma unroll
        for (int lanes = row_lanes / 2; lanes > 0; lanes >>= 1) {
            lsum += __shfl_xor_sync(0xffffffffu, lsum, lanes);
        }
        if (row_lane == 0) {
            const float corr = __expf(oldM - newM);  // 0 when oldM == -inf
            row_rescale[row] = corr;
            Mrow[row] = newM;
            Lrow[row] = Lrow[row] * corr + lsum;
        }
        __syncthreads();

        attn::tile_pv_registers<scalar_t, key_tile, head_dim, bf16, nwarps, true>(
            sP,
            tiles.values(slot),
            row_rescale,
            out,
            warp
        );
    }
    if (group_mass) {
        flush_page();
    }
}

// Sequence s's mass_sum entries for kv head g's query rows, and its weight_sum entries for kv head 0; null for both
// when the op measures no mass.
template <typename scalar_t>
__device__ __forceinline__ float* group_mass_entries(const paged_attn_params<scalar_t>& p, int s, int g) {
    return p.mass_sum ? p.mass_sum + attn::mass_row(p, s, 0) * p.n_q_heads + g * p.group : nullptr;
}

template <typename scalar_t>
__device__ __forceinline__ float* page_weight_entries(const paged_attn_params<scalar_t>& p, int s, int g) {
    return p.mass_sum && g == 0 ? p.weight_sum + attn::mass_row(p, s, 0) : nullptr;
}

// Single-CTA tensor-core decode: one CTA per (seq s, kv head g) attends the whole view.
template <typename scalar_t, int page_size, int head_dim, bool bf16, int nwarps>
__global__ void __launch_bounds__(nwarps * 32) attn_decode_tc_kernel(const paged_attn_params<scalar_t> p) {
    wait_for_prior_kernels();
    allow_next_kernel();
    const int tid = threadIdx.x;
    const int s = blockIdx.x / p.n_kv_heads;
    const int g = blockIdx.x % p.n_kv_heads;
    const int ctx_len = p.seqlens_k[s];
    const int first = attn::first_key(p, ctx_len - 1);

    extern __shared__ __align__(16) unsigned char shared_memory[];
    const auto tiles = decode_tiles<scalar_t, head_dim, nwarps>::carve(shared_memory);
    const float* Lrow = tiles.Lrow;
    float out[attn::pv_slabs_per_warp<head_dim, nwarps>() * attn::pv_slab_accums] = {};

    begin_group<scalar_t, head_dim, nwarps>(p, s, g, tiles.sQ, tiles.Mrow, tiles.Lrow);
    __syncthreads();
    tc_attend_tiles<scalar_t, page_size, head_dim, bf16, nwarps>(
        p,
        s,
        g,
        ctx_len,
        first,
        attn::sequence_key_positions(p, s, page_size),
        group_mass_entries(p, s, g),
        page_weight_entries(p, s, g),
        0,
        std::numeric_limits<int>::max(),
        tiles,
        out
    );

    const int gid = (tid & 31) >> 2;
    const float inv_low = 1.0f / Lrow[gid], inv_high = 1.0f / Lrow[gid + 8];
    scalar_t* o_group = p.o + (static_cast<int64_t>(s) * p.n_q_heads + g * p.group) * head_dim;
    attn::for_each_pv_register<head_dim, nwarps>(out, tid >> 5, [&](int row, int col, float value) {
        if (row < p.group) {
            o_group[row * head_dim + col] = static_cast<scalar_t>(value * (row < 8 ? inv_low : inv_high));
        }
    });
}

template <typename scalar_t, int page_size, int head_dim, bool bf16, int nwarps>
__global__ void __launch_bounds__(nwarps * 32) paged_decode_split_kernel(
    const paged_attn_params<scalar_t> p,
    float* __restrict__ o_partial,
    float* __restrict__ m_partial,
    float* __restrict__ l_partial,
    int nsplits
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    const int tid = threadIdx.x;
    const int split = blockIdx.x % nsplits;
    const int sg = blockIdx.x / nsplits;
    const int g = sg % p.n_kv_heads;
    const int s = sg / p.n_kv_heads;
    const int ctx_len = p.seqlens_k[s];
    const int first = attn::first_key(p, ctx_len - 1);

    extern __shared__ __align__(16) unsigned char shared_memory[];
    const auto tiles = decode_tiles<scalar_t, head_dim, nwarps>::carve(shared_memory);
    const float* Mrow = tiles.Mrow;
    const float* Lrow = tiles.Lrow;
    float out[attn::pv_slabs_per_warp<head_dim, nwarps>() * attn::pv_slab_accums] = {};

    begin_group<scalar_t, head_dim, nwarps>(p, s, g, tiles.sQ, tiles.Mrow, tiles.Lrow);
    __syncthreads();
    // Slices start on page boundaries, so every page's max is one CTA's.
    // The splits share the tiles from the window's first page to the view's last key, in whole pages.
    constexpr int tiles_per_page = decode_tiles_per_page(page_size, head_dim);
    constexpr int key_tile = decode_tiles<scalar_t, head_dim, nwarps>::key_tile;
    const int first_tile = first / key_tile / tiles_per_page * tiles_per_page;
    const int attended_tiles = (ctx_len - 1) / key_tile + 1 - first_tile;
    const int tiles_per_split = ((attended_tiles + nsplits - 1) / nsplits + tiles_per_page - 1) / tiles_per_page *
        tiles_per_page;
    const int t_start = first_tile + split * tiles_per_split;
    tc_attend_tiles<scalar_t, page_size, head_dim, bf16, nwarps>(
        p,
        s,
        g,
        ctx_len,
        first,
        attn::sequence_key_positions(p, s, page_size),
        group_mass_entries(p, s, g),
        page_weight_entries(p, s, g),
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
    float* o_slice = o_partial + base * head_dim;
    attn::for_each_pv_register<head_dim, nwarps>(out, tid >> 5, [&](int row, int col, float value) {
        if (row < p.group) {
            o_slice[row * head_dim + col] = value;
        }
    });
}

// One CTA per (seq, query head). Shared memory: cta_threads reduction floats, then one
// weight per split.
template <typename scalar_t>
__global__ void paged_decode_combine_kernel(
    const paged_attn_params<scalar_t> p,
    float* __restrict__ lse_out,
    const float* __restrict__ o_partial,
    const float* __restrict__ m_partial,
    const float* __restrict__ l_partial,
    int nsplits
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    extern __shared__ float combine_shared[];
    float* scratch = combine_shared;
    float* weight = combine_shared + cta_threads;
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
    for (int i = tid; i < nsplits; i += cta_threads) {
        local_max = fmaxf(local_max, mp[i * group]);
    }
    const float block_max = attn::block_reduce_max<cta_threads>(scratch, tid, local_max);
    float local_l = 0.0f;
    for (int i = tid; i < nsplits; i += cta_threads) {
        const float mi = mp[i * group];
        weight[i] = mi > -CUDART_INF_F ? __expf(mi - block_max) : 0.0f;
        local_l += lp[i * group] * weight[i];
    }
    const float l = attn::block_reduce_sum<cta_threads>(scratch, tid, local_l);
    const bool bad = (l == 0.0f) || !isfinite(block_max);
    const float invl = bad ? 0.0f : 1.0f / l;

    scalar_t* orow = p.o + (static_cast<int64_t>(s) * p.n_q_heads + h) * p.head_dim;
    const int64_t split_stride = static_cast<int64_t>(group) * p.head_dim;
    for (int d = tid; d < p.head_dim; d += cta_threads) {
        float acc = 0.0f;
        if (!bad) {
            for (int i = 0; i < nsplits; ++i) {
                acc += op[i * split_stride + d] * weight[i];
            }
        }
        orow[d] = static_cast<scalar_t>(acc * invl);
    }
    if (tid == 0) {
        lse_out[static_cast<int64_t>(s) * p.n_q_heads + h] = bad ? -CUDART_INF_F : (block_max + logf(l));
    }
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

template <typename scalar_t, bool bf16> int64_t resident_split_ctas(const attn::paged_attn_batch& b) {
    int64_t resident = 0;
    attn::dispatch_page_head("attn_decode", b.page_size, b.head_dim, [&](auto bs_tag, auto hd_tag) {
        constexpr int bs = decltype(bs_tag)::value;
        constexpr int hd = decltype(hd_tag)::value;
        attn::dispatch_attn_tuning([&](auto row_tag) {
            constexpr int nwarps = attn::attn_tuning_table[decltype(row_tag)::value].decode_warps;
            constexpr size_t smem = decode_tiles<scalar_t, hd, nwarps>::bytes();
            static const int64_t kernel_resident = [] {
                const auto kernel = paged_decode_split_kernel<scalar_t, bs, hd, bf16, nwarps>;
                attn::allow_shared_memory(kernel, smem);
                int per_sm = 0;
                C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, nwarps * 32, smem));
                return static_cast<int64_t>(per_sm) * at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
            }();
            resident = kernel_resident;
        });
    });
    return resident;
}

template <typename scalar_t, bool bf16>
void launch_tc_decode(const attn::paged_attn_batch& b, int64_t forced_splits, cudaStream_t stream) {
    const paged_attn_params<scalar_t> p = b.params<scalar_t>();
    const int64_t group = b.n_q_heads / b.n_kv_heads;
    const int64_t base_ctas = b.num_seqs * b.n_kv_heads;
    auto fopts = b.q.options().dtype(at::kFloat);

    const int64_t kv_capacity = b.max_pages * b.page_size;
    const int64_t key_tile = decode_key_tile(static_cast<int>(b.head_dim));
    const int64_t view_tiles = (kv_capacity + key_tile - 1) / key_tile;
    const int64_t tiles_per_page = decode_tiles_per_page(static_cast<int>(b.page_size), static_cast<int>(b.head_dim));
    // A split's slice starts on a page boundary at or before the window's first tile.
    const int64_t attended_tiles = b.sliding_window > 0
        ? std::min(view_tiles, window_tile_span(b.sliding_window, key_tile) + tiles_per_page - 1)
        : view_tiles;
    const int nsplits = forced_splits > 0
        ? static_cast<int>(forced_splits)
        : auto_splits(base_ctas, attended_tiles, resident_split_ctas<scalar_t, bf16>(b));
    const bool split = forced_splits > 0 || nsplits > 1;

    if (!split) {
        attn::dispatch_page_head("attn_decode", b.page_size, b.head_dim, [&](auto bs_tag, auto hd_tag) {
            constexpr int bs = decltype(bs_tag)::value;
            constexpr int hd = decltype(hd_tag)::value;
            attn::dispatch_attn_tuning([&](auto row_tag) {
                constexpr int nwarps = attn::attn_tuning_table[decltype(row_tag)::value].decode_warps;
                constexpr size_t smem = decode_tiles<scalar_t, hd, nwarps>::bytes();
                const auto kernel = attn_decode_tc_kernel<scalar_t, bs, hd, bf16, nwarps>;
                attn::allow_shared_memory(kernel, smem);
                launch_dependent(kernel, static_cast<int>(base_ctas), nwarps * 32, smem, stream, p);
            });
        });
        return;
    }

    auto o_partial = at::empty({b.num_seqs, b.n_kv_heads, nsplits, group, b.head_dim}, fopts);
    auto m_partial = at::empty({b.num_seqs, b.n_kv_heads, nsplits, group}, fopts);
    auto l_partial = at::empty({b.num_seqs, b.n_kv_heads, nsplits, group}, fopts);
    auto lse_out = at::empty({b.num_seqs, b.n_kv_heads, group}, fopts);

    attn::dispatch_page_head("attn_decode", b.page_size, b.head_dim, [&](auto bs_tag, auto hd_tag) {
        constexpr int bs = decltype(bs_tag)::value;
        constexpr int hd = decltype(hd_tag)::value;
        attn::dispatch_attn_tuning([&](auto row_tag) {
            constexpr int nwarps = attn::attn_tuning_table[decltype(row_tag)::value].decode_warps;
            constexpr size_t smem = decode_tiles<scalar_t, hd, nwarps>::bytes();
            const auto kernel = paged_decode_split_kernel<scalar_t, bs, hd, bf16, nwarps>;
            attn::allow_shared_memory(kernel, smem);
            launch_dependent(
                kernel,
                static_cast<int>(base_ctas * nsplits),
                nwarps * 32,
                smem,
                stream,
                p,
                o_partial.data_ptr<float>(),
                m_partial.data_ptr<float>(),
                l_partial.data_ptr<float>(),
                nsplits
            );
        });
    });
    const size_t combine_smem = sizeof(float) * (cta_threads + nsplits);
    launch_dependent(
        paged_decode_combine_kernel<scalar_t>,
        static_cast<int>(b.num_seqs * b.n_q_heads),
        cta_threads,
        combine_smem,
        stream,
        p,
        lse_out.data_ptr<float>(),
        o_partial.data_ptr<float>(),
        m_partial.data_ptr<float>(),
        l_partial.data_ptr<float>(),
        nsplits
    );
}

}  // namespace

bool attn::decode_fits(int64_t page_size, int64_t head_dim) {
    bool fits = false;
    attn::dispatch_page_head("attn_decode", page_size, head_dim, [&](auto, auto hd_tag) {
        attn::dispatch_attn_tuning([&](auto row_tag) {
            constexpr int nwarps = attn::attn_tuning_table[decltype(row_tag)::value].decode_warps;
            fits = attn::fits_shared_memory(decode_tiles<at::Half, decltype(hd_tag)::value, nwarps>::bytes());
        });
    });
    return fits;
}

namespace {

at::Tensor paged_decode(const attn::paged_attn_batch& b, int64_t num_splits) {
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
            constexpr int bs = decltype(page_size_tag)::value;
            launch_dependent(
                write_kv_kernel<scalar_t, bs>,
                static_cast<int>(num_new_tokens),
                threads,
                0,
                stream,
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
    const at::Tensor& page_positions,
    const rope_parameters& rope,
    double scale,
    std::optional<int64_t> sliding_window,
    const std::optional<at::Tensor>& k_norm_weight,
    const std::optional<at::Tensor>& mass_sum,
    const std::optional<at::Tensor>& weight_sum,
    const std::optional<at::Tensor>& cu_view_pages,
    const std::optional<at::Tensor>& query_weights,
    int64_t num_splits
) {
    const auto batch = attn::make_paged_attn_batch(
        "attn_decode",
        q,
        k_pool,
        v_pool,
        page_tables,
        std::nullopt,
        context_lens,
        page_positions,
        rope,
        scale,
        sliding_window,
        k_norm_weight,
        mass_sum,
        weight_sum,
        cu_view_pages,
        query_weights
    );
    return paged_decode(batch, num_splits);
}

}  // namespace pulsarlm
