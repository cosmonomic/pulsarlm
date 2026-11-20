#pragma once

#include "pulsarlm/kernels/dependent_launch.cuh"
#include "pulsarlm/kernels/attn_params.cuh"
#include "pulsarlm/kernels/attn_tiles.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstddef>
#include <cstdint>

// Tensor-core prefill attention. Flash-attention-style: one CTA per (query-tile,
// q head); a query tile is up to 16 query tokens of ONE q head, occupying MMA
// rows 0..tile_rows-1 (rows tile_rows..15 are zero-padded and discarded). The
// causal key range is streamed key_pages pages (key_pages * page_size keys) at a
// time: keys from rotated_k, the views' keys already rotated to their positions by
// rotate_view_keys_kernel (paged_prefill.cu), and values from the pool
// through the page table. The causal and sliding-window masks are applied per
// (query row, key) in the online softmax; pages before the window of the tile's
// first row are not read. A key tile holds whole pages, so each tile adds, per page,
// the sum over the rows attending it of the row's query weight times its max score
// over the page's attended keys into the page's mass_sum in this q head's column, and
// the rows' weights into its weight_sum. The QKᵀ/PV tiles live in attn_tiles.cuh; the
// op wrapper and the argument shapes are in paged_prefill.cu.
//
// nwarps, key_pages and ctas_per_sm are launch shape, not geometry: every one of
// them is device specific and comes from the caller's tuning row
// (attn_tuning.hpp). key_pages only changes the order in which the online
// softmax folds keys in, so o moves in its last bits with it.
// ctas_per_sm is the occupancy floor ptxas must fit the register allocation to; 1
// leaves it unconstrained.

namespace pulsarlm {
namespace attn {

// sQ, sK, sV, sP, then sS. Every size is a multiple of 16 bytes, which keeps each tile 16-byte aligned for ldmatrix.
template <typename scalar_t, int page_size, int head_dim, int key_pages> constexpr size_t prefill_tile_bytes() {
    constexpr size_t key_tile = static_cast<size_t>(page_size) * key_pages;
    return sizeof(scalar_t) * (16 * head_dim + 2 * key_tile * head_dim + 16 * key_tile) +
        sizeof(float) * 16 * (key_tile + 1);
}

// Query tiles are 16 query tokens of one sequence, numbered in sequence order. Sets the
// sequence and first local query of tile; false past the last tile.
template <typename scalar_t>
__device__ __forceinline__ bool query_tile(const paged_attn_params<scalar_t>& p, int tile, int& s, int& qbase) {
    for (s = 0; s < p.num_seqs; ++s) {
        const int tiles = (p.cu_seqlens_q[s + 1] - p.cu_seqlens_q[s] + 15) / 16;
        if (tile < tiles) {
            qbase = tile * 16;
            return true;
        }
        tile -= tiles;
    }
    return false;
}

template <typename scalar_t, int page_size, int head_dim, bool bf16, int nwarps, int key_pages, int ctas_per_sm>
__global__ void __launch_bounds__(nwarps * 32, ctas_per_sm) attn_prefill_tc_kernel(
    const paged_attn_params<scalar_t> p,
    const scalar_t* __restrict__ rotated_k  // [num_seqs * max_pages, page_size, n_kv_heads, head_dim]
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    constexpr int key_tile = page_size * key_pages;
    // sS row stride. A stride of exactly key_tile puts every row of a column on one
    // shared bank, and the two warp-0 sweeps below read a whole column at once.
    constexpr int score_stride = key_tile + 1;
    // The per-row max and denominator sweeps give every query row a group of
    // sweep_lanes consecutive threads and reduce the group on shuffles. A group's
    // threads take consecutive score pairs, which keeps the two half-warps of a
    // warp on disjoint banks at score_stride odd.
    constexpr int sweep_lanes = 4;
    constexpr int sweep_cols = key_tile / sweep_lanes;
    constexpr int sweep_threads = 16 * sweep_lanes;
    static_assert(
        sweep_threads % 32 == 0 && sweep_threads <= nwarps * 32,
        "the sweep groups must fill whole warps of the CTA"
    );
    static_assert(key_tile % sweep_lanes == 0, "key tile must split over a group");
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int sweep_row = tid / sweep_lanes;
    const int sweep_lane = tid % sweep_lanes;
    const int blk = blockIdx.x;
    const int h = blk % p.n_q_heads;
    const int g = h / p.group;
    int s, qbase;
    if (!query_tile(p, blk / p.n_q_heads, s, qbase)) {
        return;
    }
    const int q0 = p.cu_seqlens_q[s];
    const int seq_q = p.cu_seqlens_q[s + 1] - q0;
    const int key_count = p.key_counts[s];
    const int ctx_start = key_count - seq_q;
    const int tile_rows = min(16, seq_q - qbase);
    const int q_global_base = q0 + qbase;
    // Context pos of MMA row m is ctx_start + qbase + m; the tile's highest.
    const int max_p = ctx_start + qbase + tile_rows - 1;
    const int n_key_tiles = max_p / key_tile + 1;  // tiles covering keys [0, max_p]
    const int tile_first_key = first_key(p, ctx_start + qbase);
    const int first_key_tile = tile_first_key / key_tile;

    const int32_t* seq_table = p.page_tables + static_cast<int64_t>(s) * p.max_pages;
    const int64_t rotated_token_stride = static_cast<int64_t>(p.n_kv_heads) * head_dim;
    const int64_t rotated_page_stride = page_size * rotated_token_stride;

    static_assert(sizeof(scalar_t) == 2, "the tile staging copies 8 elements at a time");
    constexpr int chunk_elems = 16 / sizeof(scalar_t);
    constexpr int row_chunks = head_dim / chunk_elems;
    constexpr int page_chunks = page_size * row_chunks;

    // sQ, sK and sV carry their columns permuted by attn::swizzled_offset; sP and sS
    // are reached by scalar accesses only and stay row-major.
    extern __shared__ __align__(16) unsigned char shared_memory[];
    scalar_t* sQ = reinterpret_cast<scalar_t*>(shared_memory);
    scalar_t* sK = sQ + 16 * head_dim;
    scalar_t* sV = sK + key_tile * head_dim;
    scalar_t* sP = sV + key_tile * head_dim;
    float* sS = reinterpret_cast<float*>(sP + 16 * key_tile);
    __shared__ float Mrow[16], Lrow[16];
    // Online-softmax rescale of the row's running output, published so every warp
    // can fold it into its own share of the accumulator.
    __shared__ float row_rescale[16];
    __shared__ int qpos[16];  // context pos of each MMA row, -1 if padding
    __shared__ int qfirst[16];  // first key each MMA row attends
    __shared__ float row_weight[16];  // each MMA row's query weight, 0 if padding
    // Each (MMA row, page) of the key tile: the row's max scaled score over the page's keys it attends, -inf for none.
    __shared__ float page_max_rows[16 * key_pages];

    // Load Q for this tile's rows into the MMA M dimension; pad rows >= tile_rows.
    for (int i = tid; i < 16 * head_dim; i += nwarps * 32) {
        const int row = i / head_dim, col = i % head_dim;
        scalar_t v = static_cast<scalar_t>(0);
        if (row < tile_rows) {
            const int64_t qi = (static_cast<int64_t>(q_global_base + row) * p.n_q_heads + h) * head_dim;
            v = p.q[qi + col];
        }
        sQ[attn::swizzled_offset<head_dim>(row, col)] = v;
    }
    // This lane's share of the running output, one attn::pv_slab_accums block per dim
    // slab this warp owns. attn::tile_pv_registers states the (row, column) each
    // entry carries.
    constexpr int slabs = attn::pv_slabs_per_warp<head_dim, nwarps>();
    float out_acc[slabs * attn::pv_slab_accums] = {};
    for (int i = tid; i < 16 * key_tile; i += nwarps * 32) {
        sP[i] = static_cast<scalar_t>(0);
    }
    if (tid < 16) {
        Mrow[tid] = -CUDART_INF_F;
        Lrow[tid] = 0.0f;
        // Padding rows never enter the softmax update, so their output must stay put.
        row_rescale[tid] = 1.0f;
        const int row_pos = ctx_start + qbase + tid;  // this row's context pos
        qpos[tid] = (tid < tile_rows) ? row_pos : -1;
        qfirst[tid] = first_key(p, row_pos);
        row_weight[tid] = p.mass_sum && tid < tile_rows ? p.query_weights[q_global_base + tid] : 0.0f;
    }
    // The pages of key tile t, in the pool (lines, for V) and in rotated_k (sequence
    // s's view page b is page s * max_pages + b, for K). The staging address
    // arithmetic is a 64-bit multiply chain off these, so they are read a whole
    // iteration before the stage_tile that consumes them. A tile may reach past the
    // sequence's last page, or begin before the window; seq_table is only read for
    // the pages between.
    auto load_pages = [&](int t, int* lines, int* key_indices) {
#pragma unroll
        for (int kp = 0; kp < key_pages; ++kp) {
            const int page_base = t * key_tile + kp * page_size;
            const bool in_view = t < n_key_tiles && page_base < key_count && page_base + page_size > tile_first_key;
            lines[kp] = in_view ? seq_table[t * key_pages + kp] : 0;
            key_indices[kp] = in_view ? s * p.max_pages + t * key_pages + kp : 0;
        }
    };

    // Stage one key tile into its shared tile. Always commits a group, empty tile
    // included, so every iteration costs the same number of groups and a wait count
    // means the same thing throughout. `pool` selects K or V with its page, token and head
    // strides; both land at the same offsets, so one routine serves either.
    auto stage_tile = [&](const scalar_t* pool,
                          int64_t page_stride,
                          int64_t token_stride,
                          int64_t head_stride,
                          scalar_t* dst,
                          int t,
                          const int* pages) {
        if (t < n_key_tiles) {
#pragma unroll
            for (int kp = 0; kp < key_pages; ++kp) {
                const int page_base = t * key_tile + kp * page_size;
                const int valid = page_base + page_size > tile_first_key ? min(page_size, key_count - page_base) : 0;
                const int64_t page_first = static_cast<int64_t>(pages[kp]) * page_stride + g * head_stride;
                for (int c = tid; c < page_chunks; c += nwarps * 32) {
                    const int off = c / row_chunks;
                    const int d = (c % row_chunks) * chunk_elems;
                    const int64_t idx = page_first + static_cast<int64_t>(off) * token_stride + d;
                    attn::cp_async_16(
                        dst + attn::swizzled_offset<head_dim>(kp * page_size + off, d),
                        pool + idx,
                        off < valid ? 16 : 0
                    );
                }
            }
        }
        attn::cp_async_commit();
    };

    // K of tile t is consumed by QKᵀ at the top of the iteration and V only by PV at
    // the bottom, so each is staged one full iteration ahead of its use: K right
    // after the QKᵀ that frees its tile, V right after that PV. Two groups are in
    // flight at every wait, oldest first, so both waits leave one outstanding.
    int tile_lines[key_pages], next_lines[key_pages], tile_keys[key_pages], next_keys[key_pages];
    auto attends = [&](int m, int key) { return key >= qfirst[m] && key <= qpos[m]; };

    load_pages(first_key_tile, tile_lines, tile_keys);
    stage_tile(rotated_k, rotated_page_stride, rotated_token_stride, head_dim, sK, first_key_tile, tile_keys);
    stage_tile(p.v_pool, p.page_stride, p.token_stride, p.head_stride, sV, first_key_tile, tile_lines);
    __syncthreads();

    // Pass 1: online-softmax attention with per-(row,key) causal and window masking.
    for (int t = first_key_tile; t < n_key_tiles; ++t) {
        const int tile_base = t * key_tile;
        load_pages(t + 1, next_lines, next_keys);
        attn::cp_async_wait<1>();
        __syncthreads();

        attn::tile_qkt<scalar_t, key_tile, head_dim, bf16, score_stride, true>(sQ, sK, sS, warp, nwarps);
        __syncthreads();
        stage_tile(rotated_k, rotated_page_stride, rotated_token_stride, head_dim, sK, t + 1, next_keys);

        if (tid < sweep_threads) {
            float tmax = -CUDART_INF_F;
#pragma unroll
            for (int j = 0; j < sweep_cols; ++j) {
                const int c = sweep_lane * sweep_cols + j;
                if (attends(sweep_row, tile_base + c)) {
                    tmax = fmaxf(tmax, p.scale * sS[sweep_row * score_stride + c]);
                }
            }
#pragma unroll
            for (int off = sweep_lanes / 2; off > 0; off >>= 1) {
                tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, off));
            }
            if (sweep_lane == 0 && sweep_row < tile_rows) {
                const float newM = fmaxf(Mrow[sweep_row], tmax);
                // 0 when Mrow == -inf; 1 for a row whose window starts after this tile.
                row_rescale[sweep_row] = newM == -CUDART_INF_F ? 1.0f : __expf(Mrow[sweep_row] - newM);
                Mrow[sweep_row] = newM;
            }
        }
        if (p.mass_sum) {
            for (int i = tid; i < 16 * key_pages; i += nwarps * 32) {
                const int m = i / key_pages, kp = i % key_pages;
                float page_max = -CUDART_INF_F;
                for (int c = kp * page_size; c < (kp + 1) * page_size; ++c) {
                    if (attends(m, tile_base + c)) {
                        page_max = fmaxf(page_max, p.scale * sS[m * score_stride + c]);
                    }
                }
                page_max_rows[i] = page_max;
            }
        }
        __syncthreads();

        // A query measuring its mass apart adds its own rows; otherwise the tile's queries add to their sequence's.
        if (p.mass_sum && s < p.per_query_mass_request_count && tid < key_pages) {
            for (int m = 0; m < tile_rows; ++m) {
                const float page_max = page_max_rows[m * key_pages + tid];
                if (page_max > -CUDART_INF_F && row_weight[m] > 0.0f) {
                    const int64_t row = mass_row(p, mass_view(p, s, q_global_base + m), t * key_pages + tid);
                    atomicAdd(&p.mass_sum[row * p.n_q_heads + h], row_weight[m] * page_max);
                    if (h == 0) {
                        atomicAdd(&p.weight_sum[row], row_weight[m]);
                    }
                }
            }
        } else if (p.mass_sum && tid < key_pages) {
            float page_sum = 0.0f;
            float page_weight = 0.0f;
            for (int m = 0; m < tile_rows; ++m) {
                const float page_max = page_max_rows[m * key_pages + tid];
                if (page_max > -CUDART_INF_F) {
                    page_sum += row_weight[m] * page_max;
                    page_weight += row_weight[m];
                }
            }
            if (page_weight > 0.0f) {
                const int64_t row = mass_row(p, mass_view(p, s, q_global_base), t * key_pages + tid);
                atomicAdd(&p.mass_sum[row * p.n_q_heads + h], page_sum);
                if (h == 0) {
                    atomicAdd(&p.weight_sum[row], page_weight);
                }
            }
        }

        // One thread per (query row, key): the exponentials run across the CTA,
        // and sS carries each row's weight to the sequential sum below.
        for (int i = tid; i < 16 * key_tile; i += nwarps * 32) {
            const int m = i / key_tile, c = i % key_tile;
            float& score = sS[m * score_stride + c];
            const float pw = attends(m, tile_base + c) ? __expf(p.scale * score - Mrow[m]) : 0.0f;
            score = pw;
            sP[i] = static_cast<scalar_t>(pw);
        }
        __syncthreads();

        // Summed as a shuffle tree over the row's group, not in key order.
        if (tid < sweep_threads) {
            float lsum = 0.0f;
#pragma unroll
            for (int j = 0; j < sweep_cols; ++j) {
                lsum += sS[sweep_row * score_stride + sweep_lane * sweep_cols + j];
            }
#pragma unroll
            for (int off = sweep_lanes / 2; off > 0; off >>= 1) {
                lsum += __shfl_xor_sync(0xffffffffu, lsum, off);
            }
            if (sweep_lane == 0 && sweep_row < tile_rows) {
                Lrow[sweep_row] = Lrow[sweep_row] * row_rescale[sweep_row] + lsum;
            }
        }
        attn::cp_async_wait<1>();
        __syncthreads();

        attn::tile_pv_registers<scalar_t, key_tile, head_dim, bf16, nwarps, true>(sP, sV, row_rescale, out_acc, warp);
        __syncthreads();
        stage_tile(p.v_pool, p.page_stride, p.token_stride, p.head_stride, sV, t + 1, next_lines);
    }

    // Normalize + write o, each lane over the accumulators it owns.
    {
        const int gid = lane >> 2;
        // A row past tile_rows has Lrow 0, and its reciprocal is never read.
        const float inv_low = 1.0f / Lrow[gid], inv_high = 1.0f / Lrow[gid + 8];
        attn::for_each_pv_register<head_dim, nwarps>(out_acc, warp, [&](int row, int col, float value) {
            if (row < tile_rows) {
                p.o[(static_cast<int64_t>(q_global_base + row) * p.n_q_heads + h) * head_dim +
                    col] = static_cast<scalar_t>(value * (row < 8 ? inv_low : inv_high));
            }
        });
    }
}

// Dynamic and static shared bytes of one CTA of attn_prefill_tc_kernel.
template <typename scalar_t, int page_size, int head_dim, int key_pages> constexpr size_t prefill_smem_bytes() {
    return prefill_tile_bytes<scalar_t, page_size, head_dim, key_pages>() + sizeof(float) * (4 * 16 + 16 * key_pages) +
        sizeof(int) * 2 * 16;
}

// key_pages, unless the CTA would then need to opt in to more shared memory.
template <typename scalar_t, int page_size, int head_dim, int key_pages> constexpr int prefill_key_pages() {
    return prefill_smem_bytes<scalar_t, page_size, head_dim, key_pages>() <= shared_memory_without_opt_in ? key_pages
                                                                                                          : 1;
}

}  // namespace attn
}  // namespace pulsarlm
