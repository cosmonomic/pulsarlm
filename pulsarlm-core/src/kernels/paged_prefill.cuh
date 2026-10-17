#pragma once

#include "attn_params.cuh"
#include "attn_tiles.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstddef>
#include <cstdint>

// Tensor-core prefill attention. Flash-attention-style: one CTA per (query-tile,
// q head); a query tile is up to 16 query tokens of ONE q head, occupying MMA
// rows 0..tile_rows-1 (rows tile_rows..15 are zero-padded and discarded). The
// causal key range is streamed KEY_PAGES pages (KEY_PAGES * PAGE_SIZE keys) at a
// time: keys from rotated_k, the views' keys already rotated to their positions by
// rotate_view_keys_kernel (paged_prefill.cu), and values from the pool
// through the page table. The causal and sliding-window masks are applied per
// (query row, key) in the online softmax; pages before the window of the tile's
// first row are not read. A second streaming pass recomputes QKᵀ and atomicAdds
// the per-key mass (summed over this tile's query rows attending the key, all for
// the ONE q head) into
// this q head's own column of the key's mass row. The QKᵀ/PV tiles live in
// attn_tiles.cuh; the op wrapper and the argument shapes are in paged_prefill.cu.
//
// NWARPS, KEY_PAGES and CTAS_PER_SM are launch shape, not geometry: every one of
// them is device specific and comes from the caller's tuning row
// (attn_tuning.hpp). KEY_PAGES only changes the order in which the online
// softmax folds keys in, so o and mass move in their last bits with it.
// CTAS_PER_SM is the occupancy floor ptxas must fit the register allocation to; 1
// leaves it unconstrained.

namespace pulsarlm {
namespace attn {

// sQ, sK, sV, sP, then sS. Every size is a multiple of 16 bytes, which keeps each tile 16-byte aligned for ldmatrix.
template <typename scalar_t, int PAGE_SIZE, int HEAD_DIM, int KEY_PAGES> constexpr size_t prefill_tile_bytes() {
    constexpr size_t key_tile = static_cast<size_t>(PAGE_SIZE) * KEY_PAGES;
    return sizeof(scalar_t) * (16 * HEAD_DIM + 2 * key_tile * HEAD_DIM + 16 * key_tile) +
        sizeof(float) * 16 * (key_tile + 1);
}

// Query tiles are 16 query tokens of one sequence, numbered in sequence order. Sets the
// sequence and first local query of tile; false past the last tile.
template <typename scalar_t>
__device__ __forceinline__ bool query_tile(const PagedAttnParams<scalar_t>& p, int tile, int& s, int& qbase) {
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

template <typename scalar_t, int PAGE_SIZE, int HEAD_DIM, bool BF16, int NWARPS, int KEY_PAGES, int CTAS_PER_SM>
__global__ void __launch_bounds__(NWARPS * 32, CTAS_PER_SM) attn_prefill_tc_kernel(
    const PagedAttnParams<scalar_t> p,
    const scalar_t* __restrict__ rotated_k  // [num_seqs * max_pages, page_size, n_kv_heads, head_dim]
) {
    constexpr int KEY_TILE = PAGE_SIZE * KEY_PAGES;
    // sS row stride. A stride of exactly KEY_TILE puts every row of a column on one
    // shared bank, and the two warp-0 sweeps below read a whole column at once.
    constexpr int SCORE_STRIDE = KEY_TILE + 1;
    // The per-row max and denominator sweeps give every query row a group of
    // kSweepLanes consecutive threads and reduce the group on shuffles. A group's
    // threads take consecutive score pairs, which keeps the two half-warps of a
    // warp on disjoint banks at SCORE_STRIDE odd.
    constexpr int kSweepLanes = 4;
    constexpr int kSweepCols = KEY_TILE / kSweepLanes;
    constexpr int kSweepThreads = 16 * kSweepLanes;
    static_assert(
        kSweepThreads % 32 == 0 && kSweepThreads <= NWARPS * 32,
        "the sweep groups must fill whole warps of the CTA"
    );
    static_assert(KEY_TILE % kSweepLanes == 0, "key tile must split over a group");
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int sweep_row = tid / kSweepLanes;
    const int sweep_lane = tid % kSweepLanes;
    const int blk = blockIdx.x;
    const int h = blk % p.n_q_heads;
    const int g = h / p.group;
    int s, qbase;
    if (!query_tile(p, blk / p.n_q_heads, s, qbase)) {
        return;
    }
    // This sequence's EMA gain alpha and the retention 1 - alpha it implies.
    const float mass_decay = p.mass ? p.mass_decay[s] : 0.0f;
    const float retention = 1.0f - mass_decay;
    const int q0 = p.cu_seqlens_q[s];
    const int seq_q = p.cu_seqlens_q[s + 1] - q0;
    const int ctx_len = p.seqlens_k[s];
    const int ctx_start = ctx_len - seq_q;
    const int tile_rows = min(16, seq_q - qbase);
    const int q_global_base = q0 + qbase;
    // Context pos of MMA row m is ctx_start + qbase + m; the tile's highest.
    const int max_p = ctx_start + qbase + tile_rows - 1;
    const int n_key_tiles = max_p / KEY_TILE + 1;  // tiles covering keys [0, max_p]
    const int tile_first_key = first_key(p, ctx_start + qbase);
    const int first_key_tile = tile_first_key / KEY_TILE;

    const int32_t* seq_table = p.page_tables + static_cast<int64_t>(s) * p.max_pages;
    const int64_t rotated_token_stride = static_cast<int64_t>(p.n_kv_heads) * HEAD_DIM;
    const int64_t rotated_page_stride = PAGE_SIZE * rotated_token_stride;

    static_assert(sizeof(scalar_t) == 2, "the tile staging copies 8 elements at a time");
    constexpr int kChunkElems = 16 / sizeof(scalar_t);
    constexpr int kRowChunks = HEAD_DIM / kChunkElems;
    constexpr int kPageChunks = PAGE_SIZE * kRowChunks;

    // sQ, sK and sV carry their columns permuted by attn::swizzled_offset; sP and sS
    // are reached by scalar accesses only and stay row-major.
    extern __shared__ __align__(16) unsigned char shared_memory[];
    scalar_t* sQ = reinterpret_cast<scalar_t*>(shared_memory);
    scalar_t* sK = sQ + 16 * HEAD_DIM;
    scalar_t* sV = sK + KEY_TILE * HEAD_DIM;
    scalar_t* sP = sV + KEY_TILE * HEAD_DIM;
    float* sS = reinterpret_cast<float*>(sP + 16 * KEY_TILE);
    __shared__ float Mrow[16], Lrow[16], LSE[16];
    // Online-softmax rescale of the row's running output, published so every warp
    // can fold it into its own share of the accumulator.
    __shared__ float row_rescale[16];
    __shared__ int qpos[16];  // context pos of each MMA row, -1 if padding
    __shared__ int qfirst[16];  // first key each MMA row attends
    // retention^(offset of this row from the chunk end), the multiplier on the mass this row hands out, so a key's
    // accumulated mass is the per-token EMA of the attention it received.
    __shared__ float mass_weight[16];

    // Load Q for this tile's rows into the MMA M dimension; pad rows >= tile_rows.
    for (int i = tid; i < 16 * HEAD_DIM; i += NWARPS * 32) {
        const int row = i / HEAD_DIM, col = i % HEAD_DIM;
        scalar_t v = static_cast<scalar_t>(0);
        if (row < tile_rows) {
            const int64_t qi = (static_cast<int64_t>(q_global_base + row) * p.n_q_heads + h) * HEAD_DIM;
            v = p.q[qi + col];
        }
        sQ[attn::swizzled_offset<HEAD_DIM>(row, col)] = v;
    }
    // This lane's share of the running output, one attn::kPvSlabAccums block per dim
    // slab this warp owns. attn::tile_pv_registers states the (row, column) each
    // entry carries.
    constexpr int kSlabs = attn::pv_slabs_per_warp<HEAD_DIM, NWARPS>();
    float out_acc[kSlabs * attn::kPvSlabAccums] = {};
    for (int i = tid; i < 16 * KEY_TILE; i += NWARPS * 32) {
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
        // __powf(0, 0) is NaN, so the zero exponent is taken directly.
        const int chunk_end_offset = seq_q - 1 - qbase - tid;
        const float graded = chunk_end_offset == 0 ? 1.0f : __powf(retention, static_cast<float>(chunk_end_offset));
        mass_weight[tid] = tid < tile_rows ? graded : 0.0f;
    }
    // The pages of key tile t, in the pool (lines, for V) and in rotated_k (sequence
    // s's view page b is page s * max_pages + b, for K). The staging address
    // arithmetic is a 64-bit multiply chain off these, so they are read a whole
    // iteration before the stage_tile that consumes them. A tile may reach past the
    // sequence's last page, or begin before the window; seq_table is only read for
    // the pages between.
    auto load_pages = [&](int t, int* lines, int* key_pages) {
#pragma unroll
        for (int kp = 0; kp < KEY_PAGES; ++kp) {
            const int page_base = t * KEY_TILE + kp * PAGE_SIZE;
            const bool in_view = t < n_key_tiles && page_base < ctx_len && page_base + PAGE_SIZE > tile_first_key;
            lines[kp] = in_view ? seq_table[t * KEY_PAGES + kp] : 0;
            key_pages[kp] = in_view ? s * p.max_pages + t * KEY_PAGES + kp : 0;
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
            for (int kp = 0; kp < KEY_PAGES; ++kp) {
                const int page_base = t * KEY_TILE + kp * PAGE_SIZE;
                const int valid = page_base + PAGE_SIZE > tile_first_key ? min(PAGE_SIZE, ctx_len - page_base) : 0;
                const int64_t page_first = static_cast<int64_t>(pages[kp]) * page_stride + g * head_stride;
                for (int c = tid; c < kPageChunks; c += NWARPS * 32) {
                    const int off = c / kRowChunks;
                    const int d = (c % kRowChunks) * kChunkElems;
                    const int64_t idx = page_first + static_cast<int64_t>(off) * token_stride + d;
                    attn::cp_async_16(
                        dst + attn::swizzled_offset<HEAD_DIM>(kp * PAGE_SIZE + off, d),
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
    int tile_lines[KEY_PAGES], next_lines[KEY_PAGES], tile_keys[KEY_PAGES], next_keys[KEY_PAGES];
    auto attends = [&](int m, int key) { return key >= qfirst[m] && key <= qpos[m]; };

    load_pages(first_key_tile, tile_lines, tile_keys);
    stage_tile(rotated_k, rotated_page_stride, rotated_token_stride, HEAD_DIM, sK, first_key_tile, tile_keys);
    stage_tile(p.v_pool, p.page_stride, p.token_stride, p.head_stride, sV, first_key_tile, tile_lines);
    __syncthreads();

    // Pass 1: online-softmax attention with per-(row,key) causal and window masking.
    for (int t = first_key_tile; t < n_key_tiles; ++t) {
        const int tile_base = t * KEY_TILE;
        load_pages(t + 1, next_lines, next_keys);
        attn::cp_async_wait<1>();
        __syncthreads();

        attn::tile_qkt<scalar_t, KEY_TILE, HEAD_DIM, BF16, SCORE_STRIDE, true>(sQ, sK, sS, warp, NWARPS);
        __syncthreads();
        stage_tile(rotated_k, rotated_page_stride, rotated_token_stride, HEAD_DIM, sK, t + 1, next_keys);

        if (tid < kSweepThreads) {
            float tmax = -CUDART_INF_F;
#pragma unroll
            for (int j = 0; j < kSweepCols; ++j) {
                const int c = sweep_lane * kSweepCols + j;
                if (attends(sweep_row, tile_base + c)) {
                    tmax = fmaxf(tmax, p.scale * sS[sweep_row * SCORE_STRIDE + c]);
                }
            }
#pragma unroll
            for (int off = kSweepLanes / 2; off > 0; off >>= 1) {
                tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, off));
            }
            if (sweep_lane == 0 && sweep_row < tile_rows) {
                const float newM = fmaxf(Mrow[sweep_row], tmax);
                // 0 when Mrow == -inf; 1 for a row whose window starts after this tile.
                row_rescale[sweep_row] = newM == -CUDART_INF_F ? 1.0f : __expf(Mrow[sweep_row] - newM);
                Mrow[sweep_row] = newM;
            }
        }
        __syncthreads();

        // One thread per (query row, key): the exponentials run across the CTA,
        // and sS carries each row's weight to the sequential sum below.
        for (int i = tid; i < 16 * KEY_TILE; i += NWARPS * 32) {
            const int m = i / KEY_TILE, c = i % KEY_TILE;
            float& score = sS[m * SCORE_STRIDE + c];
            const float pw = attends(m, tile_base + c) ? __expf(p.scale * score - Mrow[m]) : 0.0f;
            score = pw;
            sP[i] = static_cast<scalar_t>(pw);
        }
        __syncthreads();

        // Summed as a shuffle tree over the row's group, not in key order.
        if (tid < kSweepThreads) {
            float lsum = 0.0f;
#pragma unroll
            for (int j = 0; j < kSweepCols; ++j) {
                lsum += sS[sweep_row * SCORE_STRIDE + sweep_lane * kSweepCols + j];
            }
#pragma unroll
            for (int off = kSweepLanes / 2; off > 0; off >>= 1) {
                lsum += __shfl_xor_sync(0xffffffffu, lsum, off);
            }
            if (sweep_lane == 0 && sweep_row < tile_rows) {
                Lrow[sweep_row] = Lrow[sweep_row] * row_rescale[sweep_row] + lsum;
            }
        }
        attn::cp_async_wait<1>();
        __syncthreads();

        attn::tile_pv_registers<scalar_t, KEY_TILE, HEAD_DIM, BF16, NWARPS, true>(sP, sV, row_rescale, out_acc, warp);
        __syncthreads();
        stage_tile(p.v_pool, p.page_stride, p.token_stride, p.head_stride, sV, t + 1, next_lines);
    }

    // Normalize + write o, each lane over the accumulators it owns.
    {
        const int gid = lane >> 2;
        // A row past tile_rows has Lrow 0, and its reciprocal is never read.
        const float inv_low = 1.0f / Lrow[gid], inv_high = 1.0f / Lrow[gid + 8];
        attn::for_each_pv_register<HEAD_DIM, NWARPS>(out_acc, warp, [&](int row, int col, float value) {
            if (row < tile_rows) {
                p.o[(static_cast<int64_t>(q_global_base + row) * p.n_q_heads + h) * HEAD_DIM +
                    col] = static_cast<scalar_t>(value * (row < 8 ? inv_low : inv_high));
            }
        });
    }

    // Stash LSE for the mass pass.
    if (tid < tile_rows) {
        LSE[tid] = Mrow[tid] + logf(Lrow[tid]);
    }
    __syncthreads();

    // Pass 2: recompute QKᵀ and accumulate per-key mass (summed over this tile's
    // query rows with p >= kp, this ONE q head), atomicAdd into this q head's own
    // column of each key's mass row. Nothing else in the kernel touches mass, so a
    // layer whose mass is never read stops here.
    if (!p.mass) {
        return;
    }
    float* mass = seq_mass(p, s, ctx_len, PAGE_SIZE);
    // Only K is re-streamed here, so one tile is in flight and QKᵀ frees the shared
    // tile for the next one.
    load_pages(first_key_tile, tile_lines, tile_keys);
    stage_tile(rotated_k, rotated_page_stride, rotated_token_stride, HEAD_DIM, sK, first_key_tile, tile_keys);
    for (int t = first_key_tile; t < n_key_tiles; ++t) {
        const int tile_base = t * KEY_TILE;
        load_pages(t + 1, next_lines, next_keys);
        attn::cp_async_wait<0>();
        __syncthreads();

        attn::tile_qkt<scalar_t, KEY_TILE, HEAD_DIM, BF16, SCORE_STRIDE, true>(sQ, sK, sS, warp, NWARPS);
        __syncthreads();
        stage_tile(rotated_k, rotated_page_stride, rotated_token_stride, HEAD_DIM, sK, t + 1, next_keys);

        // One thread per (query row, key): the exponentials run across the CTA,
        // and sS carries each row's weighted share to the per-key sum below.
        for (int i = tid; i < 16 * KEY_TILE; i += NWARPS * 32) {
            const int m = i / KEY_TILE, c = i % KEY_TILE;
            float& score = sS[m * SCORE_STRIDE + c];
            score = attends(m, tile_base + c) ? mass_weight[m] * __expf(p.scale * score - LSE[m]) : 0.0f;
        }
        __syncthreads();

        // Summed in row order, so the accumulation is the sequential one.
        for (int c = tid; c < KEY_TILE; c += NWARPS * 32) {
            float ms = 0.0f;
            for (int m = 0; m < tile_rows; ++m) {
                ms += sS[m * SCORE_STRIDE + c];
            }
            if (ms > 0.0f) {
                atomicAdd(&mass[static_cast<int64_t>(tile_base + c) * p.n_q_heads + h], mass_decay * ms);
            }
        }
        __syncthreads();
    }
}

// Dynamic and static shared bytes of one CTA of attn_prefill_tc_kernel.
template <typename scalar_t, int PAGE_SIZE, int HEAD_DIM, int KEY_PAGES> constexpr size_t prefill_smem_bytes() {
    return prefill_tile_bytes<scalar_t, PAGE_SIZE, HEAD_DIM, KEY_PAGES>() + sizeof(float) * 5 * 16 +
        sizeof(int) * 2 * 16;
}

// KEY_PAGES, unless the CTA would then need to opt in to more shared memory.
template <typename scalar_t, int PAGE_SIZE, int HEAD_DIM, int KEY_PAGES> constexpr int prefill_key_pages() {
    return prefill_smem_bytes<scalar_t, PAGE_SIZE, HEAD_DIM, KEY_PAGES>() <= kSharedMemoryWithoutOptIn ? KEY_PAGES : 1;
}

}  // namespace attn
}  // namespace pulsarlm
