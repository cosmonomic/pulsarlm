#include "attn_common.cuh"

#include "pulsarlm/kernels/attn_tiles.cuh"
#include "pulsarlm/kernels/attn_tuning.hpp"
#include "pulsarlm/kernels/paged_prefill.cuh"
#include "pulsarlm/ops.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <math_constants.h>

#include <cstdint>
#include <optional>
#include <string>

// Indexed attention: paged prefill attention over a ragged batch, decode included as one query per sequence, in which
// each query attends only the keys of its selected blocks, up to its own view index. The kernels are copies of
// paged_prefill.cuh's and paged_common.cu's without the sliding window and the key norm, reading the selection where
// those read the causal mask.

namespace pulsarlm {
namespace {

// The selection of one batch: bit b of query q's row, word b / 32, bit b % 32, selects view keys
// [b * block_size, (b + 1) * block_size).
struct Selection {
    const uint32_t* bits;
    int words;
    int block_size;

    __device__ __forceinline__ bool selects(int query, int key) const {
        const int block = key / this->block_size;
        return (this->bits[static_cast<int64_t>(query) * this->words + block / 32] >> (block % 32)) & 1u;
    }

    // The keys up to query_pos the query attends.
    __device__ int attended_keys(int query, int query_pos) const {
        const uint32_t* row = this->bits + static_cast<int64_t>(query) * this->words;
        const int last_block = query_pos / this->block_size;
        int blocks = 0;
        for (int w = 0; w < last_block / 32; ++w) {
            blocks += __popc(row[w]);
        }
        const uint32_t through_last = 0xffffffffu >> (31 - last_block % 32);
        blocks += __popc(row[last_block / 32] & through_last);
        const int keys = blocks * this->block_size;
        return this->selects(query, query_pos) ? keys - (this->block_size - 1 - query_pos % this->block_size) : keys;
    }
};

// The sequence of query token qtok, by binary search over cu_seqlens_q.
template <typename scalar_t> __device__ int sequence_of(const attn::PagedAttnParams<scalar_t>& p, int qtok) {
    int lo = 0, hi = p.num_seqs;
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (p.cu_seqlens_q[mid] <= qtok) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// One CTA per (query token, query head), as attn_scalar_kernel.
template <typename scalar_t, int PAGE_SIZE>
__global__ void attn_indexed_scalar_kernel(const attn::PagedAttnParams<scalar_t> p, const Selection selection) {
    const int tid = threadIdx.x;
    const int blk = blockIdx.x;
    const int qtok = blk / p.n_q_heads;
    const int h = blk % p.n_q_heads;
    const int g = h / p.group;
    const int head_dim = p.head_dim;
    const int half = head_dim / 2;

    const int s = sequence_of(p, qtok);
    const int q0 = p.cu_seqlens_q[s];
    const int seq_q = p.cu_seqlens_q[s + 1] - q0;
    const int ctx_len = p.seqlens_k[s];
    const int query_pos = ctx_len - seq_q + (qtok - q0);
    const attn::KeyLayout layout = attn::key_layout(p.rope_layout, s);
    const int32_t* seq_table = p.page_tables + static_cast<int64_t>(s) * p.max_pages;

    // Dynamic shared: [ qsh(head_dim) | out_acc(head_dim) | red(kThreads) ]
    extern __shared__ float smem[];
    float* qsh = smem;
    float* out_acc = qsh + head_dim;
    float* red = out_acc + head_dim;

    const scalar_t* qrow = p.q + (static_cast<int64_t>(qtok) * p.n_q_heads + h) * head_dim;
    for (int d = tid; d < head_dim; d += attn::kThreads) {
        qsh[d] = static_cast<float>(qrow[d]);
    }
    __syncthreads();

    auto key_offset = [&](int j) {
        const int line = seq_table[j / PAGE_SIZE];
        return static_cast<int64_t>(line) * p.page_stride + (j % PAGE_SIZE) * p.token_stride + g * p.head_stride;
    };
    auto score = [&](int j, int64_t offset) {
        const scalar_t* krow = p.k_pool + offset;
        const int pos = layout.position(j);
        float sc = 0.0f;
        for (int i = 0; i < half; ++i) {
            float cos_a, sin_a, lo, hi;
            attn::rope_sincos_turns(pos, p.rope_turns[i], cos_a, sin_a);
            attn::stored_key_pair(p, krow, i, half, lo, hi);
            const auto k = attn::rope_rotate<scalar_t>(lo, hi, cos_a, sin_a);
            sc += qsh[i] * static_cast<float>(k.lo) + qsh[i + half] * static_cast<float>(k.hi);
        }
        return sc * p.scale;
    };

    float m = -CUDART_INF_F;
    float l = 0.0f;
    float acc[attn::kMaxHeadDim];
    for (int d = 0; d < head_dim; ++d) {
        acc[d] = 0.0f;
    }
    for (int j = tid; j <= query_pos; j += attn::kThreads) {
        if (!selection.selects(qtok, j)) {
            continue;
        }
        const int64_t offset = key_offset(j);
        const float sc = score(j, offset);
        const float new_m = fmaxf(m, sc);
        const float corr = __expf(m - new_m);  // 0 when m == -inf
        const float pw = __expf(sc - new_m);
        l = l * corr + pw;
        const scalar_t* vrow = p.v_pool + offset;
        for (int d = 0; d < head_dim; ++d) {
            acc[d] = acc[d] * corr + pw * static_cast<float>(vrow[d]);
        }
        m = new_m;
    }

    const float M = attn::block_reduce_max<attn::kThreads>(red, tid, m);
    const float w = M == -CUDART_INF_F ? 0.0f : __expf(m - M);  // 0 when this thread saw no keys
    for (int d = tid; d < head_dim; d += attn::kThreads) {
        out_acc[d] = 0.0f;
    }
    __syncthreads();
    for (int d = 0; d < head_dim; ++d) {
        atomicAdd(&out_acc[d], acc[d] * w);
    }
    const float denom = attn::block_reduce_sum<attn::kThreads>(red, tid, l * w);

    scalar_t* orow = p.o + (static_cast<int64_t>(qtok) * p.n_q_heads + h) * head_dim;
    const float inv_denom = 1.0f / denom;
    for (int d = tid; d < head_dim; d += attn::kThreads) {
        orow[d] = static_cast<scalar_t>(out_acc[d] * inv_denom);
    }

    if (!p.mass) {
        return;
    }
    // As attn_scalar_kernel's, the length gain stated against the keys this query attends.
    const float mass_decay = p.mass_decay[s];
    const int chunk_end_offset = seq_q - 1 - (qtok - q0);
    const float graded = chunk_end_offset == 0 ? 1.0f : __powf(1.0f - mass_decay, static_cast<float>(chunk_end_offset));
    const float attended_len = p.mass_length_gain > 0.0f ? p.mass_length_gain
                                                         : static_cast<float>(selection.attended_keys(qtok, query_pos));
    const float mass_weight = mass_decay * graded * attended_len;
    float* mass = attn::seq_mass(p, s, ctx_len, PAGE_SIZE);
    for (int j = tid; j <= query_pos; j += attn::kThreads) {
        if (selection.selects(qtok, j)) {
            const float pw = __expf(score(j, key_offset(j)) - M) * inv_denom;
            atomicAdd(&mass[static_cast<int64_t>(j) * p.n_q_heads + h], mass_weight * pw);
        }
    }
}

// Copies every key of each sequence's view into rotated_k, rotated to its view position, as rotate_view_keys_kernel.
template <typename scalar_t>
__global__ void
rotate_indexed_keys_kernel(const attn::PagedAttnParams<scalar_t> p, int page_size, scalar_t* __restrict__ rotated_k) {
    const int s = blockIdx.x / p.max_pages;
    const int view_page = blockIdx.x % p.max_pages;
    const int keys = min(page_size, p.seqlens_k[s] - view_page * page_size);
    if (keys <= 0) {
        return;
    }
    const int half = p.head_dim / 2;
    const int pairs_per_key = p.n_kv_heads * half;
    const int64_t page_elems = static_cast<int64_t>(page_size) * p.n_kv_heads * p.head_dim;
    const attn::KeyLayout layout = attn::key_layout(p.rope_layout, s);
    const int line = p.page_tables[static_cast<int64_t>(s) * p.max_pages + view_page];
    const scalar_t* src = p.k_pool + line * p.page_stride;
    scalar_t* dst = rotated_k + blockIdx.x * page_elems;
    for (int i = threadIdx.x; i < keys * pairs_per_key; i += blockDim.x) {
        const int off = i / pairs_per_key;
        const int head = (i % pairs_per_key) / half;
        const int pair = i % half;
        const int64_t row = (static_cast<int64_t>(off) * p.n_kv_heads + head) * p.head_dim;
        const int64_t stored = off * p.token_stride + head * p.head_stride;
        float cos_a, sin_a, lo, hi;
        attn::rope_sincos_turns(layout.position(view_page * page_size + off), p.rope_turns[pair], cos_a, sin_a);
        attn::stored_key_pair(p, src + stored, pair, half, lo, hi);
        const auto k = attn::rope_rotate<scalar_t>(lo, hi, cos_a, sin_a);
        dst[row + pair] = k.lo;
        dst[row + pair + half] = k.hi;
    }
}

// attn_prefill_tc_kernel, each query row attending its selected keys up to its own view index.
template <typename scalar_t, int PAGE_SIZE, int HEAD_DIM, bool BF16, int NWARPS, int KEY_PAGES, int CTAS_PER_SM>
__global__ void __launch_bounds__(NWARPS * 32, CTAS_PER_SM) attn_indexed_tc_kernel(
    const attn::PagedAttnParams<scalar_t> p,
    const Selection selection,
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
    if (!attn::query_tile(p, blk / p.n_q_heads, s, qbase)) {
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
    // Per-query multiplier on the mass this row hands out, as attn_prefill_tc_kernel's, the length gain stated against
    // the keys THAT row attends.
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
        // Offset from the chunk end: last query in the chunk (seq_q-1) gets
        // retention^0. __powf(0, 0) is NaN, so the zero exponent is taken directly.
        const int chunk_end_offset = seq_q - 1 - qbase - tid;
        float attended_len = p.mass_length_gain;
        if (p.mass && p.mass_length_gain <= 0.0f && tid < tile_rows) {
            attended_len = static_cast<float>(selection.attended_keys(q_global_base + tid, row_pos));
        }
        mass_weight[tid] = (tid < tile_rows)
            ? (chunk_end_offset == 0 ? 1.0f : __powf(retention, static_cast<float>(chunk_end_offset))) * attended_len
            : 0.0f;
    }
    // The pages of key tile t, in the pool (lines, for V) and in rotated_k (sequence
    // s's view page b is page s * max_pages + b, for K). A tile may reach past the
    // sequence's last page; seq_table is only read for the pages before it.
    auto load_pages = [&](int t, int* lines, int* key_pages) {
#pragma unroll
        for (int kp = 0; kp < KEY_PAGES; ++kp) {
            const int page_base = t * KEY_TILE + kp * PAGE_SIZE;
            const bool in_view = t < n_key_tiles && page_base < ctx_len;
            lines[kp] = in_view ? seq_table[t * KEY_PAGES + kp] : 0;
            key_pages[kp] = in_view ? s * p.max_pages + t * KEY_PAGES + kp : 0;
        }
    };

    // Stage one key tile into its shared tile. Always commits a group, empty tile
    // included, so every iteration costs the same number of groups and a wait count
    // means the same thing throughout.
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
                const int valid = min(PAGE_SIZE, ctx_len - page_base);
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
    // the bottom, so each is staged one full iteration ahead of its use.
    int tile_lines[KEY_PAGES], next_lines[KEY_PAGES], tile_keys[KEY_PAGES], next_keys[KEY_PAGES];
    auto attends = [&](int m, int key) { return key <= qpos[m] && selection.selects(q_global_base + m, key); };

    load_pages(0, tile_lines, tile_keys);
    stage_tile(rotated_k, rotated_page_stride, rotated_token_stride, HEAD_DIM, sK, 0, tile_keys);
    stage_tile(p.v_pool, p.page_stride, p.token_stride, p.head_stride, sV, 0, tile_lines);
    __syncthreads();

    // Pass 1: online-softmax attention with per-(row,key) causal and selection masking.
    for (int t = 0; t < n_key_tiles; ++t) {
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
                // 1 while the row has attended no key.
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

    if (tid < tile_rows) {
        LSE[tid] = Mrow[tid] + logf(Lrow[tid]);
    }
    __syncthreads();

    // Pass 2: recompute QKᵀ and accumulate per-key mass into this q head's own column of each key's mass row.
    if (!p.mass) {
        return;
    }
    float* mass = attn::seq_mass(p, s, ctx_len, PAGE_SIZE);
    load_pages(0, tile_lines, tile_keys);
    stage_tile(rotated_k, rotated_page_stride, rotated_token_stride, HEAD_DIM, sK, 0, tile_keys);
    for (int t = 0; t < n_key_tiles; ++t) {
        const int tile_base = t * KEY_TILE;
        load_pages(t + 1, next_lines, next_keys);
        attn::cp_async_wait<0>();
        __syncthreads();

        attn::tile_qkt<scalar_t, KEY_TILE, HEAD_DIM, BF16, SCORE_STRIDE, true>(sQ, sK, sS, warp, NWARPS);
        __syncthreads();
        stage_tile(rotated_k, rotated_page_stride, rotated_token_stride, HEAD_DIM, sK, t + 1, next_keys);

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

template <typename scalar_t, bool BF16>
void launch_indexed_tc(const attn::PagedAttnBatch& b, const Selection& selection, cudaStream_t stream) {
    // A sequence of n queries takes ceil(n / 16) query tiles, at most n / 16 + 1; the CTAs past the last tile exit.
    const int64_t grid = (b.total_q / 16 + b.num_seqs) * b.n_q_heads;
    const auto params = b.params<scalar_t>();
    const int64_t view_pages = b.num_seqs * b.max_pages;
    auto rotated_k = at::empty({view_pages, b.page_size, b.n_kv_heads, b.head_dim}, b.k_pool.options());
    rotate_indexed_keys_kernel<scalar_t><<<static_cast<int>(view_pages), 256, 0, stream>>>(
        params,
        static_cast<int>(b.page_size),
        rotated_k.data_ptr<scalar_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    attn::dispatch_page_head("attn_indexed", b.page_size, b.head_dim, [&](auto bs_tag, auto hd_tag) {
        constexpr int BS = decltype(bs_tag)::value;
        constexpr int HD = decltype(hd_tag)::value;
        attn::dispatch_attn_tuning([&](auto row_tag) {
            constexpr attn::AttnTuning kTuning = attn::kAttnTuningTable[decltype(row_tag)::value];
            constexpr int NWARPS = kTuning.prefill_warps;
            constexpr int KEY_PAGES = attn::prefill_key_pages<scalar_t, BS, HD, kTuning.prefill_key_pages>();
            constexpr size_t smem = attn::prefill_tile_bytes<scalar_t, BS, HD, KEY_PAGES>();
            const auto
                kernel = attn_indexed_tc_kernel<scalar_t, BS, HD, BF16, NWARPS, KEY_PAGES, kTuning.prefill_ctas_per_sm>;
            attn::allow_shared_memory(kernel, smem);
            kernel<<<static_cast<int>(grid), NWARPS * 32, smem, stream>>>(
                params,
                selection,
                rotated_k.data_ptr<scalar_t>()
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}

void launch_indexed_scalar(const attn::PagedAttnBatch& b, const Selection& selection, cudaStream_t stream) {
    const int64_t blocks = b.total_q * b.n_q_heads;
    const size_t smem = static_cast<size_t>(2 * b.head_dim + attn::kThreads) * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, b.q.scalar_type(), "attn_indexed_scalar", [&] {
        attn::dispatch_page_size("attn_indexed", b.page_size, [&](auto page_size_tag) {
            constexpr int PS = decltype(page_size_tag)::value;
            attn_indexed_scalar_kernel<scalar_t, PS>
                <<<static_cast<int>(blocks), attn::kThreads, smem, stream>>>(b.params<scalar_t>(), selection);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}

}  // namespace

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
) {
    const auto b = attn::paged_attn_batch(
        "attn_indexed",
        q,
        k_pool,
        v_pool,
        page_tables,
        cu_seqlens_q,
        seqlens_k,
        rope_layout,
        rope,
        scale,
        std::nullopt,
        std::nullopt,
        mass,
        cu_view_pages,
        attention_mass_decay,
        mass_length_gain,
        std::nullopt
    );
    TORCH_CHECK(block_size >= 1, "attn_indexed: block_size must be at least 1");
    TORCH_CHECK(
        selected.is_cuda() && selected.scalar_type() == at::kInt && selected.dim() == 2 &&
            selected.size(0) == b.total_q && selected.is_contiguous() &&
            selected.size(1) * 32 * block_size >= b.max_pages * b.page_size,
        "attn_indexed: selected must be contiguous int32 [total_q, words] on CUDA, a bit for each block of block_size "
        "keys of the page tables"
    );
    if (b.total_q == 0 || b.num_seqs == 0) {
        return b.o;
    }
    const Selection selection{
        .bits = reinterpret_cast<const uint32_t*>(selected.const_data_ptr<int32_t>()),
        .words = static_cast<int>(selected.size(1)),
        .block_size = static_cast<int>(block_size),
    };
    auto stream = at::cuda::getCurrentCUDAStream();
    if (b.q.scalar_type() == at::kFloat) {
        launch_indexed_scalar(b, selection, stream);
        return b.o;
    }
    std::optional<std::string> unsupported = attention_unsupported_reason(
        b.q.scalar_type(),
        b.page_size,
        b.head_dim,
        b.n_q_heads,
        b.n_kv_heads
    );
    TORCH_CHECK(!unsupported, "attn_indexed: ", *unsupported);
    attn::dispatch_tensor_core_dtype(b.q.scalar_type(), [&](auto type_tag, auto bf16_tag) {
        using scalar_t = typename decltype(type_tag)::type;
        launch_indexed_tc<scalar_t, decltype(bf16_tag)::value>(b, selection, stream);
    });
    return b.o;
}

}  // namespace pulsarlm
