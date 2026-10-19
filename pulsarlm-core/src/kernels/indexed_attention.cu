#include "attn_common.cuh"

#include "attn_tiles.cuh"
#include "attn_tuning.hpp"
#include "paged_prefill.cuh"
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

    __device__ __forceinline__ uint32_t word(int query, int w) const {
        return this->bits[static_cast<int64_t>(query) * this->words + w];
    }

    __device__ __forceinline__ bool selects(int query, int key) const {
        const int block = key / this->block_size;
        return (this->bits[static_cast<int64_t>(query) * this->words + block / 32] >> (block % 32)) & 1u;
    }

    // Whether the query at view index query_pos attends a key of view page page.
    __device__ __forceinline__ bool attends_page(int query, int query_pos, int page, int page_size) const {
        for (int key = page * page_size; key < (page + 1) * page_size && key <= query_pos; key += this->block_size) {
            if (this->selects(query, key)) {
                return true;
            }
        }
        return false;
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

// One CTA per (query token, query head), as attn_scalar_kernel, its threads striding over pages up to the query's.
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
    const attn::KeyPositions layout = attn::key_positions(p, s, PAGE_SIZE);
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

    const float weight = p.mass ? p.mass_weights[qtok] : 0.0f;
    float m = -CUDART_INF_F;
    float l = 0.0f;
    float acc[attn::kMaxHeadDim];
    for (int d = 0; d < head_dim; ++d) {
        acc[d] = 0.0f;
    }
    for (int page = tid; page <= query_pos / PAGE_SIZE; page += attn::kThreads) {
        float page_max = -CUDART_INF_F;
        for (int j = page * PAGE_SIZE; j < min((page + 1) * PAGE_SIZE, query_pos + 1); ++j) {
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
            page_max = fmaxf(page_max, sc);
        }
        if (p.mass && page_max > -CUDART_INF_F) {
            const int64_t row = attn::mass_row(p, s, page);
            atomicAdd(&p.mass[row * p.n_q_heads + h], weight * page_max);
            if (h == 0) {
                atomicAdd(&p.attended_weight[row], weight);
            }
        }
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
    const float lse = M + logf(denom);
    for (int page = tid; page <= query_pos / PAGE_SIZE; page += attn::kThreads) {
        if (selection.attends_page(qtok, query_pos, page, PAGE_SIZE)) {
            atomicAdd(&p.mass[attn::mass_row(p, s, page) * p.n_q_heads + h], -weight * lse);
        }
    }
}

// One thread per (query row, selection word): marks the view pages of the row's sequence its selected blocks lie in.
template <typename scalar_t>
__global__ void mark_selected_pages_kernel(
    const attn::PagedAttnParams<scalar_t> p,
    const Selection selection,
    int total_q,
    int page_size,
    int32_t* __restrict__ selected_pages
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<int64_t>(total_q) * selection.words) {
        return;
    }
    const int row = static_cast<int>(index / selection.words);
    const int w = static_cast<int>(index % selection.words);
    const int s = sequence_of(p, row);
    for (uint32_t bits = selection.word(row, w); bits != 0; bits &= bits - 1) {
        const int page = (w * 32 + __ffs(bits) - 1) * selection.block_size / page_size;
        if (page < p.max_pages) {
            selected_pages[static_cast<int64_t>(s) * p.max_pages + page] = 1;
        }
    }
}

// Copies the keys of each sequence's selected view pages into rotated_k, rotated to their view positions, as
// rotate_view_keys_kernel.
template <typename scalar_t>
__global__ void rotate_indexed_keys_kernel(
    const attn::PagedAttnParams<scalar_t> p,
    int page_size,
    const int32_t* __restrict__ selected_pages,
    scalar_t* __restrict__ rotated_k
) {
    const int s = blockIdx.x / p.max_pages;
    const int view_page = blockIdx.x % p.max_pages;
    const int keys = min(page_size, p.seqlens_k[s] - view_page * page_size);
    if (keys <= 0 || !selected_pages[blockIdx.x]) {
        return;
    }
    const int half = p.head_dim / 2;
    const int pairs_per_key = p.n_kv_heads * half;
    const int64_t page_elems = static_cast<int64_t>(page_size) * p.n_kv_heads * p.head_dim;
    const attn::KeyPositions layout = attn::key_positions(p, s, page_size);
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

// One CTA per (query token, KV group): the MMA rows are the group's query heads, up to 16, and each key tile gathers
// the token's next selected blocks in view order. The tile, sweep and staging arithmetic is attn_prefill_tc_kernel's.
template <typename scalar_t, int PAGE_SIZE, int HEAD_DIM, bool BF16, int NWARPS, int KEY_PAGES, int CTAS_PER_SM>
__global__ void __launch_bounds__(NWARPS * 32, CTAS_PER_SM) attn_indexed_tc_kernel(
    const attn::PagedAttnParams<scalar_t> p,
    const Selection selection,
    const scalar_t* __restrict__ rotated_k  // [num_seqs * max_pages, page_size, n_kv_heads, head_dim]
) {
    constexpr int KEY_TILE = PAGE_SIZE * KEY_PAGES;
    constexpr int SCORE_STRIDE = KEY_TILE + 1;
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
    const int qtok = blockIdx.x / p.n_kv_heads;
    const int g = blockIdx.x % p.n_kv_heads;
    const int rows = p.group;
    const int first_head = g * p.group;
    const int s = sequence_of(p, qtok);
    const int q0 = p.cu_seqlens_q[s];
    const int seq_q = p.cu_seqlens_q[s + 1] - q0;
    const int ctx_len = p.seqlens_k[s];
    const int query_pos = ctx_len - seq_q + (qtok - q0);
    const int tile_blocks = KEY_TILE / selection.block_size;

    const int32_t* seq_table = p.page_tables + static_cast<int64_t>(s) * p.max_pages;
    const int64_t rotated_token_stride = static_cast<int64_t>(p.n_kv_heads) * HEAD_DIM;
    const int64_t rotated_page_stride = PAGE_SIZE * rotated_token_stride;

    static_assert(sizeof(scalar_t) == 2, "the tile staging copies 8 elements at a time");
    constexpr int kChunkElems = 16 / sizeof(scalar_t);
    constexpr int kRowChunks = HEAD_DIM / kChunkElems;
    constexpr int kTileChunks = KEY_TILE * kRowChunks;

    extern __shared__ __align__(16) unsigned char shared_memory[];
    scalar_t* sQ = reinterpret_cast<scalar_t*>(shared_memory);
    scalar_t* sK = sQ + 16 * HEAD_DIM;
    scalar_t* sV = sK + KEY_TILE * HEAD_DIM;
    scalar_t* sP = sV + KEY_TILE * HEAD_DIM;
    float* sS = reinterpret_cast<float*>(sP + 16 * KEY_TILE);
    __shared__ float Mrow[16], Lrow[16], LSE[16];
    __shared__ float row_rescale[16];
    // The view index of each key of the staged tiles, -1 past the selection or the query: the tile K holds, then the
    // one V holds, as the pipeline stages K a tile ahead of V.
    __shared__ int key_index[2][KEY_TILE];
    // The next selected block, as the selection word holding it and that word's bits from it on.
    __shared__ int cursor_word;
    __shared__ uint32_t cursor_bits;
    // Each row's max scaled score over the page its tiles are in; selected blocks come in view order, so a page's
    // keys are consecutive across tiles.
    __shared__ int run_page[16];
    __shared__ float run_max[16];
    const float weight = p.mass ? p.mass_weights[qtok] : 0.0f;
    auto flush_page = [&](int m) {
        if (run_page[m] >= 0) {
            const int64_t row = attn::mass_row(p, s, run_page[m]);
            atomicAdd(&p.mass[row * p.n_q_heads + first_head + m], weight * run_max[m]);
            if (g == 0 && m == 0) {
                atomicAdd(&p.attended_weight[row], weight);
            }
        }
    };

    for (int i = tid; i < 16 * HEAD_DIM; i += NWARPS * 32) {
        const int row = i / HEAD_DIM, col = i % HEAD_DIM;
        scalar_t v = static_cast<scalar_t>(0);
        if (row < rows) {
            v = p.q[(static_cast<int64_t>(qtok) * p.n_q_heads + first_head + row) * HEAD_DIM + col];
        }
        sQ[attn::swizzled_offset<HEAD_DIM>(row, col)] = v;
    }
    constexpr int kSlabs = attn::pv_slabs_per_warp<HEAD_DIM, NWARPS>();
    float out_acc[kSlabs * attn::kPvSlabAccums] = {};
    for (int i = tid; i < 16 * KEY_TILE; i += NWARPS * 32) {
        sP[i] = static_cast<scalar_t>(0);
    }
    if (tid < 16) {
        Mrow[tid] = -CUDART_INF_F;
        Lrow[tid] = 0.0f;
        row_rescale[tid] = 1.0f;
        run_page[tid] = -1;
    }
    if (tid == 0) {
        cursor_word = 0;
        cursor_bits = selection.word(qtok, 0);
    }
    __syncthreads();

    // Fills key_index[slot] with the next tile_blocks selected blocks at or before the query, from the cursor on; false
    // when none is left. Thread 0 walks the bits; every thread returns the same.
    __shared__ bool tile_found;
    auto next_tile = [&](int slot) {
        if (tid == 0) {
            int filled = 0;
            while (filled < tile_blocks && cursor_word < selection.words) {
                if (cursor_bits == 0) {
                    if (++cursor_word < selection.words) {
                        cursor_bits = selection.word(qtok, cursor_word);
                    }
                    continue;
                }
                const int block = cursor_word * 32 + __ffs(cursor_bits) - 1;
                if (block * selection.block_size > query_pos) {
                    cursor_word = selection.words;
                    break;
                }
                cursor_bits &= cursor_bits - 1;
                for (int k = 0; k < selection.block_size; ++k) {
                    const int key = block * selection.block_size + k;
                    key_index[slot][filled * selection.block_size + k] = key <= query_pos ? key : -1;
                }
                ++filled;
            }
            for (int c = filled * selection.block_size; c < KEY_TILE; ++c) {
                key_index[slot][c] = -1;
            }
            tile_found = filled > 0;
        }
        __syncthreads();
        return tile_found;
    };

    // Stages the keys of key_index[slot], K from rotated_k or V from the pool, zeros past the selection. Always commits
    // a group, as attn_prefill_tc_kernel's stage_tile.
    auto stage_tile = [&](bool keys, scalar_t* dst, int slot) {
        for (int c = tid; c < kTileChunks; c += NWARPS * 32) {
            const int row = c / kRowChunks;
            const int d = (c % kRowChunks) * kChunkElems;
            const int key = key_index[slot][row];
            const int view_page = max(key, 0) / PAGE_SIZE;
            const int offset = max(key, 0) % PAGE_SIZE;
            const scalar_t* src = keys
                ? rotated_k + (static_cast<int64_t>(s) * p.max_pages + view_page) * rotated_page_stride +
                    offset * rotated_token_stride + static_cast<int64_t>(g) * HEAD_DIM + d
                : p.v_pool + static_cast<int64_t>(seq_table[view_page]) * p.page_stride + offset * p.token_stride +
                    g * p.head_stride + d;
            attn::cp_async_16(dst + attn::swizzled_offset<HEAD_DIM>(row, d), src, key >= 0 ? 16 : 0);
        }
        attn::cp_async_commit();
    };

    auto attends = [&](int m, int c, int slot) { return m < rows && key_index[slot][c] >= 0; };

    // Pass 1: online softmax over the gathered tiles. Slot 0 is the tile in sK, slot 1 the one in sV.
    bool more = next_tile(0);
    if (more) {
        stage_tile(true, sK, 0);
        for (int i = tid; i < KEY_TILE; i += NWARPS * 32) {
            key_index[1][i] = key_index[0][i];
        }
        __syncthreads();
        stage_tile(false, sV, 1);
    }
    while (more) {
        attn::cp_async_wait<1>();
        __syncthreads();
        attn::tile_qkt<scalar_t, KEY_TILE, HEAD_DIM, BF16, SCORE_STRIDE, true>(sQ, sK, sS, warp, NWARPS);
        __syncthreads();
        more = next_tile(0);
        if (more) {
            stage_tile(true, sK, 0);
        } else {
            attn::cp_async_commit();
        }

        if (tid < kSweepThreads) {
            float tmax = -CUDART_INF_F;
#pragma unroll
            for (int j = 0; j < kSweepCols; ++j) {
                const int c = sweep_lane * kSweepCols + j;
                if (attends(sweep_row, c, 1)) {
                    tmax = fmaxf(tmax, p.scale * sS[sweep_row * SCORE_STRIDE + c]);
                }
            }
#pragma unroll
            for (int off = kSweepLanes / 2; off > 0; off >>= 1) {
                tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, off));
            }
            if (sweep_lane == 0 && sweep_row < rows) {
                const float newM = fmaxf(Mrow[sweep_row], tmax);
                row_rescale[sweep_row] = newM == -CUDART_INF_F ? 1.0f : __expf(Mrow[sweep_row] - newM);
                Mrow[sweep_row] = newM;
            }
        }
        if (p.mass && tid < rows) {
            for (int c = 0; c < KEY_TILE; ++c) {
                const int key = key_index[1][c];
                if (key < 0) {
                    continue;
                }
                const float score = p.scale * sS[tid * SCORE_STRIDE + c];
                if (key / PAGE_SIZE != run_page[tid]) {
                    flush_page(tid);
                    run_page[tid] = key / PAGE_SIZE;
                    run_max[tid] = score;
                } else {
                    run_max[tid] = fmaxf(run_max[tid], score);
                }
            }
        }
        __syncthreads();

        for (int i = tid; i < 16 * KEY_TILE; i += NWARPS * 32) {
            const int m = i / KEY_TILE, c = i % KEY_TILE;
            float& score = sS[m * SCORE_STRIDE + c];
            const float pw = attends(m, c, 1) ? __expf(p.scale * score - Mrow[m]) : 0.0f;
            score = pw;
            sP[i] = static_cast<scalar_t>(pw);
        }
        __syncthreads();

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
            if (sweep_lane == 0 && sweep_row < rows) {
                Lrow[sweep_row] = Lrow[sweep_row] * row_rescale[sweep_row] + lsum;
            }
        }
        attn::cp_async_wait<1>();
        __syncthreads();

        attn::tile_pv_registers<scalar_t, KEY_TILE, HEAD_DIM, BF16, NWARPS, true>(sP, sV, row_rescale, out_acc, warp);
        __syncthreads();
        if (more) {
            for (int i = tid; i < KEY_TILE; i += NWARPS * 32) {
                key_index[1][i] = key_index[0][i];
            }
            __syncthreads();
            stage_tile(false, sV, 1);
        }
    }
    attn::cp_async_wait<0>();

    {
        const int gid = lane >> 2;
        const float inv_low = 1.0f / Lrow[gid], inv_high = 1.0f / Lrow[gid + 8];
        attn::for_each_pv_register<HEAD_DIM, NWARPS>(out_acc, warp, [&](int row, int col, float value) {
            if (row < rows) {
                p.o[(static_cast<int64_t>(qtok) * p.n_q_heads + first_head + row) * HEAD_DIM +
                    col] = static_cast<scalar_t>(value * (row < 8 ? inv_low : inv_high));
            }
        });
    }

    if (!p.mass) {
        return;
    }
    if (tid < rows) {
        flush_page(tid);
        LSE[tid] = Mrow[tid] + logf(Lrow[tid]);
    }
    __syncthreads();
    for (int page = tid; page <= query_pos / PAGE_SIZE; page += NWARPS * 32) {
        if (selection.attends_page(qtok, query_pos, page, PAGE_SIZE)) {
            const int64_t row = attn::mass_row(p, s, page);
            for (int m = 0; m < rows; ++m) {
                atomicAdd(&p.mass[row * p.n_q_heads + first_head + m], -weight * LSE[m]);
            }
        }
    }
}

template <typename scalar_t, bool BF16>
void launch_indexed_tc(const attn::PagedAttnBatch& b, const Selection& selection, cudaStream_t stream) {
    const int64_t grid = b.total_q * b.n_kv_heads;
    const auto params = b.params<scalar_t>();
    const int64_t view_pages = b.num_seqs * b.max_pages;
    auto selected_pages = at::zeros({view_pages}, b.page_tables.options());
    const int64_t marks = b.total_q * selection.words;
    mark_selected_pages_kernel<scalar_t><<<static_cast<int>((marks + 255) / 256), 256, 0, stream>>>(
        params,
        selection,
        static_cast<int>(b.total_q),
        static_cast<int>(b.page_size),
        selected_pages.data_ptr<int32_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto rotated_k = at::empty({view_pages, b.page_size, b.n_kv_heads, b.head_dim}, b.k_pool.options());
    rotate_indexed_keys_kernel<scalar_t><<<static_cast<int>(view_pages), 256, 0, stream>>>(
        params,
        static_cast<int>(b.page_size),
        selected_pages.const_data_ptr<int32_t>(),
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
    const at::Tensor& page_positions,
    const rope_parameters& rope,
    double scale,
    const at::Tensor& selected,
    int64_t block_size,
    const std::optional<at::Tensor>& mass,
    const std::optional<at::Tensor>& cu_view_pages,
    const std::optional<at::Tensor>& mass_weights
) {
    const auto b = attn::paged_attn_batch(
        "attn_indexed",
        q,
        k_pool,
        v_pool,
        page_tables,
        cu_seqlens_q,
        seqlens_k,
        page_positions,
        rope,
        scale,
        std::nullopt,
        std::nullopt,
        mass,
        cu_view_pages,
        mass_weights
    );
    TORCH_CHECK(
        block_size >= 1 && b.page_size % block_size == 0,
        "attn_indexed: block_size must be at least 1 and divide the page size"
    );
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
        attn::finish_mass(b, stream);
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
    attn::finish_mass(b, stream);
    return b.o;
}

}  // namespace pulsarlm
