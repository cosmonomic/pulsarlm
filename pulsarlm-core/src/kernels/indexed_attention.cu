#include "pulsarlm/kernels/dependent_launch.cuh"
#include "pulsarlm/kernels/attn_common.cuh"

#include "pulsarlm/kernels/attn_tiles.cuh"
#include "pulsarlm/kernels/attn_tuning.hpp"
#include "pulsarlm/kernels/paged_prefill.cuh"
#include "pulsarlm/kernels/ops.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <math_constants.h>

#include <algorithm>
#include <climits>
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
struct block_selection {
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
};

// The sequence of query token qtok, by binary search over cu_seqlens_q, or qtok without it.
template <typename scalar_t> __device__ int sequence_of(const attn::paged_attn_params<scalar_t>& p, int qtok) {
    if (p.cu_seqlens_q == nullptr) {
        return qtok;
    }
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
template <typename scalar_t, int page_size>
__global__ void attn_indexed_scalar_kernel(const attn::paged_attn_params<scalar_t> p, const block_selection selection) {
    wait_for_prior_kernels();
    allow_next_kernel();
    const int tid = threadIdx.x;
    const int blk = blockIdx.x;
    const int qtok = blk / p.n_q_heads;
    const int h = blk % p.n_q_heads;
    const int g = h / p.group;
    const int head_dim = p.head_dim;
    const int half = head_dim / 2;

    const int s = sequence_of(p, qtok);
    const int q0 = p.cu_seqlens_q ? p.cu_seqlens_q[s] : s;
    const int seq_q = p.cu_seqlens_q ? p.cu_seqlens_q[s + 1] - q0 : 1;
    const int ctx_len = p.seqlens_k[s];
    const int query_pos = ctx_len - seq_q + (qtok - q0);
    const attn::key_positions layout = attn::sequence_key_positions(p, s, page_size);
    const int32_t* seq_table = p.page_tables + static_cast<int64_t>(s) * p.max_pages;

    // Dynamic shared: [ qsh(head_dim) | out_acc(head_dim) | red(cta_threads) ]
    extern __shared__ float smem[];
    float* qsh = smem;
    float* out_acc = qsh + head_dim;
    float* red = out_acc + head_dim;

    const scalar_t* qrow = p.q + (static_cast<int64_t>(qtok) * p.n_q_heads + h) * head_dim;
    for (int d = tid; d < head_dim; d += attn::cta_threads) {
        qsh[d] = static_cast<float>(qrow[d]);
    }
    __syncthreads();

    auto key_offset = [&](int j) {
        const int line = seq_table[j / page_size];
        return static_cast<int64_t>(line) * p.page_stride + (j % page_size) * p.token_stride + g * p.head_stride;
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

    const float weight = p.mass_sum ? p.query_weights[qtok] : 0.0f;
    float m = -CUDART_INF_F;
    float l = 0.0f;
    float acc[attn::max_head_dim];
    for (int d = 0; d < head_dim; ++d) {
        acc[d] = 0.0f;
    }
    for (int page = tid; page <= query_pos / page_size; page += attn::cta_threads) {
        float page_max = -CUDART_INF_F;
        for (int j = page * page_size; j < min((page + 1) * page_size, query_pos + 1); ++j) {
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
        if (p.mass_sum && page_max > -CUDART_INF_F) {
            const int64_t row = attn::mass_row(p, attn::mass_view(p, s, qtok), page);
            atomicAdd(&p.mass_sum[row * p.n_q_heads + h], weight * page_max);
            if (h == 0) {
                atomicAdd(&p.weight_sum[row], weight);
            }
        }
    }

    const float block_max = attn::block_reduce_max<attn::cta_threads>(red, tid, m);
    const float w = block_max == -CUDART_INF_F ? 0.0f : __expf(m - block_max);  // 0 when this thread saw no keys
    for (int d = tid; d < head_dim; d += attn::cta_threads) {
        out_acc[d] = 0.0f;
    }
    __syncthreads();
    for (int d = 0; d < head_dim; ++d) {
        atomicAdd(&out_acc[d], acc[d] * w);
    }
    const float denom = attn::block_reduce_sum<attn::cta_threads>(red, tid, l * w);

    scalar_t* orow = p.o + (static_cast<int64_t>(qtok) * p.n_q_heads + h) * head_dim;
    const float inv_denom = 1.0f / denom;
    for (int d = tid; d < head_dim; d += attn::cta_threads) {
        orow[d] = static_cast<scalar_t>(out_acc[d] * inv_denom);
    }
}

// One thread per (query row, selection word): marks the view pages of the row's sequence its selected blocks lie in.
template <typename scalar_t>
__global__ void mark_selected_pages_kernel(
    const attn::paged_attn_params<scalar_t> p,
    const block_selection selection,
    int total_q,
    int page_size,
    int32_t* __restrict__ selected_pages
) {
    wait_for_prior_kernels();
    allow_next_kernel();
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

constexpr int slot_threads = 1024;

// The sum of value over threads [0, threadIdx.x) of a slot_threads-thread block; sums holds slot_threads / 32 entries.
__device__ int exclusive_block_sum(int value, int* sums) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    int inclusive = value;
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
        if (lane >= offset) {
            inclusive += other;
        }
    }
    if (lane == 31) {
        sums[warp] = inclusive;
    }
    __syncthreads();
    if (warp == 0) {
        int warp_sum = sums[lane];
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int other = __shfl_up_sync(0xffffffffu, warp_sum, offset);
            if (lane >= offset) {
                warp_sum += other;
            }
        }
        sums[lane] = warp_sum;
    }
    __syncthreads();
    return inclusive - value + (warp > 0 ? sums[warp - 1] : 0);
}

// One CTA, for a batch of one query per sequence: page_slots[i] is the selected view pages up to and including view
// page i of the flattened [num_seqs, max_pages] tables, less one, a page being selected when the query of its sequence
// selects one of its blocks. Each thread takes a run of consecutive pages.
template <typename scalar_t>
__global__ void __launch_bounds__(slot_threads) single_query_page_slots_kernel(
    const attn::paged_attn_params<scalar_t> p,
    const block_selection selection,
    int page_size,
    int32_t* __restrict__ page_slots
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    __shared__ int warp_sums[slot_threads / 32];
    const int total = p.num_seqs * p.max_pages;
    const int run = (total + slot_threads - 1) / slot_threads;
    const int first = min(static_cast<int>(threadIdx.x) * run, total);
    const int last = min(first + run, total);
    const int page_blocks = page_size / selection.block_size;
    auto selected = [&](int i) {
        const int s = i / p.max_pages;
        const int first_block = i % p.max_pages * page_blocks;
        for (int b = first_block; b < first_block + page_blocks; ++b) {
            if (b / 32 < selection.words && ((selection.word(s, b / 32) >> (b % 32)) & 1u)) {
                return 1;
            }
        }
        return 0;
    };
    int count = 0;
    for (int i = first; i < last; ++i) {
        count += selected(i);
    }
    int slots = exclusive_block_sum(count, warp_sums);
    for (int i = first; i < last; ++i) {
        slots += selected(i);
        page_slots[i] = slots - 1;
    }
}

// Copies the keys of each sequence's selected view pages into rotated_k, rotated to their view positions, as
// rotate_view_keys_kernel, one CTA per row of rotated_k. page_slots[view page] is the page's row of rotated_k, the
// selected pages before it, and differs from the previous page's for a selected page, so a row's page is the first
// whose slot reaches it.
template <typename scalar_t>
__global__ void rotate_indexed_keys_kernel(
    const attn::paged_attn_params<scalar_t> p,
    int page_size,
    const int32_t* __restrict__ page_slots,
    scalar_t* __restrict__ rotated_k
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    const int slot = static_cast<int>(blockIdx.x);
    const int view_pages = p.num_seqs * p.max_pages;
    int first = 0;
    int last = view_pages;
    while (first < last) {
        const int middle = (first + last) / 2;
        if (page_slots[middle] < slot) {
            first = middle + 1;
        } else {
            last = middle;
        }
    }
    if (first == view_pages || page_slots[first] != slot) {
        return;
    }
    const int s = first / p.max_pages;
    const int view_page = first % p.max_pages;
    const int keys = min(page_size, p.seqlens_k[s] - view_page * page_size);
    if (keys <= 0) {
        return;
    }
    const int half = p.head_dim / 2;
    const int pairs_per_key = p.n_kv_heads * half;
    const int64_t page_elems = static_cast<int64_t>(page_size) * p.n_kv_heads * p.head_dim;
    const attn::key_positions layout = attn::sequence_key_positions(p, s, page_size);
    const int line = p.page_tables[static_cast<int64_t>(s) * p.max_pages + view_page];
    const scalar_t* src = p.k_pool + line * p.page_stride;
    scalar_t* dst = rotated_k + slot * page_elems;
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

// The most blocks one query may select.
constexpr int max_listed_blocks = 1024;

// A partial row holds the max scaled score, the weight sum and the weighted values [head_dim].
constexpr int partial_row_floats(int head_dim) {
    return 2 + head_dim;
}

// Runs one CTA per (query token, KV group, split). The MMA rows are the group's query heads, up to 16, and each key
// tile gathers the token's next selected blocks in view order. The tile, sweep and staging arithmetic is
// attn_prefill_tc_kernel's. A split takes whole pages of the selection.
template <typename scalar_t, int page_size, int head_dim, bool bf16, int nwarps, int key_pages, int ctas_per_sm>
__global__ void __launch_bounds__(nwarps * 32, ctas_per_sm) attn_indexed_tc_kernel(
    const attn::paged_attn_params<scalar_t> p,
    const block_selection selection,
    const scalar_t* __restrict__ rotated_k,  // [selected pages, page_size, n_kv_heads, head_dim]
    const int32_t* __restrict__ page_slots,  // [num_seqs * max_pages], each selected view page's row of rotated_k
    int splits,
    float* __restrict__ partial,  // [query tokens * KV groups * splits, 16, partial_row_floats], with splits above 1
    int32_t* __restrict__ listed_scratch  // [query tokens * KV groups * splits, max_listed_blocks]
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    constexpr int key_tile = page_size * key_pages;
    constexpr int score_stride = key_tile + 1;
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
    const int split = blockIdx.x % splits;
    const int group_cta = blockIdx.x / splits;
    const int qtok = group_cta / p.n_kv_heads;
    const int g = group_cta % p.n_kv_heads;
    const int rows = p.group;
    const int first_head = g * p.group;
    const int s = sequence_of(p, qtok);
    const int q0 = p.cu_seqlens_q ? p.cu_seqlens_q[s] : s;
    const int seq_q = p.cu_seqlens_q ? p.cu_seqlens_q[s + 1] - q0 : 1;
    const int ctx_len = p.seqlens_k[s];
    const int query_pos = ctx_len - seq_q + (qtok - q0);
    const int tile_blocks = key_tile / selection.block_size;
    const int page_blocks = page_size / selection.block_size;

    const int32_t* seq_table = p.page_tables + static_cast<int64_t>(s) * p.max_pages;
    const int32_t* seq_slots = page_slots + static_cast<int64_t>(s) * p.max_pages;
    const int64_t rotated_token_stride = static_cast<int64_t>(p.n_kv_heads) * head_dim;
    const int64_t rotated_page_stride = page_size * rotated_token_stride;

    static_assert(sizeof(scalar_t) == 2, "the tile staging copies 8 elements at a time");
    constexpr int chunk_elems = 16 / sizeof(scalar_t);
    constexpr int row_chunks = head_dim / chunk_elems;
    constexpr int tile_chunks = key_tile * row_chunks;

    extern __shared__ __align__(16) unsigned char shared_memory[];
    scalar_t* sQ = reinterpret_cast<scalar_t*>(shared_memory);
    scalar_t* sK = sQ + 16 * head_dim;
    scalar_t* sV = sK + key_tile * head_dim;
    scalar_t* sP = sV + key_tile * head_dim;
    float* sS = reinterpret_cast<float*>(sP + 16 * key_tile);
    __shared__ float Mrow[16], Lrow[16];
    __shared__ float row_rescale[16];
    // The view index of each key of the staged tiles, -1 past the selection or the query: the tile K holds, then the
    // one V holds, as the pipeline stages K a tile ahead of V.
    __shared__ int key_index[2][key_tile];
    // The CTA's selected blocks at or before the query, in view order.
    int32_t* listed_blocks = listed_scratch + static_cast<int64_t>(blockIdx.x) * max_listed_blocks;
    __shared__ int listed_counts[nwarps];
    // Each row's max scaled score over the page its tiles are in; selected blocks come in view order, so a page's
    // keys are consecutive across tiles.
    __shared__ int run_page[16];
    __shared__ float run_max[16];
    const float weight = p.mass_sum ? p.query_weights[qtok] : 0.0f;
    auto flush_page = [&](int m) {
        if (run_page[m] >= 0) {
            const int64_t row = attn::mass_row(p, attn::mass_view(p, s, qtok), run_page[m]);
            atomicAdd(&p.mass_sum[row * p.n_q_heads + first_head + m], weight * run_max[m]);
            if (g == 0 && m == 0) {
                atomicAdd(&p.weight_sum[row], weight);
            }
        }
    };

    for (int c = tid; c < 16 * row_chunks; c += nwarps * 32) {
        const int row = c / row_chunks;
        const int col = (c % row_chunks) * chunk_elems;
        const scalar_t* source = p.q + (static_cast<int64_t>(qtok) * p.n_q_heads + first_head + row) * head_dim + col;
        attn::cp_async_16(
            sQ + attn::swizzled_offset<head_dim>(row, col),
            row < rows ? source : p.q,
            row < rows ? 16 : 0
        );
    }
    attn::cp_async_commit();
    attn::cp_async_wait<0>();
    constexpr int slabs = attn::pv_slabs_per_warp<head_dim, nwarps>();
    float out_acc[slabs * attn::pv_slab_accums] = {};
    for (int i = tid; i < 16 * key_tile; i += nwarps * 32) {
        sP[i] = static_cast<scalar_t>(0);
    }
    if (tid < 16) {
        Mrow[tid] = -CUDART_INF_F;
        Lrow[tid] = 0.0f;
        row_rescale[tid] = 1.0f;
        run_page[tid] = -1;
    }
    // The blocks this CTA takes are [first_block, end_block).
    __shared__ int first_block, end_block;
    __shared__ int chunk_selected[nwarps * 32];
    if (splits == 1) {
        if (tid == 0) {
            first_block = 0;
            end_block = INT_MAX;
        }
    } else {
        const int chunk_words = (selection.words + nwarps * 32 - 1) / (nwarps * 32);
        int selected = 0;
        for (int w = tid * chunk_words; w < min((tid + 1) * chunk_words, selection.words); ++w) {
            selected += __popc(selection.word(qtok, w));
        }
        chunk_selected[tid] = selected;
        __syncthreads();
        if (tid == 0) {
            int total = 0;
            for (int c = 0; c < nwarps * 32; ++c) {
                total += chunk_selected[c];
            }
            const int split_blocks = (total + splits - 1) / splits;
            // The first block of the page holding the selected block of this rank, INT_MAX past the last one.
            auto page_start = [&](int rank) {
                if (rank >= total) {
                    return INT_MAX;
                }
                int chunk = 0;
                while (rank >= chunk_selected[chunk]) {
                    rank -= chunk_selected[chunk++];
                }
                int w = chunk * chunk_words;
                uint32_t bits = selection.word(qtok, w);
                while (rank >= __popc(bits)) {
                    rank -= __popc(bits);
                    bits = selection.word(qtok, ++w);
                }
                for (; rank > 0; --rank) {
                    bits &= bits - 1;
                }
                const int block = w * 32 + __ffs(bits) - 1;
                return block - block % page_blocks;
            };
            first_block = split == 0 ? 0 : page_start(split * split_blocks);
            end_block = split + 1 == splits ? INT_MAX : page_start((split + 1) * split_blocks);
        }
    }
    __syncthreads();
    // Each thread lists the blocks of a run of consecutive words, after the runs of the threads before it.
    const int last_block = min(end_block - 1, query_pos / selection.block_size);
    const int first_word = first_block / 32;
    const int end_word = last_block >= first_block ? min(selection.words, last_block / 32 + 1) : first_word;
    const int run_words = max(end_word - first_word, 0) / (nwarps * 32) + 1;
    auto run_bits = [&](int w) {
        uint32_t bits = selection.word(qtok, w);
        if (w == first_word) {
            bits &= ~0u << (first_block % 32);
        }
        if (w == last_block / 32 && last_block % 32 != 31) {
            bits &= (2u << (last_block % 32)) - 1u;
        }
        return bits;
    };
    const int run_first = first_word + tid * run_words;
    const int run_end = min(run_first + run_words, end_word);
    int run_count = 0;
    for (int w = run_first; w < run_end; ++w) {
        run_count += __popc(run_bits(w));
    }
    int listed_before = run_count;
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int other = __shfl_up_sync(0xffffffffu, listed_before, offset);
        if (lane >= offset) {
            listed_before += other;
        }
    }
    if (lane == 31) {
        listed_counts[warp] = listed_before;
    }
    listed_before -= run_count;
    __syncthreads();
    int listed = 0;
    for (int w = 0; w < nwarps; ++w) {
        listed_before += w < warp ? listed_counts[w] : 0;
        listed += listed_counts[w];
    }
    for (int w = run_first; w < run_end; ++w) {
        for (uint32_t bits = run_bits(w); bits != 0; bits &= bits - 1) {
            listed_blocks[listed_before++] = w * 32 + __ffs(bits) - 1;
        }
    }
    __syncthreads();

    // Fills key_index[slot] with the next tile_blocks listed blocks' keys at or before the query, -1 past them; false
    // when none is left. Every thread keeps the same next_listed.
    int next_listed = 0;
    auto next_tile = [&](int slot) {
        for (int c = tid; c < key_tile; c += nwarps * 32) {
            const int index = next_listed + c / selection.block_size;
            const int key = index < listed ? listed_blocks[index] * selection.block_size + c % selection.block_size
                                           : -1;
            key_index[slot][c] = key <= query_pos ? key : -1;
        }
        const bool found = next_listed < listed;
        next_listed += tile_blocks;
        __syncthreads();
        return found;
    };

    // Stages the keys of key_index[slot], K from rotated_k or V from the pool, zeros past the selection. Always commits
    // a group, as attn_prefill_tc_kernel's stage_tile.
    auto stage_tile = [&](bool keys, scalar_t* dst, int slot) {
        for (int c = tid; c < tile_chunks; c += nwarps * 32) {
            const int row = c / row_chunks;
            const int d = (c % row_chunks) * chunk_elems;
            const int key = key_index[slot][row];
            const int view_page = max(key, 0) / page_size;
            const int offset = max(key, 0) % page_size;
            const scalar_t* src = keys ? rotated_k + static_cast<int64_t>(seq_slots[view_page]) * rotated_page_stride +
                    offset * rotated_token_stride + static_cast<int64_t>(g) * head_dim + d
                                       : p.v_pool + static_cast<int64_t>(seq_table[view_page]) * p.page_stride +
                    offset * p.token_stride + g * p.head_stride + d;
            attn::cp_async_16(dst + attn::swizzled_offset<head_dim>(row, d), src, key >= 0 ? 16 : 0);
        }
        attn::cp_async_commit();
    };

    auto attends = [&](int m, int c, int slot) { return m < rows && key_index[slot][c] >= 0; };

    // Pass 1: online softmax over the gathered tiles. Slot 0 is the tile in sK, slot 1 the one in sV.
    bool more = next_tile(0);
    if (more) {
        stage_tile(true, sK, 0);
        for (int i = tid; i < key_tile; i += nwarps * 32) {
            key_index[1][i] = key_index[0][i];
        }
        __syncthreads();
        stage_tile(false, sV, 1);
    }
    while (more) {
        attn::cp_async_wait<1>();
        __syncthreads();
        attn::tile_qkt<scalar_t, key_tile, head_dim, bf16, score_stride, true>(sQ, sK, sS, warp, nwarps);
        __syncthreads();
        more = next_tile(0);
        if (more) {
            stage_tile(true, sK, 0);
        } else {
            attn::cp_async_commit();
        }

        if (tid < sweep_threads) {
            float tmax = -CUDART_INF_F;
#pragma unroll
            for (int j = 0; j < sweep_cols; ++j) {
                const int c = sweep_lane * sweep_cols + j;
                if (attends(sweep_row, c, 1)) {
                    tmax = fmaxf(tmax, p.scale * sS[sweep_row * score_stride + c]);
                }
            }
#pragma unroll
            for (int off = sweep_lanes / 2; off > 0; off >>= 1) {
                tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, off));
            }
            if (sweep_lane == 0 && sweep_row < rows) {
                const float newM = fmaxf(Mrow[sweep_row], tmax);
                row_rescale[sweep_row] = newM == -CUDART_INF_F ? 1.0f : __expf(Mrow[sweep_row] - newM);
                Mrow[sweep_row] = newM;
            }
        }
        if (p.mass_sum && tid < rows) {
            for (int c = 0; c < key_tile; ++c) {
                const int key = key_index[1][c];
                if (key < 0) {
                    continue;
                }
                const float score = p.scale * sS[tid * score_stride + c];
                if (key / page_size != run_page[tid]) {
                    flush_page(tid);
                    run_page[tid] = key / page_size;
                    run_max[tid] = score;
                } else {
                    run_max[tid] = fmaxf(run_max[tid], score);
                }
            }
        }
        __syncthreads();

        for (int i = tid; i < 16 * key_tile; i += nwarps * 32) {
            const int m = i / key_tile, c = i % key_tile;
            float& score = sS[m * score_stride + c];
            const float pw = attends(m, c, 1) ? __expf(p.scale * score - Mrow[m]) : 0.0f;
            score = pw;
            sP[i] = static_cast<scalar_t>(pw);
        }
        __syncthreads();

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
            if (sweep_lane == 0 && sweep_row < rows) {
                Lrow[sweep_row] = Lrow[sweep_row] * row_rescale[sweep_row] + lsum;
            }
        }
        attn::cp_async_wait<1>();
        __syncthreads();

        attn::tile_pv_registers<scalar_t, key_tile, head_dim, bf16, nwarps, true>(sP, sV, row_rescale, out_acc, warp);
        __syncthreads();
        if (more) {
            for (int i = tid; i < key_tile; i += nwarps * 32) {
                key_index[1][i] = key_index[0][i];
            }
            __syncthreads();
            stage_tile(false, sV, 1);
        }
    }
    attn::cp_async_wait<0>();

    if (splits == 1) {
        const int gid = lane >> 2;
        const float inv_low = 1.0f / Lrow[gid], inv_high = 1.0f / Lrow[gid + 8];
        attn::for_each_pv_register<head_dim, nwarps>(out_acc, warp, [&](int row, int col, float value) {
            if (row < rows) {
                p.o[(static_cast<int64_t>(qtok) * p.n_q_heads + first_head + row) * head_dim +
                    col] = static_cast<scalar_t>(value * (row < 8 ? inv_low : inv_high));
            }
        });
    } else {
        float* cta_partial = partial + static_cast<int64_t>(blockIdx.x) * 16 * partial_row_floats(head_dim);
        if (tid < rows) {
            cta_partial[tid * partial_row_floats(head_dim)] = Mrow[tid];
            cta_partial[tid * partial_row_floats(head_dim) + 1] = Lrow[tid];
        }
        attn::for_each_pv_register<head_dim, nwarps>(out_acc, warp, [&](int row, int col, float value) {
            if (row < rows) {
                cta_partial[row * partial_row_floats(head_dim) + 2 + col] = value;
            }
        });
    }

    if (p.mass_sum && tid < rows) {
        flush_page(tid);
    }
}

constexpr int decode_warps = 4;
constexpr int decode_key_tile = 32;

// The dynamic shared memory of attn_indexed_decode_kernel: Q, K and V tiles, P, the scores and the listed blocks.
template <typename scalar_t, int head_dim> constexpr size_t decode_tile_bytes() {
    return (16 * head_dim + 2 * decode_key_tile * head_dim + 16 * decode_key_tile) * sizeof(scalar_t) +
        16 * (decode_key_tile + 1) * sizeof(float) + max_listed_blocks * sizeof(int32_t);
}

// Runs one CTA per (query token, KV group, split) for batches of few query tokens. Every CTA lists its token's selected
// blocks at or before the query in shared memory and takes an equal share of them, its bounds moved forward to the
// first block of a page so that a page's keys stay in one CTA. It gathers each tile's keys and values straight from
// the pool and rotates the keys in shared memory, as rotate_indexed_keys_kernel does. Splits write partial rows as
// attn_indexed_tc_kernel's.
template <typename scalar_t, int page_size, int head_dim, bool bf16>
__global__ void __launch_bounds__(decode_warps * 32) attn_indexed_decode_kernel(
    const attn::paged_attn_params<scalar_t> p,
    const block_selection selection,
    int splits,
    float* __restrict__ partial  // [query tokens * KV groups * splits, 16, partial_row_floats], with splits above 1
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    constexpr int key_tile = decode_key_tile;
    constexpr int score_stride = key_tile + 1;
    constexpr int cta_threads = decode_warps * 32;
    constexpr int chunk_elems = 16 / sizeof(scalar_t);
    constexpr int row_chunks = head_dim / chunk_elems;
    constexpr int half = head_dim / 2;
    static_assert(sizeof(scalar_t) == 2, "the tile staging copies 8 elements at a time");
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int split = blockIdx.x % splits;
    const int group_cta = blockIdx.x / splits;
    const int qtok = group_cta / p.n_kv_heads;
    const int g = group_cta % p.n_kv_heads;
    const int rows = p.group;
    const int first_head = g * p.group;
    const int s = sequence_of(p, qtok);
    const int q0 = p.cu_seqlens_q ? p.cu_seqlens_q[s] : s;
    const int seq_q = p.cu_seqlens_q ? p.cu_seqlens_q[s + 1] - q0 : 1;
    const int query_pos = p.seqlens_k[s] - seq_q + (qtok - q0);
    const int block_size = selection.block_size;
    const int tile_blocks = key_tile / block_size;
    const int32_t* seq_table = p.page_tables + static_cast<int64_t>(s) * p.max_pages;
    const attn::key_positions layout = attn::sequence_key_positions(p, s, page_size);

    extern __shared__ __align__(16) unsigned char shared_memory[];
    scalar_t* sQ = reinterpret_cast<scalar_t*>(shared_memory);
    scalar_t* sK = sQ + 16 * head_dim;
    scalar_t* sV = sK + key_tile * head_dim;
    scalar_t* sP = sV + key_tile * head_dim;
    float* sS = reinterpret_cast<float*>(sP + 16 * key_tile);
    int32_t* listed_blocks = reinterpret_cast<int32_t*>(sS + 16 * score_stride);
    __shared__ float Mrow[16], Lrow[16], row_rescale[16];
    __shared__ int key_index[key_tile];
    __shared__ int key_position[key_tile];
    // Each key's elements from the start of the pool, K and V alike.
    __shared__ int64_t key_offset[key_tile];
    __shared__ int listed_counts[decode_warps];
    __shared__ int run_page[16];
    __shared__ float run_max[16];
    // The frequency pairs whose rotation is not the identity.
    __shared__ int rotating_pairs[half];
    __shared__ int rotating;
    static_assert(key_tile == 32, "a lane per key of a tile");

    if (tid == 0) {
        rotating = 0;
    }
    for (int i = tid; i < 16 * key_tile; i += cta_threads) {
        sP[i] = static_cast<scalar_t>(0);
    }
    __syncthreads();
    for (int pair = tid; pair < half; pair += cta_threads) {
        if (p.rope_turns[pair] != 0) {
            rotating_pairs[atomicAdd(&rotating, 1)] = pair;
        }
    }

    for (int c = tid; c < 16 * row_chunks; c += cta_threads) {
        const int row = c / row_chunks;
        const int col = (c % row_chunks) * chunk_elems;
        const scalar_t* source = p.q + (static_cast<int64_t>(qtok) * p.n_q_heads + first_head + row) * head_dim + col;
        attn::cp_async_16(sQ + attn::swizzled_offset<head_dim>(row, col), row < rows ? source : p.q, row < rows ? 16 : 0);
    }
    attn::cp_async_commit();
    if (tid < 16) {
        Mrow[tid] = -CUDART_INF_F;
        Lrow[tid] = 0.0f;
        row_rescale[tid] = 1.0f;
        run_page[tid] = -1;
    }

    // Each thread lists the blocks of a run of consecutive words, after the runs of the threads before it.
    const int last_block = query_pos / block_size;
    const int end_word = min(selection.words, last_block / 32 + 1);
    const int run_words = (end_word + cta_threads - 1) / cta_threads;
    auto run_bits = [&](int w) {
        uint32_t bits = selection.word(qtok, w);
        if (w == last_block / 32 && last_block % 32 != 31) {
            bits &= (2u << (last_block % 32)) - 1u;
        }
        return bits;
    };
    const int run_first = min(tid * run_words, end_word);
    const int run_end = min(run_first + run_words, end_word);
    int run_count = 0;
    for (int w = run_first; w < run_end; ++w) {
        run_count += __popc(run_bits(w));
    }
    int listed_before = run_count;
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int other = __shfl_up_sync(0xffffffffu, listed_before, offset);
        if (lane >= offset) {
            listed_before += other;
        }
    }
    if (lane == 31) {
        listed_counts[warp] = listed_before;
    }
    listed_before -= run_count;
    __syncthreads();
    int listed = 0;
    for (int w = 0; w < decode_warps; ++w) {
        listed_before += w < warp ? listed_counts[w] : 0;
        listed += listed_counts[w];
    }
    for (int w = run_first; w < run_end; ++w) {
        for (uint32_t bits = run_bits(w); bits != 0; bits &= bits - 1) {
            listed_blocks[listed_before++] = w * 32 + __ffs(bits) - 1;
        }
    }
    __syncthreads();

    // The listed block a split starting at index i begins at: the first at or after i that starts a page's run.
    auto page_aligned = [&](int i) {
        while (i > 0 && i < listed &&
               listed_blocks[i] * block_size / page_size == listed_blocks[i - 1] * block_size / page_size) {
            ++i;
        }
        return min(i, listed);
    };
    const int split_blocks = (listed + splits - 1) / splits;
    const int first = page_aligned(split * split_blocks);
    const int end = split + 1 == splits ? listed : page_aligned((split + 1) * split_blocks);

    const float weight = p.mass_sum ? p.query_weights[qtok] : 0.0f;
    auto flush_page = [&](int m) {
        if (run_page[m] >= 0) {
            const int64_t row = attn::mass_row(p, attn::mass_view(p, s, qtok), run_page[m]);
            atomicAdd(&p.mass_sum[row * p.n_q_heads + first_head + m], weight * run_max[m]);
            if (g == 0 && m == 0) {
                atomicAdd(&p.weight_sum[row], weight);
            }
        }
    };

    constexpr int slabs = attn::pv_slabs_per_warp<head_dim, decode_warps>();
    float out_acc[slabs * attn::pv_slab_accums] = {};
    for (int tile_first = first; tile_first < end; tile_first += tile_blocks) {
        for (int c = tid; c < key_tile; c += cta_threads) {
            const int index = tile_first + c / block_size;
            const int key = index < end ? listed_blocks[index] * block_size + c % block_size : -1;
            const bool attended = key <= query_pos;
            key_index[c] = attended ? key : -1;
            key_position[c] = attended ? layout.position(key) : 0;
            key_offset[c] = attended ? static_cast<int64_t>(seq_table[key / page_size]) * p.page_stride +
                    key % page_size * p.token_stride + g * p.head_stride
                                     : 0;
        }
        __syncthreads();
        for (int c = tid; c < key_tile * row_chunks; c += cta_threads) {
            const int row = c / row_chunks;
            const int d = (c % row_chunks) * chunk_elems;
            const int64_t offset = key_offset[row] + d;
            const int bytes = key_index[row] >= 0 ? 16 : 0;
            attn::cp_async_16(sK + attn::swizzled_offset<head_dim>(row, d), p.k_pool + offset, bytes);
            attn::cp_async_16(sV + attn::swizzled_offset<head_dim>(row, d), p.v_pool + offset, bytes);
        }
        attn::cp_async_commit();
        attn::cp_async_wait<0>();
        __syncthreads();
        for (int i = tid; i < key_tile * rotating; i += cta_threads) {
            const int row = i / rotating;
            const int pair = rotating_pairs[i % rotating];
            if (key_index[row] < 0) {
                continue;
            }
            scalar_t& low = sK[attn::swizzled_offset<head_dim>(row, pair)];
            scalar_t& high = sK[attn::swizzled_offset<head_dim>(row, pair + half)];
            float cos_a, sin_a;
            attn::rope_sincos_turns(key_position[row], p.rope_turns[pair], cos_a, sin_a);
            const auto k = attn::rope_rotate<scalar_t>(static_cast<float>(low), static_cast<float>(high), cos_a, sin_a);
            low = k.lo;
            high = k.hi;
        }
        __syncthreads();
        attn::tile_qkt<scalar_t, key_tile, head_dim, bf16, score_stride, true>(sQ, sK, sS, warp, decode_warps);
        __syncthreads();

        // A warp per query head, a lane per key of the tile.
        const int key = key_index[lane];
        for (int m = warp; m < rows; m += decode_warps) {
            const float score = key >= 0 ? p.scale * sS[m * score_stride + lane] : -CUDART_INF_F;
            if (p.mass_sum) {
                for (int c = 0; c < key_tile; ++c) {
                    const float key_score = __shfl_sync(0xffffffffu, score, c);
                    const int page = key_index[c] / page_size;
                    if (lane == 0 && key_index[c] >= 0) {
                        if (page != run_page[m]) {
                            flush_page(m);
                            run_page[m] = page;
                            run_max[m] = key_score;
                        } else {
                            run_max[m] = fmaxf(run_max[m], key_score);
                        }
                    }
                }
            }
            float tile_max = score;
            for (int offset = 16; offset > 0; offset >>= 1) {
                tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, offset));
            }
            const float new_max = fmaxf(Mrow[m], tile_max);
            const float rescale = new_max == -CUDART_INF_F ? 1.0f : __expf(Mrow[m] - new_max);
            const float pw = key >= 0 ? __expf(score - new_max) : 0.0f;
            sP[m * key_tile + lane] = static_cast<scalar_t>(pw);
            float sum = pw;
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_xor_sync(0xffffffffu, sum, offset);
            }
            __syncwarp();
            if (lane == 0) {
                Lrow[m] = Lrow[m] * rescale + sum;
                Mrow[m] = new_max;
                row_rescale[m] = rescale;
            }
        }
        __syncthreads();
        attn::tile_pv_registers<scalar_t, key_tile, head_dim, bf16, decode_warps, true>(
            sP,
            sV,
            row_rescale,
            out_acc,
            warp
        );
        __syncthreads();
    }
    attn::cp_async_wait<0>();

    if (splits == 1) {
        const int gid = lane >> 2;
        const float inv_low = 1.0f / Lrow[gid], inv_high = 1.0f / Lrow[gid + 8];
        attn::for_each_pv_register<head_dim, decode_warps>(out_acc, warp, [&](int row, int col, float value) {
            if (row < rows) {
                p.o[(static_cast<int64_t>(qtok) * p.n_q_heads + first_head + row) * head_dim + col] =
                    static_cast<scalar_t>(value * (row < 8 ? inv_low : inv_high));
            }
        });
    } else {
        float* cta_partial = partial + static_cast<int64_t>(blockIdx.x) * 16 * partial_row_floats(head_dim);
        if (tid < rows) {
            cta_partial[tid * partial_row_floats(head_dim)] = Mrow[tid];
            cta_partial[tid * partial_row_floats(head_dim) + 1] = Lrow[tid];
        }
        attn::for_each_pv_register<head_dim, decode_warps>(out_acc, warp, [&](int row, int col, float value) {
            if (row < rows) {
                cta_partial[row * partial_row_floats(head_dim) + 2 + col] = value;
            }
        });
    }
    if (p.mass_sum && tid < rows) {
        flush_page(tid);
    }
}

// The most splits combine_indexed_splits_kernel joins.
constexpr int max_combined_splits = 64;

// Joins each query head's splits into o, one CTA per (query token, query head): the first warp weighs the splits, then
// each thread sums its dims' weighted partial rows.
template <typename scalar_t>
__global__ void combine_indexed_splits_kernel(
    const attn::paged_attn_params<scalar_t> p,
    int splits,
    const float* __restrict__ partial
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    __shared__ float split_weight[max_combined_splits];
    const int qtok = blockIdx.x / p.n_q_heads;
    const int head = blockIdx.x % p.n_q_heads;
    const int row_floats = partial_row_floats(p.head_dim);
    const float* rows = partial +
        ((static_cast<int64_t>(qtok) * p.n_kv_heads + head / p.group) * splits * 16 + head % p.group) * row_floats;
    const int64_t split_stride = static_cast<int64_t>(16) * row_floats;
    if (threadIdx.x < 32) {
        const int lane = threadIdx.x;
        float max_score = -CUDART_INF_F;
        for (int j = lane; j < splits; j += 32) {
            max_score = fmaxf(max_score, rows[j * split_stride]);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            max_score = fmaxf(max_score, __shfl_xor_sync(0xffffffffu, max_score, offset));
        }
        float sum = 0.0f;
        for (int j = lane; j < splits; j += 32) {
            const float split_max = rows[j * split_stride];
            const float scale = split_max > -CUDART_INF_F ? __expf(split_max - max_score) : 0.0f;
            split_weight[j] = scale;
            sum += rows[j * split_stride + 1] * scale;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffffffu, sum, offset);
        }
        __syncwarp();
        for (int j = lane; j < splits; j += 32) {
            split_weight[j] /= sum;
        }
    }
    __syncthreads();
    scalar_t* out = p.o + (static_cast<int64_t>(qtok) * p.n_q_heads + head) * p.head_dim;
    for (int d = threadIdx.x; d < p.head_dim; d += blockDim.x) {
        float value = 0.0f;
#pragma unroll 8
        for (int j = 0; j < splits; ++j) {
            value += rows[j * split_stride + 2 + d] * split_weight[j];
        }
        out[d] = static_cast<scalar_t>(value);
    }
}

// The splits of each (query token, KV group) are the largest power of two, at most max_splits, that keeps the grid
// within the device's SMs.
constexpr int64_t max_splits = 32;

int indexed_splits(int64_t groups) {
    const int64_t sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int64_t splits = 1;
    while (splits < max_splits && groups * splits * 2 <= sms) {
        splits *= 2;
    }
    return static_cast<int>(splits);
}

template <typename scalar_t, bool bf16>
void launch_indexed_tc(
    const attn::paged_attn_batch& b,
    const block_selection& selection,
    int64_t max_selected_blocks,
    cudaStream_t stream
) {
    const int64_t grid = b.total_q * b.n_kv_heads;
    const auto params = b.params<scalar_t>();
    const int64_t view_pages = b.num_seqs * b.max_pages;
    at::Tensor page_slots;
    if (b.total_q == b.num_seqs) {
        page_slots = at::empty({view_pages}, b.page_tables.options());
        launch_dependent(
            single_query_page_slots_kernel<scalar_t>,
            1,
            slot_threads,
            0,
            stream,
            params,
            selection,
            static_cast<int>(b.page_size),
            page_slots.data_ptr<int32_t>()
        );
    } else {
        auto selected_pages = at::zeros({view_pages}, b.page_tables.options());
        const int64_t marks = b.total_q * selection.words;
        launch_dependent(
            mark_selected_pages_kernel<scalar_t>,
            static_cast<int>((marks + 255) / 256),
            256,
            0,
            stream,
            params,
            selection,
            static_cast<int>(b.total_q),
            static_cast<int>(b.page_size),
            selected_pages.data_ptr<int32_t>()
        );
        page_slots = at::cumsum(selected_pages, 0, at::kInt) - 1;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int64_t slot_count = std::min(view_pages, b.total_q * max_selected_blocks);
    auto rotated_k = at::empty({slot_count, b.page_size, b.n_kv_heads, b.head_dim}, b.k_pool.options());
    launch_dependent(
        rotate_indexed_keys_kernel<scalar_t>,
        static_cast<int>(slot_count),
        256,
        0,
        stream,
        params,
        static_cast<int>(b.page_size),
        page_slots.const_data_ptr<int32_t>(),
        rotated_k.data_ptr<scalar_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    attn::dispatch_page_head("attn_indexed", b.page_size, b.head_dim, [&](auto bs_tag, auto hd_tag) {
        constexpr int bs = decltype(bs_tag)::value;
        constexpr int hd = decltype(hd_tag)::value;
        attn::dispatch_attn_tuning([&](auto row_tag) {
            constexpr attn::attn_tuning tuning = attn::attn_tuning_table[decltype(row_tag)::value];
            constexpr int nwarps = tuning.prefill_warps;
            constexpr int key_pages = attn::prefill_key_pages<scalar_t, bs, hd, tuning.prefill_key_pages>();
            constexpr size_t smem = attn::prefill_tile_bytes<scalar_t, bs, hd, key_pages>();
            const auto
                kernel = attn_indexed_tc_kernel<scalar_t, bs, hd, bf16, nwarps, key_pages, tuning.prefill_ctas_per_sm>;
            attn::allow_shared_memory(kernel, smem);
            const int splits = indexed_splits(grid);
            at::Tensor listed_scratch = at::empty({grid * splits, max_listed_blocks}, b.page_tables.options());
            at::Tensor partial;
            if (splits > 1) {
                partial = at::empty({grid * splits, 16, partial_row_floats(hd)}, b.q.options().dtype(at::kFloat));
            }
            launch_dependent(
                kernel,
                static_cast<int>(grid * splits),
                nwarps * 32,
                smem,
                stream,
                params,
                selection,
                rotated_k.data_ptr<scalar_t>(),
                page_slots.const_data_ptr<int32_t>(),
                splits,
                splits > 1 ? partial.data_ptr<float>() : nullptr,
                listed_scratch.data_ptr<int32_t>()
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            if (splits > 1) {
                launch_dependent(
                    combine_indexed_splits_kernel<scalar_t>,
                    static_cast<int>(b.total_q * b.n_q_heads),
                    128,
                    0,
                    stream,
                    params,
                    splits,
                    partial.const_data_ptr<float>()
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        });
    });
}

// The splits of each (query token, KV group) of attn_indexed_decode_kernel: the largest power of two, at most
// max_decode_splits, that keeps the grid within two CTAs per SM.
constexpr int64_t max_decode_splits = max_combined_splits;

template <typename scalar_t, bool bf16>
void launch_indexed_decode(const attn::paged_attn_batch& b, const block_selection& selection, cudaStream_t stream) {
    const int64_t grid = b.total_q * b.n_kv_heads;
    const int64_t sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int64_t splits = 1;
    while (splits < max_decode_splits && grid * (splits * 2) <= 2 * sms) {
        splits *= 2;
    }
    attn::dispatch_page_head("attn_indexed", b.page_size, b.head_dim, [&](auto bs_tag, auto hd_tag) {
        constexpr int bs = decltype(bs_tag)::value;
        constexpr int hd = decltype(hd_tag)::value;
        constexpr size_t smem = decode_tile_bytes<scalar_t, hd>();
        const auto kernel = attn_indexed_decode_kernel<scalar_t, bs, hd, bf16>;
        // The kernel's static shared memory counts against the default limit too.
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
        at::Tensor partial;
        if (splits > 1) {
            partial = at::empty({grid * splits, 16, partial_row_floats(hd)}, b.q.options().dtype(at::kFloat));
        }
        launch_dependent(
            kernel,
            static_cast<int>(grid * splits),
            decode_warps * 32,
            smem,
            stream,
            b.params<scalar_t>(),
            selection,
            static_cast<int>(splits),
            splits > 1 ? partial.data_ptr<float>() : nullptr
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        if (splits > 1) {
            launch_dependent(
                combine_indexed_splits_kernel<scalar_t>,
                static_cast<int>(b.total_q * b.n_q_heads),
                128,
                0,
                stream,
                b.params<scalar_t>(),
                static_cast<int>(splits),
                partial.const_data_ptr<float>()
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    });
}

void launch_indexed_scalar(const attn::paged_attn_batch& b, const block_selection& selection, cudaStream_t stream) {
    const int64_t blocks = b.total_q * b.n_q_heads;
    const size_t smem = static_cast<size_t>(2 * b.head_dim + attn::cta_threads) * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, b.q.scalar_type(), "attn_indexed_scalar", [&] {
        attn::dispatch_page_size("attn_indexed", b.page_size, [&](auto page_size_tag) {
            constexpr int ps = decltype(page_size_tag)::value;
            launch_dependent(
                attn_indexed_scalar_kernel<scalar_t, ps>,
                static_cast<int>(blocks),
                attn::cta_threads,
                smem,
                stream,
                b.params<scalar_t>(),
                selection
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}

}  // namespace

namespace {

at::Tensor attn_indexed(
    const attn::paged_attn_batch& b,
    const at::Tensor& selected,
    int64_t block_size,
    int64_t max_selected_blocks
) {
    TORCH_CHECK(
        block_size >= 1 && b.page_size % block_size == 0,
        "attn_indexed: block_size must be at least 1 and divide the page size"
    );
    TORCH_CHECK(
        max_selected_blocks >= 1 && max_selected_blocks <= max_listed_blocks,
        "attn_indexed: max_selected_blocks must be between 1 and ",
        max_listed_blocks
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
    const block_selection selection{
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
        const int64_t sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
        if (b.total_q * b.n_kv_heads * 2 <= sms && decode_key_tile % block_size == 0) {
            launch_indexed_decode<scalar_t, decltype(bf16_tag)::value>(b, selection, stream);
        } else {
            launch_indexed_tc<scalar_t, decltype(bf16_tag)::value>(b, selection, max_selected_blocks, stream);
        }
    });
    return b.o;
}

}  // namespace

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
) {
    const auto b = attn::make_paged_attn_batch(
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
        mass_sum,
        weight_sum,
        cu_view_pages,
        query_weights
    );
    return attn_indexed(b, selected, block_size, max_selected_blocks);
}

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
) {
    auto b = attn::make_paged_attn_batch(
        "attn_indexed_verify",
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
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt
    );
    attn::attach_mass(b, "attn_indexed_verify", mass_sum, weight_sum, cu_view_pages, query_weights, b.total_q);
    b.mass_per_query = true;
    return attn_indexed(b, selected, block_size, max_selected_blocks);
}

}  // namespace pulsarlm
