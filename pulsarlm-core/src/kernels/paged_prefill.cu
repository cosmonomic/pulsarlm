#include "attn_common.cuh"

#include "pulsarlm/kernels/attn_tuning.hpp"
#include "pulsarlm/kernels/paged_prefill.cuh"
#include "pulsarlm/ops.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <vector>

// Paged-KV prefill attention: many query tokens per sequence, causal over the
// sequence's view, for a ragged (varlen) batch. Sequence i's queries
// q[cu_seqlens_q[i] : cu_seqlens_q[i+1]] sit at view indices [ctx_start_i,
// seqlens_k[i]) with ctx_start_i = seqlens_k[i] - seq_q_i, and the query at view
// index p attends keys [0, p], or the window of them sliding_window leaves. The
// tensor-core kernel is in paged_prefill.cuh.

namespace pulsarlm {
namespace {

// Copy every key of each sequence's view into rotated_k, rotated to its view
// position: sequence s's view page b lands at page s * max_pages + b. A page wholly
// before the window of the sequence's first query is left unwritten. One CTA per
// (sequence, view page); threads stride over the page's (key, kv head, frequency
// pair) triples. The attention kernel stages a key once per (query tile, query head),
// so rotating there would repeat each rotation that many times.
template <typename scalar_t>
__global__ void
rotate_view_keys_kernel(const attn::PagedAttnParams<scalar_t> p, int page_size, scalar_t* __restrict__ rotated_k) {
    const int s = blockIdx.x / p.max_pages;
    const int view_page = blockIdx.x % p.max_pages;
    const int ctx_len = p.seqlens_k[s];
    const int keys = min(page_size, ctx_len - view_page * page_size);
    const int first_query = ctx_len - (p.cu_seqlens_q[s + 1] - p.cu_seqlens_q[s]);
    if (keys <= 0 || (view_page + 1) * page_size <= attn::first_key(p, first_query)) {
        return;
    }
    const int half = p.head_dim / 2;
    const int pairs_per_key = p.n_kv_heads * half;
    const int64_t page_elems = static_cast<int64_t>(page_size) * p.n_kv_heads * p.head_dim;
    const attn::KeyLayout layout = attn::key_layout(p.rope_layout, s);
    const int line = p.page_tables[static_cast<int64_t>(s) * p.max_pages + view_page];
    const scalar_t* src = p.k_pool + line * page_elems;
    scalar_t* dst = rotated_k + blockIdx.x * page_elems;
    for (int i = threadIdx.x; i < keys * pairs_per_key; i += blockDim.x) {
        const int off = i / pairs_per_key;
        const int head = (i % pairs_per_key) / half;
        const int pair = i % half;
        const int64_t row = (static_cast<int64_t>(off) * p.n_kv_heads + head) * p.head_dim;
        float cos_a, sin_a, lo, hi;
        attn::rope_sincos_turns(layout.position(view_page * page_size + off), p.rope_turns[pair], cos_a, sin_a);
        attn::stored_key_pair(p, src + row, pair, half, lo, hi);
        const auto k = attn::rope_rotate<scalar_t>(lo, hi, cos_a, sin_a);
        dst[row + pair] = k.lo;
        dst[row + pair + half] = k.hi;
    }
}

template <typename scalar_t, bool BF16> void launch_tc_prefill(const attn::PagedAttnBatch& b, cudaStream_t stream) {
    // The work list on the host: each query tile is up to 16 query tokens of one
    // sequence. Grid = ntiles * n_q_heads.
    auto cu_cpu = b.cu_seqlens_q.to(at::kCPU);
    const int32_t* cu = cu_cpu.data_ptr<int32_t>();
    std::vector<int32_t> tile_seq, tile_qbase;
    for (int64_t s = 0; s < b.num_seqs; ++s) {
        for (int32_t qb = 0; qb < cu[s + 1] - cu[s]; qb += 16) {
            tile_seq.push_back(static_cast<int32_t>(s));
            tile_qbase.push_back(qb);
        }
    }
    const int64_t ntiles = static_cast<int64_t>(tile_seq.size());
    if (ntiles == 0) {
        return;
    }
    auto iopts = at::TensorOptions().dtype(at::kInt);
    auto tile_seq_dev = at::from_blob(tile_seq.data(), {ntiles}, iopts).to(b.q.device());
    auto tile_qbase_dev = at::from_blob(tile_qbase.data(), {ntiles}, iopts).to(b.q.device());
    const int64_t grid = ntiles * b.n_q_heads;

    const auto params = b.params<scalar_t>();
    const int64_t view_pages = b.num_seqs * b.max_pages;
    auto rotated_k = at::empty({view_pages, b.page_size, b.n_kv_heads, b.head_dim}, b.k_pool.options());
    rotate_view_keys_kernel<scalar_t><<<static_cast<int>(view_pages), 256, 0, stream>>>(
        params,
        static_cast<int>(b.page_size),
        rotated_k.data_ptr<scalar_t>()
    );

    attn::dispatch_page_head("attn_prefill", b.page_size, b.head_dim, [&](auto bs_tag, auto hd_tag) {
        constexpr int BS = decltype(bs_tag)::value;
        constexpr int HD = decltype(hd_tag)::value;
        attn::dispatch_attn_tuning([&](auto row_tag) {
            constexpr attn::AttnTuning kTuning = attn::kAttnTuningTable[decltype(row_tag)::value];
            constexpr int NWARPS = kTuning.prefill_warps;
            constexpr int KEY_PAGES = attn::prefill_key_pages<scalar_t, BS, HD, kTuning.prefill_key_pages>();
            constexpr size_t smem = attn::prefill_tile_bytes<scalar_t, BS, HD, KEY_PAGES>();
            const auto kernel = attn::
                attn_prefill_tc_kernel<scalar_t, BS, HD, BF16, NWARPS, KEY_PAGES, kTuning.prefill_ctas_per_sm>;
            attn::allow_shared_memory(kernel, smem);
            kernel<<<static_cast<int>(grid), NWARPS * 32, smem, stream>>>(
                params,
                rotated_k.data_ptr<scalar_t>(),
                tile_seq_dev.data_ptr<int32_t>(),
                tile_qbase_dev.data_ptr<int32_t>()
            );
        });
    });
}

at::Tensor paged_prefill(const attn::PagedAttnBatch& b, bool force_scalar) {
    if (b.total_q == 0 || b.num_seqs == 0) {
        return b.o;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    bool tensor_core = !force_scalar && b.tensor_core_eligible();
    if (tensor_core) {
        attn::dispatch_page_head("attn_prefill", b.page_size, b.head_dim, [&](auto bs_tag, auto hd_tag) {
            constexpr int BS = decltype(bs_tag)::value;
            constexpr int HD = decltype(hd_tag)::value;
            attn::dispatch_attn_tuning([&](auto row_tag) {
                constexpr int KEY_PAGES = attn::prefill_key_pages<
                    at::Half,
                    BS,
                    HD,
                    attn::kAttnTuningTable[decltype(row_tag)::value].prefill_key_pages>();
                tensor_core = attn::fits_shared_memory(attn::prefill_smem_bytes<at::Half, BS, HD, KEY_PAGES>());
            });
        });
    }
    if (!tensor_core) {
        attn::launch_attn_scalar(b, stream);
        return b.o;
    }
    attn::dispatch_tensor_core_dtype(b.q.scalar_type(), [&](auto type_tag, auto bf16_tag) {
        using scalar_t = typename decltype(type_tag)::type;
        launch_tc_prefill<scalar_t, decltype(bf16_tag)::value>(b, stream);
    });
    return b.o;
}

}  // namespace

at::Tensor attn_prefill_cuda(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& page_tables,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& seqlens_k,
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
        "attn_prefill",
        q,
        k_pool,
        v_pool,
        page_tables,
        cu_seqlens_q,
        seqlens_k,
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
    return paged_prefill(batch, /*force_scalar=*/false);
}

at::Tensor attn_prefill_scalar_cuda(
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& page_tables,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& seqlens_k,
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
        "attn_prefill_scalar",
        q,
        k_pool,
        v_pool,
        page_tables,
        cu_seqlens_q,
        seqlens_k,
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
    return paged_prefill(batch, /*force_scalar=*/true);
}

}  // namespace pulsarlm
