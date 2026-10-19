#include "attn_common.cuh"

#include "pulsarlm/ops.hpp"
#include "rope_tables.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <math_constants.h>

#include <optional>
#include <string>

// The argument contract the paged attention ops share, the scalar kernel float32 runs
// on, and the step that turns each page's S and W into its mass. Decode is the case of one query per sequence at the context's last
// position, so one kernel serves both.

namespace pulsarlm {
namespace attn {
namespace {

// One CTA per (query token, query head). Threads stride over the view pages the query
// attends, each running an online softmax over its pages' keys and adding weight times
// each page's max score into the page's S; the softmaxes combine with the flash rescale,
// and the LSE they give is then taken off every attended page's S.
template <typename scalar_t, int PAGE_SIZE> __global__ void attn_scalar_kernel(const PagedAttnParams<scalar_t> p) {
    const int tid = threadIdx.x;
    const int blk = blockIdx.x;
    const int qtok = blk / p.n_q_heads;
    const int h = blk % p.n_q_heads;
    const int g = h / p.group;
    const int head_dim = p.head_dim;
    const int half = head_dim / 2;

    int s = qtok;
    if (p.cu_seqlens_q) {
        int lo = 0, hi = p.num_seqs;
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (p.cu_seqlens_q[mid] <= qtok) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        s = lo;
    }
    const int q0 = p.cu_seqlens_q ? p.cu_seqlens_q[s] : s;
    const int seq_q = p.cu_seqlens_q ? p.cu_seqlens_q[s + 1] - q0 : 1;
    const int ctx_len = p.seqlens_k[s];
    const int query_pos = ctx_len - seq_q + (qtok - q0);
    const int first = first_key(p, query_pos);
    const int first_page = first / PAGE_SIZE;
    const int last_page = query_pos / PAGE_SIZE;
    const KeyPositions layout = key_positions(p, s, PAGE_SIZE);
    const int32_t* seq_table = p.page_tables + static_cast<int64_t>(s) * p.max_pages;
    const float weight = p.mass ? p.mass_weights[qtok] : 0.0f;

    // Dynamic shared: [ qsh(head_dim) | out_acc(head_dim) | red(kThreads) ]
    extern __shared__ float smem[];
    float* qsh = smem;
    float* out_acc = qsh + head_dim;
    float* red = out_acc + head_dim;

    const scalar_t* qrow = p.q + (static_cast<int64_t>(qtok) * p.n_q_heads + h) * head_dim;
    for (int d = tid; d < head_dim; d += kThreads) {
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
            rope_sincos_turns(pos, p.rope_turns[i], cos_a, sin_a);
            stored_key_pair(p, krow, i, half, lo, hi);
            const auto k = rope_rotate<scalar_t>(lo, hi, cos_a, sin_a);
            sc += qsh[i] * static_cast<float>(k.lo) + qsh[i + half] * static_cast<float>(k.hi);
        }
        return sc * p.scale;
    };

    float m = -CUDART_INF_F;
    float l = 0.0f;
    float acc[kMaxHeadDim];
    for (int d = 0; d < head_dim; ++d) {
        acc[d] = 0.0f;
    }
    for (int page = first_page + tid; page <= last_page; page += kThreads) {
        float page_max = -CUDART_INF_F;
        const int end = min((page + 1) * PAGE_SIZE, query_pos + 1);
        for (int j = max(page * PAGE_SIZE, first); j < end; ++j) {
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
        if (p.mass) {
            const int64_t row = mass_row(p, s, page);
            atomicAdd(&p.mass[row * p.n_q_heads + h], weight * page_max);
            if (h == 0) {
                atomicAdd(&p.attended_weight[row], weight);
            }
        }
    }

    const float M = block_reduce_max<kThreads>(red, tid, m);
    const float w = __expf(m - M);  // 0 when this thread saw no keys
    for (int d = tid; d < head_dim; d += kThreads) {
        out_acc[d] = 0.0f;
    }
    __syncthreads();
    for (int d = 0; d < head_dim; ++d) {
        atomicAdd(&out_acc[d], acc[d] * w);
    }
    const float denom = block_reduce_sum<kThreads>(red, tid, l * w);

    scalar_t* orow = p.o + (static_cast<int64_t>(qtok) * p.n_q_heads + h) * head_dim;
    const float inv_denom = 1.0f / denom;
    for (int d = tid; d < head_dim; d += kThreads) {
        orow[d] = static_cast<scalar_t>(out_acc[d] * inv_denom);
    }

    if (!p.mass) {
        return;
    }
    const float lse = M + logf(denom);
    for (int page = first_page + tid; page <= last_page; page += kThreads) {
        atomicAdd(&p.mass[mass_row(p, s, page) * p.n_q_heads + h], -weight * lse);
    }
}

// One thread per (view page, query head) of the batch.
__global__ void finish_mass_kernel(float* mass, const float* attended_weight, int64_t pages, int n_q_heads) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= pages * n_q_heads) {
        return;
    }
    const float total = attended_weight[i / n_q_heads];
    mass[i] = total > 0.0f ? total * expf(mass[i] / total) : 0.0f;
}

}  // namespace

void check_pools(const char* op, const at::Tensor& k_pool, const at::Tensor& v_pool) {
    TORCH_CHECK(
        k_pool.dim() == 4 && k_pool.sizes() == v_pool.sizes() && k_pool.strides() == v_pool.strides(),
        op,
        ": k_pool and v_pool must be 4-D [num_pages, page_size, n_kv_heads, head_dim] of one shape and stride"
    );
    TORCH_CHECK(k_pool.stride(3) == 1, op, ": a pool's head_dim must be contiguous");
    auto aligned = [](const at::Tensor& pool) { return reinterpret_cast<std::uintptr_t>(pool.data_ptr()) % 16 == 0; };
    auto aligned_stride = [&](int64_t dim) { return k_pool.stride(dim) * k_pool.element_size() % 16 == 0; };
    TORCH_CHECK(
        aligned(k_pool) && aligned(v_pool) && aligned_stride(0) && aligned_stride(1) && aligned_stride(2),
        op,
        ": pool pages, tokens and heads must start at multiples of 16 bytes"
    );
}

PagedAttnBatch paged_attn_batch(
    const char* op,
    const at::Tensor& q,
    const at::Tensor& k_pool,
    const at::Tensor& v_pool,
    const at::Tensor& page_tables,
    const std::optional<at::Tensor>& cu_seqlens_q,
    const at::Tensor& seqlens_k,
    const at::Tensor& page_positions,
    const rope_parameters& rope,
    double scale,
    std::optional<int64_t> sliding_window,
    const std::optional<at::Tensor>& k_norm_weight,
    const std::optional<at::Tensor>& mass,
    const std::optional<at::Tensor>& cu_view_pages,
    const std::optional<at::Tensor>& mass_weights
) {
    auto on_cuda = [](const at::Tensor& t) { return t.is_cuda(); };
    auto int32_dims = [](const at::Tensor& t, int64_t dims) { return t.scalar_type() == at::kInt && t.dim() == dims; };

    TORCH_CHECK(
        on_cuda(q) && on_cuda(k_pool) && on_cuda(v_pool) && on_cuda(page_tables) && on_cuda(seqlens_k) &&
            on_cuda(page_positions),
        op,
        ": all inputs must be CUDA tensors"
    );
    TORCH_CHECK(
        q.scalar_type() == k_pool.scalar_type() && k_pool.scalar_type() == v_pool.scalar_type(),
        op,
        ": q, k_pool, v_pool must share a dtype"
    );
    TORCH_CHECK(q.dim() == 3, op, ": q must be 3-D [total_q, n_q_heads, head_dim]");
    check_pools(op, k_pool, v_pool);

    PagedAttnBatch b;
    b.total_q = q.size(0);
    b.n_q_heads = q.size(1);
    b.head_dim = q.size(2);
    b.page_size = k_pool.size(1);
    b.n_kv_heads = k_pool.size(2);
    b.num_seqs = seqlens_k.size(0);
    b.scale = scale;
    TORCH_CHECK(!sliding_window || *sliding_window >= 1, op, ": sliding_window must be at least 1");
    b.sliding_window = sliding_window.value_or(0);

    TORCH_CHECK(k_pool.size(3) == b.head_dim, op, ": pool head_dim must match q");
    TORCH_CHECK(b.head_dim % 2 == 0 && b.head_dim <= kMaxHeadDim, op, ": head_dim must be even and <= ", kMaxHeadDim);
    TORCH_CHECK(
        b.n_kv_heads > 0 && b.n_q_heads % b.n_kv_heads == 0,
        op,
        ": n_q_heads must be a multiple of n_kv_heads"
    );
    TORCH_CHECK(int32_dims(seqlens_k, 1), op, ": seqlens_k must be int32 [num_seqs]");
    TORCH_CHECK(
        int32_dims(page_tables, 2) && page_tables.size(0) == b.num_seqs,
        op,
        ": page_tables must be int32 [num_seqs, max_pages]"
    );
    TORCH_CHECK(
        int32_dims(page_positions, 2) && page_positions.sizes() == page_tables.sizes(),
        op,
        ": page_positions must be int32 [num_seqs, max_pages], as page_tables"
    );
    if (cu_seqlens_q.has_value()) {
        TORCH_CHECK(
            on_cuda(*cu_seqlens_q) && int32_dims(*cu_seqlens_q, 1) && cu_seqlens_q->size(0) == b.num_seqs + 1,
            op,
            ": cu_seqlens_q must be int32 [num_seqs + 1] on CUDA"
        );
        b.cu_seqlens_q = cu_seqlens_q->contiguous();
    } else {
        TORCH_CHECK(b.total_q == b.num_seqs, op, ": one query row per sequence");
    }
    b.max_pages = page_tables.size(1);

    const bool measures = mass.has_value();
    TORCH_CHECK(
        cu_view_pages.has_value() == measures && mass_weights.has_value() == measures,
        op,
        ": mass, cu_view_pages and mass_weights are passed together or not at all"
    );
    if (measures) {
        TORCH_CHECK(
            on_cuda(*mass) && on_cuda(*cu_view_pages) && on_cuda(*mass_weights),
            op,
            ": all inputs must be CUDA tensors"
        );
        TORCH_CHECK(
            mass->scalar_type() == at::kFloat && mass->dim() == 2 && mass->size(1) == b.n_q_heads &&
                mass->is_contiguous(),
            op,
            ": mass must be contiguous fp32 [total_view_pages, n_q_heads]"
        );
        TORCH_CHECK(
            int32_dims(*cu_view_pages, 1) && cu_view_pages->size(0) == b.num_seqs + 1,
            op,
            ": cu_view_pages must be int32 [num_seqs + 1]"
        );
        TORCH_CHECK(
            mass_weights->scalar_type() == at::kFloat && mass_weights->dim() == 1 && mass_weights->size(0) == b.total_q,
            op,
            ": mass_weights must be fp32 [total_q]"
        );
        b.mass = *mass;
        b.cu_view_pages = cu_view_pages->contiguous();
        b.mass_weights = mass_weights->contiguous();
        b.attended_weight = at::zeros({mass->size(0)}, mass->options());
    }
    if (k_norm_weight.has_value()) {
        TORCH_CHECK(
            on_cuda(*k_norm_weight) && k_norm_weight->dim() == 1 && k_norm_weight->size(0) == b.head_dim,
            op,
            ": k_norm_weight must be a CUDA tensor of [head_dim]"
        );
        b.k_norm_weight = k_norm_weight->to(at::kFloat).contiguous();
    }

    b.q = q.contiguous();
    b.k_pool = k_pool;
    b.v_pool = v_pool;
    b.page_tables = page_tables.contiguous();
    b.seqlens_k = seqlens_k.contiguous();
    b.page_positions = page_positions.contiguous();
    b.rope_turns = rope_turns_per_position(rope, b.head_dim, q.device());
    b.o = at::empty_like(b.q);
    return b;
}

void launch_attn_scalar(const PagedAttnBatch& batch, cudaStream_t stream) {
    if (batch.total_q == 0 || batch.num_seqs == 0) {
        return;
    }
    const int64_t blocks = batch.total_q * batch.n_q_heads;
    const size_t smem = static_cast<size_t>(2 * batch.head_dim + kThreads) * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, batch.q.scalar_type(), "attn_scalar", [&] {
        dispatch_page_size("attn_scalar", batch.page_size, [&](auto page_size_tag) {
            constexpr int PS = decltype(page_size_tag)::value;
            attn_scalar_kernel<scalar_t, PS>
                <<<static_cast<int>(blocks), kThreads, smem, stream>>>(batch.params<scalar_t>());
        });
    });
}

void finish_mass(const PagedAttnBatch& batch, cudaStream_t stream) {
    if (!batch.mass.defined() || batch.mass.numel() == 0) {
        return;
    }
    const int64_t count = batch.mass.numel();
    finish_mass_kernel<<<static_cast<int>((count + 255) / 256), 256, 0, stream>>>(
        batch.mass.data_ptr<float>(),
        batch.attended_weight.const_data_ptr<float>(),
        batch.mass.size(0),
        static_cast<int>(batch.n_q_heads)
    );
}

}  // namespace attn

std::optional<std::string> attention_unsupported_reason(
    at::ScalarType dtype,
    int64_t page_size,
    int64_t head_dim,
    int64_t n_q_heads,
    int64_t n_kv_heads
) {
    if (dtype == at::kFloat) {
        return std::nullopt;
    }
    if (dtype != at::kHalf && dtype != at::kBFloat16) {
        return c10::str("dtype ", c10::toString(dtype), " is not float32, float16 or bfloat16");
    }
    if (head_dim != 64 && head_dim != 128 && head_dim != 256 && head_dim != 512) {
        return c10::str("head dim ", head_dim, " is not 64, 128, 256 or 512");
    }
    if (page_size != 16 && page_size != 32) {
        return c10::str("page size ", page_size, " is not 16 or 32");
    }
    if (n_kv_heads <= 0 || n_q_heads % n_kv_heads != 0 || n_q_heads / n_kv_heads > 16) {
        return c10::str(n_q_heads, " query heads over ", n_kv_heads, " KV heads is not a group of at most 16");
    }
    if (!attn::decode_fits(page_size, head_dim) || !attn::prefill_fits(page_size, head_dim)) {
        return c10::str("page size ", page_size, " and head dim ", head_dim, " exceed the device's shared memory");
    }
    return std::nullopt;
}

}  // namespace pulsarlm
