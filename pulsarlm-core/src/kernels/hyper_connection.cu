#include "pulsarlm/kernels/dependent_launch.cuh"
#include "pulsarlm/kernels/ops.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <tuple>
#include <type_traits>

namespace pulsarlm {
namespace {

constexpr int kThreads = 256;

template <typename scalar_t> __device__ __forceinline__ float rounded(float value) {
    return static_cast<float>(static_cast<scalar_t>(value));
}

__device__ __forceinline__ float sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

// One CTA per (token, stream): the stream's RMSNorm over hidden in fp32, times weight, rounded once.
template <typename scalar_t>
__global__ void stream_rmsnorm_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ weight,
    scalar_t* __restrict__ y,
    int64_t streams,
    int64_t hidden,
    float eps
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    __shared__ float partial[kThreads];
    const int64_t offset = static_cast<int64_t>(blockIdx.x) * hidden;
    const int64_t stream_offset = static_cast<int64_t>(blockIdx.x) % streams * hidden;
    float squares = 0.0f;
    for (int64_t i = threadIdx.x; i < hidden; i += kThreads) {
        const float v = static_cast<float>(x[offset + i]);
        squares += v * v;
    }
    partial[threadIdx.x] = squares;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(partial[0] / static_cast<float>(hidden) + eps);
    for (int64_t i = threadIdx.x; i < hidden; i += kThreads) {
        y[offset + i] = static_cast<scalar_t>(static_cast<float>(x[offset + i]) * inv_rms * weight[stream_offset + i]);
    }
}

// One thread per (token, hidden index): the mean over streams of the rounded sigmoid(gate) times the stream, each
// product rounded; threads of the first hidden indices also write the token's injection weights.
template <typename scalar_t>
__global__ void hyper_connection_mix_kernel(
    const scalar_t* __restrict__ normed,
    const scalar_t* __restrict__ gate_logits,
    const scalar_t* __restrict__ inject_logits,
    int64_t inject_row_stride,
    scalar_t* __restrict__ input,
    scalar_t* __restrict__ injection_weights,
    int64_t tokens,
    int64_t streams,
    int64_t hidden
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= tokens * hidden) {
        return;
    }
    const int64_t token = index / hidden;
    const int64_t h = index % hidden;
    const int64_t row = token * streams * hidden;
    float sum = 0.0f;
    for (int64_t s = 0; s < streams; s++) {
        const int64_t i = row + s * hidden + h;
        const float gate = rounded<scalar_t>(sigmoid(static_cast<float>(gate_logits[i])));
        sum += rounded<scalar_t>(__fmul_rn(gate, static_cast<float>(normed[i])));
    }
    const float scale = 1.0f / static_cast<float>(streams);
    input[index] = static_cast<scalar_t>(sum * scale);
    if (h < streams) {
        const float logit = rounded<scalar_t>(static_cast<float>(inject_logits[token * inject_row_stride + h]) * scale);
        injection_weights[token * streams + h] = static_cast<scalar_t>(2.0f * rounded<scalar_t>(sigmoid(logit)));
    }
}

// One thread per element of x: x plus the rounded product of the block output and the stream's injection weight.
template <typename scalar_t>
__global__ void hyper_connection_inject_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ out,
    const scalar_t* __restrict__ injection_weights,
    scalar_t* __restrict__ y,
    int64_t elements,
    int64_t streams,
    int64_t hidden
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const int64_t token = index / (streams * hidden);
    const int64_t s = index / hidden % streams;
    const int64_t h = index % hidden;
    const float injected = rounded<scalar_t>(__fmul_rn(
        static_cast<float>(out[token * hidden + h]),
        static_cast<float>(injection_weights[token * streams + s])
    ));
    y[index] = static_cast<scalar_t>(static_cast<float>(x[index]) + injected);
}

unsigned blocks_for(int64_t threads) {
    return static_cast<unsigned>((threads + kThreads - 1) / kThreads);
}

// The token counts the fused matrix-vector kernels serve; larger batches take the separate kernels and cuBLAS.
constexpr int64_t kVectorTokens = 4;
constexpr int kWarps = kThreads / 32;
// The weight rows a CTA of stream_rmsnorm_linear_kernel takes; its warps split each row's columns.
constexpr int kLinearRows = 2;

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

// Adds to sums[t] the dot products of 8 consecutive values of row with those of each token's staged row.
template <typename scalar_t, int TOKENS>
__device__ __forceinline__ void
add_dots(const scalar_t* __restrict__ row, const scalar_t* staged, int64_t staged_stride, float (&sums)[TOKENS]) {
    const uint4 loaded = *reinterpret_cast<const uint4*>(row);
    const auto* weights = reinterpret_cast<const scalar_t*>(&loaded);
#pragma unroll
    for (int t = 0; t < TOKENS; t++) {
        const uint4 staged_vector = *reinterpret_cast<const uint4*>(staged + t * staged_stride);
        const auto* values = reinterpret_cast<const scalar_t*>(&staged_vector);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            sums[t] += static_cast<float>(weights[i]) * static_cast<float>(values[i]);
        }
    }
}

// Each CTA takes kLinearRows rows of weight. One pass over the columns sums, per (token, stream), the squares of x and
// the products of x, norm_weight and each row, so the stream's inverse RMS scales the products once at the end; the
// CTA then writes its share of normed as stream_rmsnorm_kernel computes it.
template <typename scalar_t, int TOKENS, int STREAMS>
__global__ void __launch_bounds__(kThreads) stream_rmsnorm_linear_kernel(
    const scalar_t* __restrict__ x,
    const float* __restrict__ norm_weight,
    const scalar_t* __restrict__ weight,
    int64_t weight_row_stride,
    scalar_t* __restrict__ normed,
    scalar_t* __restrict__ projected,
    int hidden,
    int64_t rows,
    float eps
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    __shared__ float warp_dots[kWarps][kLinearRows][TOKENS][STREAMS];
    __shared__ float warp_squares[kWarps][TOKENS][STREAMS];
    __shared__ float inv_rms[TOKENS][STREAMS];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int width = STREAMS * hidden;
    const int64_t first_row = static_cast<int64_t>(blockIdx.x) * kLinearRows;
    float dots[kLinearRows][TOKENS][STREAMS] = {};
    float squares[TOKENS][STREAMS] = {};
    for (int k = threadIdx.x * 8; k < width; k += kThreads * 8) {
        const int stream = k / hidden;
        const float4 low = *reinterpret_cast<const float4*>(norm_weight + k);
        const float4 high = *reinterpret_cast<const float4*>(norm_weight + k + 4);
        const float gains[8] = {low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w};
        float scaled[TOKENS][8];
#pragma unroll
        for (int t = 0; t < TOKENS; t++) {
            const uint4 loaded = *reinterpret_cast<const uint4*>(x + t * width + k);
            const auto* values = reinterpret_cast<const scalar_t*>(&loaded);
            float token_squares = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const float v = static_cast<float>(values[i]);
                token_squares += v * v;
                scaled[t][i] = v * gains[i];
            }
#pragma unroll
            for (int s = 0; s < STREAMS; s++) {
                squares[t][s] += s == stream ? token_squares : 0.0f;
            }
        }
#pragma unroll
        for (int r = 0; r < kLinearRows; r++) {
            if (first_row + r >= rows) {
                continue;
            }
            const uint4 loaded = *reinterpret_cast<const uint4*>(weight + (first_row + r) * weight_row_stride + k);
            const auto* weights = reinterpret_cast<const scalar_t*>(&loaded);
#pragma unroll
            for (int t = 0; t < TOKENS; t++) {
                float dot = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    dot += scaled[t][i] * static_cast<float>(weights[i]);
                }
#pragma unroll
                for (int s = 0; s < STREAMS; s++) {
                    dots[r][t][s] += s == stream ? dot : 0.0f;
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < TOKENS; t++) {
#pragma unroll
        for (int s = 0; s < STREAMS; s++) {
            const float total = warp_sum(squares[t][s]);
            if (lane == 0) {
                warp_squares[warp][t][s] = total;
            }
#pragma unroll
            for (int r = 0; r < kLinearRows; r++) {
                const float dot = warp_sum(dots[r][t][s]);
                if (lane == 0) {
                    warp_dots[warp][r][t][s] = dot;
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < TOKENS * STREAMS) {
        const int t = threadIdx.x / STREAMS;
        const int s = threadIdx.x % STREAMS;
        float total = 0.0f;
        for (int w = 0; w < kWarps; w++) {
            total += warp_squares[w][t][s];
        }
        inv_rms[t][s] = rsqrtf(total / static_cast<float>(hidden) + eps);
    }
    __syncthreads();
    if (threadIdx.x < kLinearRows * TOKENS) {
        const int r = threadIdx.x / TOKENS;
        const int t = threadIdx.x % TOKENS;
        if (first_row + r < rows) {
            float sum = 0.0f;
            for (int s = 0; s < STREAMS; s++) {
                float dot = 0.0f;
                for (int w = 0; w < kWarps; w++) {
                    dot += warp_dots[w][r][t][s];
                }
                sum += inv_rms[t][s] * dot;
            }
            projected[t * rows + first_row + r] = static_cast<scalar_t>(sum);
        }
    }
    // The CTA's share of normed, in 8-element vectors.
    const int vectors = width / 8;
    const int share = (TOKENS * vectors + static_cast<int>(gridDim.x) - 1) / static_cast<int>(gridDim.x);
    for (int index = static_cast<int>(blockIdx.x) * share + threadIdx.x;
         index < min((static_cast<int>(blockIdx.x) + 1) * share, TOKENS * vectors);
         index += kThreads) {
        const int t = index / vectors;
        const int k = index % vectors * 8;
        const float scale = inv_rms[t][k / hidden];
        const uint4 loaded = *reinterpret_cast<const uint4*>(x + t * width + k);
        const auto* values = reinterpret_cast<const scalar_t*>(&loaded);
        uint4 stored;
        auto* out = reinterpret_cast<scalar_t*>(&stored);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            out[i] = static_cast<scalar_t>(static_cast<float>(values[i]) * scale * norm_weight[k + i]);
        }
        *reinterpret_cast<uint4*>(normed + t * width + k) = stored;
    }
}

// Each CTA stages the tokens' silu(mix_logits / hc), each rounded as the ATen ops round; then each warp takes hidden
// indices h, computes the gate logits of rows h + s * hidden of up_weight, rounded as at::linear returns them, and
// mixes the streams as hyper_connection_mix_kernel does.
template <typename scalar_t, int TOKENS, int STREAMS>
__global__ void __launch_bounds__(kThreads) hyper_connection_mix_linear_kernel(
    const scalar_t* __restrict__ normed,
    const scalar_t* __restrict__ mix_logits,
    int64_t mix_row_stride,
    const scalar_t* __restrict__ up_weight,
    int64_t up_row_stride,
    const scalar_t* __restrict__ inject_logits,
    int64_t inject_row_stride,
    scalar_t* __restrict__ input,
    scalar_t* __restrict__ injection_weights,
    int64_t hidden,
    int64_t rank
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    extern __shared__ __align__(16) unsigned char shared_memory[];
    auto* staged = reinterpret_cast<scalar_t*>(shared_memory);
    constexpr float kScale = 1.0f / static_cast<float>(STREAMS);
    for (int64_t i = threadIdx.x; i < TOKENS * rank; i += kThreads) {
        const float scaled = rounded<scalar_t>(
            static_cast<float>(mix_logits[i / rank * mix_row_stride + i % rank]) * kScale
        );
        staged[i] = static_cast<scalar_t>(scaled / (1.0f + expf(-scaled)));
    }
    __syncthreads();
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int64_t row_vectors = rank / 8;
    for (int64_t h = blockIdx.x * kWarps + warp; h < hidden; h += gridDim.x * kWarps) {
        float sums[STREAMS][TOKENS] = {};
#pragma unroll
        for (int s = 0; s < STREAMS; s++) {
            for (int64_t v = lane; v < row_vectors; v += 32) {
                add_dots<scalar_t, TOKENS>(
                    up_weight + (s * hidden + h) * up_row_stride + v * 8,
                    staged + v * 8,
                    rank,
                    sums[s]
                );
            }
        }
#pragma unroll
        for (int t = 0; t < TOKENS; t++) {
            float mixed = 0.0f;
#pragma unroll
            for (int s = 0; s < STREAMS; s++) {
                const float logit = rounded<scalar_t>(warp_sum(sums[s][t]));
                const float gate = rounded<scalar_t>(sigmoid(logit));
                mixed += rounded<scalar_t>(__fmul_rn(gate, static_cast<float>(normed[(t * STREAMS + s) * hidden + h])));
            }
            if (lane == 0) {
                input[t * hidden + h] = static_cast<scalar_t>(mixed * kScale);
                if (h < STREAMS) {
                    const float logit = rounded<scalar_t>(
                        static_cast<float>(inject_logits[t * inject_row_stride + h]) * kScale
                    );
                    injection_weights[t * STREAMS + h] = static_cast<scalar_t>(
                        2.0f * rounded<scalar_t>(sigmoid(logit))
                    );
                }
            }
        }
    }
}

// Calls body with std::integral_constant<int, tokens> for tokens from 1 to kVectorTokens.
template <typename Body> void dispatch_tokens(int64_t tokens, Body body) {
    switch (tokens) {
    case 1:
        body(std::integral_constant<int, 1>{});
        break;
    case 2:
        body(std::integral_constant<int, 2>{});
        break;
    case 3:
        body(std::integral_constant<int, 3>{});
        break;
    default:
        body(std::integral_constant<int, 4>{});
        break;
    }
}

bool vector_rows(const at::Tensor& t, int64_t row_stride) {
    return t.stride(-1) == 1 && row_stride % 8 == 0 && reinterpret_cast<uintptr_t>(t.const_data_ptr()) % 16 == 0;
}

std::tuple<at::Tensor, at::Tensor>
mix_streams(const at::Tensor& normed, const at::Tensor& gate_logits, const at::Tensor& inject_logits) {
    const int64_t tokens = normed.size(0);
    const int64_t streams = inject_logits.size(1);
    const int64_t hidden = normed.size(1) / streams;
    at::Tensor input = at::empty({tokens, hidden}, normed.options());
    at::Tensor injection_weights = at::empty({tokens, streams}, normed.options());
    if (tokens == 0) {
        return {input, injection_weights};
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, normed.scalar_type(), "hyper_connection_mix_cuda", [&] {
        launch_dependent(
            hyper_connection_mix_kernel<scalar_t>,
            blocks_for(tokens * hidden),
            kThreads,
            0,
            stream,
            normed.const_data_ptr<scalar_t>(),
            gate_logits.const_data_ptr<scalar_t>(),
            inject_logits.const_data_ptr<scalar_t>(),
            inject_logits.stride(0),
            input.mutable_data_ptr<scalar_t>(),
            injection_weights.mutable_data_ptr<scalar_t>(),
            tokens,
            streams,
            hidden
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return {input, injection_weights};
}

}  // namespace

at::Tensor stream_rmsnorm_cuda(const at::Tensor& x, const at::Tensor& weight, int64_t n_streams, double eps) {
    TORCH_CHECK(
        x.is_cuda() && x.dim() == 2 && x.is_contiguous() && n_streams >= 1 && x.size(1) % n_streams == 0,
        "stream_rmsnorm: x must be contiguous CUDA [tokens, n_streams * hidden]"
    );
    TORCH_CHECK(
        weight.is_cuda() && weight.scalar_type() == at::kFloat && weight.dim() == 1 && weight.size(0) == x.size(1) &&
            weight.is_contiguous(),
        "stream_rmsnorm: weight must be contiguous CUDA fp32 [n_streams * hidden]"
    );
    at::Tensor y = at::empty_like(x);
    if (x.numel() == 0) {
        return y;
    }
    const int64_t hidden = x.size(1) / n_streams;
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "stream_rmsnorm_cuda", [&] {
        launch_dependent(
            stream_rmsnorm_kernel<scalar_t>,
            static_cast<unsigned>(x.size(0) * n_streams),
            kThreads,
            0,
            stream,
            x.const_data_ptr<scalar_t>(),
            weight.const_data_ptr<float>(),
            y.mutable_data_ptr<scalar_t>(),
            n_streams,
            hidden,
            static_cast<float>(eps)
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return y;
}

std::tuple<at::Tensor, at::Tensor> stream_rmsnorm_linear_cuda(
    const at::Tensor& x,
    const at::Tensor& norm_weight,
    int64_t n_streams,
    double eps,
    const at::Tensor& weight
) {
    TORCH_CHECK(
        weight.is_cuda() && weight.dim() == 2 && weight.size(1) == x.size(1) && weight.scalar_type() == x.scalar_type(),
        "stream_rmsnorm_linear: weight must be CUDA [out, n_streams * hidden] in x's dtype"
    );
    // The fused kernel is compiled for Qwen3.8-Flash-Next's 4 streams.
    const bool vectors = x.size(0) >= 1 && x.size(0) <= kVectorTokens && x.scalar_type() != at::kFloat &&
        x.is_contiguous() && n_streams == 4 && x.size(1) % n_streams == 0 && (x.size(1) / n_streams) % 8 == 0 &&
        vector_rows(x, x.size(1)) && vector_rows(weight, weight.stride(0)) &&
        reinterpret_cast<uintptr_t>(norm_weight.const_data_ptr()) % 16 == 0;
    if (!vectors) {
        at::Tensor normed = stream_rmsnorm_cuda(x, norm_weight, n_streams, eps);
        return {normed, at::linear(normed, weight)};
    }
    TORCH_CHECK(
        norm_weight.is_cuda() && norm_weight.scalar_type() == at::kFloat && norm_weight.dim() == 1 &&
            norm_weight.size(0) == x.size(1) && norm_weight.is_contiguous() && x.size(1) % n_streams == 0,
        "stream_rmsnorm_linear: norm_weight must be contiguous CUDA fp32 [n_streams * hidden]"
    );
    const int64_t tokens = x.size(0);
    const int64_t rows = weight.size(0);
    at::Tensor normed = at::empty_like(x);
    at::Tensor projected = at::empty({tokens, rows}, x.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_REDUCED_FLOATING_TYPES(x.scalar_type(), "stream_rmsnorm_linear_cuda", [&] {
        dispatch_tokens(tokens, [&](auto tokens_tag) {
            launch_dependent(
                stream_rmsnorm_linear_kernel<scalar_t, decltype(tokens_tag)::value, 4>,
                static_cast<unsigned>((rows + kLinearRows - 1) / kLinearRows),
                kThreads,
                0,
                stream,
                x.const_data_ptr<scalar_t>(),
                norm_weight.const_data_ptr<float>(),
                weight.const_data_ptr<scalar_t>(),
                weight.stride(0),
                normed.mutable_data_ptr<scalar_t>(),
                projected.mutable_data_ptr<scalar_t>(),
                static_cast<int>(x.size(1) / n_streams),
                rows,
                static_cast<float>(eps)
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
    return {normed, projected};
}

std::tuple<at::Tensor, at::Tensor> hyper_connection_mix_cuda(
    const at::Tensor& normed,
    const at::Tensor& mix_logits,
    const at::Tensor& up_weight,
    const at::Tensor& inject_logits
) {
    TORCH_CHECK(
        normed.is_cuda() && normed.dim() == 2 && normed.is_contiguous(),
        "hyper_connection_mix: normed must be contiguous CUDA [tokens, hc * hidden]"
    );
    TORCH_CHECK(
        mix_logits.dim() == 2 && mix_logits.size(0) == normed.size(0) && mix_logits.stride(1) == 1 &&
            mix_logits.scalar_type() == normed.scalar_type() && up_weight.dim() == 2 &&
            up_weight.size(0) == normed.size(1) && up_weight.size(1) == mix_logits.size(1) &&
            up_weight.scalar_type() == normed.scalar_type(),
        "hyper_connection_mix: mix_logits must be [tokens, rank], each row contiguous, and up_weight [hc * hidden, "
        "rank], in normed's dtype"
    );
    TORCH_CHECK(
        inject_logits.is_cuda() && inject_logits.dim() == 2 && inject_logits.size(0) == normed.size(0) &&
            inject_logits.scalar_type() == normed.scalar_type() && inject_logits.stride(1) == 1,
        "hyper_connection_mix: inject_logits must be CUDA [tokens, hc] in normed's dtype, each row contiguous"
    );
    const int64_t tokens = normed.size(0);
    const int64_t streams = inject_logits.size(1);
    TORCH_CHECK(
        streams >= 1 && normed.size(1) % streams == 0 && streams <= normed.size(1) / streams,
        "hyper_connection_mix: normed must hold hc whole streams, hc at most the hidden size"
    );
    const int64_t hidden = normed.size(1) / streams;
    const int64_t rank = mix_logits.size(1);
    // The fused kernel is compiled for Qwen3.8-Flash-Next's 4 streams.
    const bool vectors = tokens >= 1 && tokens <= kVectorTokens && streams == 4 && normed.scalar_type() != at::kFloat &&
        rank % 8 == 0 && vector_rows(up_weight, up_weight.stride(0));
    if (!vectors) {
        at::Tensor gate_logits = at::linear(at::silu(mix_logits / static_cast<double>(streams)), up_weight);
        return mix_streams(normed, gate_logits, inject_logits);
    }
    at::Tensor input = at::empty({tokens, hidden}, normed.options());
    at::Tensor injection_weights = at::empty({tokens, streams}, normed.options());
    const size_t shared = static_cast<size_t>(tokens * rank) * normed.element_size();
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_REDUCED_FLOATING_TYPES(normed.scalar_type(), "hyper_connection_mix_cuda", [&] {
        dispatch_tokens(tokens, [&](auto tokens_tag) {
            auto kernel = hyper_connection_mix_linear_kernel<scalar_t, decltype(tokens_tag)::value, 4>;
            C10_CUDA_CHECK(
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared))
            );
            launch_dependent(
                kernel,
                static_cast<unsigned>((hidden + kWarps - 1) / kWarps),
                kThreads,
                shared,
                stream,
                normed.const_data_ptr<scalar_t>(),
                mix_logits.const_data_ptr<scalar_t>(),
                mix_logits.stride(0),
                up_weight.const_data_ptr<scalar_t>(),
                up_weight.stride(0),
                inject_logits.const_data_ptr<scalar_t>(),
                inject_logits.stride(0),
                input.mutable_data_ptr<scalar_t>(),
                injection_weights.mutable_data_ptr<scalar_t>(),
                hidden,
                rank
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
    return {input, injection_weights};
}

at::Tensor
hyper_connection_inject_cuda(const at::Tensor& x, const at::Tensor& out, const at::Tensor& injection_weights) {
    TORCH_CHECK(
        x.is_cuda() && x.dim() == 2 && x.is_contiguous(),
        "hyper_connection_inject: x must be contiguous CUDA [tokens, hc * hidden]"
    );
    TORCH_CHECK(
        out.dim() == 2 && out.size(0) == x.size(0) && out.scalar_type() == x.scalar_type() && out.is_contiguous(),
        "hyper_connection_inject: out must be contiguous [tokens, hidden] in x's dtype"
    );
    TORCH_CHECK(
        injection_weights.dim() == 2 && injection_weights.size(0) == x.size(0) &&
            injection_weights.scalar_type() == x.scalar_type() && injection_weights.is_contiguous() &&
            injection_weights.size(1) * out.size(1) == x.size(1),
        "hyper_connection_inject: injection_weights must be contiguous [tokens, hc] in x's dtype, hc * hidden x's width"
    );
    at::Tensor y = at::empty_like(x);
    if (x.numel() == 0) {
        return y;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "hyper_connection_inject_cuda", [&] {
        launch_dependent(
            hyper_connection_inject_kernel<scalar_t>,
            blocks_for(x.numel()),
            kThreads,
            0,
            stream,
            x.const_data_ptr<scalar_t>(),
            out.const_data_ptr<scalar_t>(),
            injection_weights.const_data_ptr<scalar_t>(),
            y.mutable_data_ptr<scalar_t>(),
            x.numel(),
            injection_weights.size(1),
            out.size(1)
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return y;
}

}  // namespace pulsarlm
