#include "pulsarlm/ops/cuda/dependent_launch.cuh"
#include "pulsarlm/ops/cuda/device.hpp"
#include "pulsarlm/ops/cuda/device_math.cuh"
#include "pulsarlm/ops/hyper_connection.hpp"
#include "pulsarlm/ops/normalization.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <optional>
#include <tuple>
#include <type_traits>

namespace pulsarlm::ops::cuda {
namespace {

constexpr int cta_threads = 256;

}  // namespace

namespace kernel {

// One CTA per (token, stream): the stream's RMSNorm over hidden in fp32, times weight, rounded once.
template <typename scalar_t>
__global__ void stream_rmsnorm(
    const scalar_t* __restrict__ x,
    const float* __restrict__ weight,
    scalar_t* __restrict__ y,
    int64_t streams,
    int64_t hidden,
    float eps
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    __shared__ float partial[cta_threads];
    const int64_t offset = static_cast<int64_t>(blockIdx.x) * hidden;
    const int64_t stream_offset = static_cast<int64_t>(blockIdx.x) % streams * hidden;
    float squares = 0.0f;
    for (int64_t i = threadIdx.x; i < hidden; i += cta_threads) {
        const float v = static_cast<float>(x[offset + i]);
        squares += v * v;
    }
    partial[threadIdx.x] = squares;
    __syncthreads();
    for (int stride = cta_threads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(partial[0] / static_cast<float>(hidden) + eps);
    for (int64_t i = threadIdx.x; i < hidden; i += cta_threads) {
        y[offset + i] = static_cast<scalar_t>(static_cast<float>(x[offset + i]) * inv_rms * weight[stream_offset + i]);
    }
}

// One thread per (token, hidden index): the mean over streams of the rounded sigmoid(gate) times the stream, each
// product rounded; threads of the first hidden indices also write the token's injection weights.
template <typename scalar_t>
__global__ void hyper_connection_mix(
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
__global__ void hyper_connection_inject(
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

}  // namespace kernel

namespace {

// The token counts the fused matrix-vector kernels serve; larger batches take the separate kernels and cuBLAS.
constexpr int64_t vector_tokens = 4;
constexpr int warps = cta_threads / 32;
// The weight rows a CTA of kernel::stream_rmsnorm_linear takes; its warps split each row's columns.
constexpr int linear_rows = 2;

}  // namespace

namespace kernel {

// Adds to sums[t] the dot products of 8 consecutive values of row with those of each token's staged row.
template <typename scalar_t, int tokens>
__device__ __forceinline__ void add_staged_dots(
    const scalar_t* __restrict__ row,
    const scalar_t* staged,
    int64_t staged_stride,
    float (&sums)[tokens]
) {
    const uint4 loaded = *reinterpret_cast<const uint4*>(row);
    const auto* weights = reinterpret_cast<const scalar_t*>(&loaded);
#pragma unroll
    for (int t = 0; t < tokens; t++) {
        const uint4 staged_vector = *reinterpret_cast<const uint4*>(staged + t * staged_stride);
        const auto* values = reinterpret_cast<const scalar_t*>(&staged_vector);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            sums[t] += static_cast<float>(weights[i]) * static_cast<float>(values[i]);
        }
    }
}

// The 8 streams' values of token t at column k: x's, or with INJECTED, x plus the block output at the column's hidden
// index times the stream's injection weight, as hyper_connection_inject rounds them.
template <typename scalar_t, bool has_injection, int streams>
__device__ __forceinline__ void stream_values(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ block_out,
    const scalar_t* __restrict__ injection_weights,
    int t,
    int k,
    int hidden,
    float (&values)[8]
) {
    const uint4 loaded = *reinterpret_cast<const uint4*>(x + t * streams * hidden + k);
    const auto* x_values = reinterpret_cast<const scalar_t*>(&loaded);
    if constexpr (has_injection) {
        const uint4 out_loaded = *reinterpret_cast<const uint4*>(block_out + t * hidden + k % hidden);
        const auto* out_values = reinterpret_cast<const scalar_t*>(&out_loaded);
        const float injection = static_cast<float>(injection_weights[t * streams + k / hidden]);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float injected = rounded<scalar_t>(__fmul_rn(static_cast<float>(out_values[i]), injection));
            values[i] = rounded<scalar_t>(static_cast<float>(x_values[i]) + injected);
        }
    } else {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            values[i] = static_cast<float>(x_values[i]);
        }
    }
}

// Each CTA takes linear_rows rows of weight. One pass over the columns sums, per (token, stream), the squares of the
// streams' values (stream_values) and their products with norm_weight and each row, so the stream's inverse RMS scales
// the products once at the end; the CTA then writes its share of normed as stream_rmsnorm computes it and, with
// INJECTED, of the values into summed.
template <typename scalar_t, int tokens, int streams, bool has_injection>
__global__ void __launch_bounds__(cta_threads) stream_rmsnorm_linear(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ block_out,
    const scalar_t* __restrict__ injection_weights,
    const float* __restrict__ norm_weight,
    const scalar_t* __restrict__ weight,
    int64_t weight_row_stride,
    scalar_t* __restrict__ summed,
    scalar_t* __restrict__ normed,
    scalar_t* __restrict__ projected,
    int hidden,
    int64_t rows,
    float eps
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    __shared__ float warp_dots[warps][linear_rows][tokens][streams];
    __shared__ float warp_squares[warps][tokens][streams];
    __shared__ float inv_rms[tokens][streams];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int width = streams * hidden;
    const int64_t first_row = static_cast<int64_t>(blockIdx.x) * linear_rows;
    float dots[linear_rows][tokens][streams] = {};
    float squares[tokens][streams] = {};
    for (int k = threadIdx.x * 8; k < width; k += cta_threads * 8) {
        const int stream = k / hidden;
        const float4 low = *reinterpret_cast<const float4*>(norm_weight + k);
        const float4 high = *reinterpret_cast<const float4*>(norm_weight + k + 4);
        const float gains[8] = {low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w};
        float scaled[tokens][8];
#pragma unroll
        for (int t = 0; t < tokens; t++) {
            float values[8];
            stream_values<scalar_t, has_injection, streams>(x, block_out, injection_weights, t, k, hidden, values);
            float token_squares = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                token_squares += values[i] * values[i];
                scaled[t][i] = values[i] * gains[i];
            }
#pragma unroll
            for (int s = 0; s < streams; s++) {
                squares[t][s] += s == stream ? token_squares : 0.0f;
            }
        }
#pragma unroll
        for (int r = 0; r < linear_rows; r++) {
            if (first_row + r >= rows) {
                continue;
            }
            const uint4 loaded = *reinterpret_cast<const uint4*>(weight + (first_row + r) * weight_row_stride + k);
            const auto* weights = reinterpret_cast<const scalar_t*>(&loaded);
#pragma unroll
            for (int t = 0; t < tokens; t++) {
                float dot = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    dot += scaled[t][i] * static_cast<float>(weights[i]);
                }
#pragma unroll
                for (int s = 0; s < streams; s++) {
                    dots[r][t][s] += s == stream ? dot : 0.0f;
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < tokens; t++) {
#pragma unroll
        for (int s = 0; s < streams; s++) {
            const float total = warp_sum(squares[t][s]);
            if (lane == 0) {
                warp_squares[warp][t][s] = total;
            }
#pragma unroll
            for (int r = 0; r < linear_rows; r++) {
                const float dot = warp_sum(dots[r][t][s]);
                if (lane == 0) {
                    warp_dots[warp][r][t][s] = dot;
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < tokens * streams) {
        const int t = threadIdx.x / streams;
        const int s = threadIdx.x % streams;
        float total = 0.0f;
        for (int w = 0; w < warps; w++) {
            total += warp_squares[w][t][s];
        }
        inv_rms[t][s] = rsqrtf(total / static_cast<float>(hidden) + eps);
    }
    __syncthreads();
    if (threadIdx.x < linear_rows * tokens) {
        const int r = threadIdx.x / tokens;
        const int t = threadIdx.x % tokens;
        if (first_row + r < rows) {
            float sum = 0.0f;
            for (int s = 0; s < streams; s++) {
                float dot = 0.0f;
                for (int w = 0; w < warps; w++) {
                    dot += warp_dots[w][r][t][s];
                }
                sum += inv_rms[t][s] * dot;
            }
            projected[t * rows + first_row + r] = static_cast<scalar_t>(sum);
        }
    }
    // The CTA's share of normed and summed, in 8-element vectors.
    const int vectors = width / 8;
    const int share = (tokens * vectors + static_cast<int>(gridDim.x) - 1) / static_cast<int>(gridDim.x);
    for (int index = static_cast<int>(blockIdx.x) * share + threadIdx.x;
         index < min((static_cast<int>(blockIdx.x) + 1) * share, tokens * vectors);
         index += cta_threads) {
        const int t = index / vectors;
        const int k = index % vectors * 8;
        const float scale = inv_rms[t][k / hidden];
        float values[8];
        stream_values<scalar_t, has_injection, streams>(x, block_out, injection_weights, t, k, hidden, values);
        uint4 stored;
        auto* out = reinterpret_cast<scalar_t*>(&stored);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            out[i] = static_cast<scalar_t>(values[i] * scale * norm_weight[k + i]);
        }
        *reinterpret_cast<uint4*>(normed + t * width + k) = stored;
        if constexpr (has_injection) {
#pragma unroll
            for (int i = 0; i < 8; i++) {
                out[i] = static_cast<scalar_t>(values[i]);
            }
            *reinterpret_cast<uint4*>(summed + t * width + k) = stored;
        }
    }
}

// Each CTA stages the tokens' silu(mix_logits / hc), each rounded as the ATen ops round; then each warp takes hidden
// indices h, computes the gate logits of rows h + s * hidden of up_weight, rounded as at::linear returns them, and
// mixes the streams as hyper_connection_mix does.
template <typename scalar_t, int tokens, int streams>
__global__ void __launch_bounds__(cta_threads) hyper_connection_mix_linear(
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
    constexpr float scale = 1.0f / static_cast<float>(streams);
    for (int64_t i = threadIdx.x; i < tokens * rank; i += cta_threads) {
        const float scaled = rounded<scalar_t>(
            static_cast<float>(mix_logits[i / rank * mix_row_stride + i % rank]) * scale
        );
        staged[i] = static_cast<scalar_t>(scaled / (1.0f + expf(-scaled)));
    }
    __syncthreads();
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int64_t row_vectors = rank / 8;
    for (int64_t h = blockIdx.x * warps + warp; h < hidden; h += gridDim.x * warps) {
        float sums[streams][tokens] = {};
#pragma unroll
        for (int s = 0; s < streams; s++) {
            for (int64_t v = lane; v < row_vectors; v += 32) {
                add_staged_dots<scalar_t, tokens>(
                    up_weight + (s * hidden + h) * up_row_stride + v * 8,
                    staged + v * 8,
                    rank,
                    sums[s]
                );
            }
        }
#pragma unroll
        for (int t = 0; t < tokens; t++) {
            float mixed = 0.0f;
#pragma unroll
            for (int s = 0; s < streams; s++) {
                const float logit = rounded<scalar_t>(warp_sum(sums[s][t]));
                const float gate = rounded<scalar_t>(sigmoid(logit));
                mixed += rounded<scalar_t>(__fmul_rn(gate, static_cast<float>(normed[(t * streams + s) * hidden + h])));
            }
            if (lane == 0) {
                input[t * hidden + h] = static_cast<scalar_t>(mixed * scale);
                if (h < streams) {
                    const float logit = rounded<scalar_t>(
                        static_cast<float>(inject_logits[t * inject_row_stride + h]) * scale
                    );
                    injection_weights[t * streams + h] = static_cast<scalar_t>(
                        2.0f * rounded<scalar_t>(sigmoid(logit))
                    );
                }
            }
        }
    }
}

}  // namespace kernel

namespace {

// Calls body with std::integral_constant<int, tokens> for tokens from 1 to vector_tokens.
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
    case 4:
        body(std::integral_constant<int, 4>{});
        break;
    default:
        TORCH_CHECK(false, "hyper_connection: the kernel takes 1 to 4 tokens, got ", tokens);
    }
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
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, normed.scalar_type(), "hyper_connection_mix", [&] {
        launch_dependent(
            kernel::hyper_connection_mix<scalar_t>,
            cta_count(tokens * hidden, cta_threads),
            cta_threads,
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

at::Tensor stream_rmsnorm(const at::Tensor& x, const at::Tensor& weight, int64_t n_streams, double eps) {
    check_on_current_device("stream_rmsnorm", {x, weight});
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
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "stream_rmsnorm", [&] {
        launch_dependent(
            kernel::stream_rmsnorm<scalar_t>,
            static_cast<unsigned>(x.size(0) * n_streams),
            cta_threads,
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

namespace {

// Whether x, a block output and injection weights when injected, norm_weight and weight take the fused kernel: 1 to
// vector_tokens fp16 or bf16 tokens of 4 streams whose hidden size and rows hold whole 16-byte vectors.
bool fused_norm_linear(
    const at::Tensor& x,
    const std::optional<at::Tensor>& block_out,
    const at::Tensor& norm_weight,
    int64_t n_streams,
    const at::Tensor& weight
) {
    return x.size(0) >= 1 && x.size(0) <= vector_tokens &&
        (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf) && x.is_contiguous() && n_streams == 4 &&
        x.size(1) % n_streams == 0 && (x.size(1) / n_streams) % 8 == 0 && vector_rows(x, x.size(1)) &&
        vector_rows(weight, weight.stride(0)) && reinterpret_cast<uintptr_t>(norm_weight.const_data_ptr()) % 16 == 0 &&
        (!block_out || (block_out->is_contiguous() && vector_rows(*block_out, block_out->size(1))));
}

template <bool has_injection>
void launch_norm_linear(
    const at::Tensor& x,
    const at::Tensor& block_out,
    const at::Tensor& injection_weights,
    const at::Tensor& norm_weight,
    int64_t n_streams,
    double eps,
    const at::Tensor& weight,
    const at::Tensor& summed,
    const at::Tensor& normed,
    const at::Tensor& projected
) {
    const int64_t rows = weight.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_REDUCED_FLOATING_TYPES(x.scalar_type(), "stream_rmsnorm_linear", [&] {
        dispatch_tokens(x.size(0), [&](auto tokens_tag) {
            launch_dependent(
                kernel::stream_rmsnorm_linear<scalar_t, decltype(tokens_tag)::value, 4, has_injection>,
                static_cast<unsigned>((rows + linear_rows - 1) / linear_rows),
                cta_threads,
                0,
                stream,
                x.const_data_ptr<scalar_t>(),
                has_injection ? block_out.const_data_ptr<scalar_t>() : nullptr,
                has_injection ? injection_weights.const_data_ptr<scalar_t>() : nullptr,
                norm_weight.const_data_ptr<float>(),
                weight.const_data_ptr<scalar_t>(),
                weight.stride(0),
                has_injection ? summed.mutable_data_ptr<scalar_t>() : nullptr,
                normed.mutable_data_ptr<scalar_t>(),
                projected.mutable_data_ptr<scalar_t>(),
                static_cast<int>(x.size(1) / n_streams),
                rows,
                static_cast<float>(eps)
            );
        });
    });
}

void check_norm_linear(
    const at::Tensor& x,
    const at::Tensor& norm_weight,
    int64_t n_streams,
    const at::Tensor& weight
) {
    TORCH_CHECK(
        x.is_cuda() && x.dim() == 2 && n_streams >= 1 && x.size(1) % n_streams == 0,
        "stream_rmsnorm_linear: x must be CUDA [tokens, n_streams * hidden]"
    );
    TORCH_CHECK(
        weight.is_cuda() && weight.dim() == 2 && weight.size(1) == x.size(1) && weight.scalar_type() == x.scalar_type(),
        "stream_rmsnorm_linear: weight must be CUDA [out, n_streams * hidden] in x's dtype"
    );
    TORCH_CHECK(
        norm_weight.is_cuda() && norm_weight.scalar_type() == at::kFloat && norm_weight.dim() == 1 &&
            norm_weight.size(0) == x.size(1) && norm_weight.is_contiguous(),
        "stream_rmsnorm_linear: norm_weight must be contiguous CUDA fp32 [n_streams * hidden]"
    );
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> stream_rmsnorm_linear(
    const at::Tensor& x,
    const at::Tensor& norm_weight,
    int64_t n_streams,
    double eps,
    const at::Tensor& weight
) {
    check_on_current_device("stream_rmsnorm_linear", {x, norm_weight, weight});
    check_norm_linear(x, norm_weight, n_streams, weight);
    if (!fused_norm_linear(x, std::nullopt, norm_weight, n_streams, weight)) {
        at::Tensor normed = ops::cuda::stream_rmsnorm(x, norm_weight, n_streams, eps);
        return {normed, at::linear(normed, weight)};
    }
    at::Tensor normed = at::empty_like(x);
    at::Tensor projected = at::empty({x.size(0), weight.size(0)}, x.options());
    launch_norm_linear<
        false>(x, at::Tensor(), at::Tensor(), norm_weight, n_streams, eps, weight, at::Tensor(), normed, projected);
    return {normed, projected};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> injected_stream_rmsnorm_linear(
    const at::Tensor& x,
    const at::Tensor& out,
    const at::Tensor& injection_weights,
    const at::Tensor& norm_weight,
    int64_t n_streams,
    double eps,
    const at::Tensor& weight
) {
    check_on_current_device("injected_stream_rmsnorm_linear", {x, out, injection_weights, norm_weight, weight});
    check_norm_linear(x, norm_weight, n_streams, weight);
    if (!fused_norm_linear(x, out, norm_weight, n_streams, weight) || !injection_weights.is_contiguous() ||
        out.dim() != 2 || out.size(1) * n_streams != x.size(1)) {
        at::Tensor summed = ops::cuda::hyper_connection_inject(x, out, injection_weights);
        auto [normed, projected] = ops::cuda::stream_rmsnorm_linear(summed, norm_weight, n_streams, eps, weight);
        return {summed, normed, projected};
    }
    TORCH_CHECK(
        out.size(0) == x.size(0) && out.scalar_type() == x.scalar_type() && injection_weights.dim() == 2 &&
            injection_weights.size(0) == x.size(0) && injection_weights.size(1) == n_streams &&
            injection_weights.scalar_type() == x.scalar_type(),
        "injected_stream_rmsnorm_linear: out must be [tokens, hidden] and injection_weights [tokens, n_streams] in x's "
        "dtype"
    );
    at::Tensor summed = at::empty_like(x);
    at::Tensor normed = at::empty_like(x);
    at::Tensor projected = at::empty({x.size(0), weight.size(0)}, x.options());
    launch_norm_linear<true>(x, out, injection_weights, norm_weight, n_streams, eps, weight, summed, normed, projected);
    return {summed, normed, projected};
}

std::tuple<at::Tensor, at::Tensor> hyper_connection_mix(
    const at::Tensor& normed,
    const at::Tensor& mix_logits,
    const at::Tensor& up_weight,
    const at::Tensor& inject_logits
) {
    check_on_current_device("hyper_connection_mix", {normed, mix_logits, up_weight, inject_logits});
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
    const bool vectors = tokens >= 1 && tokens <= vector_tokens && streams == 4 &&
        (normed.scalar_type() == at::kBFloat16 || normed.scalar_type() == at::kHalf) && rank % 8 == 0 &&
        vector_rows(up_weight, up_weight.stride(0));
    if (!vectors) {
        at::Tensor gate_logits = at::linear(at::silu(mix_logits / static_cast<double>(streams)), up_weight);
        return mix_streams(normed, gate_logits, inject_logits);
    }
    at::Tensor input = at::empty({tokens, hidden}, normed.options());
    at::Tensor injection_weights = at::empty({tokens, streams}, normed.options());
    const size_t shared = static_cast<size_t>(tokens * rank) * normed.element_size();
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_REDUCED_FLOATING_TYPES(normed.scalar_type(), "hyper_connection_mix", [&] {
        dispatch_tokens(tokens, [&](auto tokens_tag) {
            auto kernel = kernel::hyper_connection_mix_linear<scalar_t, decltype(tokens_tag)::value, 4>;
            allow_shared_memory(kernel, shared);
            launch_dependent(
                kernel,
                static_cast<unsigned>((hidden + warps - 1) / warps),
                cta_threads,
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

at::Tensor hyper_connection_inject(const at::Tensor& x, const at::Tensor& out, const at::Tensor& injection_weights) {
    check_on_current_device("hyper_connection_inject", {x, out, injection_weights});
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
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "hyper_connection_inject", [&] {
        launch_dependent(
            kernel::hyper_connection_inject<scalar_t>,
            cta_count(x.numel(), cta_threads),
            cta_threads,
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

}  // namespace pulsarlm::ops::cuda
