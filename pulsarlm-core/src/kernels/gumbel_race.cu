#include "pulsarlm/ops.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <curand_philox4x32_x.h>

#include <cstdint>

namespace pulsarlm {
namespace {

constexpr int kRaceThreads = 256;
constexpr int64_t kRaceElementsPerCta = 16 * 4 * kRaceThreads;

// A larger key is a larger score, then a smaller index: the score's bits mapped so that they order as the floats do.
__device__ __forceinline__ unsigned long long race_key(float score, int64_t index) {
    const uint32_t bits = __float_as_uint(score);
    const uint32_t ordered = bits & 0x80000000u ? ~bits : bits | 0x80000000u;
    return (static_cast<unsigned long long>(ordered) << 32) | (0xffffffffull - index);
}

// CTA (row, chunk) races the chunk's kRaceElementsPerCta scores and merges its best into best[row] by atomicMax.
// Philox is keyed by the row's seed and counts (element / 4, the row's draw count), so equal seeds and draw counts
// draw equal noise.
template <typename scalar_t>
__global__ void gumbel_race_kernel(
    const scalar_t* __restrict__ scores,
    const float* __restrict__ temperatures,
    const int64_t* __restrict__ draws,
    int64_t rows,
    int64_t vocab,
    unsigned long long* __restrict__ best
) {
    const int64_t row = blockIdx.y;
    const scalar_t* row_scores = scores + row * vocab;
    const float temperature = temperatures[row];
    const auto seed = static_cast<uint64_t>(draws[row]);
    const auto count = static_cast<uint64_t>(draws[rows + row]);
    const uint2 key = make_uint2(static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32));
    const int64_t first_quad = blockIdx.x * (kRaceElementsPerCta / 4);
    const int64_t end_quad = min(first_quad + kRaceElementsPerCta / 4, (vocab + 3) / 4);

    unsigned long long local = 0;
    for (int64_t quad = first_quad + threadIdx.x; quad < end_quad; quad += kRaceThreads) {
        uint32_t bits[4] = {};
        if (temperature > 0.0f) {
            const uint4 counter = make_uint4(
                static_cast<uint32_t>(quad),
                static_cast<uint32_t>(static_cast<uint64_t>(quad) >> 32),
                static_cast<uint32_t>(count),
                static_cast<uint32_t>(count >> 32)
            );
            const uint4 random = curand_Philox4x32_10(counter, key);
            bits[0] = random.x;
            bits[1] = random.y;
            bits[2] = random.z;
            bits[3] = random.w;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int64_t index = quad * 4 + j;
            if (index < vocab) {
                float score = static_cast<float>(row_scores[index]);
                if (temperature > 0.0f) {
                    const float uniform = (static_cast<float>(bits[j]) + 0.5f) * 0x1.0p-32f;
                    score = score / temperature - __logf(-__logf(uniform));
                }
                local = max(local, race_key(score, index));
            }
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local = max(local, __shfl_down_sync(0xffffffff, local, offset));
    }
    if (threadIdx.x % 32 == 0) {
        atomicMax(best + row, local);
    }
}

}  // namespace

at::Tensor gumbel_race_cuda(const at::Tensor& scores, const at::Tensor& temperatures, const at::Tensor& draws) {
    TORCH_CHECK(
        scores.is_cuda() && scores.dim() == 2 && scores.is_contiguous(),
        "gumbel_race: scores must be contiguous CUDA [rows, vocab]"
    );
    const int64_t rows = scores.size(0);
    const int64_t vocab = scores.size(1);
    TORCH_CHECK(vocab < 0xffffffffll, "gumbel_race: vocab must fit 32 bits");
    TORCH_CHECK(
        temperatures.device() == scores.device() && temperatures.scalar_type() == at::kFloat &&
            temperatures.dim() == 1 && temperatures.size(0) == rows && temperatures.is_contiguous(),
        "gumbel_race: temperatures must be fp32 [rows]"
    );
    TORCH_CHECK(
        draws.device() == scores.device() && draws.scalar_type() == at::kLong && draws.dim() == 2 &&
            draws.size(0) == 2 && draws.size(1) == rows && draws.is_contiguous(),
        "gumbel_race: draws must be int64 [2, rows]"
    );
    at::Tensor best = at::zeros({rows}, draws.options());
    if (rows == 0) {
        return best;
    }
    const dim3 grid(
        static_cast<unsigned>((vocab + kRaceElementsPerCta - 1) / kRaceElementsPerCta),
        static_cast<unsigned>(rows)
    );
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, scores.scalar_type(), "gumbel_race_cuda", [&] {
        gumbel_race_kernel<scalar_t><<<grid, kRaceThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
            scores.data_ptr<scalar_t>(),
            temperatures.data_ptr<float>(),
            draws.data_ptr<int64_t>(),
            rows,
            vocab,
            reinterpret_cast<unsigned long long*>(best.data_ptr<int64_t>())
        );
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return 0xffffffffll - best.bitwise_and(0xffffffffll);
}

}  // namespace pulsarlm
