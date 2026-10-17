#include "pulsarlm/ops.hpp"
#include "rope_tables.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>

// Rotary position embedding (RoPE), HF/Qwen2 "rotate_half" convention.
//
// For head_dim = 2h and inv_freq[j] the frequencies rope_parameters states, each row
// s at pos p = pos[s] is rotated by angle a = p * inv_freq[j]:
//   out[j]   = x[j]   * cos(a) - x[j+h] * sin(a)
//   out[j+h] = x[j+h] * cos(a) + x[j]   * sin(a)
// cos 0 and sin 0 are exact, so a pair of frequency 0 comes out unchanged. The angle is formed
// in fp64 (see rope_sincos), the rotation runs in fp32, and the result is cast
// back to x's dtype on store.
//
// pos is int64 [seq] and may be arbitrary / non-contiguous. One block handles one
// (head, row) pair; threads stride over the h frequency pairs.

namespace pulsarlm {
namespace {

// cos/sin of a RoPE angle given in QUADRANTS (the angle divided by pi/2), rounded
// to fp32.
//
// The argument arrives in fp64 and the whole-quadrant part is removed there: that
// subtraction is exact, so only the remainder, bounded by half a quadrant, is
// rounded to fp32. Rounding the unreduced angle instead would scale the rounding
// error by the angle's magnitude, which reaches the trained context length.
__device__ __forceinline__ void rope_sincos(double quadrants, float& cos_out, float& sin_out) {
    const double whole = nearbyint(quadrants);
    const float reduced = static_cast<float>(quadrants - whole) * 1.57079632679489662f;

    float sine, cosine;
    sincosf(reduced, &sine, &cosine);
    switch (static_cast<int>(static_cast<long long>(whole)) & 3) {
    case 0:
        cos_out = cosine;
        sin_out = sine;
        break;
    case 1:
        cos_out = -sine;
        sin_out = cosine;
        break;
    case 2:
        cos_out = -cosine;
        sin_out = -sine;
        break;
    default:
        cos_out = sine;
        sin_out = -cosine;
        break;
    }
}

template <typename scalar_t>
__global__ void rope_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ pos,
    const double* __restrict__ quadrants_per_position,
    scalar_t* __restrict__ out,
    int64_t seq,
    int64_t head_dim
) {
    const int64_t blk = blockIdx.x;
    const int64_t s = blk % seq;
    const int64_t h = head_dim / 2;

    const scalar_t* xr = x + blk * head_dim;
    scalar_t* outr = out + blk * head_dim;
    const double p = static_cast<double>(pos[s]);

    for (int64_t j = threadIdx.x; j < h; j += blockDim.x) {
        float c, sn;
        rope_sincos(p * quadrants_per_position[j], c, sn);
        const float lo = static_cast<float>(xr[j]);
        const float hi = static_cast<float>(xr[j + h]);
        outr[j] = static_cast<scalar_t>(lo * c - hi * sn);
        outr[j + h] = static_cast<scalar_t>(hi * c + lo * sn);
    }
}

}  // namespace

at::Tensor rope_cuda(const at::Tensor& x, const at::Tensor& pos, const rope_parameters& rope) {
    TORCH_CHECK(x.is_cuda() && pos.is_cuda(), "rope: inputs must be CUDA tensors");
    TORCH_CHECK(x.dim() == 3, "rope: x must be 3-D [n_heads, seq, head_dim]");
    TORCH_CHECK(pos.dim() == 1, "rope: pos must be 1-D [seq]");
    TORCH_CHECK(pos.scalar_type() == at::kLong, "rope: pos must be int64");

    const int64_t n_heads = x.size(0);
    const int64_t seq = x.size(1);
    const int64_t head_dim = x.size(2);

    TORCH_CHECK(head_dim % 2 == 0, "rope: head_dim must be even");
    TORCH_CHECK(pos.size(0) == seq, "rope: pos length must match x seq dim");

    auto xc = x.contiguous();
    auto pc = pos.contiguous();
    auto out = at::empty_like(xc);
    const auto& quadrants_per_position = rope_quadrants_per_position(rope, head_dim, xc.device());

    const int64_t blocks = n_heads * seq;
    const int threads = static_cast<int>(std::min<int64_t>(256, head_dim / 2));
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, xc.scalar_type(), "rope_cuda", [&] {
        rope_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            xc.data_ptr<scalar_t>(),
            pc.data_ptr<int64_t>(),
            quadrants_per_position.data_ptr<double>(),
            out.data_ptr<scalar_t>(),
            seq,
            head_dim
        );
    });
    return out;
}

}  // namespace pulsarlm
