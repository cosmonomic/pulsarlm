#pragma once

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <tuple>

namespace pulsarlm {

// Fused int4 dequant-GEMM: y = x @ dequant(W)^T for a linear with group-wise
// symmetric int4 weights. Computes y[m,n] = sum_k x[m,k] * w_deq[n,k] with
// w_deq[n,k] = int4(weight_packed[n,k]) * weight_scale[n, k/group_size].
//   x             [M, K]    fp16/bf16, contiguous
//   weight_packed [N, K/8]  int32, 8 signed int4 per int32 along K (LSB-first)
//   weight_scale  [N, K/group_size]  per-group scale (upcast to fp32 internally)
//   group_size    quant group along K (multiple of 16; K a multiple of it)
// Returns y [M, N], same dtype as x.
at::Tensor gemm_int4_cuda(
    const at::Tensor& x,
    const at::Tensor& weight_packed,
    const at::Tensor& weight_scale,
    int64_t group_size
);

// Split-K variant of gemm_int4 with the K-split count forced to num_splits
// (>= 1). Same semantics and result; always routes through the split+combine path
// (even at num_splits == 1; num_splits may exceed the K-chunk count, the extra
// splits being empty and contributing 0).
at::Tensor gemm_int4_split_cuda(
    const at::Tensor& x,
    const at::Tensor& weight_packed,
    const at::Tensor& weight_scale,
    int64_t group_size,
    int64_t num_splits
);

// Marlin dequant-GEMMs: y [M, N] = x [M, K] @ dequant(W)^T in x's dtype (fp16 or bf16), over weights repacked once
// into an opaque layout. w_deq[n, k] = w[n, k] * weight_scale[n / block_n, k / group_size]; group_size is a multiple
// of 16 dividing K. The repacks return (weight_marlin, scale_marlin), which the GEMM takes with the same N, block_n
// and group_size.
//   repack_int4_marlin: weight_packed int32 [N, K/8], 8 signed int4 per int32 along K, least significant first.
//   repack_fp8_marlin:  weight float8_e4m3fn [N, K]; the NaN codes are not supported.
//   weight_scale [ceil(N / block_n), K / group_size], upcast to fp32.
std::tuple<at::Tensor, at::Tensor> repack_int4_marlin_cuda(
    const at::Tensor& weight_packed,
    const at::Tensor& weight_scale,
    int64_t block_n,
    int64_t group_size
);

std::tuple<at::Tensor, at::Tensor>
repack_fp8_marlin_cuda(const at::Tensor& weight, const at::Tensor& weight_scale, int64_t block_n, int64_t group_size);

at::Tensor gemm_int4_marlin_cuda(
    const at::Tensor& x,
    const at::Tensor& weight_marlin,
    const at::Tensor& scale_marlin,
    int64_t n,
    int64_t block_n,
    int64_t group_size
);

at::Tensor gemm_fp8_marlin_cuda(
    const at::Tensor& x,
    const at::Tensor& weight_marlin,
    const at::Tensor& scale_marlin,
    int64_t n,
    int64_t block_n,
    int64_t group_size
);

}  // namespace pulsarlm
