#pragma once

#include <ATen/core/Tensor.h>

namespace pulsarlm::ops::cuda {

// x [tokens, in] times weight^T [out, in] in x's dtype, as at::linear; batches of 1 to 4 fp16 or bf16 tokens run as one
// matrix-vector kernel, others through cuBLAS.
at::Tensor small_batch_linear(const at::Tensor& x, const at::Tensor& weight);

// silu(x gate^T) * (x up^T) for gate_up [2 * out, in], the gate rows first, as ops::cuda::silu_mul(at::linear(x,
// gate_up)); batches of 1 to 4 fp16 or bf16 tokens run as one kernel, others through cuBLAS and ops::cuda::silu_mul.
at::Tensor small_batch_gated_linear(const at::Tensor& x, const at::Tensor& gate_up);

}  // namespace pulsarlm::ops::cuda
