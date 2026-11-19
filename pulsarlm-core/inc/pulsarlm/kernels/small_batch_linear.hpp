#pragma once

#include <ATen/core/Tensor.h>

namespace pulsarlm {

// x [tokens, in] times weight^T [out, in] in x's dtype, as at::linear; batches of 1 to 4 fp16 or bf16 tokens run as one
// matrix-vector kernel, others through cuBLAS.
at::Tensor small_batch_linear_cuda(const at::Tensor& x, const at::Tensor& weight);

// silu(x gate^T) * (x up^T) for gate_up [2 * out, in], the gate rows first, as silu_mul_cuda(at::linear(x, gate_up));
// batches of 1 to 4 fp16 or bf16 tokens run as one kernel, others through cuBLAS and silu_mul_cuda.
at::Tensor small_batch_gated_linear_cuda(const at::Tensor& x, const at::Tensor& gate_up);

}  // namespace pulsarlm
