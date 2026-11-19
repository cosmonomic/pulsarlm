#pragma once

#include <ATen/core/Tensor.h>

namespace pulsarlm {

// x + sigmoid(gate_logits) * y for x and y contiguous [rows, columns] and gate_logits [rows, 1], whose rows may lie
// apart, each operation rounded to the dtype as the ATen ops round it.
at::Tensor sigmoid_gated_add_cuda(const at::Tensor& x, const at::Tensor& gate_logits, const at::Tensor& y);

// y * sigmoid(gate_logits) for y contiguous [rows, columns] or [rows, middle, columns] and gate_logits of its sizes with
// a contiguous last axis, each operation rounded to the dtype as the ATen ops round it.
at::Tensor sigmoid_gated_cuda(const at::Tensor& y, const at::Tensor& gate_logits);

// silu(x[:, :n]) * x[:, n:] for contiguous x [tokens, 2 * n], silu rounded to the dtype before the product.
at::Tensor silu_mul_cuda(const at::Tensor& x);

}  // namespace pulsarlm
