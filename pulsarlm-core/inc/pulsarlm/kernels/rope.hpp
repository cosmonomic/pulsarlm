#pragma once

#include <ATen/core/Tensor.h>

#include <compare>
#include <cstdint>
#include <optional>

#include "pulsarlm/kernels/rope_parameters.hpp"
namespace pulsarlm {

// x [n_heads, seq, head_dim], head_dim even, with row i rotated to position pos[i] (int64 [seq]) in the rotate_half
// convention with the frequencies rope states.
at::Tensor rope_cuda(const at::Tensor& x, const at::Tensor& pos, const rope_parameters& rope);

// x with its trailing head_dim axis rotated to position in the rotate_half convention rope_cuda uses, to the accuracy
// rope_cuda reaches: the angle is formed in fp64 and only cos and sin are rounded to fp32. position broadcasts against
// x's leading axes once given a trailing frequency axis, so a 0-dim tensor turns every row by one angle and a [rows, 1]
// tensor turns each row by its own. Rotations compose: rotating a key stored at p by delta lands it at p + delta.
at::Tensor rope_rotate(const at::Tensor& x, const at::Tensor& position, const rope_parameters& rope);

}  // namespace pulsarlm
