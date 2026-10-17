#pragma once

#include "pulsarlm/rope.hpp"

#include <ATen/core/Tensor.h>

#include <cstdint>

namespace pulsarlm {

// Device fp64 table of head_dim / 2 RoPE inverse frequencies in quadrants per unit of position: entry j is pair j's
// inverse frequency times 2 / pi, so a position times it is the angle in units of pi / 2. It stays fp64: its relative
// error scales by the position into the angle, so an fp32 table puts 2e-3 radians on a 32k-position angle.
const at::Tensor& rope_quadrants_per_position(const rope_parameters& rope, int64_t head_dim, at::Device device);

// Device table of head_dim / 2 RoPE inverse frequencies in turns per unit of position, as 64-bit fixed point: entry j
// is pair j's inverse frequency / (2 pi) * 2^64, stored bit for bit in int64. A position times an entry, wrapped mod
// 2^64, is the angle's fraction of a turn to 2^-64 at any position.
const at::Tensor& rope_turns_per_position(const rope_parameters& rope, int64_t head_dim, at::Device device);

}  // namespace pulsarlm
