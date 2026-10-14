#pragma once

#include <ATen/core/Tensor.h>

#include <compare>
#include <cstdint>
#include <optional>

// Host-side RoPE angle arithmetic, shared by the CUDA kernels (rope.cu, and the paged
// attention kernels through pulsarlm/kernels/attn_params.cuh) and by rope_rotate. Every
// table derives from the frequencies rope_parameters states.

namespace pulsarlm {

// Llama 3's rescaling of each inverse frequency f, by its wavelength 2 pi / f against
// original_max_position_embeddings / high_freq_factor and original_max_position_embeddings / low_freq_factor: a
// shorter wavelength keeps f, a longer one takes f / factor, and one between takes
// (1 - s) f / factor + s f with s = (original_max_position_embeddings / wavelength - low_freq_factor) /
// (high_freq_factor - low_freq_factor).
struct llama3_rope_scaling {
    double factor;
    double low_freq_factor;
    double high_freq_factor;
    double original_max_position_embeddings;

    auto operator<=>(const llama3_rope_scaling&) const = default;
};

// The first floor(partial_rotary_factor * head_dim / 2) rotate_half pairs rotate, pair j at the inverse frequency
// theta^(-2j/head_dim), rescaled by llama3_scaling when present; the other pairs have frequency 0 and pass through
// unrotated.
struct rope_parameters {
    double theta;
    double partial_rotary_factor = 1.0;
    std::optional<llama3_rope_scaling> llama3_scaling = std::nullopt;

    auto operator<=>(const rope_parameters&) const = default;
};

// Device fp64 table of head_dim/2 RoPE inverse frequencies, carried in QUADRANTS per
// unit of position: entry j is pair j's inverse frequency * 2/pi, so multiplying by a
// position yields the angle in units of pi/2 and rope_sincos reduces it with one
// subtraction.
//
// The table MUST stay fp64: its relative error scales by the position into the
// angle, so an fp32 table puts 2e-3 radians on a 32k-position angle.
const at::Tensor& rope_quadrants_per_position(const rope_parameters& rope, int64_t head_dim, at::Device device);

// Device table of head_dim/2 RoPE inverse frequencies, carried in TURNS per unit of
// position as 64-bit fixed point: entry j is pair j's inverse frequency / (2 pi) *
// 2^64, stored bit for bit in int64 (ATen has no uint64). A position times an entry,
// wrapped mod 2^64, is the angle's fraction of a turn to 2^-64 at any position, with
// no fp64 arithmetic on the device.
const at::Tensor& rope_turns_per_position(const rope_parameters& rope, int64_t head_dim, at::Device device);

// Rotate x's trailing head_dim axis by `position` TOKENS, in the rotate_half
// convention rope_cuda uses, and to the accuracy rope_cuda reaches: the angle is
// formed in fp64 off the shared table and only cos/sin are rounded to fp32. An
// fp32 angle would instead scale its rounding by the position, reaching 2e-3
// radians at position 32k.
//
// position must broadcast against x's leading axes once given a trailing frequency
// axis, so a 0-dim tensor turns every row by the same angle and a [rows, 1] tensor
// turns each row by its own. RoPE rotations compose, so rotating a key stored at p
// by a delta lands it at p + delta.
at::Tensor rope_rotate(const at::Tensor& x, const at::Tensor& position, const rope_parameters& rope);

}  // namespace pulsarlm
