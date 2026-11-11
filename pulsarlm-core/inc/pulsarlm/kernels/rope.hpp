#pragma once

#include <ATen/core/Tensor.h>

#include <compare>
#include <cstdint>
#include <optional>

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

// x with its trailing head_dim axis rotated to position in the rotate_half convention rope_cuda uses, to the accuracy
// rope_cuda reaches: the angle is formed in fp64 and only cos and sin are rounded to fp32. position broadcasts against
// x's leading axes once given a trailing frequency axis, so a 0-dim tensor turns every row by one angle and a [rows, 1]
// tensor turns each row by its own. Rotations compose: rotating a key stored at p by delta lands it at p + delta.
at::Tensor rope_rotate(const at::Tensor& x, const at::Tensor& position, const rope_parameters& rope);

}  // namespace pulsarlm
