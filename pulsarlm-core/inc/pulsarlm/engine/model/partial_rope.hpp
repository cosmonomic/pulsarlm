#pragma once

#include <ATen/core/Tensor.h>

#include <cstdint>

// Hugging Face's partial RoPE rotates pairs (j, j + rotary_dim / 2) of a head's first rotary_dim dims; the paged ops
// rotate pairs (j, j + head_dim / 2). Permuting each query and key head's dims at load, and raising theta to
// head_dim / rotary_dim, rotates the same pairs at the same angles.

namespace pulsarlm {

// Head dim i of the paged ops holds Hugging Face's head dim permutation[i]: the paged ops rotate pairs (j, j + half)
// for j below rotary_dim / 2, which hold Hugging Face's rotated pairs (j, j + rotary_dim / 2); the dims past rotary_dim
// fill the rest in order.
at::Tensor rope_permutation(std::int64_t head_dim, std::int64_t rotary_dim, at::Device device);

// weight [heads * head_dim, in] with each head's output dims permuted.
at::Tensor permute_heads(const at::Tensor& weight, std::int64_t heads, const at::Tensor& permutation);

// q_proj [heads * 2 * head_dim, in] with the query half of each head permuted and the gate half kept.
at::Tensor permute_queries(const at::Tensor& weight, std::int64_t heads, const at::Tensor& permutation);

}  // namespace pulsarlm
