#pragma once

#include <ATen/core/Tensor.h>

namespace pulsarlm::ops::cuda {

// int64 [rows]: per row of scores [rows, vocab], argmax(x / T + g) with each g drawn from Gumbel(0, 1), which is
// index i with probability softmax(x / T)_i; a row with T <= 0 takes argmax(x). temperatures is fp32 [rows]; draws is
// int64 [2, rows], each row's seed and draw count, of which the noise is a function.
at::Tensor gumbel_race(const at::Tensor& scores, const at::Tensor& temperatures, const at::Tensor& draws);

// Replaces in place each negative token -(r + 1) of tokens (contiguous int64 [rows]) by samples[r], an earlier
// step's sampled tokens (int64) on tokens' device; the other tokens stay.
void fill_awaited_tokens(const at::Tensor& tokens, const at::Tensor& samples);

}  // namespace pulsarlm::ops::cuda
