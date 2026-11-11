#pragma once

#include <ATen/core/Tensor.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

#include "pulsarlm/engine/sampling_params.hpp"

namespace pulsarlm {

// A row's seed and the number of tokens drawn with it before. Rows with equal draws and probabilities draw equal
// tokens, whatever the other rows are.
struct sampling_draw {
    std::uint64_t seed;
    std::uint64_t count;
};

// seed, or a nondeterministic one without it.
std::uint64_t sampling_seed(std::optional<std::uint64_t> seed);

// One token per row of logits [rows, vocab], drawn with that row's params and draw: int64 [rows] on the logits'
// device. Queues its work on the current stream and does not wait for it.
at::Tensor
sample(const at::Tensor& logits, std::span<const sampling_params> params, std::span<const sampling_draw> draws);

}  // namespace pulsarlm
