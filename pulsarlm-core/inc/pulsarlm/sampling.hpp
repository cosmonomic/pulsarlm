#pragma once

#include <ATen/core/Tensor.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace pulsarlm {

// temperature <= 0 samples the argmax. top_k 0 and top_p outside (0, 1) keep every token; with both set, a token must
// pass both. top_p keeps a token while the probability of the tokens ranked above it is at most top_p.
struct sampling_params {
    double temperature = 0.0;
    double top_p = 1.0;
    std::size_t top_k = 0;
    std::optional<std::uint64_t> seed;
};

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
