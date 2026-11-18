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

// The device tensors a sample draws by: per row its temperature (fp32, 0 for argmax), top_k (int64, the largest int64
// keeps every token) and top_p (fp32, 2 keeps every token), its seed then its draw count (int64 [2, rows]), and whether any drawing
// row keeps fewer tokens than the vocabulary.
struct sampling_inputs {
    at::Tensor temperatures;
    at::Tensor top_k;
    at::Tensor top_p;
    at::Tensor draws;
    bool filters;
};

// The sampling inputs of params and draws on device, copied there on the current stream.
sampling_inputs
sampling_inputs_of(std::span<const sampling_params> params, std::span<const sampling_draw> draws, at::Device device);

// One token per row of logits [rows, vocab], drawn by inputs: int64 [rows] on the logits' device. Reads nothing back to
// the host, so a CUDA graph can capture it.
at::Tensor sample(const at::Tensor& logits, const sampling_inputs& inputs);

// sample by the inputs of params and draws. Queues its work on the current stream and does not wait for it.
at::Tensor
sample(const at::Tensor& logits, std::span<const sampling_params> params, std::span<const sampling_draw> draws);
// sample, each row's draw count moved by draw_shifts[row], int64 [rows] on the logits' device.
at::Tensor sample(
    const at::Tensor& logits,
    std::span<const sampling_params> params,
    std::span<const sampling_draw> draws,
    const at::Tensor& draw_shifts
);

}  // namespace pulsarlm
