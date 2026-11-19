#include "pulsarlm/engine/sampling.hpp"

#include "pulsarlm/kernels/ops.hpp"

#include <ATen/ATen.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace pulsarlm {

namespace {

template <typename value_type>
at::Tensor rows_on_device(const std::vector<value_type>& values, at::ScalarType dtype, at::Device device) {
    at::Tensor pinned = at::empty(
        {static_cast<std::int64_t>(values.size())},
        at::TensorOptions().dtype(dtype).pinned_memory(true)
    );
    std::ranges::copy(values, static_cast<value_type*>(pinned.data_ptr()));
    return pinned.to(device, /*non_blocking=*/true);
}

}  // namespace

std::uint64_t sampling_seed(std::optional<std::uint64_t> seed) {
    if (seed) {
        return *seed;
    }
    std::random_device device;
    return (static_cast<std::uint64_t>(device()) << 32) | device();
}

sampling_inputs
sampling_inputs_of(std::span<const sampling_params> params, std::span<const sampling_draw> draws, at::Device device) {
    if (draws.size() != params.size()) {
        throw std::invalid_argument("sampling_inputs_of takes one draw per row of params");
    }
    std::vector<float> temperature;
    std::vector<std::int64_t> top_k;
    std::vector<float> top_p;
    std::vector<std::int64_t> seeds_then_counts(2 * params.size());
    bool filters = false;
    for (std::size_t row = 0; row < params.size(); row++) {
        const sampling_params& p = params[row];
        const bool draws_row = p.temperature > 0.0;
        const bool keeps_top_k = p.top_k > 0;
        const bool keeps_top_p = p.top_p > 0.0 && p.top_p < 1.0;
        filters |= draws_row && (keeps_top_k || keeps_top_p);
        temperature.push_back(draws_row ? static_cast<float>(p.temperature) : 0.0f);
        top_k.push_back(keeps_top_k ? static_cast<std::int64_t>(p.top_k) : std::numeric_limits<std::int64_t>::max());
        top_p.push_back(keeps_top_p ? static_cast<float>(p.top_p) : 2.0f);
        seeds_then_counts[row] = static_cast<std::int64_t>(draws[row].seed);
        seeds_then_counts[params.size() + row] = static_cast<std::int64_t>(draws[row].count);
    }
    const auto rows = static_cast<std::int64_t>(params.size());
    return sampling_inputs{
        .temperatures = rows_on_device(temperature, at::kFloat, device),
        .top_k = rows_on_device(top_k, at::kLong, device),
        .top_p = rows_on_device(top_p, at::kFloat, device),
        .draws = rows_on_device(seeds_then_counts, at::kLong, device).view({2, rows}),
        .filters = filters,
    };
}

// With top_k or top_p the race runs over the log of the kept probabilities in rank order, at temperature 1 for a
// drawing row and 0 for an argmax row.
at::Tensor sample(const at::Tensor& logits, const sampling_inputs& inputs) {
    if (!inputs.filters) {
        return gumbel_race_cuda(logits.contiguous(), inputs.temperatures, inputs.draws);
    }
    const std::int64_t vocab = logits.size(1);
    at::Tensor drawing = inputs.temperatures > 0.0f;
    at::Tensor divisor = at::where(drawing, inputs.temperatures, at::ones_like(inputs.temperatures)).unsqueeze(1);
    auto [sorted, order] = at::sort(at::softmax(logits.to(at::kFloat) / divisor, -1), -1, /*descending=*/true);
    at::Tensor ranks = at::arange(vocab, sorted.options().dtype(at::kLong)).unsqueeze(0);
    at::Tensor kept = ranks < inputs.top_k.unsqueeze(1);
    kept &= at::cumsum(sorted, -1) - sorted <= inputs.top_p.unsqueeze(1);
    at::Tensor rank = gumbel_race_cuda(at::log(sorted * kept).contiguous(), drawing.to(at::kFloat), inputs.draws);
    return order.gather(1, rank.unsqueeze(1)).squeeze(1);
}

at::Tensor sample(
    const at::Tensor& logits,
    std::span<const sampling_params> params,
    std::span<const sampling_draw> draws,
    const at::Tensor& draw_shifts
) {
    if (params.size() != static_cast<std::size_t>(logits.size(0))) {
        throw std::invalid_argument("sample takes one sampling_params per row of logits");
    }
    sampling_inputs inputs = sampling_inputs_of(params, draws, logits.device());
    if (draw_shifts.defined()) {
        inputs.draws.select(0, 1).add_(draw_shifts);
    }
    return sample(logits, inputs);
}

at::Tensor
sample(const at::Tensor& logits, std::span<const sampling_params> params, std::span<const sampling_draw> draws) {
    return sample(logits, params, draws, at::Tensor());
}

// A sequence accepts while sample t equals draft t + 1, then the sample after them.
at::Tensor count_accepted_tokens(
    const at::Tensor& verified,
    const at::Tensor& drafted,
    const at::Tensor& stop_tokens,
    const at::Tensor& max_accepted_token_counts
) {
    const std::int64_t rows = verified.size(1);
    at::Tensor accepted_counts = verified.narrow(1, 0, rows - 1).eq(drafted).to(at::kLong).cumprod(1).sum(1) + 1;
    if (stop_tokens.numel() > 0) {
        at::Tensor stops = at::isin(verified, stop_tokens);
        at::Tensor first_stop = at::where(stops.any(1), stops.to(at::kLong).argmax(1), rows);
        accepted_counts = at::minimum(accepted_counts, first_stop + 1);
    }
    return at::minimum(accepted_counts, max_accepted_token_counts);
}

}  // namespace pulsarlm
