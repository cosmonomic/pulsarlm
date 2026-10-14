#include "pulsarlm/sampling.hpp"

#include "pulsarlm/ops.hpp"

#include <ATen/ATen.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <span>
#include <vector>

namespace pulsarlm {

namespace {

template <typename T> at::Tensor rows_on_device(const std::vector<T>& values, at::ScalarType dtype, at::Device device) {
    at::Tensor pinned = at::empty(
        {static_cast<std::int64_t>(values.size())},
        at::TensorOptions().dtype(dtype).pinned_memory(true)
    );
    std::ranges::copy(values, static_cast<T*>(pinned.data_ptr()));
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

// A row keeping every token gets top_k vocab and top_p 2. With top_k or top_p the race runs over the log of the kept
// probabilities in rank order, at temperature 1 for a drawing row and 0 for an argmax row.
at::Tensor
sample(const at::Tensor& logits, std::span<const sampling_params> params, std::span<const sampling_draw> draws) {
    assert(params.size() == static_cast<std::size_t>(logits.size(0)) && draws.size() == params.size());
    const std::int64_t vocab = logits.size(1);
    const auto rows = static_cast<std::int64_t>(params.size());
    std::vector<float> temperature;
    std::vector<std::int64_t> top_k;
    std::vector<float> top_p;
    std::vector<std::int64_t> seeds_then_counts(2 * params.size());
    bool any_filter = false;
    for (std::size_t row = 0; row < params.size(); row++) {
        const sampling_params& p = params[row];
        const bool draws_row = p.temperature > 0.0;
        const bool keeps_top_k = p.top_k > 0;
        const bool keeps_top_p = p.top_p > 0.0 && p.top_p < 1.0;
        any_filter |= draws_row && (keeps_top_k || keeps_top_p);
        temperature.push_back(draws_row ? static_cast<float>(p.temperature) : 0.0f);
        top_k.push_back(keeps_top_k ? static_cast<std::int64_t>(p.top_k) : vocab);
        top_p.push_back(keeps_top_p ? static_cast<float>(p.top_p) : 2.0f);
        seeds_then_counts[row] = static_cast<std::int64_t>(draws[row].seed);
        seeds_then_counts[params.size() + row] = static_cast<std::int64_t>(draws[row].count);
    }
    const at::Device device = logits.device();
    at::Tensor draw_rows = rows_on_device(seeds_then_counts, at::kLong, device).view({2, rows});
    at::Tensor temperatures = rows_on_device(temperature, at::kFloat, device);
    if (!any_filter) {
        return gumbel_race_cuda(logits.contiguous(), temperatures, draw_rows);
    }
    at::Tensor drawing = temperatures > 0.0f;
    at::Tensor divisor = at::where(drawing, temperatures, at::ones_like(temperatures)).unsqueeze(1);
    auto [sorted, order] = at::sort(at::softmax(logits.to(at::kFloat) / divisor, -1), -1, /*descending=*/true);
    at::Tensor ranks = at::arange(vocab, sorted.options().dtype(at::kLong)).unsqueeze(0);
    at::Tensor kept = ranks < rows_on_device(top_k, at::kLong, device).unsqueeze(1);
    kept &= at::cumsum(sorted, -1) - sorted <= rows_on_device(top_p, at::kFloat, device).unsqueeze(1);
    at::Tensor rank = gumbel_race_cuda(at::log(sorted * kept).contiguous(), drawing.to(at::kFloat), draw_rows);
    return order.gather(1, rank.unsqueeze(1)).squeeze(1);
}

}  // namespace pulsarlm
