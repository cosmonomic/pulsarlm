#include "pulsarlm/sampling.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace pulsarlm {

namespace {

// argmax(p / e), with each e drawn from Exp(1), is index i with probability p_i / sum(p). Unlike at::multinomial, it
// reads nothing back to the host.
at::Tensor draw(const at::Tensor& probabilities, at::Generator& generator) {
    at::Tensor race = at::empty_like(probabilities).exponential_(1.0, generator);
    return (probabilities / race).argmax();
}

at::Tensor sample_row(const at::Tensor& logits, const sampling_params& params, at::Generator& generator) {
    at::Tensor probabilities = at::softmax(logits.to(at::kFloat) / params.temperature, -1);
    bool top_k = params.top_k > 0;
    bool top_p = params.top_p > 0.0 && params.top_p < 1.0;
    if (!top_k && !top_p) {
        return draw(probabilities, generator);
    }
    auto [sorted, order] = at::sort(probabilities, -1, /*descending=*/true);
    at::Tensor kept = at::ones_like(sorted, at::kBool);
    if (top_k) {
        kept &= at::arange(sorted.size(0), sorted.options().dtype(at::kLong)) < static_cast<std::int64_t>(params.top_k);
    }
    if (top_p) {
        kept &= at::cumsum(sorted, -1) - sorted <= params.top_p;
    }
    return order.index_select(0, draw(sorted * kept, generator).unsqueeze(0)).squeeze(0);
}

}  // namespace

at::Generator make_sampling_generator(at::Device device, std::optional<std::uint64_t> seed) {
    at::Generator generator = device.is_cuda() ? at::make_generator<at::CUDAGeneratorImpl>(device.index())
                                               : at::make_generator<at::CPUGeneratorImpl>();
    if (seed) {
        generator.set_current_seed(*seed);
    } else {
        generator.seed();
    }
    return generator;
}

at::Tensor sample(const at::Tensor& logits, std::span<const sampling_params> params, std::span<at::Generator> generators) {
    assert(params.size() == static_cast<std::size_t>(logits.size(0)) && generators.size() == params.size());
    at::Tensor tokens = logits.argmax(-1);
    for (std::size_t row = 0; row < params.size(); row++) {
        if (params[row].temperature > 0.0) {
            auto index = static_cast<std::int64_t>(row);
            tokens.select(0, index).copy_(sample_row(logits.select(0, index), params[row], generators[row]));
        }
    }
    return tokens;
}

}  // namespace pulsarlm
