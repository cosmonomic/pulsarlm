#include "pulsarlm/rope.hpp"

#include "kernels/rope_tables.hpp"

#include <ATen/ATen.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

namespace pulsarlm {

namespace {

constexpr long double two_pi = 6.28318530717958647692528676655900577L;

double quadrants_per_position(long double frequency) {
    constexpr long double two_over_pi = 0.63661977236758134307553505349005745L;
    return static_cast<double>(frequency * two_over_pi);
}

int64_t fixed_point_turns_per_position(long double frequency) {
    const long double fixed_point = std::nearbyint(std::ldexp(frequency / two_pi, 64));
    return std::bit_cast<int64_t>(static_cast<uint64_t>(fixed_point));
}

long double llama3_scaled(long double frequency, const llama3_rope_scaling& scaling) {
    const long double wavelength = two_pi / frequency;
    const long double original = scaling.original_max_position_embeddings;
    if (wavelength < original / scaling.high_freq_factor) {
        return frequency;
    }
    if (wavelength > original / scaling.low_freq_factor) {
        return frequency / scaling.factor;
    }
    const long double smooth = (original / wavelength - scaling.low_freq_factor) /
        (scaling.high_freq_factor - scaling.low_freq_factor);
    return (1.0L - smooth) * frequency / scaling.factor + smooth * frequency;
}

long double rope_frequency(const rope_parameters& rope, int64_t head_dim, int64_t pair) {
    const auto rotated_pairs = static_cast<int64_t>(std::floor(rope.partial_rotary_factor * head_dim / 2));
    if (pair >= rotated_pairs) {
        return 0.0L;
    }
    const long double exponent = -2.0L * static_cast<long double>(pair) / static_cast<long double>(head_dim);
    const long double frequency = std::pow(static_cast<long double>(rope.theta), exponent);
    return rope.llama3_scaling ? llama3_scaled(frequency, *rope.llama3_scaling) : frequency;
}

// Device table of head_dim/2 entries, entry j = per_frequency(rope_frequency(rope, head_dim, j)), built once per
// (rope, head_dim, device) and never freed; the reference stays valid for the process.
template <typename Element, Element (*per_frequency)(long double)>
const at::Tensor&
rope_frequency_table(const rope_parameters& rope, int64_t head_dim, at::Device device, at::ScalarType dtype) {
    using Key = std::tuple<rope_parameters, int64_t, int8_t>;
    static std::mutex guard;
    static auto* cache = new std::map<Key, at::Tensor>();

    const Key key{rope, head_dim, device.index()};
    const std::lock_guard<std::mutex> lock(guard);
    if (const auto found = cache->find(key); found != cache->end()) {
        return found->second;
    }

    const int64_t pairs = head_dim / 2;
    std::vector<Element> entries(pairs);
    for (int64_t j = 0; j < pairs; ++j) {
        entries[j] = per_frequency(rope_frequency(rope, head_dim, j));
    }
    auto tensor = at::tensor(entries, at::dtype(dtype)).to(device);
    return cache->emplace(key, std::move(tensor)).first->second;
}

}  // namespace

const at::Tensor& rope_quadrants_per_position(const rope_parameters& rope, int64_t head_dim, at::Device device) {
    return rope_frequency_table<double, quadrants_per_position>(rope, head_dim, device, at::kDouble);
}

const at::Tensor& rope_turns_per_position(const rope_parameters& rope, int64_t head_dim, at::Device device) {
    return rope_frequency_table<int64_t, fixed_point_turns_per_position>(rope, head_dim, device, at::kLong);
}

at::Tensor rope_rotate(const at::Tensor& x, const at::Tensor& position, const rope_parameters& rope) {
    constexpr double quadrant_radians = 1.57079632679489661923;
    const int64_t head_dim = x.size(-1), half = head_dim / 2;
    at::Tensor angle = position.to(at::kDouble).unsqueeze(-1) *
        rope_quadrants_per_position(rope, head_dim, x.device()) * quadrant_radians;
    at::Tensor cs = at::cos(angle).to(at::kFloat);
    at::Tensor sn = at::sin(angle).to(at::kFloat);
    at::Tensor first = x.slice(-1, 0, half), second = x.slice(-1, half, head_dim);
    return at::cat({first * cs - second * sn, second * cs + first * sn}, -1);
}

}  // namespace pulsarlm
