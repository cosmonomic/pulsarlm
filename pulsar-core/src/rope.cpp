#include "pulsar/rope.hpp"

#include <ATen/ATen.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

namespace pulsar {

namespace {

double quadrants_per_position(long double frequency) {
    constexpr long double two_over_pi = 0.63661977236758134307553505349005745L;
    return static_cast<double>(frequency * two_over_pi);
}

int64_t fixed_point_turns_per_position(long double frequency) {
    constexpr long double two_pi = 6.28318530717958647692528676655900577L;
    const long double fixed_point = std::nearbyint(std::ldexp(frequency / two_pi, 64));
    return std::bit_cast<int64_t>(static_cast<uint64_t>(fixed_point));
}

// Device table of head_dim/2 entries, entry j = per_frequency(theta^(-2j/head_dim)),
// built once per (theta, head_dim, device) and never freed; the reference stays valid
// for the process.
template <typename Element, Element (*per_frequency)(long double)>
const at::Tensor& rope_frequency_table(double theta, int64_t head_dim, at::Device device, at::ScalarType dtype) {
    using Key = std::tuple<double, int64_t, int8_t>;
    static std::mutex guard;
    static auto* cache = new std::map<Key, at::Tensor>();

    const Key key{theta, head_dim, device.index()};
    const std::lock_guard<std::mutex> lock(guard);
    if (const auto found = cache->find(key); found != cache->end()) {
        return found->second;
    }

    const int64_t pairs = head_dim / 2;
    std::vector<Element> entries(pairs);
    for (int64_t j = 0; j < pairs; ++j) {
        const long double exponent = -2.0L * static_cast<long double>(j) / static_cast<long double>(head_dim);
        entries[j] = per_frequency(std::pow(static_cast<long double>(theta), exponent));
    }
    auto tensor = at::tensor(entries, at::dtype(dtype)).to(device);
    return cache->emplace(key, std::move(tensor)).first->second;
}

}  // namespace

const at::Tensor& rope_quadrants_per_position(double theta, int64_t head_dim, at::Device device) {
    return rope_frequency_table<double, quadrants_per_position>(theta, head_dim, device, at::kDouble);
}

const at::Tensor& rope_turns_per_position(double theta, int64_t head_dim, at::Device device) {
    return rope_frequency_table<int64_t, fixed_point_turns_per_position>(theta, head_dim, device, at::kLong);
}

at::Tensor rope_rotate(const at::Tensor& x, const at::Tensor& position, double theta) {
    constexpr double quadrant_radians = 1.57079632679489661923;
    const int64_t head_dim = x.size(-1), half = head_dim / 2;
    at::Tensor angle = position.to(at::kDouble).unsqueeze(-1) *
        rope_quadrants_per_position(theta, head_dim, x.device()) * quadrant_radians;
    at::Tensor cs = at::cos(angle).to(at::kFloat);
    at::Tensor sn = at::sin(angle).to(at::kFloat);
    at::Tensor first = x.slice(-1, 0, half), second = x.slice(-1, half, head_dim);
    return at::cat({first * cs - second * sn, second * cs + first * sn}, -1);
}

}  // namespace pulsar
