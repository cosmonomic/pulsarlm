#include "digest_bytes.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <string_view>

namespace pulsarlm {

namespace {

constexpr std::size_t digest_size = 16;

void write_le(std::uint64_t value, char* out) {
    for (std::size_t i = 0; i < 8; i++) {
        out[i] = static_cast<char>(value >> (8 * i));
    }
}

std::uint64_t read_le(const char* in) {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < 8; i++) {
        value |= static_cast<std::uint64_t>(static_cast<unsigned char>(in[i])) << (8 * i);
    }
    return value;
}

}  // namespace

pybind11::bytes to_bytes(const digest& d) {
    std::array<char, digest_size> out;
    write_le(d.hi, out.data());
    write_le(d.lo, out.data() + 8);
    return pybind11::bytes(out.data(), out.size());
}

digest digest_from_bytes(const pybind11::bytes& bytes) {
    std::string_view view = bytes;
    if (view.size() != digest_size) {
        throw std::invalid_argument(std::format("a digest is {} bytes, not {}", digest_size, view.size()));
    }
    return {read_le(view.data()), read_le(view.data() + 8)};
}

pybind11::list to_bytes_list(std::span<const digest> digests) {
    pybind11::list list(digests.size());
    for (std::size_t i = 0; i < digests.size(); i++) {
        list[i] = to_bytes(digests[i]);
    }
    return list;
}

}  // namespace pulsarlm
