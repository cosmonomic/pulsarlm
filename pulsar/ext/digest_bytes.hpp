#pragma once

#include <pybind11/pybind11.h>

#include <span>
#include <vector>

#include "pulsar/digest.hpp"

namespace pulsar {

// A digest as bytes is the hash's first 16 output bytes: hi, then lo, each little-endian (digest.hpp).
pybind11::bytes to_bytes(const digest& d);
// Throws std::invalid_argument when bytes is not 16 bytes long.
digest digest_from_bytes(const pybind11::bytes& bytes);

pybind11::list to_bytes_list(std::span<const digest> digests);
std::vector<digest> digests_from_bytes(const std::vector<pybind11::bytes>& list);

}  // namespace pulsar
