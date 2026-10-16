#pragma once

#include <pybind11/pybind11.h>

#include <span>

#include "pulsarlm/digest.hpp"

namespace pulsarlm {

// A digest as bytes is the hash's first 16 output bytes: hi, then lo, each little-endian (digest.hpp).
pybind11::bytes to_bytes(const digest& d);
// Throws std::invalid_argument when bytes is not 16 bytes long.
digest digest_from_bytes(const pybind11::bytes& bytes);

pybind11::list to_bytes_list(std::span<const digest> digests);

}  // namespace pulsarlm
