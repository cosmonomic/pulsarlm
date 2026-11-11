#pragma once

#include <ATen/core/Tensor.h>

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

#include "pulsarlm/engine/communicator.hpp"
#include "pulsarlm/engine/model/config.hpp"
#include "pulsarlm/engine/model/interface.hpp"

namespace pulsarlm {

std::size_t vocab_size(const any_model_config& config);

// weights carry Hugging Face's names and are on the engine's device; they are comm's rank's slices when comm has more
// than one rank, which only Qwen3.5 and Qwen4-Exp split across.
std::unique_ptr<model> make_model(
    const any_model_config& config,
    const std::unordered_map<std::string, at::Tensor>& weights,
    const communicator& comm = {}
);

}  // namespace pulsarlm
