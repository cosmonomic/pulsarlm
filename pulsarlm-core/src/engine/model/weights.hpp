#pragma once

#include <ATen/core/Dict.h>
#include <ATen/core/Tensor.h>

#include <string>
#include <string_view>

namespace pulsarlm {

// The weight named name. Throws std::invalid_argument naming model_name when the weight is missing.
at::Tensor required_weight(
    const c10::Dict<std::string, at::Tensor>& weights,
    std::string_view model_name,
    const std::string& name
);

}  // namespace pulsarlm
