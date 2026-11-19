#include "weights.hpp"

#include <format>
#include <stdexcept>

namespace pulsarlm {

at::Tensor required_weight(
    const c10::Dict<std::string, at::Tensor>& weights,
    std::string_view model_name,
    const std::string& name
) {
    if (!weights.contains(name)) {
        throw std::invalid_argument(std::format("{}: missing weight {:?}", model_name, name));
    }
    return weights.at(name);
}

}  // namespace pulsarlm
