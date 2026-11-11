#pragma once

#include <string>
#include <variant>

#include "pulsarlm/engine/model/dense_config.hpp"
#include "pulsarlm/engine/model/gemma4_config.hpp"
#include "pulsarlm/engine/model/qwen3_5_config.hpp"
#include "pulsarlm/engine/model/qwen3_5_moe_config.hpp"
#include "pulsarlm/engine/model/qwen4_exp_config.hpp"

namespace pulsarlm {

// architecture is "qwen2", "qwen3" or "llama".
struct model_config {
    std::string architecture;
    dense_config dense_model;
};

using any_model_config = std::variant<model_config, gemma4_config, qwen3_5_config, qwen3_5_moe_config, qwen4_exp_config>;

}  // namespace pulsarlm
