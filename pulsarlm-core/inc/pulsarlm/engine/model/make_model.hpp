#pragma once

#include <ATen/core/Tensor.h>

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>

#include "pulsarlm/engine/communicator.hpp"
#include "pulsarlm/engine/config_value.hpp"
#include "pulsarlm/engine/model/dense_config.hpp"
#include "pulsarlm/engine/model/gemma4.hpp"
#include "pulsarlm/engine/model/interface.hpp"
#include "pulsarlm/engine/model/qwen3_5.hpp"
#include "pulsarlm/engine/model/qwen3_5_moe.hpp"
#include "pulsarlm/engine/model/qwen4_exp.hpp"

namespace pulsarlm {

// architecture is "qwen2", "qwen3" or "llama".
struct model_config {
    std::string architecture;
    dense_config dense_model;
};

using any_model_config = std::variant<model_config, gemma4_config, qwen3_5_config, qwen3_5_moe_config, qwen4_exp_config>;

// The model of a checkpoint's config.json, read from its text config: text_config for a multimodal architecture, the
// config itself otherwise. Throws std::invalid_argument for a config the engine does not run.
any_model_config parse_model_config(const config_dict& config);

std::size_t vocab_size(const any_model_config& config);

// weights carry Hugging Face's names and are on the engine's device; they are comm's rank's slices when comm has more
// than one rank, which only Qwen3.5 and Qwen4-Exp split across.
std::unique_ptr<model> make_model(
    const any_model_config& config,
    const std::unordered_map<std::string, at::Tensor>& weights,
    const communicator& comm = {}
);

}  // namespace pulsarlm
