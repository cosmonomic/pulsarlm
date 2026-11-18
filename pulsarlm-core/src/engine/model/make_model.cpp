#include "pulsarlm/engine/model/make_model.hpp"

#include "pulsarlm/engine/model/gemma4.hpp"
#include "pulsarlm/engine/model/llama.hpp"
#include "pulsarlm/engine/model/qwen2.hpp"
#include "pulsarlm/engine/model/qwen3.hpp"
#include "pulsarlm/engine/model/qwen3_5.hpp"
#include "pulsarlm/engine/model/qwen3_5_moe.hpp"
#include "pulsarlm/engine/model/qwen4_exp.hpp"

#include <ATen/core/Dict.h>

#include <format>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace pulsarlm {

std::size_t vocab_size(const any_model_config& config) {
    return std::visit(
        [](const auto& c) {
            using config_type = std::decay_t<decltype(c)>;
            if constexpr (std::is_same_v<config_type, model_config>) {
                return static_cast<std::size_t>(c.dense_model.vocab_size);
            } else {
                return static_cast<std::size_t>(c.vocab_size);
            }
        },
        config
    );
}

std::unique_ptr<model> make_model(
    const any_model_config& any_config,
    const std::unordered_map<std::string, at::Tensor>& weights,
    const communicator& comm
) {
    c10::Dict<std::string, at::Tensor> weight_dict;
    for (const auto& [name, tensor] : weights) {
        weight_dict.insert(name, tensor);
    }
    if (const auto* qwen = std::get_if<qwen3_5_config>(&any_config)) {
        return std::make_unique<qwen3_5_model>(*qwen, weight_dict, comm);
    }
    if (const auto* qwen = std::get_if<qwen4_exp_config>(&any_config)) {
        return std::make_unique<qwen4_exp_model>(*qwen, weight_dict, comm);
    }
    if (comm.size() > 1) {
        throw std::invalid_argument("only Qwen3.5 and Qwen4-Exp models split across GPUs");
    }
    if (const auto* gemma = std::get_if<gemma4_config>(&any_config)) {
        return std::make_unique<gemma4_model>(*gemma, weight_dict);
    }
    if (const auto* qwen = std::get_if<qwen3_5_moe_config>(&any_config)) {
        return std::make_unique<qwen3_5_moe_model>(*qwen, weight_dict);
    }
    const model_config& config = std::get<model_config>(any_config);
    if (config.architecture == "qwen2") {
        return std::make_unique<qwen2_model>(config.dense_model, weight_dict);
    }
    if (config.architecture == "qwen3") {
        return std::make_unique<qwen3_model>(config.dense_model, weight_dict);
    }
    if (config.architecture == "llama") {
        return std::make_unique<llama_model>(config.dense_model, weight_dict);
    }
    throw std::invalid_argument(std::format("unknown architecture {:?}", config.architecture));
}

}  // namespace pulsarlm
