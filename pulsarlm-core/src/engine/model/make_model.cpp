#include "pulsarlm/engine/model/make_model.hpp"

#include "pulsarlm/engine/model/llama.hpp"
#include "pulsarlm/engine/model/qwen2.hpp"
#include "pulsarlm/engine/model/qwen3.hpp"

#include <ATen/core/Dict.h>

#include <algorithm>
#include <cstdint>
#include <format>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace pulsarlm {

namespace {

constexpr std::string_view sliding = "sliding_attention";
constexpr std::string_view full = "full_attention";
constexpr std::string_view linear = "linear_attention";
constexpr std::string_view indexed = "indexed_attention";

bool is_one_of(std::string_view architecture, std::initializer_list<std::string_view> architectures) {
    return std::ranges::find(architectures, architecture) != architectures.end();
}

const config_value* find(const config_dict& dict, std::string_view key) {
    auto it = dict.find(key);
    return it == dict.end() ? nullptr : &it->second;
}

bool is_null(const config_value* value) {
    return value == nullptr || std::holds_alternative<std::nullptr_t>(value->value);
}

// Python's truth value of a JSON value; a missing key is false.
bool truthy(const config_value* value) {
    if (value == nullptr) {
        return false;
    }
    return std::visit(
        [](const auto& v) {
            using value_type = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<value_type, std::nullptr_t>) {
                return false;
            } else if constexpr (std::is_same_v<value_type, bool>) {
                return v;
            } else if constexpr (std::is_same_v<value_type, std::int64_t> || std::is_same_v<value_type, double>) {
                return v != 0;
            } else {
                return !v.empty();
            }
        },
        value->value
    );
}

// The value as Python formats it in an f-string.
std::string describe(const config_value* value) {
    if (is_null(value)) {
        return "None";
    }
    return std::visit(
        [](const auto& v) -> std::string {
            using value_type = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<value_type, std::nullptr_t>) {
                return "None";
            } else if constexpr (std::is_same_v<value_type, bool>) {
                return v ? "True" : "False";
            } else if constexpr (std::is_same_v<value_type, std::int64_t> || std::is_same_v<value_type, double>) {
                return std::format("{}", v);
            } else if constexpr (std::is_same_v<value_type, std::string>) {
                return v;
            } else if constexpr (std::is_same_v<value_type, config_list>) {
                std::string text = "[";
                for (std::size_t i = 0; i < v.size(); i++) {
                    text += (i == 0 ? "" : ", ") + describe(&v[i]);
                }
                return text + "]";
            } else {
                return "{...}";
            }
        },
        value->value
    );
}

const config_value& required(const config_dict& dict, std::string_view key) {
    const config_value* value = find(dict, key);
    if (value == nullptr) {
        throw std::invalid_argument(std::format("the config has no {}", key));
    }
    return *value;
}

std::int64_t as_int(const config_value& value, std::string_view key) {
    if (const auto* integer = std::get_if<std::int64_t>(&value.value)) {
        return *integer;
    }
    throw std::invalid_argument(std::format("{} must be an integer, got {}", key, describe(&value)));
}

double as_number(const config_value& value, std::string_view key) {
    if (const auto* integer = std::get_if<std::int64_t>(&value.value)) {
        return static_cast<double>(*integer);
    }
    if (const auto* number = std::get_if<double>(&value.value)) {
        return *number;
    }
    throw std::invalid_argument(std::format("{} must be a number, got {}", key, describe(&value)));
}

const std::string& as_string(const config_value& value, std::string_view key) {
    if (const auto* text = std::get_if<std::string>(&value.value)) {
        return *text;
    }
    throw std::invalid_argument(std::format("{} must be a string, got {}", key, describe(&value)));
}

const config_list& as_list(const config_value& value, std::string_view key) {
    if (const auto* list = std::get_if<config_list>(&value.value)) {
        return *list;
    }
    throw std::invalid_argument(std::format("{} must be a list, got {}", key, describe(&value)));
}

const config_dict& as_dict(const config_value& value, std::string_view key) {
    if (const auto* dict = std::get_if<config_dict>(&value.value)) {
        return *dict;
    }
    throw std::invalid_argument(std::format("{} must be a dict, got {}", key, describe(&value)));
}

std::int64_t int_of(const config_dict& dict, std::string_view key) {
    return as_int(required(dict, key), key);
}

double number_of(const config_dict& dict, std::string_view key) {
    return as_number(required(dict, key), key);
}

// dict.get(key, fallback): the fallback only when key is missing.
std::int64_t int_or(const config_dict& dict, std::string_view key, std::int64_t fallback) {
    const config_value* value = find(dict, key);
    return value == nullptr ? fallback : as_int(*value, key);
}

double number_or(const config_dict& dict, std::string_view key, double fallback) {
    const config_value* value = find(dict, key);
    return value == nullptr ? fallback : as_number(*value, key);
}

std::string string_or(const config_dict& dict, std::string_view key, std::string_view fallback) {
    const config_value* value = find(dict, key);
    return value == nullptr ? std::string(fallback) : as_string(*value, key);
}

// dict.get(key) or empty, for a dict.
const config_dict& dict_or_empty(const config_dict& dict, std::string_view key) {
    static const config_dict empty;
    const config_value* value = find(dict, key);
    return truthy(value) ? as_dict(*value, key) : empty;
}

std::vector<std::string> strings_of(const config_list& list, std::string_view key) {
    std::vector<std::string> strings;
    for (const config_value& value : list) {
        strings.push_back(as_string(value, key));
    }
    return strings;
}

gemma4_layer_type gemma4_layer(std::string_view name) {
    if (name == sliding) {
        return gemma4_layer_type::sliding_attention;
    }
    if (name == full) {
        return gemma4_layer_type::full_attention;
    }
    throw std::invalid_argument(std::format("unknown Gemma 4 layer type {}", name));
}

qwen3_5_layer_type qwen3_5_layer(std::string_view name) {
    if (name == linear) {
        return qwen3_5_layer_type::linear_attention;
    }
    if (name == full) {
        return qwen3_5_layer_type::full_attention;
    }
    throw std::invalid_argument(std::format("unknown Qwen3.5 layer type {}", name));
}

qwen4_exp_layer_type qwen4_exp_layer(std::string_view name) {
    if (name == linear) {
        return qwen4_exp_layer_type::linear_attention;
    }
    if (name == indexed) {
        return qwen4_exp_layer_type::indexed_attention;
    }
    throw std::invalid_argument(std::format("unknown Qwen4-Exp layer type {}", name));
}

gate_activation activation(std::string_view name) {
    if (name == "silu") {
        return gate_activation::silu;
    }
    if (name == "sigmoid") {
        return gate_activation::sigmoid;
    }
    throw std::invalid_argument(std::format("unknown gate activation {}", name));
}

// Hugging Face's default is five sliding layers to one full layer, and the last layer is full in any case.
std::vector<gemma4_layer_type> gemma4_layer_types(const config_dict& text) {
    std::vector<std::string> names;
    if (const config_value* types = find(text, "layer_types"); truthy(types)) {
        names = strings_of(as_list(*types, "layer_types"), "layer_types");
    } else {
        std::int64_t n_layers = int_of(text, "num_hidden_layers");
        for (std::int64_t i = 0; i < n_layers; i++) {
            names.emplace_back((i + 1) % 6 ? sliding : full);
        }
    }
    if (names.empty()) {
        names.emplace_back(full);
    } else {
        names.back() = full;
    }
    std::vector<gemma4_layer_type> layer_types;
    for (const std::string& name : names) {
        layer_types.push_back(gemma4_layer(name));
    }
    return layer_types;
}

rope_parameters gemma4_rope(const config_dict& text, std::string_view layer_type) {
    const config_dict& parameters = as_dict(required(as_dict(required(text, "rope_parameters"), "rope_parameters"), layer_type), layer_type);
    std::string rope_type = string_or(parameters, "rope_type", "default");
    if (rope_type != "default" && rope_type != "proportional") {
        throw std::invalid_argument(std::format("unsupported RoPE {}", rope_type));
    }
    if (number_or(parameters, "factor", 1.0) != 1.0) {
        throw std::invalid_argument("unsupported RoPE scaling factor");
    }
    return rope_parameters{
        .theta = number_of(parameters, "rope_theta"),
        .partial_rotary_factor = number_or(parameters, "partial_rotary_factor", 1.0),
    };
}

gemma4_config parse_gemma4(std::string_view architecture, const config_dict& text) {
    if (const config_value* bidirectional = find(text, "use_bidirectional_attention");
        bidirectional != nullptr && std::holds_alternative<std::string>(bidirectional->value) &&
        std::get<std::string>(bidirectional->value) == "all") {
        throw std::invalid_argument("bidirectional attention is not supported");
    }
    if (std::string hidden_activation = string_or(text, "hidden_activation", "gelu_pytorch_tanh");
        hidden_activation != "gelu_pytorch_tanh") {
        throw std::invalid_argument(std::format("unsupported activation {}", hidden_activation));
    }
    bool k_eq_v = truthy(find(text, "attention_k_eq_v"));
    const config_value* global_kv_heads = find(text, "num_global_key_value_heads");
    std::int64_t n_kv_heads = int_of(text, "num_key_value_heads");
    const config_value* per_layer_input = find(text, "hidden_size_per_layer_input");
    std::int64_t hidden_size_per_layer_input = per_layer_input == nullptr ? (architecture == "gemma4" ? 256 : 0)
                                               : truthy(per_layer_input)
                                                   ? as_int(*per_layer_input, "hidden_size_per_layer_input")
                                                   : 0;
    const config_value* softcapping = find(text, "final_logit_softcapping");
    const config_value* top_k = find(text, "top_k_experts");
    return gemma4_config{
        .vocab_size = int_of(text, "vocab_size"),
        .hidden_size = int_of(text, "hidden_size"),
        .n_heads = int_of(text, "num_attention_heads"),
        .n_kv_heads = n_kv_heads,
        .head_dim = int_of(text, "head_dim"),
        .global_n_kv_heads = k_eq_v && !is_null(global_kv_heads)
                                 ? as_int(*global_kv_heads, "num_global_key_value_heads")
                                 : n_kv_heads,
        .global_head_dim = int_or(text, "global_head_dim", 512),
        .attention_k_eq_v = k_eq_v,
        .layer_types = gemma4_layer_types(text),
        .sliding_window = int_of(text, "sliding_window"),
        .sliding_rope = gemma4_rope(text, sliding),
        .full_rope = gemma4_rope(text, full),
        .rms_norm_eps = number_of(text, "rms_norm_eps"),
        .final_logit_softcapping = is_null(softcapping)
                                       ? std::nullopt
                                       : std::optional(as_number(*softcapping, "final_logit_softcapping")),
        .num_kv_shared_layers = int_or(text, "num_kv_shared_layers", 0),
        .hidden_size_per_layer_input = hidden_size_per_layer_input,
        .enable_moe_block = truthy(find(text, "enable_moe_block")),
        .top_k_experts = truthy(top_k) ? as_int(*top_k, "top_k_experts") : 0,
    };
}

// Hugging Face's default is a full-attention layer every full_attention_interval layers, the rest linear attention.
std::vector<std::string> qwen3_5_layer_names(const config_dict& text) {
    if (const config_value* types = find(text, "layer_types"); truthy(types)) {
        return strings_of(as_list(*types, "layer_types"), "layer_types");
    }
    std::int64_t interval = int_or(text, "full_attention_interval", 4);
    std::int64_t n_layers = int_of(text, "num_hidden_layers");
    std::vector<std::string> names;
    for (std::int64_t i = 0; i < n_layers; i++) {
        names.emplace_back((i + 1) % interval == 0 ? full : linear);
    }
    return names;
}

void check_qwen_attention(const config_dict& text) {
    if (truthy(find(text, "attention_bias"))) {
        throw std::invalid_argument("unsupported attention bias");
    }
    if (std::string hidden_act = string_or(text, "hidden_act", "silu"); hidden_act != "silu") {
        throw std::invalid_argument(std::format("unsupported activation {}", hidden_act));
    }
}

void check_default_rope(const config_dict& parameters) {
    if (std::string rope_type = string_or(parameters, "rope_type", "default"); rope_type != "default") {
        throw std::invalid_argument(std::format("unsupported RoPE {}", rope_type));
    }
}

// float(parameters.get("rope_theta") or text["rope_theta"])
double qwen_rope_theta(const config_dict& parameters, const config_dict& text) {
    const config_value* theta = find(parameters, "rope_theta");
    return truthy(theta) ? as_number(*theta, "rope_theta") : number_of(text, "rope_theta");
}

// float(parameters.get("partial_rotary_factor", text.get("partial_rotary_factor", 1.0)))
double qwen_partial_rotary_factor(const config_dict& parameters, const config_dict& text) {
    if (const config_value* factor = find(parameters, "partial_rotary_factor")) {
        return as_number(*factor, "partial_rotary_factor");
    }
    return number_or(text, "partial_rotary_factor", 1.0);
}

// The fields qwen3_5_config and qwen3_5_moe_config share.
template <typename config_type> config_type parse_qwen3_5_fields(const config_dict& text) {
    check_qwen_attention(text);
    if (const config_value* gate = find(text, "attn_output_gate"); gate != nullptr && !truthy(gate)) {
        throw std::invalid_argument("attention without an output gate is not supported");
    }
    if (std::string gate = string_or(text, "output_gate_type", "swish"); gate != "swish" && gate != "silu") {
        throw std::invalid_argument(std::format("unsupported linear attention output gate {}", gate));
    }
    const config_dict& parameters = dict_or_empty(text, "rope_parameters");
    check_default_rope(parameters);
    std::vector<qwen3_5_layer_type> layer_types;
    for (const std::string& name : qwen3_5_layer_names(text)) {
        layer_types.push_back(qwen3_5_layer(name));
    }
    config_type config{};
    config.vocab_size = int_of(text, "vocab_size");
    config.hidden_size = int_of(text, "hidden_size");
    config.layer_types = std::move(layer_types);
    config.n_heads = int_of(text, "num_attention_heads");
    config.n_kv_heads = int_of(text, "num_key_value_heads");
    config.head_dim = int_of(text, "head_dim");
    config.rope_theta = qwen_rope_theta(parameters, text);
    config.partial_rotary_factor = qwen_partial_rotary_factor(parameters, text);
    config.linear_n_key_heads = int_of(text, "linear_num_key_heads");
    config.linear_n_value_heads = int_of(text, "linear_num_value_heads");
    config.linear_key_head_dim = int_of(text, "linear_key_head_dim");
    config.linear_value_head_dim = int_of(text, "linear_value_head_dim");
    config.linear_conv_kernel = int_of(text, "linear_conv_kernel_dim");
    config.rms_norm_eps = number_of(text, "rms_norm_eps");
    return config;
}

// The [rows, columns] each scale of a block-quantized fp8 checkpoint covers.
fp8_block fp8_weight_block(const config_dict& quantization) {
    const config_value* method = find(quantization, "quant_method");
    const config_value* fmt = find(quantization, "fmt");
    const config_value* block = find(quantization, "weight_block_size");
    bool supported = method != nullptr && std::holds_alternative<std::string>(method->value) &&
                     std::get<std::string>(method->value) == "fp8" && string_or(quantization, "fmt", "e4m3") == "e4m3" &&
                     !is_null(block) && as_list(*block, "weight_block_size").size() == 2;
    if (!supported) {
        throw std::invalid_argument(std::format(
            "unsupported quantization {} {} with blocks {}",
            describe(method),
            fmt == nullptr ? "e4m3" : describe(fmt),
            describe(block)
        ));
    }
    const config_list& sizes = as_list(*block, "weight_block_size");
    return fp8_block{.n = as_int(sizes[0], "weight_block_size"), .k = as_int(sizes[1], "weight_block_size")};
}

// The checkpoint's full_attention layers are indexed attention layers, as transformers reads them.
qwen4_exp_config parse_qwen4_exp(const config_dict& text) {
    check_qwen_attention(text);
    if (const config_value* normalize = find(text, "norm_topk_prob"); normalize != nullptr && !truthy(normalize)) {
        throw std::invalid_argument("unnormalized top-k router weights are unsupported");
    }
    const config_dict& parameters = dict_or_empty(text, "rope_parameters");
    check_default_rope(parameters);
    std::vector<qwen4_exp_layer_type> layer_types;
    for (const std::string& name : qwen3_5_layer_names(text)) {
        layer_types.push_back(qwen4_exp_layer(name == full ? indexed : std::string_view(name)));
    }
    const config_value* gate = find(text, "output_gate_type");
    std::string gate_name = truthy(gate) ? as_string(*gate, "output_gate_type") : "silu";
    std::vector<std::int64_t> ple_layers;
    if (const config_value* ids = find(text, "ple_layer_ids"); truthy(ids)) {
        for (const config_value& id : as_list(*ids, "ple_layer_ids")) {
            ple_layers.push_back(as_int(id, "ple_layer_ids") - 1);
        }
    }
    const config_value& eos = required(text, "eos_token_id");
    std::int64_t eos_token_id = std::holds_alternative<config_list>(eos.value)
                                    ? as_int(as_list(eos, "eos_token_id").at(0), "eos_token_id")
                                    : as_int(eos, "eos_token_id");
    return qwen4_exp_config{
        .vocab_size = int_of(text, "vocab_size"),
        .hidden_size = int_of(text, "hidden_size"),
        .layer_types = std::move(layer_types),
        .n_heads = int_of(text, "num_attention_heads"),
        .n_kv_heads = int_of(text, "num_key_value_heads"),
        .head_dim = int_of(text, "head_dim"),
        .rope_theta = qwen_rope_theta(parameters, text),
        .partial_rotary_factor = qwen_partial_rotary_factor(parameters, text),
        .linear_n_key_heads = int_of(text, "linear_num_key_heads"),
        .linear_n_value_heads = int_of(text, "linear_num_value_heads"),
        .linear_key_head_dim = int_of(text, "linear_key_head_dim"),
        .linear_value_head_dim = int_of(text, "linear_value_head_dim"),
        .linear_conv_kernel = int_of(text, "linear_conv_kernel_dim"),
        .linear_output_gate = activation(gate_name == "swish" ? "silu" : gate_name),
        .rms_norm_eps = number_of(text, "rms_norm_eps"),
        .num_experts_per_tok = int_of(text, "num_experts_per_tok"),
        .hc_count = int_of(text, "hc_count"),
        .ple_layers = std::move(ple_layers),
        .ngram_size = int_of(text, "ngram_size"),
        .eos_token_id = eos_token_id,
        .indexer_n_heads = int_of(text, "indexer_n_heads"),
        .indexer_head_dim = int_of(text, "indexer_head_dim"),
        .indexer_budget = int_of(text, "indexer_budget"),
        .indexer_compress_ratio = int_of(text, "indexer_compress_ratio"),
    };
}

// Reads Hugging Face's rope_parameters, or the rope_scaling and rope_theta that preceded it.
rope_parameters dense_rope(const config_dict& text) {
    const config_dict* parameters = &dict_or_empty(text, "rope_parameters");
    if (parameters->empty()) {
        parameters = &dict_or_empty(text, "rope_scaling");
    }
    const config_value* theta_value = find(*parameters, "rope_theta");
    double theta = truthy(theta_value) ? as_number(*theta_value, "rope_theta") : number_of(text, "rope_theta");
    std::string rope_type = find(*parameters, "rope_type") != nullptr ? string_or(*parameters, "rope_type", "default")
                                                                       : string_or(*parameters, "type", "default");
    if (rope_type == "default") {
        return rope_parameters{.theta = theta};
    }
    if (rope_type == "llama3") {
        return rope_parameters{
            .theta = theta,
            .llama3_scaling =
                llama3_rope_scaling{
                    .factor = number_of(*parameters, "factor"),
                    .low_freq_factor = number_of(*parameters, "low_freq_factor"),
                    .high_freq_factor = number_of(*parameters, "high_freq_factor"),
                    .original_max_position_embeddings = number_of(*parameters, "original_max_position_embeddings"),
                },
        };
    }
    throw std::invalid_argument(std::format("unsupported RoPE {}", rope_type));
}

model_config parse_dense(std::string_view architecture, const config_dict& text) {
    if (truthy(find(text, "attention_bias")) || truthy(find(text, "mlp_bias"))) {
        throw std::invalid_argument("unsupported attention or MLP bias");
    }
    std::int64_t n_heads = int_of(text, "num_attention_heads");
    const config_value* head_dim = find(text, "head_dim");
    return model_config{
        .architecture = std::string(architecture),
        .dense_model =
            dense_config{
                .vocab_size = int_of(text, "vocab_size"),
                .n_layers = int_of(text, "num_hidden_layers"),
                .n_heads = n_heads,
                .n_kv_heads = int_of(text, "num_key_value_heads"),
                .head_dim = truthy(head_dim) ? as_int(*head_dim, "head_dim") : int_of(text, "hidden_size") / n_heads,
                .rope = dense_rope(text),
                .rms_norm_eps = number_of(text, "rms_norm_eps"),
            },
    };
}

}  // namespace

any_model_config parse_model_config(const config_dict& config) {
    const std::string& architecture = as_string(required(config, "model_type"), "model_type");
    bool multimodal = is_one_of(architecture, {"gemma4", "gemma4_unified", "qwen3_5", "qwen3_5_moe", "qwen4_exp"});
    const config_dict& text = multimodal ? as_dict(required(config, "text_config"), "text_config") : config;
    const config_value* quantization = find(config, "quantization_config");
    bool quantized = !is_null(quantization);
    bool qwen3_5_dense = is_one_of(architecture, {"qwen3_5", "qwen3_5_text"});
    if (quantized && !qwen3_5_dense) {
        throw std::invalid_argument(std::format("quantized {} checkpoints are not supported", architecture));
    }
    if (is_one_of(architecture, {"gemma4", "gemma4_unified"})) {
        return parse_gemma4(architecture, text);
    }
    if (is_one_of(architecture, {"qwen3_5_moe", "qwen3_5_moe_text"})) {
        auto config = parse_qwen3_5_fields<qwen3_5_moe_config>(text);
        config.num_experts_per_tok = int_of(text, "num_experts_per_tok");
        return config;
    }
    if (qwen3_5_dense) {
        auto config = parse_qwen3_5_fields<qwen3_5_config>(text);
        if (quantized) {
            config.fp8_weight_block = fp8_weight_block(as_dict(*quantization, "quantization_config"));
        }
        return config;
    }
    if (architecture == "qwen4_exp") {
        return parse_qwen4_exp(text);
    }
    if (is_one_of(architecture, {"qwen2", "qwen3", "llama"})) {
        return parse_dense(architecture, text);
    }
    throw std::invalid_argument(std::format("unsupported architecture {}", architecture));
}

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
