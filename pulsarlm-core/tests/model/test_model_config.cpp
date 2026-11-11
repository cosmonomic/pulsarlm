#include "pulsarlm/engine/model/make_model.hpp"

#undef CHECK
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

using Catch::Matchers::ContainsSubstring;
using pulsarlm::config_dict;
using pulsarlm::config_list;
using pulsarlm::config_value;

config_value integer(std::int64_t x) {
    return {x};
}

config_value number(double x) {
    return {x};
}

config_value text(std::string x) {
    return {std::move(x)};
}

config_value boolean(bool x) {
    return {x};
}

config_value null() {
    return {nullptr};
}

config_value list(std::initializer_list<config_value> items) {
    return {config_list(items)};
}

config_value dict(config_dict d) {
    return {std::move(d)};
}

config_value strings(const std::vector<std::string>& items) {
    config_list values;
    for (const std::string& item : items) {
        values.push_back(text(item));
    }
    return {std::move(values)};
}

config_dict with(config_dict base, const config_dict& changes) {
    for (const auto& [key, value] : changes) {
        base.insert_or_assign(key, value);
    }
    return base;
}

std::vector<std::string> repeated(std::initializer_list<std::pair<std::string, int>> runs, int times) {
    std::vector<std::string> names;
    for (int t = 0; t < times; t++) {
        for (const auto& [name, count] : runs) {
            names.insert(names.end(), count, name);
        }
    }
    return names;
}

const config_dict gemma4_rope_parameters{
    {"sliding_attention", dict({{"rope_type", text("default")}, {"rope_theta", number(10000.0)}})},
    {"full_attention",
     dict(
         {{"rope_type", text("proportional")},
          {"partial_rotary_factor", number(0.25)},
          {"rope_theta", number(1000000.0)}}
     )},
};

config_dict gemma4(const std::string& model_type, const config_dict& changes = {}) {
    config_dict base{
        {"vocab_size", integer(262144)},
        {"hidden_size", integer(1536)},
        {"num_hidden_layers", integer(12)},
        {"num_attention_heads", integer(8)},
        {"num_key_value_heads", integer(1)},
        {"head_dim", integer(256)},
        {"global_head_dim", integer(512)},
        {"sliding_window", integer(512)},
        {"rope_parameters", dict(gemma4_rope_parameters)},
        {"rms_norm_eps", number(1e-6)},
        {"final_logit_softcapping", number(30.0)},
        {"max_position_embeddings", integer(131072)},
    };
    return {
        {"model_type", text(model_type)},
        {"eos_token_id", list({integer(1), integer(106)})},
        {"text_config", dict(with(base, changes))},
    };
}

config_dict qwen3_5(const config_dict& changes = {}, const config_dict& top = {}) {
    config_dict base{
        {"vocab_size", integer(248320)},
        {"hidden_size", integer(1024)},
        {"intermediate_size", integer(3584)},
        {"num_hidden_layers", integer(8)},
        {"num_attention_heads", integer(8)},
        {"num_key_value_heads", integer(2)},
        {"head_dim", integer(256)},
        {"full_attention_interval", integer(4)},
        {"linear_num_key_heads", integer(16)},
        {"linear_num_value_heads", integer(16)},
        {"linear_key_head_dim", integer(128)},
        {"linear_value_head_dim", integer(128)},
        {"linear_conv_kernel_dim", integer(4)},
        {"rope_parameters",
         dict(
             {{"rope_type", text("default")},
              {"rope_theta", integer(10000000)},
              {"partial_rotary_factor", number(0.25)},
              {"mrope_section", list({integer(11), integer(11), integer(10)})}}
         )},
        {"rms_norm_eps", number(1e-6)},
    };
    config_dict text_config = with(base, changes);
    return with(
        {
            {"model_type", text(text_config.contains("num_experts") ? "qwen3_5_moe" : "qwen3_5")},
            {"eos_token_id", list({integer(248044)})},
            {"text_config", dict(text_config)},
        },
        top
    );
}

const config_dict moe{
    {"num_experts", integer(256)},
    {"num_experts_per_tok", integer(8)},
    {"moe_intermediate_size", integer(512)},
};

const config_dict fp8{
    {"quant_method", text("fp8")},
    {"fmt", text("e4m3")},
    {"activation_scheme", text("dynamic")},
    {"weight_block_size", list({integer(128), integer(128)})},
};

template <typename config_type> config_type parsed_as(const config_dict& config) {
    pulsarlm::any_model_config parsed = pulsarlm::parse_model_config(config);
    REQUIRE(std::holds_alternative<config_type>(parsed));
    return std::get<config_type>(parsed);
}

std::vector<std::string> qwen3_5_names(const std::vector<pulsarlm::qwen3_5_layer_type>& types) {
    std::vector<std::string> names;
    for (pulsarlm::qwen3_5_layer_type type : types) {
        names.emplace_back(type == pulsarlm::qwen3_5_layer_type::linear_attention ? "linear_attention" : "full_attention");
    }
    return names;
}

std::vector<std::string> gemma4_names(const std::vector<pulsarlm::gemma4_layer_type>& types) {
    std::vector<std::string> names;
    for (pulsarlm::gemma4_layer_type type : types) {
        names.emplace_back(type == pulsarlm::gemma4_layer_type::sliding_attention ? "sliding_attention" : "full_attention");
    }
    return names;
}

}  // namespace

TEST_CASE("a qwen3_5 checkpoint maps its text config") {
    auto config = parsed_as<pulsarlm::qwen3_5_config>(qwen3_5());
    REQUIRE(qwen3_5_names(config.layer_types) == repeated({{"linear_attention", 3}, {"full_attention", 1}}, 2));
}

TEST_CASE("a qwen3_5_moe checkpoint maps its experts per token") {
    auto config = parsed_as<pulsarlm::qwen3_5_moe_config>(qwen3_5(moe));
    REQUIRE(config.num_experts_per_tok == 8);
}

TEST_CASE("a qwen3_5 fp8 checkpoint maps its weight block") {
    auto config = parsed_as<pulsarlm::qwen3_5_config>(qwen3_5({}, {{"quantization_config", dict(fp8)}}));
    REQUIRE(config.fp8_weight_block.has_value());
    REQUIRE(config.fp8_weight_block->n == 128);
    REQUIRE(config.fp8_weight_block->k == 128);
}

TEST_CASE("an unquantized qwen3_5 checkpoint has no weight block") {
    REQUIRE(!parsed_as<pulsarlm::qwen3_5_config>(qwen3_5()).fp8_weight_block.has_value());
    REQUIRE(
        !parsed_as<pulsarlm::qwen3_5_config>(qwen3_5({}, {{"quantization_config", null()}})).fp8_weight_block.has_value()
    );
}

TEST_CASE("a qwen3_5 checkpoint of another quantization is rejected") {
    config_dict awq{{"quant_method", text("awq")}, {"bits", integer(4)}, {"group_size", integer(128)}};
    REQUIRE_THROWS_WITH(
        pulsarlm::parse_model_config(qwen3_5({}, {{"quantization_config", dict(awq)}})),
        ContainsSubstring("unsupported quantization awq")
    );
}

TEST_CASE("a quantized qwen3_5_moe checkpoint is rejected") {
    REQUIRE_THROWS_WITH(
        pulsarlm::parse_model_config(qwen3_5(moe, {{"quantization_config", dict(fp8)}})),
        ContainsSubstring("quantized qwen3_5_moe")
    );
}

TEST_CASE("a qwen3_5 checkpoint without an output gate is rejected") {
    REQUIRE_THROWS_WITH(
        pulsarlm::parse_model_config(qwen3_5({{"attn_output_gate", boolean(false)}})),
        ContainsSubstring("output gate")
    );
}

TEST_CASE("a qwen3_5 checkpoint with a sigmoid linear attention gate is rejected") {
    REQUIRE_THROWS_WITH(
        pulsarlm::parse_model_config(qwen3_5({{"output_gate_type", text("sigmoid")}})),
        ContainsSubstring("output gate sigmoid")
    );
}

TEST_CASE("a qwen4_exp checkpoint maps its ple layers and text eos") {
    config_dict text_config{
        {"vocab_size", integer(248320)},
        {"hidden_size", integer(2560)},
        {"num_hidden_layers", integer(8)},
        {"layer_types", strings(repeated({{"linear_attention", 3}, {"full_attention", 1}}, 2))},
        {"num_attention_heads", integer(24)},
        {"num_key_value_heads", integer(2)},
        {"head_dim", integer(256)},
        {"linear_num_key_heads", integer(16)},
        {"linear_num_value_heads", integer(48)},
        {"linear_key_head_dim", integer(128)},
        {"linear_value_head_dim", integer(128)},
        {"linear_conv_kernel_dim", integer(4)},
        {"num_experts", integer(512)},
        {"num_experts_per_tok", integer(10)},
        {"hc_count", integer(4)},
        {"ple_layer_ids", list({integer(2)})},
        {"ngram_size", integer(3)},
        {"output_gate_type", text("sigmoid")},
        {"rope_parameters",
         dict(
             {{"rope_type", text("default")},
              {"rope_theta", integer(10000000)},
              {"partial_rotary_factor", number(0.25)}}
         )},
        {"rms_norm_eps", number(1e-6)},
        {"eos_token_id", integer(248044)},
        {"indexer_n_heads", integer(4)},
        {"indexer_kv_heads", integer(1)},
        {"indexer_head_dim", integer(128)},
        {"indexer_budget", integer(2048)},
        {"indexer_compress_ratio", integer(4)},
    };
    auto config = parsed_as<pulsarlm::qwen4_exp_config>({
        {"model_type", text("qwen4_exp")},
        {"eos_token_id", list({integer(248046), integer(248044)})},
        {"text_config", dict(text_config)},
    });
    REQUIRE(config.ple_layers == std::vector<std::int64_t>{1});
    REQUIRE(config.eos_token_id == 248044);
    std::vector<std::string> names;
    for (pulsarlm::qwen4_exp_layer_type type : config.layer_types) {
        names.emplace_back(type == pulsarlm::qwen4_exp_layer_type::linear_attention ? "linear_attention" : "indexed_attention");
    }
    REQUIRE(names == repeated({{"linear_attention", 3}, {"indexed_attention", 1}}, 2));
    REQUIRE(config.linear_output_gate == pulsarlm::gate_activation::sigmoid);
}

TEST_CASE("a gemma4 checkpoint maps its text config") {
    auto config = parsed_as<pulsarlm::gemma4_config>(
        gemma4("gemma4", {{"num_kv_shared_layers", integer(4)}, {"hidden_size_per_layer_input", integer(256)}})
    );
    REQUIRE(gemma4_names(config.layer_types) == repeated({{"sliding_attention", 5}, {"full_attention", 1}}, 2));
    REQUIRE(config.global_n_kv_heads == 1);
    REQUIRE(config.hidden_size_per_layer_input == 256);
    REQUIRE(config.full_rope.partial_rotary_factor == 0.25);
    REQUIRE(config.full_rope.theta == 1e6);
    REQUIRE(config.sliding_rope.theta == 1e4);
}

TEST_CASE("k_eq_v takes the global kv heads and the last layer is full") {
    auto config = parsed_as<pulsarlm::gemma4_config>(gemma4(
        "gemma4_unified",
        {{"attention_k_eq_v", boolean(true)},
         {"num_global_key_value_heads", integer(2)},
         {"layer_types", strings(repeated({{"sliding_attention", 12}}, 1))}}
    ));
    REQUIRE(config.global_n_kv_heads == 2);
    REQUIRE(config.layer_types.back() == pulsarlm::gemma4_layer_type::full_attention);
    REQUIRE(config.hidden_size_per_layer_input == 0);
}

TEST_CASE("global kv heads apply only with k_eq_v") {
    auto config = parsed_as<pulsarlm::gemma4_config>(gemma4("gemma4", {{"num_global_key_value_heads", integer(4)}}));
    REQUIRE(config.global_n_kv_heads == 1);
}

TEST_CASE("an unsupported gemma4 config is rejected") {
    REQUIRE_THROWS_AS(
        pulsarlm::parse_model_config(gemma4("gemma4", {{"use_bidirectional_attention", text("all")}})),
        std::invalid_argument
    );
    config_dict scaled = gemma4_rope_parameters;
    scaled.insert_or_assign(
        "full_attention",
        dict(with(std::get<config_dict>(gemma4_rope_parameters.at("full_attention").value), {{"factor", number(8.0)}}))
    );
    REQUIRE_THROWS_AS(
        pulsarlm::parse_model_config(gemma4("gemma4", {{"rope_parameters", dict(scaled)}})),
        std::invalid_argument
    );
}

TEST_CASE("a qwen3 checkpoint maps to a dense model") {
    auto config = parsed_as<pulsarlm::model_config>({
        {"model_type", text("qwen3")},
        {"eos_token_id", integer(151645)},
        {"vocab_size", integer(151936)},
        {"hidden_size", integer(4096)},
        {"intermediate_size", integer(12288)},
        {"num_hidden_layers", integer(36)},
        {"num_attention_heads", integer(32)},
        {"num_key_value_heads", integer(8)},
        {"head_dim", integer(128)},
        {"rope_theta", integer(1000000)},
        {"rms_norm_eps", number(1e-6)},
    });
    REQUIRE(config.architecture == "qwen3");
    REQUIRE(config.dense_model.head_dim == 128);
}

namespace {

config_dict llama(const config_dict& changes = {}) {
    return with(
        {
            {"model_type", text("llama")},
            {"eos_token_id", list({integer(128001), integer(128008), integer(128009)})},
            {"vocab_size", integer(128256)},
            {"hidden_size", integer(4096)},
            {"intermediate_size", integer(14336)},
            {"num_hidden_layers", integer(32)},
            {"num_attention_heads", integer(32)},
            {"num_key_value_heads", integer(8)},
            {"rope_theta", number(500000.0)},
            {"rope_scaling", null()},
            {"rms_norm_eps", number(1e-5)},
        },
        changes
    );
}

const config_dict llama3_scaling{
    {"factor", number(8.0)},
    {"low_freq_factor", number(1.0)},
    {"high_freq_factor", number(4.0)},
    {"original_max_position_embeddings", integer(8192)},
    {"rope_type", text("llama3")},
};

}  // namespace

TEST_CASE("a llama checkpoint without rope scaling takes plain rope") {
    auto config = parsed_as<pulsarlm::model_config>(llama());
    REQUIRE(config.dense_model.rope.theta == 500000.0);
    REQUIRE(!config.dense_model.rope.llama3_scaling.has_value());
    REQUIRE(config.dense_model.head_dim == 128);
}

TEST_CASE("llama3 rope scaling is read from either config layout") {
    for (const config_dict& layout :
         {config_dict{{"rope_scaling", dict(llama3_scaling)}},
          config_dict{{"rope_parameters", dict(with(llama3_scaling, {{"rope_theta", number(500000.0)}}))}}}) {
        auto config = parsed_as<pulsarlm::model_config>(llama(layout));
        REQUIRE(config.dense_model.rope.theta == 500000.0);
        REQUIRE(config.dense_model.rope.llama3_scaling.has_value());
        const pulsarlm::llama3_rope_scaling& scaling = *config.dense_model.rope.llama3_scaling;
        REQUIRE(scaling.factor == 8.0);
        REQUIRE(scaling.low_freq_factor == 1.0);
        REQUIRE(scaling.high_freq_factor == 4.0);
        REQUIRE(scaling.original_max_position_embeddings == 8192);
    }
}

TEST_CASE("an unsupported llama config is rejected") {
    for (const config_dict& changes :
         {config_dict{{"rope_scaling", dict({{"type", text("yarn")}, {"factor", number(4.0)}})}},
          config_dict{{"rope_scaling", dict({{"rope_type", text("dynamic")}, {"factor", number(2.0)}})}},
          config_dict{{"attention_bias", boolean(true)}},
          config_dict{{"mlp_bias", boolean(true)}}}) {
        REQUIRE_THROWS_AS(pulsarlm::parse_model_config(llama(changes)), std::invalid_argument);
    }
}

TEST_CASE("an unknown architecture is rejected") {
    REQUIRE_THROWS_WITH(
        pulsarlm::parse_model_config({{"model_type", text("mistral")}, {"eos_token_id", integer(2)}}),
        ContainsSubstring("unsupported architecture")
    );
}
