#include <torch/csrc/utils/pybind.h>

#include <pybind11/gil_safe_call_once.h>
#include <pybind11/native_enum.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "digest_bytes.hpp"
#include "python_engine.hpp"
#include "python_store.hpp"

namespace py = pybind11;

namespace pulsarlm {

namespace {

PYBIND11_CONSTINIT py::gil_safe_call_once_and_store<py::object> request_failed_type;

// An exception class whose error property is its first argument.
py::object make_request_failed_type() {
    auto type = py::reinterpret_steal<py::object>(
        PyErr_NewException("pulsarlm.pulsarlm_core.RequestFailed", PyExc_Exception, nullptr)
    );
    py::cpp_function first_argument([](const py::handle& self) { return self.attr("args")[py::int_(0)]; });
    type.attr("error") = py::module_::import("builtins").attr("property")(first_argument);
    return type;
}

gemma4_layer_type layer_type(const std::string& name) {
    if (name == "sliding_attention") {
        return gemma4_layer_type::sliding_attention;
    }
    if (name == "full_attention") {
        return gemma4_layer_type::full_attention;
    }
    throw std::invalid_argument("unknown Gemma 4 layer type " + name);
}

qwen3_5_layer_type qwen3_5_layer(const std::string& name) {
    if (name == "linear_attention") {
        return qwen3_5_layer_type::linear_attention;
    }
    if (name == "full_attention") {
        return qwen3_5_layer_type::full_attention;
    }
    throw std::invalid_argument("unknown Qwen3.5 layer type " + name);
}

qwen4_exp_layer_type qwen4_exp_layer(const std::string& name) {
    if (name == "linear_attention") {
        return qwen4_exp_layer_type::linear_attention;
    }
    if (name == "indexed_attention") {
        return qwen4_exp_layer_type::indexed_attention;
    }
    throw std::invalid_argument("unknown Qwen4-Exp layer type " + name);
}

gate_activation activation(const std::string& name) {
    if (name == "silu") {
        return gate_activation::silu;
    }
    if (name == "sigmoid") {
        return gate_activation::sigmoid;
    }
    throw std::invalid_argument("unknown gate activation " + name);
}

// Shard r of the model runs on devices[r] with the weights load_model_shard(r, len(devices)) returns; kv_cache_gib
// sizes each device's KV cache.
std::unique_ptr<python_engine> make_engine(
    py::object store,
    const any_model_config& model_config,
    py::object load_model_shard,
    std::vector<at::Device> devices,
    const std::string& model_name,
    const std::string& model_hash,
    std::size_t page_size,
    at::ScalarType kv_dtype,
    double kv_cache_gib,
    double store_gib,
    std::vector<std::int64_t> stop_ids,
    std::size_t prefill_tokens_per_step
) {
    if (!(kv_cache_gib > 0.0) || !(store_gib >= 0.0)) {
        throw std::invalid_argument("kv_cache_gib must be positive and store_gib not negative");
    }
    engine_config engine_settings{
        .model_name = model_name,
        .model_hash = model_hash,
        .page_size = page_size,
        .kv_dtype = kv_dtype,
        .kv_cache_pool_bytes = static_cast<std::size_t>(kv_cache_gib * (1ull << 30)),
        .devices = std::move(devices),
        .stop_ids = std::move(stop_ids),
        .prefill_tokens_per_step = prefill_tokens_per_step,
    };
    auto adapter = std::make_unique<python_store>(std::move(store), static_cast<std::size_t>(store_gib * (1ull << 30)));
    model_shard_loader loader = [&load_model_shard](std::size_t rank, std::size_t size) {
        py::gil_scoped_acquire gil;
        return load_model_shard(rank, size).cast<std::unordered_map<std::string, at::Tensor>>();
    };
    py::gil_scoped_release nogil;
    return std::make_unique<python_engine>(model_config, loader, engine_settings, std::move(adapter));
}

}  // namespace

void raise_request_failed(request_error error) {
    PyErr_SetObject(request_failed_type.get_stored().ptr(), py::cast(error).ptr());
    throw py::error_already_set();
}

}  // namespace pulsarlm

PYBIND11_MODULE(pulsarlm_core, m) {
    using namespace pulsarlm;
    using namespace py::literals;

    py::class_<moments>(m, "Moments")
        .def(
            py::init([](std::uint64_t count, double mean, double m2) {
                return moments{.count = count, .mean = mean, .m2 = m2};
            }),
            "count"_a = 0,
            "mean"_a = 0.0,
            "m2"_a = 0.0
        )
        .def_readwrite("count", &moments::count)
        .def_readwrite("mean", &moments::mean)
        .def_readwrite("m2", &moments::m2)
        .def(py::self == py::self);

    py::class_<active_page>(m, "ActivePage")
        .def(
            py::init([](const py::bytes& hash, std::size_t index, std::vector<float> mass_sum, float weight_sum) {
                return active_page{
                    .hash = digest_from_bytes(hash),
                    .index = index,
                    .mass_sum = std::move(mass_sum),
                    .weight_sum = weight_sum,
                };
            }),
            "hash"_a,
            "index"_a,
            "mass_sum"_a = std::vector<float>{},
            "weight_sum"_a = 0.0f
        )
        .def_property(
            "hash",
            [](const active_page& p) { return to_bytes(p.hash); },
            [](active_page& p, const py::bytes& hash) { p.hash = digest_from_bytes(hash); }
        )
        .def_readwrite("index", &active_page::index)
        .def_readwrite("mass_sum", &active_page::mass_sum)
        .def_readwrite("weight_sum", &active_page::weight_sum)
        .def(py::self == py::self);

    py::class_<layer_sequence>(m, "LayerSequence")
        .def(
            py::init([](std::vector<active_page> pages, std::vector<moments> density_moments, moments score_moments) {
                return layer_sequence{
                    .pages = std::move(pages),
                    .density_moments = std::move(density_moments),
                    .score_moments = score_moments,
                };
            }),
            "pages"_a = std::vector<active_page>(),
            "density_moments"_a = std::vector<moments>(),
            "score_moments"_a = moments{}
        )
        .def_readwrite("pages", &layer_sequence::pages)
        .def_readwrite("density_moments", &layer_sequence::density_moments)
        .def_readwrite("score_moments", &layer_sequence::score_moments)
        .def(py::self == py::self);

    py::class_<state>(m, "State")
        .def(
            py::init([](const std::optional<py::bytes>& head,
                        std::vector<layer_sequence> layers,
                        std::vector<std::int64_t> tail) {
                return state{
                    .head = head ? std::optional(digest_from_bytes(*head)) : std::nullopt,
                    .layers = std::move(layers),
                    .tail = std::move(tail),
                };
            }),
            "head"_a = py::none(),
            "layers"_a = std::vector<layer_sequence>(),
            "tail"_a = std::vector<std::int64_t>()
        )
        .def_property(
            "head",
            [](const state& s) -> std::optional<py::bytes> {
                return s.head ? std::optional(to_bytes(*s.head)) : std::nullopt;
            },
            [](state& s, const std::optional<py::bytes>& head) {
                s.head = head ? std::optional(digest_from_bytes(*head)) : std::nullopt;
            }
        )
        .def_readwrite("layers", &state::layers)
        .def_readwrite("tail", &state::tail);

    py::class_<cycle_end>(m, "CycleEnd")
        .def_readonly("state", &cycle_end::state)
        .def_readonly("token_count", &cycle_end::token_count);

    py::class_<request_result>(m, "RequestResult")
        .def_readonly("state", &request_result::state)
        .def_readonly("total_tokens", &request_result::total_tokens)
        .def_readonly("cached_tokens", &request_result::cached_tokens)
        .def_readonly("device_cached_tokens", &request_result::device_cached_tokens);

    py::class_<engine_stat>(m, "EngineStat").def_readonly("kv_cache_bytes", &engine_stat::kv_cache_bytes);

    py::native_enum<request_error>(m, "RequestError", "enum.Enum")
        .value("MISS", request_error::miss)
        .value("OOM", request_error::oom)
        .value("CANCELLED", request_error::cancelled)
        .value("LOOP_EXITED", request_error::loop_exited)
        .finalize();

    m.attr("RequestFailed") = request_failed_type.call_once_and_store_result(make_request_failed_type).get_stored();

    py::class_<sampling_params>(m, "SamplingParams")
        .def(
            py::init([](double temperature, double top_p, std::size_t top_k, std::optional<std::uint64_t> seed) {
                return sampling_params{temperature, top_p, top_k, seed};
            }),
            py::kw_only(),
            "temperature"_a = 0.0,
            "top_p"_a = 1.0,
            "top_k"_a = 0,
            "seed"_a = py::none()
        )
        .def_readwrite("temperature", &sampling_params::temperature)
        .def_readwrite("top_p", &sampling_params::top_p)
        .def_readwrite("top_k", &sampling_params::top_k)
        .def_readwrite("seed", &sampling_params::seed);

    py::class_<eviction_policy>(m, "EvictionPolicy")
        .def(
            py::init([](std::size_t n_sink,
                        std::size_t n_working,
                        std::size_t cycle_tokens,
                        std::optional<std::size_t> max_context_window,
                        double eviction_threshold,
                        std::size_t eviction_warmup,
                        double attention_mass_decay) {
                return eviction_policy{
                    n_sink,
                    n_working,
                    cycle_tokens,
                    max_context_window,
                    eviction_threshold,
                    eviction_warmup,
                    attention_mass_decay
                };
            }),
            py::kw_only(),
            "n_sink"_a = eviction_policy{}.n_sink,
            "n_working"_a = eviction_policy{}.n_working,
            "cycle_tokens"_a = eviction_policy{}.cycle_tokens,
            "max_context_window"_a = eviction_policy{}.max_context_window,
            "eviction_threshold"_a = eviction_policy{}.eviction_threshold,
            "eviction_warmup"_a = eviction_policy{}.eviction_warmup,
            "attention_mass_decay"_a = eviction_policy{}.attention_mass_decay
        )
        .def_readwrite("n_sink", &eviction_policy::n_sink)
        .def_readwrite("n_working", &eviction_policy::n_working)
        .def_readwrite("cycle_tokens", &eviction_policy::cycle_tokens)
        .def_readwrite("max_context_window", &eviction_policy::max_context_window)
        .def_readwrite("eviction_threshold", &eviction_policy::eviction_threshold)
        .def_readwrite("eviction_warmup", &eviction_policy::eviction_warmup)
        .def_readwrite("attention_mass_decay", &eviction_policy::attention_mass_decay);

    py::class_<layer_scores>(m, "LayerScores")
        .def_readonly("layer", &layer_scores::layer)
        .def_readonly("threshold", &layer_scores::threshold)
        .def_readonly("page_scores", &layer_scores::page_scores);

    py::class_<eviction_report>(m, "EvictionReport")
        .def_readonly("cap_tokens", &eviction_report::cap_tokens)
        .def_readonly("layers", &eviction_report::layers);

    py::class_<model_config>(m, "ModelConfig")
        .def(
            py::init([](std::string architecture,
                        std::int64_t vocab_size,
                        std::int64_t n_layers,
                        std::int64_t n_heads,
                        std::int64_t n_kv_heads,
                        std::int64_t head_dim,
                        const rope_parameters& rope,
                        double rms_norm_eps) {
                return model_config{
                    .architecture = std::move(architecture),
                    .dense_model = dense_config{
                        .vocab_size = vocab_size,
                        .n_layers = n_layers,
                        .n_heads = n_heads,
                        .n_kv_heads = n_kv_heads,
                        .head_dim = head_dim,
                        .rope = rope,
                        .rms_norm_eps = rms_norm_eps,
                    },
                };
            }),
            py::kw_only(),
            "architecture"_a,
            "vocab_size"_a,
            "n_layers"_a,
            "n_heads"_a,
            "n_kv_heads"_a,
            "head_dim"_a,
            "rope"_a,
            "rms_norm_eps"_a
        )
        .def_property_readonly("rope", [](const model_config& c) { return c.dense_model.rope; });

    py::class_<llama3_rope_scaling>(m, "Llama3RopeScaling")
        .def(
            py::init([](double factor,
                        double low_freq_factor,
                        double high_freq_factor,
                        double original_max_position_embeddings) {
                return llama3_rope_scaling{factor, low_freq_factor, high_freq_factor, original_max_position_embeddings};
            }),
            py::kw_only(),
            "factor"_a,
            "low_freq_factor"_a,
            "high_freq_factor"_a,
            "original_max_position_embeddings"_a
        )
        .def_readonly("factor", &llama3_rope_scaling::factor)
        .def_readonly("low_freq_factor", &llama3_rope_scaling::low_freq_factor)
        .def_readonly("high_freq_factor", &llama3_rope_scaling::high_freq_factor)
        .def_readonly("original_max_position_embeddings", &llama3_rope_scaling::original_max_position_embeddings);

    py::class_<rope_parameters>(m, "RopeParameters")
        .def(
            py::init([](double theta, double partial_rotary_factor, std::optional<llama3_rope_scaling> llama3_scaling) {
                return rope_parameters{theta, partial_rotary_factor, llama3_scaling};
            }),
            py::kw_only(),
            "theta"_a,
            "partial_rotary_factor"_a = 1.0,
            "llama3_scaling"_a = py::none()
        )
        .def_readonly("theta", &rope_parameters::theta)
        .def_readonly("partial_rotary_factor", &rope_parameters::partial_rotary_factor)
        .def_readonly("llama3_scaling", &rope_parameters::llama3_scaling);

    py::class_<gemma4_config>(m, "Gemma4Config")
        .def(
            py::init([](std::int64_t vocab_size,
                        std::int64_t hidden_size,
                        std::int64_t n_heads,
                        std::int64_t n_kv_heads,
                        std::int64_t head_dim,
                        std::int64_t global_n_kv_heads,
                        std::int64_t global_head_dim,
                        bool attention_k_eq_v,
                        const std::vector<std::string>& layer_types,
                        std::int64_t sliding_window,
                        const rope_parameters& sliding_rope,
                        const rope_parameters& full_rope,
                        double rms_norm_eps,
                        std::optional<double> final_logit_softcapping,
                        std::int64_t num_kv_shared_layers,
                        std::int64_t hidden_size_per_layer_input,
                        bool enable_moe_block,
                        std::int64_t top_k_experts) {
                std::vector<gemma4_layer_type> types;
                for (const std::string& name : layer_types) {
                    types.push_back(layer_type(name));
                }
                return gemma4_config{
                    vocab_size,
                    hidden_size,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    global_n_kv_heads,
                    global_head_dim,
                    attention_k_eq_v,
                    std::move(types),
                    sliding_window,
                    sliding_rope,
                    full_rope,
                    rms_norm_eps,
                    final_logit_softcapping,
                    num_kv_shared_layers,
                    hidden_size_per_layer_input,
                    enable_moe_block,
                    top_k_experts,
                };
            }),
            py::kw_only(),
            "vocab_size"_a,
            "hidden_size"_a,
            "n_heads"_a,
            "n_kv_heads"_a,
            "head_dim"_a,
            "global_n_kv_heads"_a,
            "global_head_dim"_a,
            "attention_k_eq_v"_a,
            "layer_types"_a,
            "sliding_window"_a,
            "sliding_rope"_a,
            "full_rope"_a,
            "rms_norm_eps"_a,
            "final_logit_softcapping"_a,
            "num_kv_shared_layers"_a,
            "hidden_size_per_layer_input"_a,
            "enable_moe_block"_a,
            "top_k_experts"_a
        )
        .def_property_readonly(
            "layer_types",
            [](const gemma4_config& c) {
                std::vector<std::string> names;
                for (gemma4_layer_type type : c.layer_types) {
                    names.push_back(
                        type == gemma4_layer_type::sliding_attention ? "sliding_attention" : "full_attention"
                    );
                }
                return names;
            }
        )
        .def_readonly("global_n_kv_heads", &gemma4_config::global_n_kv_heads)
        .def_readonly("hidden_size_per_layer_input", &gemma4_config::hidden_size_per_layer_input)
        .def_readonly("sliding_rope", &gemma4_config::sliding_rope)
        .def_readonly("full_rope", &gemma4_config::full_rope);

    py::class_<qwen3_5_config>(m, "Qwen3_5Config")
        .def(
            py::init([](std::int64_t vocab_size,
                        std::int64_t hidden_size,
                        const std::vector<std::string>& layer_types,
                        std::int64_t n_heads,
                        std::int64_t n_kv_heads,
                        std::int64_t head_dim,
                        double rope_theta,
                        double partial_rotary_factor,
                        std::int64_t linear_n_key_heads,
                        std::int64_t linear_n_value_heads,
                        std::int64_t linear_key_head_dim,
                        std::int64_t linear_value_head_dim,
                        std::int64_t linear_conv_kernel,
                        double rms_norm_eps,
                        std::optional<std::pair<std::int64_t, std::int64_t>> fp8_weight_block) {
                std::vector<qwen3_5_layer_type> types;
                for (const std::string& name : layer_types) {
                    types.push_back(qwen3_5_layer(name));
                }
                return qwen3_5_config{
                    .vocab_size = vocab_size,
                    .hidden_size = hidden_size,
                    .layer_types = std::move(types),
                    .n_heads = n_heads,
                    .n_kv_heads = n_kv_heads,
                    .head_dim = head_dim,
                    .rope_theta = rope_theta,
                    .partial_rotary_factor = partial_rotary_factor,
                    .linear_n_key_heads = linear_n_key_heads,
                    .linear_n_value_heads = linear_n_value_heads,
                    .linear_key_head_dim = linear_key_head_dim,
                    .linear_value_head_dim = linear_value_head_dim,
                    .linear_conv_kernel = linear_conv_kernel,
                    .rms_norm_eps = rms_norm_eps,
                    .fp8_weight_block = fp8_weight_block.transform(
                        [](const std::pair<std::int64_t, std::int64_t>& block) {
                            return fp8_block{.n = block.first, .k = block.second};
                        }
                    ),
                };
            }),
            py::kw_only(),
            "vocab_size"_a,
            "hidden_size"_a,
            "layer_types"_a,
            "n_heads"_a,
            "n_kv_heads"_a,
            "head_dim"_a,
            "rope_theta"_a,
            "partial_rotary_factor"_a,
            "linear_n_key_heads"_a,
            "linear_n_value_heads"_a,
            "linear_key_head_dim"_a,
            "linear_value_head_dim"_a,
            "linear_conv_kernel"_a,
            "rms_norm_eps"_a,
            "fp8_weight_block"_a = py::none()
        )
        .def_property_readonly(
            "fp8_weight_block",
            [](const qwen3_5_config& c) {
                return c.fp8_weight_block.transform([](const fp8_block& block) { return std::pair{block.n, block.k}; });
            }
        )
        .def_property_readonly("layer_types", [](const qwen3_5_config& c) {
            std::vector<std::string> names;
            for (qwen3_5_layer_type type : c.layer_types) {
                names.push_back(type == qwen3_5_layer_type::linear_attention ? "linear_attention" : "full_attention");
            }
            return names;
        });

    py::class_<qwen3_5_moe_config>(m, "Qwen3_5MoeConfig")
        .def(
            py::init([](std::int64_t vocab_size,
                        std::int64_t hidden_size,
                        const std::vector<std::string>& layer_types,
                        std::int64_t n_heads,
                        std::int64_t n_kv_heads,
                        std::int64_t head_dim,
                        double rope_theta,
                        double partial_rotary_factor,
                        std::int64_t linear_n_key_heads,
                        std::int64_t linear_n_value_heads,
                        std::int64_t linear_key_head_dim,
                        std::int64_t linear_value_head_dim,
                        std::int64_t linear_conv_kernel,
                        double rms_norm_eps,
                        std::int64_t num_experts_per_tok) {
                std::vector<qwen3_5_layer_type> types;
                for (const std::string& name : layer_types) {
                    types.push_back(qwen3_5_layer(name));
                }
                return qwen3_5_moe_config{
                    .vocab_size = vocab_size,
                    .hidden_size = hidden_size,
                    .layer_types = std::move(types),
                    .n_heads = n_heads,
                    .n_kv_heads = n_kv_heads,
                    .head_dim = head_dim,
                    .rope_theta = rope_theta,
                    .partial_rotary_factor = partial_rotary_factor,
                    .linear_n_key_heads = linear_n_key_heads,
                    .linear_n_value_heads = linear_n_value_heads,
                    .linear_key_head_dim = linear_key_head_dim,
                    .linear_value_head_dim = linear_value_head_dim,
                    .linear_conv_kernel = linear_conv_kernel,
                    .rms_norm_eps = rms_norm_eps,
                    .num_experts_per_tok = num_experts_per_tok,
                };
            }),
            py::kw_only(),
            "vocab_size"_a,
            "hidden_size"_a,
            "layer_types"_a,
            "n_heads"_a,
            "n_kv_heads"_a,
            "head_dim"_a,
            "rope_theta"_a,
            "partial_rotary_factor"_a,
            "linear_n_key_heads"_a,
            "linear_n_value_heads"_a,
            "linear_key_head_dim"_a,
            "linear_value_head_dim"_a,
            "linear_conv_kernel"_a,
            "rms_norm_eps"_a,
            "num_experts_per_tok"_a
        )
        .def_readonly("num_experts_per_tok", &qwen3_5_moe_config::num_experts_per_tok)
        .def_property_readonly("layer_types", [](const qwen3_5_moe_config& c) {
            std::vector<std::string> names;
            for (qwen3_5_layer_type type : c.layer_types) {
                names.push_back(type == qwen3_5_layer_type::linear_attention ? "linear_attention" : "full_attention");
            }
            return names;
        });

    py::class_<qwen4_exp_config>(m, "Qwen4ExpConfig")
        .def(
            py::init([](std::int64_t vocab_size,
                        std::int64_t hidden_size,
                        const std::vector<std::string>& layer_types,
                        std::int64_t n_heads,
                        std::int64_t n_kv_heads,
                        std::int64_t head_dim,
                        double rope_theta,
                        double partial_rotary_factor,
                        std::int64_t linear_n_key_heads,
                        std::int64_t linear_n_value_heads,
                        std::int64_t linear_key_head_dim,
                        std::int64_t linear_value_head_dim,
                        std::int64_t linear_conv_kernel,
                        const std::string& linear_output_gate,
                        double rms_norm_eps,
                        std::int64_t num_experts_per_tok,
                        std::int64_t hc_count,
                        std::vector<std::int64_t> ple_layers,
                        std::int64_t ngram_size,
                        std::int64_t eos_token_id,
                        std::int64_t indexer_n_heads,
                        std::int64_t indexer_head_dim,
                        std::int64_t indexer_budget,
                        std::int64_t indexer_compress_ratio) {
                std::vector<qwen4_exp_layer_type> types;
                for (const std::string& name : layer_types) {
                    types.push_back(qwen4_exp_layer(name));
                }
                return qwen4_exp_config{
                    .vocab_size = vocab_size,
                    .hidden_size = hidden_size,
                    .layer_types = std::move(types),
                    .n_heads = n_heads,
                    .n_kv_heads = n_kv_heads,
                    .head_dim = head_dim,
                    .rope_theta = rope_theta,
                    .partial_rotary_factor = partial_rotary_factor,
                    .linear_n_key_heads = linear_n_key_heads,
                    .linear_n_value_heads = linear_n_value_heads,
                    .linear_key_head_dim = linear_key_head_dim,
                    .linear_value_head_dim = linear_value_head_dim,
                    .linear_conv_kernel = linear_conv_kernel,
                    .linear_output_gate = activation(linear_output_gate),
                    .rms_norm_eps = rms_norm_eps,
                    .num_experts_per_tok = num_experts_per_tok,
                    .hc_count = hc_count,
                    .ple_layers = std::move(ple_layers),
                    .ngram_size = ngram_size,
                    .eos_token_id = eos_token_id,
                    .indexer_n_heads = indexer_n_heads,
                    .indexer_head_dim = indexer_head_dim,
                    .indexer_budget = indexer_budget,
                    .indexer_compress_ratio = indexer_compress_ratio,
                };
            }),
            py::kw_only(),
            "vocab_size"_a,
            "hidden_size"_a,
            "layer_types"_a,
            "n_heads"_a,
            "n_kv_heads"_a,
            "head_dim"_a,
            "rope_theta"_a,
            "partial_rotary_factor"_a,
            "linear_n_key_heads"_a,
            "linear_n_value_heads"_a,
            "linear_key_head_dim"_a,
            "linear_value_head_dim"_a,
            "linear_conv_kernel"_a,
            "linear_output_gate"_a,
            "rms_norm_eps"_a,
            "num_experts_per_tok"_a,
            "hc_count"_a,
            "ple_layers"_a,
            "ngram_size"_a,
            "eos_token_id"_a,
            "indexer_n_heads"_a,
            "indexer_head_dim"_a,
            "indexer_budget"_a,
            "indexer_compress_ratio"_a
        )
        .def_readonly("ple_layers", &qwen4_exp_config::ple_layers)
        .def_property_readonly(
            "layer_types",
            [](const qwen4_exp_config& c) {
                std::vector<std::string> names;
                for (qwen4_exp_layer_type type : c.layer_types) {
                    names.push_back(
                        type == qwen4_exp_layer_type::linear_attention ? "linear_attention" : "indexed_attention"
                    );
                }
                return names;
            }
        )
        .def_readonly("eos_token_id", &qwen4_exp_config::eos_token_id);

    py::class_<request_handle>(m, "Request").def("take", &request_handle::take).def("cancel", &request_handle::cancel);

    py::class_<python_engine>(m, "Engine")
        .def(
            py::init(&make_engine),
            "store"_a,
            "config"_a,
            "load_model_shard"_a,
            py::kw_only(),
            "devices"_a,
            "model_name"_a,
            "model_hash"_a,
            "page_size"_a,
            "kv_dtype"_a,
            "kv_cache_gib"_a = 4.0,
            "store_gib"_a,
            "stop_ids"_a,
            "prefill_tokens_per_step"_a
        )
        .def(
            "submit",
            &python_engine::submit,
            "initial_state"_a,
            "tokens"_a,
            py::kw_only(),
            "sampling"_a = sampling_params{},
            "max_tokens"_a = py::none(),
            "eviction"_a = eviction_policy{},
            "request_id"_a = 0,
            "on_cycle_end"_a = py::none(),
            "on_end"_a = py::none()
        )
        .def(
            "page_digests",
            [](const python_engine& e, const std::vector<std::int64_t>& tokens) {
                return to_bytes_list(e.page_digests(tokens));
            },
            "tokens"_a
        )
        .def(
            "page_keys",
            [](const python_engine& e, const state& s) { return to_bytes_list(e.page_keys(s)); },
            "state"_a
        )
        .def("eviction_report_of", &python_engine::eviction_report_of, "state"_a, "eviction"_a)
        .def("stat", &python_engine::stat)
        .def("fileno", &python_engine::fileno)
        .def("clear_wake", &python_engine::clear_wake)
        .def("exception", &python_engine::exception);
}
