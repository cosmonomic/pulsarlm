#include <torch/csrc/utils/pybind.h>

#include <pybind11/native_enum.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

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

py::tuple take_to_python(request_handle& handle) {
    taken_results taken = handle.take();
    py::object end = py::none();
    if (taken.end) {
        end = *taken.end ? py::cast(std::move(**taken.end)) : py::cast(taken.end->error());
    }
    return py::make_tuple(std::move(taken.tokens), py::cast(std::move(taken.cycle_ends)), std::move(end));
}

std::size_t vocab_size(const any_model_config& model) {
    if (const auto* gemma = std::get_if<gemma4_config>(&model)) {
        return static_cast<std::size_t>(gemma->vocab_size);
    }
    if (const auto* qwen = std::get_if<qwen3_5_config>(&model)) {
        return static_cast<std::size_t>(qwen->vocab_size);
    }
    return static_cast<std::size_t>(std::get<model_config>(model).vocab_size);
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

std::unique_ptr<python_engine> make_engine(
    py::object store,
    const any_model_config& model,
    const std::unordered_map<std::string, at::Tensor>& weights,
    const std::string& model_name,
    std::size_t page_size,
    at::ScalarType kv_dtype,
    double kv_cache_gib,
    double store_gib,
    at::Device device,
    std::vector<std::int64_t> stop_ids,
    std::size_t prefill_tokens_per_step
) {
    if (!(kv_cache_gib > 0.0) || !(store_gib >= 0.0)) {
        throw std::invalid_argument("kv_cache_gib must be positive and store_gib not negative");
    }
    engine_config config{
        model_name,
        page_size,
        kv_dtype,
        static_cast<std::size_t>(kv_cache_gib * (1ull << 30)),
        static_cast<std::size_t>(store_gib * (1ull << 30)),
        device,
        std::move(stop_ids),
        vocab_size(model),
        prefill_tokens_per_step,
    };
    auto adapter = std::make_unique<python_store>(std::move(store));
    py::gil_scoped_release nogil;
    return std::make_unique<python_engine>(make_model(model, weights), config, std::move(adapter));
}

}  // namespace

}  // namespace pulsarlm

PYBIND11_MODULE(pulsarlm_core, m) {
    using namespace pulsarlm;
    using namespace py::literals;

    py::class_<active_page>(m, "ActivePage")
        .def(
            py::init([](const py::bytes& hash, float density) {
                return active_page{.hash = digest_from_bytes(hash), .density = density};
            }),
            "hash"_a,
            "density"_a = 0.0f
        )
        .def_property(
            "hash",
            [](const active_page& p) { return to_bytes(p.hash); },
            [](active_page& p, const py::bytes& hash) { p.hash = digest_from_bytes(hash); }
        )
        .def_readwrite("density", &active_page::density)
        .def(py::self == py::self);

    py::class_<layer_sequence>(m, "LayerSequence")
        .def(
            py::init([](std::vector<active_page> pages) { return layer_sequence{.pages = std::move(pages)}; }),
            "pages"_a = std::vector<active_page>()
        )
        .def_readwrite("pages", &layer_sequence::pages)
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
        .def_readonly("reached", &cycle_end::reached)
        .def_readonly("token_count", &cycle_end::token_count);

    py::native_enum<request_error>(m, "RequestError", "enum.Enum")
        .value("MISS", request_error::miss)
        .value("OOM", request_error::oom)
        .value("CANCELLED", request_error::cancelled)
        .value("LOOP_EXITED", request_error::loop_exited)
        .finalize();

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
                        std::optional<std::size_t> max_pages,
                        double evict_tau,
                        double attention_mass_decay) {
                return eviction_policy{n_sink, n_working, cycle_tokens, max_pages, evict_tau, attention_mass_decay};
            }),
            py::kw_only(),
            "n_sink"_a = eviction_policy{}.n_sink,
            "n_working"_a = eviction_policy{}.n_working,
            "cycle_tokens"_a = eviction_policy{}.cycle_tokens,
            "max_pages"_a = eviction_policy{}.max_pages,
            "evict_tau"_a = eviction_policy{}.evict_tau,
            "attention_mass_decay"_a = eviction_policy{}.attention_mass_decay
        )
        .def_readwrite("n_sink", &eviction_policy::n_sink)
        .def_readwrite("n_working", &eviction_policy::n_working)
        .def_readwrite("cycle_tokens", &eviction_policy::cycle_tokens)
        .def_readwrite("max_pages", &eviction_policy::max_pages)
        .def_readwrite("evict_tau", &eviction_policy::evict_tau)
        .def_readwrite("attention_mass_decay", &eviction_policy::attention_mass_decay);

    py::class_<model_config>(m, "ModelConfig")
        .def(
            py::init([](std::string architecture,
                        std::int64_t n_layers,
                        std::int64_t n_heads,
                        std::int64_t n_kv_heads,
                        std::int64_t head_dim,
                        std::int64_t hidden,
                        std::int64_t intermediate,
                        std::int64_t vocab_size,
                        const rope_parameters& rope,
                        double rms_eps) {
                return model_config{
                    std::move(architecture),
                    n_layers,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    hidden,
                    intermediate,
                    vocab_size,
                    rope,
                    rms_eps,
                };
            }),
            py::kw_only(),
            "architecture"_a,
            "n_layers"_a,
            "n_heads"_a,
            "n_kv_heads"_a,
            "head_dim"_a,
            "hidden"_a,
            "intermediate"_a,
            "vocab_size"_a,
            "rope"_a,
            "rms_eps"_a
        )
        .def_readonly("rope", &model_config::rope);

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
                        std::int64_t intermediate_size,
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
                        double rms_norm_eps) {
                std::vector<qwen3_5_layer_type> types;
                for (const std::string& name : layer_types) {
                    types.push_back(qwen3_5_layer(name));
                }
                return qwen3_5_config{
                    .vocab_size = vocab_size,
                    .hidden_size = hidden_size,
                    .intermediate_size = intermediate_size,
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
                };
            }),
            py::kw_only(),
            "vocab_size"_a,
            "hidden_size"_a,
            "intermediate_size"_a,
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
            "rms_norm_eps"_a
        )
        .def_property_readonly("layer_types", [](const qwen3_5_config& c) {
            std::vector<std::string> names;
            for (qwen3_5_layer_type type : c.layer_types) {
                names.push_back(type == qwen3_5_layer_type::linear_attention ? "linear_attention" : "full_attention");
            }
            return names;
        });

    py::class_<request_handle>(m, "Request").def("take", &take_to_python).def("cancel", &request_handle::cancel);

    py::class_<python_engine>(m, "Engine")
        .def(
            py::init(&make_engine),
            "store"_a,
            "model"_a,
            "weights"_a,
            py::kw_only(),
            "model_name"_a,
            "page_size"_a,
            "kv_dtype"_a,
            "kv_cache_gib"_a = 4.0,
            "store_gib"_a,
            "device"_a,
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
            "request_id"_a = 0
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
        .def("fileno", &python_engine::fileno)
        .def("clear_wake", &python_engine::clear_wake)
        .def("exception", &python_engine::exception);
}
