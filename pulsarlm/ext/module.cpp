#include <torch/csrc/utils/pybind.h>

#include <pybind11/native_enum.h>
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

std::size_t vocab_size(const std::variant<model_config, gemma4_model_config>& model) {
    if (const auto* gemma = std::get_if<gemma4_model_config>(&model)) {
        return static_cast<std::size_t>(gemma->text.vocab_size);
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

std::unique_ptr<python_engine> make_engine(
    py::object store,
    const std::variant<model_config, gemma4_model_config>& model,
    const std::unordered_map<std::string, at::Tensor>& weights,
    const std::string& model_name,
    std::size_t page_size,
    at::ScalarType kv_dtype,
    double kv_cache_gib,
    double staging_gib,
    std::size_t max_position_embeddings,
    at::Device device,
    std::vector<std::int64_t> stop_ids,
    std::size_t prefill_tokens_per_step
) {
    if (!(kv_cache_gib > 0.0) || !(staging_gib > 0.0)) {
        throw std::invalid_argument("kv_cache_gib and staging_gib must be positive");
    }
    engine_config config{
        model_name,
        page_size,
        kv_dtype,
        static_cast<std::size_t>(kv_cache_gib * (1ull << 30)),
        static_cast<std::size_t>(staging_gib * (1ull << 30)),
        max_position_embeddings,
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

    py::class_<state>(m, "State")
        .def(
            py::init([](const std::vector<py::bytes>& page_hashes, std::vector<std::int64_t> tail) {
                return state{digests_from_bytes(page_hashes), std::move(tail)};
            }),
            "page_hashes"_a = std::vector<py::bytes>(),
            "tail"_a = std::vector<std::int64_t>()
        )
        .def_property(
            "page_hashes",
            [](const state& s) { return to_bytes_list(s.page_hashes); },
            [](state& s, const std::vector<py::bytes>& page_hashes) { s.page_hashes = digests_from_bytes(page_hashes); }
        )
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
                        double rope_theta,
                        double rms_eps,
                        std::vector<std::int64_t> relevance_layers) {
                return model_config{
                    std::move(architecture),
                    n_layers,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    hidden,
                    intermediate,
                    vocab_size,
                    rope_theta,
                    rms_eps,
                    std::move(relevance_layers),
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
            "rope_theta"_a,
            "rms_eps"_a,
            "relevance_layers"_a
        );

    py::class_<rope_parameters>(m, "RopeParameters")
        .def(
            py::init([](double theta, double partial_rotary_factor) {
                return rope_parameters{theta, partial_rotary_factor};
            }),
            py::kw_only(),
            "theta"_a,
            "partial_rotary_factor"_a = 1.0
        )
        .def_readonly("theta", &rope_parameters::theta)
        .def_readonly("partial_rotary_factor", &rope_parameters::partial_rotary_factor);

    py::class_<gemma4_model_config>(m, "Gemma4Config")
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
                        std::int64_t top_k_experts,
                        std::vector<std::int64_t> relevance_layers) {
                std::vector<gemma4_layer_type> types;
                for (const std::string& name : layer_types) {
                    types.push_back(layer_type(name));
                }
                return gemma4_model_config{
                    gemma4_config{
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
                    },
                    std::move(relevance_layers),
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
            "top_k_experts"_a,
            "relevance_layers"_a
        )
        .def_property_readonly(
            "layer_types",
            [](const gemma4_model_config& c) {
                std::vector<std::string> names;
                for (gemma4_layer_type type : c.text.layer_types) {
                    names.push_back(
                        type == gemma4_layer_type::sliding_attention ? "sliding_attention" : "full_attention"
                    );
                }
                return names;
            }
        )
        .def_property_readonly("global_n_kv_heads", [](const gemma4_model_config& c) { return c.text.global_n_kv_heads; })
        .def_property_readonly(
            "hidden_size_per_layer_input",
            [](const gemma4_model_config& c) { return c.text.hidden_size_per_layer_input; }
        )
        .def_property_readonly("sliding_rope", [](const gemma4_model_config& c) { return c.text.sliding_rope; })
        .def_property_readonly("full_rope", [](const gemma4_model_config& c) { return c.text.full_rope; })
        .def_readonly("relevance_layers", &gemma4_model_config::relevance_layers);

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
            "kv_cache_gib"_a,
            "staging_gib"_a,
            "max_position_embeddings"_a,
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
            "eviction"_a = eviction_policy{}
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
