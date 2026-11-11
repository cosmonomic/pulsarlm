#include <torch/csrc/utils/pybind.h>

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

py::tuple take_to_python(request_handle& handle) {
    taken_results taken = handle.take();
    py::object end = py::none();
    if (taken.end) {
        end = *taken.end ? py::cast(std::move(**taken.end)) : py::cast(taken.end->error());
    }
    return py::make_tuple(std::move(taken.tokens), py::cast(std::move(taken.cycle_ends)), std::move(end));
}

// A JSON value from a Python object: None, a bool, an int, a float, a str, or a list or a dict with str keys of them.
config_value to_config_value(const py::handle& object) {
    if (object.is_none()) {
        return {nullptr};
    }
    if (py::isinstance<py::bool_>(object)) {
        return {object.cast<bool>()};
    }
    if (py::isinstance<py::int_>(object)) {
        return {object.cast<std::int64_t>()};
    }
    if (py::isinstance<py::float_>(object)) {
        return {object.cast<double>()};
    }
    if (py::isinstance<py::str>(object)) {
        return {object.cast<std::string>()};
    }
    if (py::isinstance<py::list>(object) || py::isinstance<py::tuple>(object)) {
        config_list list;
        for (const py::handle& item : object) {
            list.push_back(to_config_value(item));
        }
        return {std::move(list)};
    }
    if (py::isinstance<py::dict>(object)) {
        config_dict dict;
        for (const auto& [key, item] : object.cast<py::dict>()) {
            if (!py::isinstance<py::str>(key)) {
                throw py::type_error("a config's keys are strings, got " + py::repr(key).cast<std::string>());
            }
            dict.emplace(key.cast<std::string>(), to_config_value(item));
        }
        return {std::move(dict)};
    }
    throw py::type_error("a config holds None, bool, int, float, str, list and dict, got " + py::repr(object).cast<std::string>());
}

// Shard r of the model runs on devices[r] with the weights load_model_shard(r, len(devices)) returns; kv_cache_gib
// sizes each device's KV cache.
std::unique_ptr<python_engine> make_engine(
    py::object store,
    const py::dict& config,
    py::object load_model_shard,
    std::vector<at::Device> devices,
    const std::string& model_name,
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
    config_dict model_config = std::get<config_dict>(to_config_value(config).value);
    engine_config engine_settings{
        .model_name = model_name,
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

    py::class_<engine_stat>(m, "EngineStat").def_readonly("kv_cache_bytes", &engine_stat::kv_cache_bytes);

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

    py::class_<request_handle>(m, "Request").def("take", &take_to_python).def("cancel", &request_handle::cancel);

    py::class_<python_engine>(m, "Engine")
        .def(
            py::init(&make_engine),
            "store"_a,
            "config"_a,
            "load_model_shard"_a,
            py::kw_only(),
            "devices"_a,
            "model_name"_a,
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
        .def("eviction_report_of", &python_engine::eviction_report_of, "state"_a, "eviction"_a)
        .def("stat", &python_engine::stat)
        .def("fileno", &python_engine::fileno)
        .def("clear_wake", &python_engine::clear_wake)
        .def("exception", &python_engine::exception);
}
