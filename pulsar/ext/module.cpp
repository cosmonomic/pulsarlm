#include <torch/csrc/utils/pybind.h>

#include <pybind11/native_enum.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "digest_bytes.hpp"
#include "python_engine.hpp"

namespace py = pybind11;

namespace pulsar {

namespace {

py::tuple take_to_python(request_handle& handle) {
    taken_results taken = handle.take();
    py::object end = py::none();
    if (taken.end) {
        end = *taken.end ? py::cast(std::move(**taken.end)) : py::cast(taken.end->error());
    }
    return py::make_tuple(std::move(taken.tokens), py::cast(std::move(taken.cycle_ends)), std::move(end));
}

std::unique_ptr<python_engine> make_engine(
    py::object store,
    const model_config& model,
    const std::unordered_map<std::string, at::Tensor>& weights,
    const std::string& model_name,
    std::size_t page_size,
    at::ScalarType kv_dtype,
    std::size_t n_lines,
    at::Device device,
    std::vector<std::int64_t> stop_ids,
    std::size_t prefill_tokens_per_step
) {
    engine_config config{
        model_name,
        page_shape{
            static_cast<std::size_t>(model.n_layers),
            page_size,
            static_cast<std::size_t>(model.n_kv_heads),
            static_cast<std::size_t>(model.head_dim),
            kv_dtype,
        },
        n_lines,
        device,
        std::move(stop_ids),
        static_cast<std::size_t>(model.vocab_size),
        prefill_tokens_per_step,
    };
    python_store adapter(std::move(store));
    py::gil_scoped_release nogil;
    return std::make_unique<python_engine>(make_model(model, weights), config, std::move(adapter));
}

}  // namespace

}  // namespace pulsar

PYBIND11_MODULE(pulsar_core, m) {
    using namespace pulsar;
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
                        double density_bar,
                        double attention_mass_decay) {
                return eviction_policy{n_sink, n_working, cycle_tokens, max_pages, density_bar, attention_mass_decay};
            }),
            py::kw_only(),
            "n_sink"_a = eviction_policy{}.n_sink,
            "n_working"_a = eviction_policy{}.n_working,
            "cycle_tokens"_a = eviction_policy{}.cycle_tokens,
            "max_pages"_a = eviction_policy{}.max_pages,
            "density_bar"_a = eviction_policy{}.density_bar,
            "attention_mass_decay"_a = eviction_policy{}.attention_mass_decay
        )
        .def_readwrite("n_sink", &eviction_policy::n_sink)
        .def_readwrite("n_working", &eviction_policy::n_working)
        .def_readwrite("cycle_tokens", &eviction_policy::cycle_tokens)
        .def_readwrite("max_pages", &eviction_policy::max_pages)
        .def_readwrite("density_bar", &eviction_policy::density_bar)
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
            "n_lines"_a,
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
        .def("fileno", &python_engine::fileno)
        .def("clear_wake", &python_engine::clear_wake)
        .def("exception", &python_engine::exception);
}
