#include "python_engine.hpp"

#include <ATen/core/Dict.h>

#include <exception>
#include <format>
#include <stdexcept>
#include <utility>

#include "pulsar/model/qwen2.hpp"
#include "pulsar/model/qwen3.hpp"

namespace pulsar {

namespace py = pybind11;

template class engine<python_store>;

std::unique_ptr<Model>
make_model(const model_config& config, const std::unordered_map<std::string, at::Tensor>& weights) {
    c10::Dict<std::string, at::Tensor> weight_dict;
    for (const auto& [name, tensor] : weights) {
        weight_dict.insert(name, tensor);
    }
    std::unique_ptr<Model> model;
    if (config.architecture == "qwen2") {
        model = std::make_unique<Qwen2Model>(
            config.n_layers,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            config.hidden,
            config.intermediate,
            config.vocab_size,
            config.rope_theta,
            config.rms_eps,
            std::move(weight_dict)
        );
    } else if (config.architecture == "qwen3") {
        model = std::make_unique<Qwen3Model>(
            config.n_layers,
            config.n_heads,
            config.n_kv_heads,
            config.head_dim,
            config.hidden,
            config.intermediate,
            config.vocab_size,
            config.rope_theta,
            config.rms_eps,
            std::move(weight_dict)
        );
    } else {
        throw std::invalid_argument(std::format("unknown architecture {:?}", config.architecture));
    }
    model->set_relevance_layers(config.relevance_layers);
    return model;
}

request_handle::request_handle(std::shared_ptr<request_buffer> buffer) noexcept
: buffer(std::move(buffer)) {}

request_handle::~request_handle() {
    if (buffer) {
        buffer->cancel();
    }
}

taken_results request_handle::take() {
    return buffer->take();
}

void request_handle::cancel() noexcept {
    buffer->cancel();
}

python_engine::python_engine(std::unique_ptr<Model> model, const engine_config& config, python_store&& store)
: wake(std::make_shared<wake_fd>()),
  served(std::make_unique<engine<python_store>>(std::move(model), config, std::move(store))) {}

python_engine::~python_engine() {
    py::gil_scoped_release nogil;
    served.reset();
}

request_handle python_engine::submit(
    const state& initial_state,
    std::vector<std::int64_t>&& tokens,
    const sampling_params& sampling,
    std::optional<std::size_t> max_tokens,
    const eviction_policy& eviction
) {
    auto buffer = std::make_shared<request_buffer>();
    request r{initial_state, std::move(tokens), sampling, max_tokens, eviction};
    {
        py::gil_scoped_release nogil;
        served->submit(std::move(r), std::make_unique<buffered_callbacks>(buffer, wake));
    }
    return request_handle(std::move(buffer));
}

std::vector<digest> python_engine::page_digests(const std::vector<std::int64_t>& tokens) const {
    return served->page_digests(tokens);
}

int python_engine::fileno() const noexcept {
    return wake->fileno();
}

void python_engine::clear_wake() noexcept {
    wake->clear();
}

std::optional<std::string> python_engine::exception() const {
    std::exception_ptr thrown = served->exception();
    if (!thrown) {
        return std::nullopt;
    }
    try {
        std::rethrow_exception(thrown);
    } catch (const std::exception& e) {
        return e.what();
    } catch (...) {
        return "an exception not derived from std::exception";
    }
}

}  // namespace pulsar
