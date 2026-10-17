#include "python_engine.hpp"

#include <ATen/core/Dict.h>

#include <exception>
#include <format>
#include <stdexcept>
#include <utility>

#include "pulsarlm/model/llama.hpp"
#include "pulsarlm/model/qwen2.hpp"
#include "pulsarlm/model/qwen3.hpp"

namespace pulsarlm {

namespace py = pybind11;

std::unique_ptr<Model>
make_model(const any_model_config& any_config, const std::unordered_map<std::string, at::Tensor>& weights) {
    c10::Dict<std::string, at::Tensor> weight_dict;
    for (const auto& [name, tensor] : weights) {
        weight_dict.insert(name, tensor);
    }
    if (const auto* gemma = std::get_if<gemma4_config>(&any_config)) {
        return std::make_unique<Gemma4Model>(*gemma, weight_dict);
    }
    if (const auto* qwen = std::get_if<qwen3_5_config>(&any_config)) {
        return std::make_unique<Qwen3_5Model>(*qwen, weight_dict);
    }
    if (const auto* qwen = std::get_if<qwen3_5_moe_config>(&any_config)) {
        return std::make_unique<Qwen3_5MoeModel>(*qwen, weight_dict);
    }
    if (const auto* qwen = std::get_if<qwen4_exp_config>(&any_config)) {
        return std::make_unique<Qwen4ExpModel>(*qwen, weight_dict);
    }
    const model_config& config = std::get<model_config>(any_config);
    if (config.architecture == "qwen2") {
        return std::make_unique<Qwen2Model>(config.dense, weight_dict);
    }
    if (config.architecture == "qwen3") {
        return std::make_unique<Qwen3Model>(config.dense, weight_dict);
    }
    if (config.architecture == "llama") {
        return std::make_unique<LlamaModel>(config.dense, weight_dict);
    }
    throw std::invalid_argument(std::format("unknown architecture {:?}", config.architecture));
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

python_engine::python_engine(
    std::unique_ptr<Model> model,
    const engine_config& config,
    std::unique_ptr<page_store>&& store
)
: wake(std::make_shared<wake_fd>()),
  served(std::make_unique<engine>(std::move(model), config, std::move(store))) {}

python_engine::~python_engine() {
    py::gil_scoped_release nogil;
    served.reset();
}

request_handle python_engine::submit(
    const state& initial_state,
    std::vector<std::int64_t>&& tokens,
    const sampling_params& sampling,
    std::optional<std::size_t> max_tokens,
    const eviction_policy& eviction,
    std::uint64_t request_id
) {
    auto buffer = std::make_shared<request_buffer>();
    request r{initial_state, std::move(tokens), sampling, max_tokens, eviction, request_id};
    {
        py::gil_scoped_release nogil;
        served->submit(std::move(r), std::make_unique<buffered_callbacks>(buffer, wake));
    }
    return request_handle(std::move(buffer));
}

std::vector<digest> python_engine::page_digests(const std::vector<std::int64_t>& tokens) const {
    return served->page_digests(tokens);
}

std::vector<digest> python_engine::page_keys(const state& s) const {
    return served->page_keys(s);
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

}  // namespace pulsarlm
