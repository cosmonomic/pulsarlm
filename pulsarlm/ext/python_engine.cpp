#include "python_engine.hpp"

#include <ATen/core/Dict.h>

#include <exception>
#include <format>
#include <stdexcept>
#include <utility>

#include "pulsarlm/engine.hpp"

namespace pulsarlm {

namespace py = pybind11;

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
    const any_model_config& model_config,
    const model_shard_loader& load_model_shard,
    const engine_config& config,
    std::unique_ptr<python_store>&& store
)
: wake_signal(std::make_shared<wake_fd>()) {
    python_store& host_store = *store;
    core_engine = std::make_unique<engine>(model_config, load_model_shard, config, std::move(store));
    host_store.size_host_pages(core_engine->largest_page_bytes());
}

python_engine::~python_engine() {
    py::gil_scoped_release nogil;
    core_engine.reset();
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
        core_engine->submit(std::move(r), std::make_unique<buffered_callbacks>(buffer, wake_signal));
    }
    return request_handle(std::move(buffer));
}

std::vector<digest> python_engine::page_digests(const std::vector<std::int64_t>& tokens) const {
    return core_engine->page_digests(tokens);
}

std::vector<digest> python_engine::page_keys(const state& s) const {
    return core_engine->page_keys(s);
}

eviction_report python_engine::eviction_report_of(const state& s, const eviction_policy& policy) const {
    return core_engine->eviction_report_of(s, policy);
}

engine_stat python_engine::stat() const {
    return core_engine->stat();
}

int python_engine::fileno() const noexcept {
    return wake_signal->fileno();
}

void python_engine::clear_wake() noexcept {
    wake_signal->clear();
}

std::optional<std::string> python_engine::exception() const {
    std::exception_ptr thrown = core_engine->exception();
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
