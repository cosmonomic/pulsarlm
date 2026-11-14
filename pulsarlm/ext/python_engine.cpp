#include "python_engine.hpp"

#include <ATen/core/Dict.h>
#include <pybind11/stl.h>

#include <exception>
#include <format>
#include <stdexcept>
#include <utility>

#include "digest_bytes.hpp"
#include "pulsarlm/engine.hpp"

namespace pulsarlm {

namespace py = pybind11;

request_handle::request_handle(
    std::shared_ptr<request_buffer> buffer,
    py::object&& on_cycle_end,
    py::object&& on_end
) noexcept
: buffer(std::move(buffer)),
  on_cycle_end(std::move(on_cycle_end)),
  on_end(std::move(on_end)) {}

request_handle::~request_handle() {
    if (buffer) {
        buffer->cancel();
    }
}

py::object request_handle::take() {
    if (failure) {
        raise_request_failed(*failure);
    }
    if (ended) {
        return py::none();
    }
    taken_results taken = buffer->take();
    for (cycle_end& ended_cycle : taken.cycle_ends) {
        if (!on_cycle_end.is_none()) {
            on_cycle_end(std::move(ended_cycle));
        }
    }
    if (taken.end) {
        if (*taken.end) {
            ended = true;
            if (!on_end.is_none()) {
                on_end(std::move(**taken.end));
            }
        } else {
            failure = taken.end->error();
        }
    }
    if (taken.tokens.empty()) {
        if (failure) {
            raise_request_failed(*failure);
        }
        if (ended) {
            return py::none();
        }
    }
    return py::cast(std::move(taken.tokens));
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
    std::vector<std::int64_t>&& tokens,
    const std::optional<py::bytes>& initial_digest,
    const sampling_params& sampling,
    std::optional<std::size_t> max_tokens,
    const eviction_policy& eviction,
    py::object on_cycle_end,
    py::object on_end
) {
    auto buffer = std::make_shared<request_buffer>();
    request r{
        .initial_digest = initial_digest ? std::optional(digest_from_bytes(*initial_digest)) : std::nullopt,
        .tokens = std::move(tokens),
        .sampling = sampling,
        .max_tokens = max_tokens,
        .eviction = eviction,
    };
    {
        py::gil_scoped_release nogil;
        core_engine->submit(std::move(r), std::make_unique<buffered_callbacks>(buffer, wake_signal));
    }
    return request_handle(std::move(buffer), std::move(on_cycle_end), std::move(on_end));
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
