#pragma once

#include <pybind11/pybind11.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "pulsarlm/config.hpp"
#include "pulsarlm/engine.hpp"
#include "python_store.hpp"
#include "request_buffer.hpp"
#include "wake_fd.hpp"

namespace pulsarlm {

// Raises RequestFailed(error) in Python.
[[noreturn]] void raise_request_failed(request_error error);

// Python's handle on a request. Dropping it cancels the request.
class request_handle {
    std::shared_ptr<request_buffer> buffer;
    pybind11::object on_cycle_end;
    pybind11::object on_end;
    bool ended = false;
    std::optional<request_error> failure;

  public:
    request_handle(
        std::shared_ptr<request_buffer> buffer,
        pybind11::object&& on_cycle_end,
        pybind11::object&& on_end
    ) noexcept;
    request_handle(const request_handle&) = delete;
    request_handle& operator=(const request_handle&) = delete;
    request_handle(request_handle&&) noexcept = default;
    request_handle& operator=(request_handle&&) = delete;
    ~request_handle();

    // The tokens sampled since the last take, None once the request ended and every token was taken. Passes each cycle
    // end to on_cycle_end and the result to on_end first, when they are not None. Raises RequestFailed once every
    // token before an error was taken.
    pybind11::object take();
    void cancel() noexcept;
};

// The engine as Python uses it. The loop thread takes the GIL to call the store, so every call that waits for the
// loop thread releases the GIL first.
class python_engine {
    std::shared_ptr<wake_fd> wake_signal;
    std::unique_ptr<engine> core_engine;

  public:
    // Builds the engine with load_model_shard, then sizes store's pool to its pages.
    python_engine(
        const any_model_config& model_config,
        const model_shard_loader& load_model_shard,
        const engine_config& config,
        std::unique_ptr<python_store>&& store
    );
    python_engine(const python_engine&) = delete;
    python_engine& operator=(const python_engine&) = delete;
    ~python_engine();

    request_handle submit(
        const state& initial_state,
        std::vector<std::int64_t>&& tokens,
        const sampling_params& sampling,
        std::optional<std::size_t> max_tokens,
        const eviction_policy& eviction,
        std::uint64_t request_id,
        pybind11::object on_cycle_end,
        pybind11::object on_end
    );
    std::vector<digest> page_digests(const std::vector<std::int64_t>& tokens) const;
    std::vector<digest> page_keys(const state& s) const;
    eviction_report eviction_report_of(const state& s, const eviction_policy& policy) const;
    engine_stat stat() const;
    int fileno() const noexcept;
    void clear_wake() noexcept;
    // The message of the exception that panicked the loop; nullopt while the loop runs or when it was stopped.
    std::optional<std::string> exception() const;
};

}  // namespace pulsarlm
