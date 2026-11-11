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
#include "request_buffer.hpp"
#include "wake_fd.hpp"

namespace pulsarlm {

// architecture is "qwen2", "qwen3" or "llama".
struct model_config {
    std::string architecture;
    dense_config dense_model;
};

using any_model_config = std::
    variant<model_config, gemma4_config, qwen3_5_config, qwen3_5_moe_config, qwen4_exp_config>;

// weights carry Hugging Face's names and are on the engine's device; they are comm's rank's slices when comm has more
// than one rank, which only Qwen3.5 and Qwen4-Exp split across.
std::unique_ptr<model> make_model(
    const any_model_config& config,
    const std::unordered_map<std::string, at::Tensor>& weights,
    const communicator& comm = {}
);

// One rank's model as Python holds it until an engine takes it.
struct model_handle {
    std::unique_ptr<model> served_model;
    at::Device device;
    std::size_t vocab_size;
};

// Python's handle on a request. Dropping it cancels the request.
class request_handle {
    std::shared_ptr<request_buffer> buffer;

  public:
    explicit request_handle(std::shared_ptr<request_buffer> buffer) noexcept;
    request_handle(const request_handle&) = delete;
    request_handle& operator=(const request_handle&) = delete;
    request_handle(request_handle&&) noexcept = default;
    request_handle& operator=(request_handle&&) = delete;
    ~request_handle();

    taken_results take();
    void cancel() noexcept;
};

// The engine as Python uses it. The loop thread takes the GIL to call the store, so every call that waits for the
// loop thread releases the GIL first.
class python_engine {
    std::shared_ptr<wake_fd> wake_signal;
    std::unique_ptr<engine> core_engine;

  public:
    python_engine(
        std::vector<std::unique_ptr<model>>&& shards,
        const engine_config& config,
        std::unique_ptr<page_store>&& store
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
        std::uint64_t request_id
    );
    std::vector<digest> page_digests(const std::vector<std::int64_t>& tokens) const;
    std::vector<digest> page_keys(const state& s) const;
    engine_stat stat() const;
    int fileno() const noexcept;
    void clear_wake() noexcept;
    // The message of the exception that panicked the loop; nullopt while the loop runs or when it was stopped.
    std::optional<std::string> exception() const;
};

}  // namespace pulsarlm
