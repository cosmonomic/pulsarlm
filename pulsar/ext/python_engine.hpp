#pragma once

#include <pybind11/pybind11.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "pulsar/engine.hpp"
#include "pulsar/eviction.hpp"
#include "pulsar/sampling.hpp"
#include "python_store.hpp"
#include "request_buffer.hpp"
#include "wake_fd.hpp"

namespace pulsar {

struct model_config {
    std::string architecture;
    std::int64_t n_layers;
    std::int64_t n_heads;
    std::int64_t n_kv_heads;
    std::int64_t head_dim;
    std::int64_t hidden;
    std::int64_t intermediate;
    std::int64_t vocab_size;
    double rope_theta;
    double rms_eps;
    std::vector<std::int64_t> relevance_layers;
};

// architecture is "qwen2" or "qwen3"; weights use HF naming and are on the engine's device.
std::unique_ptr<Model>
make_model(const model_config& config, const std::unordered_map<std::string, at::Tensor>& weights);

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
    std::shared_ptr<wake_fd> wake;
    std::unique_ptr<engine<python_store>> served;

  public:
    python_engine(std::unique_ptr<Model> model, const engine_config& config, python_store&& store);
    python_engine(const python_engine&) = delete;
    python_engine& operator=(const python_engine&) = delete;
    ~python_engine();

    request_handle submit(
        const state& initial_state,
        std::vector<std::int64_t>&& tokens,
        const sampling_params& sampling,
        std::optional<std::size_t> max_tokens,
        const eviction_policy& eviction
    );
    int fileno() const noexcept;
    void clear_wake() noexcept;
    // The message of the exception that panicked the loop; nullopt while the loop runs or when it was stopped.
    std::optional<std::string> exception() const;
};

}  // namespace pulsar
