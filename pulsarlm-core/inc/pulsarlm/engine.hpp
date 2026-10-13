#pragma once

#include <c10/core/Device.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <latch>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "pulsarlm/event_loop.hpp"
#include "pulsarlm/kv_cache.hpp"
#include "pulsarlm/model/interface.hpp"
#include "pulsarlm/request_callbacks.hpp"
#include "pulsarlm/scheduler.hpp"

namespace pulsarlm {

// model_name names the model and its weights version; with shape it makes the root digest. The KV cache takes as many
// lines as fit in kv_cache_bytes, and its pinned staging buffer as many pages as fit in staging_bytes. A step forwards
// every decoding sequence and at most prefill_tokens_per_step tokens of the others.
struct engine_config {
    std::string model_name;
    page_shape shape;
    std::size_t kv_cache_bytes;
    std::size_t staging_bytes;
    at::Device device;
    std::vector<std::int64_t> stop_ids;
    std::size_t vocab_size;
    std::size_t prefill_tokens_per_step;
};

// Throws std::invalid_argument when r breaks a precondition of scheduler::submit or holds a token outside
// [0, vocab_size).
void validate(const request& r, std::size_t page_size, std::size_t vocab_size);

// The lines that fit in config's KV cache size. Throws std::invalid_argument when not one does.
std::size_t line_count(const engine_config& config);
// The pages that fit in config's staging size. Throws std::invalid_argument when not one does.
std::size_t staging_page_count(const engine_config& config);

// Serves requests on a loop thread it owns. Once the loop exits, after a panic or on destruction, the requests it was
// serving end with loop_exited and the engine is shut down: calling submit after a request ended with loop_exited is
// undefined.
class engine {
    std::unique_ptr<Model> model;
    at::Device device;
    cudaStream_t stream;
    std::size_t page_size;
    std::size_t vocab_size;
    digest root;
    scheduler sched;
    mutable std::mutex mutex;
    // Set while the loop runs.
    event_loop* loop = nullptr;
    std::exception_ptr loop_exception;
    std::thread loop_thread;

    void run_loop(std::latch& started);

  public:
    engine(std::unique_ptr<Model> model, const engine_config& config, std::unique_ptr<page_store>&& store);
    engine(const engine&) = delete;
    engine& operator=(const engine&) = delete;
    ~engine();

    // Serves r, passing its results to callbacks. Safe to call from any thread. Throws as validate does, without
    // calling callbacks.
    void submit(request&& r, std::unique_ptr<request_callbacks> callbacks);
    // The exception that panicked the loop; null while the loop runs or when it was stopped.
    std::exception_ptr exception() const;
    // The digests of the full pages of a token history from its start, the page hashes a state reaching it would end
    // with.
    std::vector<digest> page_digests(std::span<const std::int64_t> tokens) const;
};

}  // namespace pulsarlm
