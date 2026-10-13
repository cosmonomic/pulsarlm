#pragma once

#include <c10/core/Device.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <latch>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include "pulsar/event_loop.hpp"
#include "pulsar/kv_cache.hpp"
#include "pulsar/model/interface.hpp"
#include "pulsar/request_callbacks.hpp"
#include "pulsar/scheduler.hpp"

namespace pulsar {

// model_name names the model and its weights version; with shape it makes the root digest. A step forwards every
// decoding sequence and at most prefill_tokens_per_step tokens of the others.
struct engine_config {
    std::string model_name;
    page_shape shape;
    std::size_t n_lines;
    at::Device device;
    std::int64_t eos_id;
    std::size_t vocab_size;
    std::size_t prefill_tokens_per_step;
};

// Throws std::invalid_argument when r breaks a precondition of scheduler::submit or holds a token outside
// [0, vocab_size).
void validate(const request& r, std::size_t page_size, std::size_t vocab_size);

// Serves requests on a loop thread it owns. Once the loop exits, after a panic or on destruction, the requests it was
// serving end with loop_exited and the engine is shut down: calling submit after a request ended with loop_exited is
// undefined.
template <page_store t_store> class engine {
    std::unique_ptr<Model> model;
    at::Device device;
    cudaStream_t stream;
    std::size_t page_size;
    std::size_t vocab_size;
    scheduler<t_store> sched;
    mutable std::mutex mutex;
    // Set while the loop runs.
    event_loop* loop = nullptr;
    std::exception_ptr loop_exception;
    std::thread loop_thread;

    void run_loop(std::latch& started);

  public:
    engine(std::unique_ptr<Model> model, const engine_config& config, t_store&& store);
    engine(const engine&) = delete;
    engine& operator=(const engine&) = delete;
    ~engine();

    // Serves r, passing its results to callbacks. Safe to call from any thread. Throws as validate does, without
    // calling callbacks.
    void submit(request&& r, std::unique_ptr<request_callbacks> callbacks);
    // The exception that panicked the loop; null while the loop runs or when it was stopped.
    std::exception_ptr exception() const;
};

template <page_store t_store>
engine<t_store>::engine(std::unique_ptr<Model> model, const engine_config& config, t_store&& store)
: model(std::move(model)),
  device(config.device),
  stream(c10::cuda::getStreamFromPool(false, config.device.index()).stream()),
  page_size(config.shape.page_size),
  vocab_size(config.vocab_size),
  sched(
      *this->model,
      kv_cache<t_store>(config.shape, config.n_lines, config.device, std::move(store)),
      stream,
      make_root_digest(config.model_name, config.shape),
      config.eos_id,
      config.prefill_tokens_per_step
  ) {
    std::latch started(1);
    loop_thread = std::thread([this, &started] { run_loop(started); });
    started.wait();
}

template <page_store t_store> engine<t_store>::~engine() {
    {
        std::lock_guard lock(mutex);
        if (loop != nullptr) {
            loop->stop();
        }
    }
    loop_thread.join();
}

// The loop's frames hold staging buffers the stream may still use, so they are destroyed after it finishes. Their
// writers end their callbacks with loop_exited after loop_exception is set.
template <page_store t_store> void engine<t_store>::run_loop(std::latch& started) {
    c10::cuda::CUDAGuard device_guard(device);
    {
        event_loop running;
        running.spawn(sched.run());
        {
            std::lock_guard lock(mutex);
            loop = &running;
        }
        started.count_down();
        std::exception_ptr thrown = running.run();
        cudaStreamSynchronize(stream);
        {
            std::lock_guard lock(mutex);
            loop_exception = std::move(thrown);
            loop = nullptr;
        }
    }
}

// A submit racing the loop's exit finds loop null; its writer then ends the callbacks with loop_exited.
template <page_store t_store> void engine<t_store>::submit(request&& r, std::unique_ptr<request_callbacks> callbacks) {
    validate(r, page_size, vocab_size);
    result_writer writer(std::move(callbacks));
    std::lock_guard lock(mutex);
    if (loop != nullptr) {
        loop->spawn(sched.submit(std::move(r), std::move(writer)));
    }
}

template <page_store t_store> std::exception_ptr engine<t_store>::exception() const {
    std::lock_guard lock(mutex);
    return loop_exception;
}

}  // namespace pulsar
