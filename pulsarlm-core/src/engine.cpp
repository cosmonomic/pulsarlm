#include "pulsarlm/engine.hpp"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <latch>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace pulsarlm {

namespace {

void check_tokens(std::span<const std::int64_t> tokens, std::size_t vocab_size) {
    for (std::int64_t token : tokens) {
        if (token < 0 || static_cast<std::uint64_t>(token) >= vocab_size) {
            throw std::invalid_argument(std::format("token id {} is outside [0, {})", token, vocab_size));
        }
    }
}

}  // namespace

std::size_t line_count(const engine_config& config) {
    std::size_t bytes = line_bytes(config.shape);
    if (config.kv_cache_bytes < bytes) {
        throw std::invalid_argument(
            std::format("a KV cache of {} bytes holds no line; a line takes {} bytes", config.kv_cache_bytes, bytes)
        );
    }
    return config.kv_cache_bytes / bytes;
}

std::size_t staging_page_count(const engine_config& config) {
    std::size_t bytes = line_bytes(config.shape);
    if (config.staging_bytes < bytes) {
        throw std::invalid_argument(
            std::format("a staging buffer of {} bytes holds no page; a page takes {} bytes", config.staging_bytes, bytes)
        );
    }
    return config.staging_bytes / bytes;
}

void validate(const request& r, std::size_t page_size, std::size_t vocab_size) {
    if (r.initial_state.tail.size() >= page_size) {
        throw std::invalid_argument(std::format(
            "the initial state's tail holds {} tokens; it must be shorter than a page ({})",
            r.initial_state.tail.size(),
            page_size
        ));
    }
    if (r.initial_state.tail.empty() && r.tokens.empty()) {
        throw std::invalid_argument("the request has no token to forward");
    }
    check_tokens(r.initial_state.tail, vocab_size);
    check_tokens(r.tokens, vocab_size);
    if (r.eviction.n_working < 1) {
        throw std::invalid_argument("eviction.n_working must be at least 1");
    }
    if (r.eviction.cycle_tokens < 1) {
        throw std::invalid_argument("eviction.cycle_tokens must be at least 1");
    }
    if (!(r.eviction.attention_mass_decay > 0.0 && r.eviction.attention_mass_decay <= 1.0)) {
        throw std::invalid_argument(
            std::format("eviction.attention_mass_decay {} is outside (0, 1]", r.eviction.attention_mass_decay)
        );
    }
}

engine::engine(std::unique_ptr<Model> model, const engine_config& config, std::unique_ptr<page_store>&& store)
: model(std::move(model)),
  device(config.device),
  stream(c10::cuda::getStreamFromPool(false, config.device.index()).stream()),
  page_size(config.shape.page_size),
  vocab_size(config.vocab_size),
  root(make_root_digest(config.model_name, config.shape)),
  sched(
      *this->model,
      kv_cache(config.shape, line_count(config), staging_page_count(config), config.device, std::move(store)),
      stream,
      root,
      std::vector<std::int64_t>(config.stop_ids),
      config.prefill_tokens_per_step
  ) {
    std::latch started(1);
    loop_thread = std::thread([this, &started] { run_loop(started); });
    started.wait();
}

engine::~engine() {
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
void engine::run_loop(std::latch& started) {
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
void engine::submit(request&& r, std::unique_ptr<request_callbacks> callbacks) {
    validate(r, page_size, vocab_size);
    result_writer writer(std::move(callbacks));
    std::lock_guard lock(mutex);
    if (loop != nullptr) {
        loop->spawn(sched.submit(std::move(r), std::move(writer)));
    }
}

std::exception_ptr engine::exception() const {
    std::lock_guard lock(mutex);
    return loop_exception;
}

std::vector<digest> engine::page_digests(std::span<const std::int64_t> tokens) const {
    return make_page_digests(root, tokens, page_size);
}

}  // namespace pulsarlm
