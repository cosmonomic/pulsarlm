#include "pulsarlm/engine.hpp"

#include "pulsarlm/ops.hpp"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <latch>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
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

void check_attention(const engine_config& config, std::span<const kv_layer> layers) {
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        if (layers[layer].state) {
            continue;
        }
        std::optional<std::string> unsupported = attention_unsupported_reason(
            config.kv_dtype,
            static_cast<std::int64_t>(config.page_size),
            static_cast<std::int64_t>(layers[layer].head_dim),
            static_cast<std::int64_t>(layers[layer].n_q_heads),
            static_cast<std::int64_t>(layers[layer].n_kv_heads)
        );
        if (unsupported) {
            throw std::invalid_argument(std::format("KV layer {} cannot run attention: {}", layer, *unsupported));
        }
    }
}

void validate(const request& r, std::size_t page_size, std::size_t vocab_size, std::optional<std::size_t> max_context) {
    if (r.initial_state.tail.size() >= page_size) {
        throw std::invalid_argument(
            std::format(
                "the initial state's tail holds {} tokens; it must be shorter than a page ({})",
                r.initial_state.tail.size(),
                page_size
            )
        );
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
    if (!(r.eviction.eviction_target > 0.0 && r.eviction.eviction_target <= 1.0)) {
        throw std::invalid_argument(
            std::format("eviction.eviction_target {} is outside (0, 1]", r.eviction.eviction_target)
        );
    }
    if (r.eviction.eviction_target < 1.0 && !max_context) {
        throw std::invalid_argument("eviction.eviction_target below 1 needs the model's max context");
    }
}

kv_cache make_cache(const engine_config& config, std::span<const kv_layer> layers, page_store& store) {
    check_attention(config, layers);
    std::vector<page_shape> shapes = page_shapes(config, layers);
    return kv_cache(shapes, config.kv_cache_bytes, config.host_page_bytes, config.device, store);
}

}  // namespace

std::vector<page_shape> page_shapes(const engine_config& config, std::span<const kv_layer> layers) {
    std::vector<page_shape> shapes;
    for (const kv_layer& layer : layers) {
        if (layer.state) {
            shapes.push_back(
                page_shape{
                    .n_layers = 1,
                    .page_size = config.page_size,
                    .n_kv_heads = 0,
                    .head_dim = 0,
                    .dtype = layer.state->dtype,
                    .values_only = true,
                    .state_sizes = layer.state->sizes,
                }
            );
            continue;
        }
        shapes.push_back(
            page_shape{
                .n_layers = 1,
                .page_size = config.page_size,
                .n_kv_heads = layer.n_kv_heads,
                .head_dim = layer.head_dim,
                .dtype = config.kv_dtype,
                .values_only = layer.values_only,
                .index_head_dim = layer.index_head_dim,
            }
        );
    }
    return shapes;
}

engine::engine(std::unique_ptr<Model> model, const engine_config& config, std::unique_ptr<page_store>&& store)
: model(std::move(model)),
  device(config.device),
  stream(c10::cuda::getStreamFromPool(false, config.device.index()).stream()),
  page_size(config.page_size),
  vocab_size(config.vocab_size),
  max_context(config.max_context),
  root(make_root_digest(config.model_name, page_shapes(config, this->model->kv_layers()))),
  store(std::move(store)),
  sched(
      *this->model,
      make_cache(config, this->model->kv_layers(), *this->store),
      stream,
      root,
      std::vector<std::int64_t>(config.stop_ids),
      config.prefill_tokens_per_step,
      config.max_context
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

// The loop's frames hold host buffers the stream may still use, so they are destroyed after it finishes. Their
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
    validate(r, page_size, vocab_size, max_context);
    sched.validate_state(r.initial_state);
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

std::vector<digest> engine::page_keys(const state& s) const {
    std::vector<digest> keys;
    for (std::size_t layer = 0; layer < s.layers.size(); layer++) {
        for (const active_page& page : s.layers[layer].pages) {
            keys.push_back(store_key(page.hash, layer));
        }
    }
    return keys;
}

}  // namespace pulsarlm
