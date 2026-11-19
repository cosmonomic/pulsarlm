#include "pulsarlm/engine/engine.hpp"
#include "pulsarlm/engine/eviction.hpp"
#include "pulsarlm/engine/model/make_model.hpp"

#include "pulsarlm/kernels/paged_attention.hpp"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
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
#include <variant>
#include <vector>

namespace pulsarlm {

namespace {

constexpr std::size_t sampled_row_limit = 4096;

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

void validate(const request& r, std::size_t vocab_size) {
    if (r.tokens.empty()) {
        throw std::invalid_argument("the request has no token to forward");
    }
    check_tokens(r.tokens, vocab_size);
    if (r.eviction.n_working < 1) {
        throw std::invalid_argument("eviction.n_working must be at least 1");
    }
    if (r.eviction.cycle_tokens < 1) {
        throw std::invalid_argument("eviction.cycle_tokens must be at least 1");
    }
    if (r.eviction.mass_tokens && (*r.eviction.mass_tokens < 1 || *r.eviction.mass_tokens > r.eviction.cycle_tokens)) {
        throw std::invalid_argument(
            std::format(
                "eviction.mass_tokens {} is outside [1, cycle_tokens {}]",
                *r.eviction.mass_tokens,
                r.eviction.cycle_tokens
            )
        );
    }
    if (!(r.eviction.attention_mass_decay > 0.0 && r.eviction.attention_mass_decay <= 1.0)) {
        throw std::invalid_argument(
            std::format("eviction.attention_mass_decay {} is outside (0, 1]", r.eviction.attention_mass_decay)
        );
    }
    if (!(r.eviction.eviction_threshold >= 0.0 && r.eviction.eviction_threshold <= 1.0)) {
        throw std::invalid_argument(
            std::format("eviction.eviction_threshold {} is outside [0, 1]", r.eviction.eviction_threshold)
        );
    }
}

std::vector<kv_layer> full_layers(const model& shard) {
    std::vector<kv_layer> layers;
    for (const kv_layer_shard& layer : shard.kv_layer_shards()) {
        layers.push_back(layer.full);
    }
    return layers;
}

std::vector<at::Device> checked_devices(const engine_config& config, std::size_t shard_count) {
    if (shard_count == 0 || config.devices.size() != shard_count) {
        throw std::invalid_argument(
            std::format(
                "the engine runs {} model shards on {} devices; it takes one per shard",
                shard_count,
                config.devices.size()
            )
        );
    }
    return config.devices;
}

std::size_t checked_draft_tokens(const engine_config& config, const std::vector<std::unique_ptr<model>>& shards) {
    if (config.draft_tokens > 0 && shards.front()->drafter() == nullptr) {
        throw std::invalid_argument("the engine drafts tokens, and the model has no drafter");
    }
    if (config.draft_tokens + 1 > sampled_row_limit) {
        throw std::invalid_argument(
            std::format(
                "the engine drafts {} tokens a step; a step samples at most {} rows, draft_tokens + 1 of one sequence",
                config.draft_tokens,
                sampled_row_limit
            )
        );
    }
    return config.draft_tokens;
}

std::size_t checked_prefill_tokens_per_step(const engine_config& config) {
    if (config.prefill_tokens_per_step < 1) {
        throw std::invalid_argument("prefill_tokens_per_step must be at least 1");
    }
    return config.prefill_tokens_per_step;
}

const any_model_config& checked_model_config(const any_model_config& model_config, const engine_config& config) {
    if (const qwen4_exp_config* qwen4_exp = std::get_if<qwen4_exp_config>(&model_config)) {
        const auto ratio = static_cast<std::size_t>(qwen4_exp->indexer_compress_ratio);
        if (qwen4_exp->indexer_compress_ratio > 0 && config.page_size % ratio != 0) {
            throw std::invalid_argument(
                std::format(
                    "a page of {} tokens does not hold whole index blocks of {} tokens",
                    config.page_size,
                    ratio
                )
            );
        }
    }
    return model_config;
}

// From the high-priority pool with high_priority, apart from the compute streams of the low-priority one.
std::vector<cudaStream_t> make_streams(std::span<const at::Device> devices, bool high_priority) {
    std::vector<cudaStream_t> streams;
    for (const at::Device& device : devices) {
        streams.push_back(c10::cuda::getStreamFromPool(high_priority, device.index()).stream());
    }
    return streams;
}

std::vector<std::unique_ptr<event_loop>> make_worker_loops(std::size_t shard_count) {
    std::vector<std::unique_ptr<event_loop>> loops;
    for (std::size_t shard = 1; shard < shard_count; shard++) {
        loops.push_back(std::make_unique<event_loop>());
    }
    return loops;
}

std::vector<model*> model_pointers(std::span<const std::unique_ptr<model>> shards) {
    std::vector<model*> pointers;
    for (const std::unique_ptr<model>& shard : shards) {
        pointers.push_back(shard.get());
    }
    return pointers;
}

std::vector<event_loop*> loop_pointers(std::span<const std::unique_ptr<event_loop>> loops) {
    std::vector<event_loop*> pointers;
    for (const std::unique_ptr<event_loop>& loop : loops) {
        pointers.push_back(loop.get());
    }
    return pointers;
}

kv_cache make_cache(const engine_config& config, std::span<const std::unique_ptr<model>> shards, page_store& store) {
    std::vector<page_shape> full_shapes = page_shapes(config, full_layers(*shards.front()));
    std::vector<kv_cache_shard> cache_shards;
    for (std::size_t shard = 0; shard < shards.size(); shard++) {
        std::vector<kv_layer> layers = shards[shard]->kv_layers();
        c10::cuda::CUDAGuard device_guard(config.devices[shard]);
        check_attention(config, layers);
        kv_cache_shard cache_shard{.device = config.devices[shard], .shapes = page_shapes(config, layers)};
        for (const kv_layer_shard& layer : shards[shard]->kv_layer_shards()) {
            cache_shard.rows.push_back(layer.rows);
        }
        cache_shards.push_back(std::move(cache_shard));
    }
    return kv_cache(full_shapes, cache_shards, config.kv_cache_pool_bytes, store);
}

std::vector<communicator> make_communicators(std::span<const at::Device> devices) {
    if (devices.size() == 1) {
        return {communicator{}};
    }
    if (std::ranges::all_of(devices, [&](const at::Device& device) { return device == devices.front(); })) {
        return communicator::make_on_one_device(devices.front(), devices.size());
    }
    for (std::size_t i = 0; i < devices.size(); i++) {
        for (std::size_t j = i + 1; j < devices.size(); j++) {
            if (devices[i] == devices[j]) {
                throw std::invalid_argument("an engine's devices are distinct, or all one device");
            }
        }
    }
    return communicator::make(devices);
}

// The pages of a view of max_context_window tokens, its last page partly filled, when decode graphs are on.
std::optional<std::size_t> graph_pages(const engine_config& config) {
    if (!config.decode_graphs || !config.max_context_window) {
        return std::nullopt;
    }
    return (*config.max_context_window + config.page_size - 1) / config.page_size + 1;
}

std::size_t largest_line(const engine_config& config, const model& shard) {
    std::size_t largest = 0;
    for (const page_shape& shape : page_shapes(config, full_layers(shard))) {
        largest = std::max(largest, line_bytes(shape));
    }
    return largest;
}

std::vector<std::unique_ptr<model>> one_shard(std::unique_ptr<model> served_model) {
    std::vector<std::unique_ptr<model>> shards;
    shards.push_back(std::move(served_model));
    return shards;
}

}  // namespace

std::vector<page_shape> page_shapes(const engine_config& config, std::span<const kv_layer> layers) {
    std::vector<page_shape> shapes;
    for (const kv_layer& layer : layers) {
        if (layer.state) {
            std::vector<std::int64_t> state_sizes = layer.state->sizes;
            if (layer.state->input_window) {
                state_sizes.back() += static_cast<std::int64_t>(config.draft_tokens);
            }
            shapes.push_back(
                page_shape{
                    .n_layers = 1,
                    .page_size = config.page_size,
                    .n_kv_heads = 0,
                    .head_dim = 0,
                    .dtype = layer.state->dtype,
                    .values_only = true,
                    .state_sizes = state_sizes,
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
                .last_hidden_dim = layer.last_hidden_dim,
            }
        );
    }
    return shapes;
}

struct engine::loaded_model {
    std::vector<std::unique_ptr<model>> shards;
    std::size_t vocab_size;
};

engine::loaded_model engine::load_model(
    const any_model_config& model_config,
    const model_shard_loader& load_model_shard,
    std::span<const at::Device> devices
) {
    if (devices.empty()) {
        throw std::invalid_argument("an engine takes at least one device");
    }
    if (devices.size() > 1 && !std::holds_alternative<qwen3_5_config>(model_config) &&
        !std::holds_alternative<qwen4_exp_config>(model_config)) {
        throw std::invalid_argument("only Qwen3.5 and Qwen4-Exp models split across GPUs");
    }
    std::vector<communicator> communicators = make_communicators(devices);
    loaded_model loaded{.shards = {}, .vocab_size = pulsarlm::vocab_size(model_config)};
    for (std::size_t rank = 0; rank < devices.size(); rank++) {
        // Loading launches kernels, such as the fp8 Marlin repack, on the current device's stream.
        c10::cuda::CUDAGuard device_guard(devices[rank]);
        loaded.shards.push_back(make_model(model_config, load_model_shard(rank, devices.size()), communicators[rank]));
    }
    return loaded;
}

engine::engine(
    const any_model_config& model_config,
    const model_shard_loader& load_model_shard,
    const engine_config& config,
    std::unique_ptr<page_store>&& store
)
: engine(
      load_model(checked_model_config(model_config, config), load_model_shard, config.devices),
      config,
      std::move(store)
  ) {}

engine::engine(loaded_model&& loaded, const engine_config& config, std::unique_ptr<page_store>&& store)
: engine(std::move(loaded.shards), loaded.vocab_size, config, std::move(store)) {}

engine::engine(
    std::vector<std::unique_ptr<model>>&& shards,
    std::size_t vocab_size,
    const engine_config& config,
    std::unique_ptr<page_store>&& store
)
: shards(std::move(shards)),
  devices(checked_devices(config, this->shards.size())),
  streams(make_streams(devices, false)),
  copy_streams(make_streams(devices, true)),
  worker_loops(make_worker_loops(devices.size())),
  page_size(config.page_size),
  vocab_size(vocab_size),
  max_page_bytes(largest_line(config, *this->shards.front())),
  store(std::move(store)),
  exec(
      model_pointers(this->shards),
      make_cache(config, this->shards, *this->store),
      *this->store,
      std::vector<cudaStream_t>(streams),
      std::vector<cudaStream_t>(copy_streams),
      loop_pointers(worker_loops),
      make_root_digest(config.model_name, config.model_hash, page_shapes(config, full_layers(*this->shards.front()))),
      std::vector<std::int64_t>(config.stop_ids),
      graph_pages(config),
      checked_draft_tokens(config, this->shards),
      sampled_row_limit
  ),
  sched(exec, checked_prefill_tokens_per_step(config)) {
    for (std::size_t shard = 1; shard < devices.size(); shard++) {
        worker_threads.emplace_back([this, shard] { run_worker_loop(shard); });
    }
    std::latch started(1);
    loop_thread = std::thread([this, &started] { run_loop(started); });
    started.wait();
}

engine::engine(
    std::unique_ptr<model> served_model,
    std::size_t vocab_size,
    const engine_config& config,
    std::unique_ptr<page_store>&& store
)
: engine(one_shard(std::move(served_model)), vocab_size, config, std::move(store)) {}

engine::~engine() {
    {
        std::lock_guard lock(mutex);
        if (loop != nullptr) {
            loop->stop();
        }
    }
    loop_thread.join();
}

// The worker loops' tasks hold references into the loop's frames, so the workers stop before those frames are
// destroyed; the frames hold host buffers the streams may still use, so they are destroyed after the streams finish.
// The requests end with loop_exited after loop_exception is set.
void engine::run_loop(std::latch& started) {
    c10::cuda::CUDAGuard device_guard(devices.front());
    {
        event_loop running;
        running.spawn(sched.run());
        {
            std::lock_guard lock(mutex);
            loop = &running;
        }
        started.count_down();
        std::exception_ptr thrown = running.run();
        for (const std::unique_ptr<event_loop>& worker : worker_loops) {
            worker->stop();
        }
        for (std::thread& worker : worker_threads) {
            worker.join();
        }
        for (const std::vector<cudaStream_t>* shard_streams : {&streams, &copy_streams}) {
            for (cudaStream_t stream : *shard_streams) {
                const cudaError_t status = cudaStreamSynchronize(stream);
                if (status != cudaSuccess && !thrown) {
                    thrown = std::make_exception_ptr(
                        std::runtime_error(std::string("engine: ") + cudaGetErrorString(status))
                    );
                }
            }
        }
        {
            std::lock_guard lock(mutex);
            if (!loop_exception) {
                loop_exception = std::move(thrown);
            }
            loop = nullptr;
        }
        exec.end_requests(request_error::loop_exited);
    }
}

// A panic on a worker loop stops the engine's loop with the worker's exception.
void engine::run_worker_loop(std::size_t shard) {
    c10::cuda::CUDAGuard device_guard(devices[shard]);
    std::exception_ptr thrown = worker_loops[shard - 1]->run();
    if (thrown) {
        std::lock_guard lock(mutex);
        if (!loop_exception) {
            loop_exception = std::move(thrown);
        }
        if (loop != nullptr) {
            loop->stop();
        }
    }
}

// A submit racing the loop's exit finds loop null; the task is then destroyed without running, and the
// scheduler::unstarted_callbacks in its frame ends the callbacks with loop_exited.
void engine::submit(request&& r, std::unique_ptr<request_callbacks> callbacks) {
    validate(r, vocab_size);
    task<void> serving = sched.submit(std::move(r), std::move(callbacks));
    std::lock_guard lock(mutex);
    if (loop != nullptr) {
        loop->spawn(std::move(serving));
    }
}

std::exception_ptr engine::exception() const {
    std::lock_guard lock(mutex);
    return loop_exception;
}

std::size_t engine::largest_page_bytes() const noexcept {
    return max_page_bytes;
}

engine_stat engine::stat() const {
    return engine_stat{
        .kv_cache_bytes = exec.kv_cache_bytes(),
        .drafted_token_count = exec.drafted_token_count(),
        .accepted_draft_count = exec.accepted_draft_count(),
    };
}

eviction_report engine::eviction_report_of(const state& s, const eviction_policy& policy) const {
    return make_eviction_report(s, policy, page_size);
}

}  // namespace pulsarlm
