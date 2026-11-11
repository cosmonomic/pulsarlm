#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <latch>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "pulsarlm/engine/event_loop.hpp"
#include "pulsarlm/engine/engine_config.hpp"
#include "pulsarlm/engine/eviction_policy.hpp"
#include "pulsarlm/engine/eviction_report.hpp"
#include "pulsarlm/engine/model/config.hpp"
#include "pulsarlm/engine/kv_cache.hpp"
#include "pulsarlm/engine/model/interface.hpp"
#include "pulsarlm/engine/request_callbacks.hpp"
#include "pulsarlm/engine/scheduler.hpp"

namespace pulsarlm {

// kv_cache_bytes is the bytes of the KV cache's pool that lines in use take: those of running requests, not those only
// caching a page.
struct engine_stat {
    std::size_t kv_cache_bytes;
};

// Loads rank's part of a model split over size ranks: its weights, by Hugging Face's names, on the rank's device.
using model_shard_loader =
    std::function<std::unordered_map<std::string, at::Tensor>(std::size_t rank, std::size_t size)>;

// The page shape of each KV layer, one layer each.
std::vector<page_shape> page_shapes(const engine_config& config, std::span<const kv_layer> layers);

// Serves requests on a loop thread it owns, with one more loop thread per shard past the first. Once the loops exit,
// after a panic in any of them or on destruction, the requests it was serving end with loop_exited and the engine is
// shut down: calling submit after a request ended with loop_exited is undefined.
class engine {
    std::vector<std::unique_ptr<model>> shards;
    std::vector<at::Device> devices;
    std::vector<cudaStream_t> streams;
    std::vector<std::unique_ptr<event_loop>> worker_loops;
    std::size_t page_size;
    std::size_t vocab_size;
    std::size_t max_page_bytes;
    digest root;
    std::unique_ptr<page_store> store;
    scheduler sched;
    mutable std::mutex mutex;
    // Set while the loop runs.
    event_loop* loop = nullptr;
    std::exception_ptr loop_exception;
    std::vector<std::thread> worker_threads;
    std::thread loop_thread;

    struct loaded_model;

    static loaded_model load_model(
        const any_model_config& model_config,
        const model_shard_loader& load_model_shard,
        std::span<const at::Device> devices
    );
    engine(loaded_model&& loaded, const engine_config& config, std::unique_ptr<page_store>&& store);

    void run_loop(std::latch& started);
    void run_worker_loop(std::size_t shard);

  public:
    // The model of model_config, split over config.devices: shard r runs on config.devices[r] with the weights
    // load_model_shard(r, config.devices.size()) gives, loaded one shard after another. Shards on distinct devices reduce
    // over NCCL; several on one device, for tests, without it. Throws std::invalid_argument for a model the engine does
    // not split over config.devices.
    engine(
        const any_model_config& model_config,
        const model_shard_loader& load_model_shard,
        const engine_config& config,
        std::unique_ptr<page_store>&& store
    );
    // shards[s] holds shard s of the model, run on config.devices[s]; tokens are in [0, vocab_size).
    engine(
        std::vector<std::unique_ptr<model>>&& shards,
        std::size_t vocab_size,
        const engine_config& config,
        std::unique_ptr<page_store>&& store
    );
    // One shard on config.devices[0].
    engine(
        std::unique_ptr<model> served_model,
        std::size_t vocab_size,
        const engine_config& config,
        std::unique_ptr<page_store>&& store
    );
    engine(const engine&) = delete;
    engine& operator=(const engine&) = delete;
    ~engine();

    // Serves r, passing its results to callbacks. Safe to call from any thread. Throws std::invalid_argument, without
    // calling callbacks, when r breaks a precondition of scheduler::submit, holds a token outside [0, vocab_size), or
    // starts from a state scheduler::validate_state rejects.
    void submit(request&& r, std::unique_ptr<request_callbacks> callbacks);
    // The exception that panicked the loop; null while the loop runs or when it was stopped.
    std::exception_ptr exception() const;
    // The digests of the full pages of a token history from its start; the last is the head of a state reaching it.
    std::vector<digest> page_digests(std::span<const std::int64_t> tokens) const;
    // The store keys of the pages of s's sequences, which a request resuming from s reads. Safe to call from any thread.
    std::vector<digest> page_keys(const state& s) const;
    // The scores a cycle's end under policy would evict s's pages by. Safe to call from any thread.
    eviction_report eviction_report_of(const state& s, const eviction_policy& policy) const;
    // Safe to call from any thread.
    engine_stat stat() const;
    // The bytes of the largest page the engine fetches or puts. Safe to call from any thread.
    std::size_t largest_page_bytes() const noexcept;
};

}  // namespace pulsarlm
