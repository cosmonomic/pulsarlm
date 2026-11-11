#include "pulsarlm/digest.hpp"
#include "pulsarlm/engine/engine.hpp"
#include "pulsarlm/engine/event_loop.hpp"
#include "pulsarlm/engine/kv_cache.hpp"
#include "pulsarlm/engine/model/interface.hpp"
#include "pulsarlm/engine/model/paged_forward.hpp"
#include "pulsarlm/engine/request_callbacks.hpp"
#include "pulsarlm/engine/scheduler.hpp"

#include <ATen/ATen.h>

#undef CHECK
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <expected>
#include <latch>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using pulsarlm::digest;
using pulsarlm::page;
using pulsarlm::request;
using pulsarlm::request_error;
using pulsarlm::sharded_page;
using pulsarlm::task;

using token_list = std::vector<std::int64_t>;

constexpr std::size_t page_size = 4;
const std::vector<pulsarlm::kv_layer> one_layer{pulsarlm::kv_layer{.n_kv_heads = 1, .head_dim = 2, .n_q_heads = 1}};
constexpr std::size_t vocab = 100;
constexpr auto end_timeout = std::chrono::seconds(30);

// Drops puts. A fetch never completes; it counts down fetch_started.
struct stalling_store final : pulsarlm::page_store {
    std::shared_ptr<std::latch> fetch_start_latch = std::make_shared<std::latch>(1);

    task<std::vector<bool>>
    fetch(std::span<const digest>, std::vector<sharded_page>&&, std::span<const cudaStream_t>) override {
        auto [never, resolver] = pulsarlm::external_task<>::make();
        fetch_start_latch->count_down();
        co_await std::move(never);
        co_return {};
    }

    task<std::expected<void, pulsarlm::page_error>>
    put(std::vector<digest>&&,
        std::vector<sharded_page>&&,
        std::vector<pulsarlm::page_owner>&&,
        std::span<const cudaStream_t>) override {
        co_return {};
    }
};

// Predicts each token's successor, mod vocab.
class successor_model final : public pulsarlm::model {
  public:
    std::vector<pulsarlm::kv_layer> kv_layers() const override {
        return one_layer;
    }

    pulsarlm::step_logits forward(const pulsarlm::step_batch& batch, std::span<const pulsarlm::kv>) override {
        pulsarlm::step_logits logits;
        if (batch.prefill_group) {
            logits.prefill_logits = predict(*batch.prefill_group);
        }
        if (batch.decode_group) {
            logits.decode_logits = predict(*batch.decode_group);
        }
        return logits;
    }

  private:
    static at::Tensor predict(const pulsarlm::group_batch& g) {
        return pulsarlm::logit_inputs(
            at::one_hot((g.tokens + 1) % static_cast<std::int64_t>(vocab), vocab).to(at::kFloat),
            g
        );
    }
};

class failing_model final : public pulsarlm::model {
  public:
    std::vector<pulsarlm::kv_layer> kv_layers() const override {
        return one_layer;
    }

    pulsarlm::step_logits forward(const pulsarlm::step_batch&, std::span<const pulsarlm::kv>) override {
        throw std::runtime_error("forward failed");
    }
};

// Filled on the loop thread, read on the test thread.
struct request_results {
    std::mutex mutex;
    std::condition_variable end_condition;
    token_list tokens;
    std::optional<std::expected<pulsarlm::state, request_error>> end;
};

class recording_callbacks final : public pulsarlm::request_callbacks {
    std::shared_ptr<request_results> recorded_results;

  public:
    explicit recording_callbacks(std::shared_ptr<request_results> recorded_results)
    : recorded_results(std::move(recorded_results)) {}

    void on_token(std::int64_t token) noexcept override {
        std::lock_guard lock(recorded_results->mutex);
        recorded_results->tokens.push_back(token);
    }

    void on_cycle_end(pulsarlm::cycle_end&&) noexcept override {}

    void on_end(std::expected<pulsarlm::request_result, request_error>&& result) noexcept override {
        {
            std::lock_guard lock(recorded_results->mutex);
            recorded_results->end.emplace(std::move(result).transform([](pulsarlm::request_result&& ended) {
                return std::move(ended.state);
            }));
        }
        recorded_results->end_condition.notify_all();
    }

    bool cancelled() const noexcept override {
        return false;
    }
};

std::unique_ptr<recording_callbacks> record_into(const std::shared_ptr<request_results>& recorded_results) {
    return std::make_unique<recording_callbacks>(recorded_results);
}

void wait_for_end(request_results& recorded_results) {
    std::unique_lock lock(recorded_results.mutex);
    REQUIRE(recorded_results.end_condition.wait_for(lock, end_timeout, [&] { return recorded_results.end.has_value(); }));
}

pulsarlm::engine_config config(std::size_t n_lines) {
    std::size_t line_bytes = 2 * page_size * 2 * sizeof(float);
    return pulsarlm::engine_config{
        .model_name = "successor",
        .page_size = page_size,
        .kv_dtype = at::kFloat,
        .kv_cache_pool_bytes = n_lines * line_bytes,
        .devices = {at::Device(at::kCUDA, 0)},
        .stop_ids = {},
        .prefill_tokens_per_step = 64,
    };
}

bool ended_with(const request_results& recorded_results, request_error error) {
    return recorded_results.end && !recorded_results.end->has_value() && recorded_results.end->error() == error;
}

// A state of one_layer's model whose one page is hash, with mass sum 0 and weight_sum.
pulsarlm::state one_page_state(const digest& hash, float weight_sum) {
    return pulsarlm::state{
        .head = hash,
        .layers = {{.pages = {{.hash = hash, .mass_sum = {0.0f}, .weight_sum = weight_sum}}}},
    };
}

digest root() {
    return pulsarlm::make_root_digest("successor", pulsarlm::page_shapes(config(16), one_layer));
}

TEST_CASE("an engine whose KV cache holds no line is rejected") {
    pulsarlm::engine_config sized = config(16);
    sized.kv_cache_pool_bytes = pulsarlm::line_bytes(pulsarlm::page_shapes(sized, one_layer)[0]) - 1;
    REQUIRE_THROWS_AS(
        pulsarlm::engine(std::make_unique<successor_model>(), vocab, sized, std::make_unique<stalling_store>()),
        std::invalid_argument
    );
}

TEST_CASE("an engine whose bfloat16 attention the kernels cannot run is rejected") {
    pulsarlm::engine_config bfloat16 = config(16);
    bfloat16.kv_dtype = at::kBFloat16;
    REQUIRE_THROWS_AS(
        pulsarlm::engine(std::make_unique<successor_model>(), vocab, bfloat16, std::make_unique<stalling_store>()),
        std::invalid_argument
    );
}

TEST_CASE("a submitted request passes its tokens and its end state to its callbacks") {
    pulsarlm::engine e(std::make_unique<successor_model>(), vocab, config(16), std::make_unique<stalling_store>());
    auto recorded_results = std::make_shared<request_results>();
    e.submit(request{.tokens = {1, 2, 3}, .max_tokens = 4}, record_into(recorded_results));
    wait_for_end(*recorded_results);

    REQUIRE(recorded_results->tokens == token_list{4, 5, 6, 7});
    REQUIRE(recorded_results->end->has_value());
    std::vector<digest> hashes = pulsarlm::make_page_digests(root(), token_list{1, 2, 3, 4}, page_size);
    REQUIRE(((*recorded_results->end)->head == hashes.back()));
    REQUIRE(
        ((*recorded_results->end)->layers ==
         std::vector{pulsarlm::layer_sequence{
             .pages = {{.hash = hashes[0], .mass_sum = {0.0f}}},
             .density_moments = {pulsarlm::moments{}},
         }})
    );
    REQUIRE((*recorded_results->end)->tail == token_list{5, 6});
    REQUIRE(e.exception() == nullptr);
}

TEST_CASE("submit rejects an invalid request without calling its callbacks") {
    pulsarlm::engine e(std::make_unique<successor_model>(), vocab, config(16), std::make_unique<stalling_store>());
    auto recorded_results = std::make_shared<request_results>();

    REQUIRE_THROWS_AS(e.submit(request{}, record_into(recorded_results)), std::invalid_argument);
    REQUIRE_THROWS_AS(
        e.submit(request{.initial_state = {.tail = {1, 2, 3, 4}}}, record_into(recorded_results)),
        std::invalid_argument
    );
    REQUIRE_THROWS_AS(e.submit(request{.tokens = {-1}}, record_into(recorded_results)), std::invalid_argument);
    REQUIRE_THROWS_AS(
        e.submit(request{.tokens = {static_cast<std::int64_t>(vocab)}}, record_into(recorded_results)),
        std::invalid_argument
    );
    REQUIRE_THROWS_AS(
        e.submit(request{.initial_state = {.tail = {200}}, .tokens = {1}}, record_into(recorded_results)),
        std::invalid_argument
    );
    REQUIRE_THROWS_AS(
        e.submit(request{.initial_state = {.head = digest{.hi = 1, .lo = 2}}, .tokens = {1}}, record_into(recorded_results)),
        std::invalid_argument
    );
    REQUIRE_THROWS_AS(
        e.submit(
            request{.initial_state = one_page_state(digest{.hi = 1, .lo = 2}, -1.0f), .tokens = {1}},
            record_into(recorded_results)
        ),
        std::invalid_argument
    );
    REQUIRE_THROWS_AS(
        e.submit(request{.tokens = {1}, .eviction = {.n_working = 0}}, record_into(recorded_results)),
        std::invalid_argument
    );
    REQUIRE_THROWS_AS(
        e.submit(request{.tokens = {1}, .eviction = {.cycle_tokens = 0}}, record_into(recorded_results)),
        std::invalid_argument
    );
    REQUIRE_THROWS_AS(
        e.submit(request{.tokens = {1}, .eviction = {.attention_mass_decay = 0.0}}, record_into(recorded_results)),
        std::invalid_argument
    );
    REQUIRE_THROWS_AS(
        e.submit(request{.tokens = {1}, .eviction = {.attention_mass_decay = 1.5}}, record_into(recorded_results)),
        std::invalid_argument
    );
    REQUIRE_THROWS_AS(
        e.submit(request{.tokens = {1}, .eviction = {.eviction_threshold = -0.5}}, record_into(recorded_results)),
        std::invalid_argument
    );
    REQUIRE_THROWS_AS(
        e.submit(request{.tokens = {1}, .eviction = {.eviction_threshold = 1.5}}, record_into(recorded_results)),
        std::invalid_argument
    );
    REQUIRE(recorded_results->tokens.empty());
    REQUIRE(!recorded_results->end);
}

TEST_CASE("destroying the engine ends the requests it serves with loop_exited") {
    auto store = std::make_unique<stalling_store>();
    std::shared_ptr<std::latch> fetch_start_latch = store->fetch_start_latch;
    std::optional<pulsarlm::engine> e;
    e.emplace(std::make_unique<successor_model>(), vocab, config(16), std::move(store));
    auto recorded_results = std::make_shared<request_results>();
    e->submit(
        request{.initial_state = one_page_state(digest{.hi = 1, .lo = 2}, 0.0f), .tokens = {1}, .max_tokens = 1},
        record_into(recorded_results)
    );
    fetch_start_latch->wait();
    e.reset();

    REQUIRE(ended_with(*recorded_results, request_error::loop_exited));
}

TEST_CASE("a panic ends the requests with loop_exited and leaves its exception") {
    pulsarlm::engine e(std::make_unique<failing_model>(), vocab, config(16), std::make_unique<stalling_store>());
    auto recorded_results = std::make_shared<request_results>();
    e.submit(request{.tokens = {1, 2, 3}, .max_tokens = 1}, record_into(recorded_results));
    wait_for_end(*recorded_results);

    REQUIRE(ended_with(*recorded_results, request_error::loop_exited));
    std::exception_ptr thrown = e.exception();
    REQUIRE(thrown != nullptr);
    REQUIRE_THROWS_AS(std::rethrow_exception(thrown), std::runtime_error);
}

}  // namespace
