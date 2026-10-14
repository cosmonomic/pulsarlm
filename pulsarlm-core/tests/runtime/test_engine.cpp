#include "pulsarlm/digest.hpp"
#include "pulsarlm/engine.hpp"
#include "pulsarlm/event_loop.hpp"
#include "pulsarlm/kv_cache.hpp"
#include "pulsarlm/model/interface.hpp"
#include "pulsarlm/model/paged_forward.hpp"
#include "pulsarlm/request_callbacks.hpp"
#include "pulsarlm/scheduler.hpp"

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
using pulsarlm::task;

using token_list = std::vector<std::int64_t>;

constexpr std::size_t page_size = 4;
const std::vector<pulsarlm::kv_layer> one_layer{pulsarlm::kv_layer{.n_kv_heads = 1, .head_dim = 2}};
constexpr std::size_t vocab = 100;
constexpr auto end_timeout = std::chrono::seconds(30);

// Drops puts. A fetch never completes; it counts down fetch_started.
struct stalling_store final : pulsarlm::page_store {
    std::shared_ptr<std::latch> fetch_started = std::make_shared<std::latch>(1);

    task<std::vector<std::optional<page>>> fetch(std::span<const digest>) override {
        auto [never, resolver] = pulsarlm::external_task<>::make();
        fetch_started->count_down();
        co_await std::move(never);
        co_return {};
    }

    void put(std::vector<digest>&&, std::vector<page>&&, std::vector<pulsarlm::page_owner>&&) override {}
};

// Predicts each token's successor, mod vocab.
class successor_model final : public pulsarlm::Model {
  public:
    std::vector<pulsarlm::kv_layer> kv_layers() const override {
        return one_layer;
    }

    pulsarlm::StepLogits forward(const pulsarlm::StepBatch& batch, std::span<const pulsarlm::kv>) override {
        pulsarlm::StepLogits logits;
        if (batch.prefill) {
            logits.prefill = predict(*batch.prefill);
        }
        if (batch.decode) {
            logits.decode = predict(*batch.decode);
        }
        return logits;
    }

  private:
    static at::Tensor predict(const pulsarlm::GroupBatch& g) {
        return pulsarlm::logit_inputs(
            at::one_hot((g.tokens + 1) % static_cast<std::int64_t>(vocab), vocab).to(at::kFloat),
            g
        );
    }
};

class failing_model final : public pulsarlm::Model {
  public:
    std::vector<pulsarlm::kv_layer> kv_layers() const override {
        return one_layer;
    }

    pulsarlm::StepLogits forward(const pulsarlm::StepBatch&, std::span<const pulsarlm::kv>) override {
        throw std::runtime_error("forward failed");
    }
};

// Filled on the loop thread, read on the test thread.
struct request_results {
    std::mutex mutex;
    std::condition_variable ended;
    token_list tokens;
    std::optional<std::expected<pulsarlm::state, request_error>> end;
};

class recording_callbacks final : public pulsarlm::request_callbacks {
    std::shared_ptr<request_results> recorded;

  public:
    explicit recording_callbacks(std::shared_ptr<request_results> recorded)
    : recorded(std::move(recorded)) {}

    void on_token(std::int64_t token) noexcept override {
        std::lock_guard lock(recorded->mutex);
        recorded->tokens.push_back(token);
    }

    void on_cycle_end(pulsarlm::cycle_end&&) noexcept override {}

    void on_end(std::expected<pulsarlm::state, request_error>&& result) noexcept override {
        {
            std::lock_guard lock(recorded->mutex);
            recorded->end.emplace(std::move(result));
        }
        recorded->ended.notify_all();
    }

    bool cancelled() const noexcept override {
        return false;
    }
};

std::unique_ptr<recording_callbacks> record_into(const std::shared_ptr<request_results>& recorded) {
    return std::make_unique<recording_callbacks>(recorded);
}

void wait_for_end(request_results& recorded) {
    std::unique_lock lock(recorded.mutex);
    REQUIRE(recorded.ended.wait_for(lock, end_timeout, [&] { return recorded.end.has_value(); }));
}

pulsarlm::engine_config config(std::size_t n_lines) {
    std::size_t line_bytes = 2 * page_size * 2 * sizeof(float);
    return pulsarlm::engine_config{
        .model_name = "successor",
        .page_size = page_size,
        .kv_dtype = at::kFloat,
        .kv_cache_bytes = n_lines * line_bytes,
        .host_page_bytes = 64 * line_bytes,
        .device = at::Device(at::kCUDA, 0),
        .stop_ids = {},
        .vocab_size = vocab,
        .prefill_tokens_per_step = 64,
    };
}

bool ended_with(const request_results& recorded, request_error error) {
    return recorded.end && !recorded.end->has_value() && recorded.end->error() == error;
}

digest root() {
    return pulsarlm::make_root_digest("successor", pulsarlm::page_shapes(config(16), one_layer));
}

TEST_CASE("an engine whose KV cache holds no line is rejected") {
    pulsarlm::engine_config sized = config(16);
    sized.kv_cache_bytes = pulsarlm::line_bytes(pulsarlm::page_shapes(sized, one_layer)[0]) - 1;
    REQUIRE_THROWS_AS(
        pulsarlm::engine(std::make_unique<successor_model>(), sized, std::make_unique<stalling_store>()),
        std::invalid_argument
    );
}

TEST_CASE("an engine whose bfloat16 attention the kernels cannot run is rejected") {
    pulsarlm::engine_config bfloat16 = config(16);
    bfloat16.kv_dtype = at::kBFloat16;
    REQUIRE_THROWS_AS(
        pulsarlm::engine(std::make_unique<successor_model>(), bfloat16, std::make_unique<stalling_store>()),
        std::invalid_argument
    );
}

TEST_CASE("a submitted request passes its tokens and its end state to its callbacks") {
    pulsarlm::engine e(std::make_unique<successor_model>(), config(16), std::make_unique<stalling_store>());
    auto recorded = std::make_shared<request_results>();
    e.submit(request{.tokens = {1, 2, 3}, .max_tokens = 4}, record_into(recorded));
    wait_for_end(*recorded);

    REQUIRE(recorded->tokens == token_list{4, 5, 6, 7});
    REQUIRE(recorded->end->has_value());
    REQUIRE(((*recorded->end)->page_hashes == pulsarlm::make_page_digests(root(), token_list{1, 2, 3, 4}, page_size)));
    REQUIRE((*recorded->end)->tail == token_list{5, 6});
    REQUIRE(e.exception() == nullptr);
}

TEST_CASE("submit rejects an invalid request without calling its callbacks") {
    pulsarlm::engine e(std::make_unique<successor_model>(), config(16), std::make_unique<stalling_store>());
    auto recorded = std::make_shared<request_results>();

    REQUIRE_THROWS_AS(e.submit(request{}, record_into(recorded)), std::invalid_argument);
    REQUIRE_THROWS_AS(
        e.submit(request{.initial_state = {.tail = {1, 2, 3, 4}}}, record_into(recorded)),
        std::invalid_argument
    );
    REQUIRE_THROWS_AS(e.submit(request{.tokens = {-1}}, record_into(recorded)), std::invalid_argument);
    REQUIRE_THROWS_AS(
        e.submit(request{.tokens = {static_cast<std::int64_t>(vocab)}}, record_into(recorded)),
        std::invalid_argument
    );
    REQUIRE_THROWS_AS(
        e.submit(request{.initial_state = {.tail = {200}}, .tokens = {1}}, record_into(recorded)),
        std::invalid_argument
    );
    REQUIRE_THROWS_AS(
        e.submit(request{.tokens = {1}, .eviction = {.n_working = 0}}, record_into(recorded)),
        std::invalid_argument
    );
    REQUIRE_THROWS_AS(
        e.submit(request{.tokens = {1}, .eviction = {.cycle_tokens = 0}}, record_into(recorded)),
        std::invalid_argument
    );
    REQUIRE_THROWS_AS(
        e.submit(request{.tokens = {1}, .eviction = {.attention_mass_decay = 0.0}}, record_into(recorded)),
        std::invalid_argument
    );
    REQUIRE_THROWS_AS(
        e.submit(request{.tokens = {1}, .eviction = {.attention_mass_decay = 1.5}}, record_into(recorded)),
        std::invalid_argument
    );
    REQUIRE(recorded->tokens.empty());
    REQUIRE(!recorded->end);
}

TEST_CASE("destroying the engine ends the requests it serves with loop_exited") {
    auto store = std::make_unique<stalling_store>();
    std::shared_ptr<std::latch> fetch_started = store->fetch_started;
    std::optional<pulsarlm::engine> e;
    e.emplace(std::make_unique<successor_model>(), config(16), std::move(store));
    auto recorded = std::make_shared<request_results>();
    e->submit(
        request{.initial_state = {.page_hashes = {digest{.hi = 1, .lo = 2}}}, .tokens = {1}, .max_tokens = 1},
        record_into(recorded)
    );
    fetch_started->wait();
    e.reset();

    REQUIRE(ended_with(*recorded, request_error::loop_exited));
}

TEST_CASE("a panic ends the requests with loop_exited and leaves its exception") {
    pulsarlm::engine e(std::make_unique<failing_model>(), config(16), std::make_unique<stalling_store>());
    auto recorded = std::make_shared<request_results>();
    e.submit(request{.tokens = {1, 2, 3}, .max_tokens = 1}, record_into(recorded));
    wait_for_end(*recorded);

    REQUIRE(ended_with(*recorded, request_error::loop_exited));
    std::exception_ptr thrown = e.exception();
    REQUIRE(thrown != nullptr);
    REQUIRE_THROWS_AS(std::rethrow_exception(thrown), std::runtime_error);
}

}  // namespace
