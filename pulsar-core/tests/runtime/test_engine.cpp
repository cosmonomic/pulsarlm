#include "pulsar/digest.hpp"
#include "pulsar/engine.hpp"
#include "pulsar/event_loop.hpp"
#include "pulsar/kv_cache.hpp"
#include "pulsar/model/interface.hpp"
#include "pulsar/request_callbacks.hpp"
#include "pulsar/scheduler.hpp"

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

using pulsar::digest;
using pulsar::page;
using pulsar::request;
using pulsar::request_error;
using pulsar::task;

using token_list = std::vector<std::int64_t>;

constexpr pulsar::page_shape shape{.n_layers = 1, .page_size = 4, .n_kv_heads = 1, .head_dim = 2, .dtype = at::kFloat};
constexpr std::size_t vocab = 100;
constexpr auto end_timeout = std::chrono::seconds(30);

// Drops puts. A fetch never completes; it counts down fetch_started.
struct stalling_store {
    std::shared_ptr<std::latch> fetch_started = std::make_shared<std::latch>(1);

    task<std::vector<std::optional<page>>> fetch(std::span<const digest>) {
        auto [never, resolver] = pulsar::external_task<>::make();
        fetch_started->count_down();
        co_await std::move(never);
        co_return {};
    }

    void put(std::vector<digest>&&, std::vector<page>&&) {}
};

static_assert(pulsar::page_store<stalling_store>);

using test_engine = pulsar::engine<stalling_store>;

// Predicts each token's successor, mod vocab.
class successor_model final : public pulsar::Model {
  public:
    pulsar::StepLogits forward(const pulsar::StepBatch& batch, const pulsar::kv_lines&) override {
        pulsar::StepLogits logits;
        if (batch.prefill) {
            logits.prefill = predict(*batch.prefill);
        }
        if (batch.decode) {
            logits.decode = predict(*batch.decode);
        }
        return logits;
    }

  private:
    static at::Tensor predict(const pulsar::GroupBatch& g) {
        return at::one_hot((g.tokens + 1) % static_cast<std::int64_t>(vocab), vocab).to(at::kFloat);
    }
};

class failing_model final : public pulsar::Model {
  public:
    pulsar::StepLogits forward(const pulsar::StepBatch&, const pulsar::kv_lines&) override {
        throw std::runtime_error("forward failed");
    }
};

// Filled on the loop thread, read on the test thread.
struct request_results {
    std::mutex mutex;
    std::condition_variable ended;
    token_list tokens;
    std::optional<std::expected<pulsar::state, request_error>> end;
};

class recording_callbacks final : public pulsar::request_callbacks {
    std::shared_ptr<request_results> recorded;

  public:
    explicit recording_callbacks(std::shared_ptr<request_results> recorded)
    : recorded(std::move(recorded)) {}

    void on_token(std::int64_t token) noexcept override {
        std::lock_guard lock(recorded->mutex);
        recorded->tokens.push_back(token);
    }

    void on_cycle_end(pulsar::cycle_end&&) noexcept override {}

    void on_end(std::expected<pulsar::state, request_error>&& result) noexcept override {
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

digest root() {
    return pulsar::make_root_digest("successor", shape);
}

pulsar::engine_config config(std::size_t n_lines) {
    return pulsar::engine_config{
        .model_name = "successor",
        .shape = shape,
        .n_lines = n_lines,
        .device = at::Device(at::kCUDA, 0),
        .eos_id = static_cast<std::int64_t>(vocab),
        .vocab_size = vocab,
        .prefill_tokens_per_step = 64,
    };
}

bool ended_with(const request_results& recorded, request_error error) {
    return recorded.end && !recorded.end->has_value() && recorded.end->error() == error;
}

TEST_CASE("a submitted request passes its tokens and its end state to its callbacks") {
    test_engine e(std::make_unique<successor_model>(), config(16), stalling_store{});
    auto recorded = std::make_shared<request_results>();
    e.submit(request{.tokens = {1, 2, 3}, .max_tokens = 4}, record_into(recorded));
    wait_for_end(*recorded);

    REQUIRE(recorded->tokens == token_list{4, 5, 6, 7});
    REQUIRE(recorded->end->has_value());
    REQUIRE((
        (*recorded->end)->page_hashes == pulsar::make_page_digests(root(), token_list{1, 2, 3, 4}, shape.page_size)
    ));
    REQUIRE((*recorded->end)->tail == token_list{5, 6});
    REQUIRE(e.exception() == nullptr);
}

TEST_CASE("submit rejects an invalid request without calling its callbacks") {
    test_engine e(std::make_unique<successor_model>(), config(16), stalling_store{});
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
    stalling_store store;
    std::shared_ptr<std::latch> fetch_started = store.fetch_started;
    std::optional<test_engine> e;
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
    test_engine e(std::make_unique<failing_model>(), config(16), stalling_store{});
    auto recorded = std::make_shared<request_results>();
    e.submit(request{.tokens = {1, 2, 3}, .max_tokens = 1}, record_into(recorded));
    wait_for_end(*recorded);

    REQUIRE(ended_with(*recorded, request_error::loop_exited));
    std::exception_ptr thrown = e.exception();
    REQUIRE(thrown != nullptr);
    REQUIRE_THROWS_AS(std::rethrow_exception(thrown), std::runtime_error);
}

}  // namespace
