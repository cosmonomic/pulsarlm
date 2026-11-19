#include "pulsarlm/engine/request_callbacks.hpp"

#undef CHECK
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <utility>

namespace {

using pulsarlm::request_error;

struct request_results {
    std::optional<std::expected<std::shared_ptr<const pulsarlm::state>, request_error>> end;
    pulsarlm::request_statistics statistics{};
    std::size_t call_count = 0;
};

class recording_callbacks final : public pulsarlm::request_callbacks {
    std::shared_ptr<request_results> recorded_results;

  public:
    explicit recording_callbacks(std::shared_ptr<request_results> recorded_results)
    : recorded_results(std::move(recorded_results)) {}

    void on_prefill(const pulsarlm::prefill_statistics&) noexcept override {
        recorded_results->call_count++;
    }

    void on_decode(std::int64_t, const pulsarlm::decode_statistics&) noexcept override {
        recorded_results->call_count++;
    }

    void on_cycle(pulsarlm::cycle_end&&) noexcept override {
        recorded_results->call_count++;
    }

    void on_end(
        std::expected<std::shared_ptr<const pulsarlm::state>, request_error>&& result,
        const pulsarlm::request_statistics& statistics
    ) noexcept override {
        recorded_results->end.emplace(std::move(result));
        recorded_results->statistics = statistics;
        recorded_results->call_count++;
    }

    bool cancelled() const noexcept override {
        return false;
    }
};

TEST_CASE("end_with_loop_exited ends the callbacks once with loop_exited and value-initialized statistics") {
    auto recorded_results = std::make_shared<request_results>();
    std::unique_ptr<pulsarlm::request_callbacks> callbacks = std::make_unique<recording_callbacks>(recorded_results);
    pulsarlm::end_with_loop_exited(callbacks);

    REQUIRE(recorded_results->call_count == 1);
    REQUIRE(recorded_results->end.has_value());
    REQUIRE(!recorded_results->end->has_value());
    REQUIRE(recorded_results->end->error() == request_error::loop_exited);
    REQUIRE(recorded_results->statistics.total_token_count == 0);
    REQUIRE(recorded_results->statistics.cached_token_count == 0);
    REQUIRE(recorded_results->statistics.device_cached_token_count == 0);
    REQUIRE(recorded_results->statistics.peak_device_memory == 0);
    REQUIRE(recorded_results->statistics.drafted_token_count == 0);
    REQUIRE(recorded_results->statistics.accepted_draft_count == 0);
    REQUIRE(recorded_results->statistics.step_duration == std::chrono::steady_clock::duration::zero());
}

}  // namespace
