#include "pulsarlm/digest.hpp"
#include "pulsarlm/engine/request_callbacks.hpp"

#undef CHECK
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace {

using pulsarlm::request_error;
using pulsarlm::result_writer;

using token_list = std::vector<std::int64_t>;

struct request_results {
    token_list tokens;
    std::vector<pulsarlm::cycle_end> cycle_ends;
    std::optional<std::expected<pulsarlm::state, request_error>> end;
    std::size_t end_count = 0;
    bool cancelled = false;
    bool callbacks_destroyed = false;
};

class recording_callbacks final : public pulsarlm::request_callbacks {
    std::shared_ptr<request_results> recorded_results;

  public:
    explicit recording_callbacks(std::shared_ptr<request_results> recorded_results)
    : recorded_results(std::move(recorded_results)) {}

    ~recording_callbacks() override {
        recorded_results->callbacks_destroyed = true;
    }

    void on_token(std::int64_t token) noexcept override {
        recorded_results->tokens.push_back(token);
    }

    void on_cycle_end(pulsarlm::cycle_end&& ended) noexcept override {
        recorded_results->cycle_ends.push_back(std::move(ended));
    }

    void on_end(std::expected<pulsarlm::request_result, request_error>&& result) noexcept override {
        recorded_results->end.emplace(std::move(result).transform([](pulsarlm::request_result&& ended) {
            return *ended.state;
        }));
        recorded_results->end_count++;
    }

    bool cancelled() const noexcept override {
        return recorded_results->cancelled;
    }
};

result_writer make_writer(const std::shared_ptr<request_results>& recorded_results) {
    return result_writer(std::make_unique<recording_callbacks>(recorded_results));
}

bool ended_with(const request_results& recorded_results, request_error error) {
    return recorded_results.end && !recorded_results.end->has_value() && recorded_results.end->error() == error;
}

TEST_CASE("the writer passes tokens and cycle ends to its callbacks") {
    auto recorded_results = std::make_shared<request_results>();
    result_writer writer = make_writer(recorded_results);
    pulsarlm::digest page_hash{.hi = 3, .lo = 4};
    writer.append(1);
    writer.append(
        pulsarlm::cycle_end{.state = std::make_shared<const pulsarlm::state>(pulsarlm::state{.head = page_hash, .tail = {1}})}
    );
    writer.append(2);

    REQUIRE(recorded_results->tokens == token_list{1, 2});
    REQUIRE(recorded_results->cycle_ends.size() == 1);
    REQUIRE((recorded_results->cycle_ends[0].state->head == page_hash));
    REQUIRE(recorded_results->cycle_ends[0].state->tail == token_list{1});
    REQUIRE(!recorded_results->end);
}

TEST_CASE("the writer ends its callbacks once, then destroys them") {
    auto recorded_results = std::make_shared<request_results>();
    {
        result_writer writer = make_writer(recorded_results);
        pulsarlm::digest page_hash{.hi = 1, .lo = 2};
        writer.end(
            pulsarlm::request_result{
                .state = std::make_shared<const pulsarlm::state>(pulsarlm::state{.head = page_hash, .tail = {5}}),
                .total_tokens = 5,
                .cached_tokens = 4,
                .device_cached_tokens = 4,
            }
        );

        REQUIRE(recorded_results->callbacks_destroyed);
        REQUIRE(recorded_results->end->has_value());
        REQUIRE(((*recorded_results->end)->head == page_hash));
        REQUIRE((*recorded_results->end)->tail == token_list{5});
    }

    REQUIRE(recorded_results->end_count == 1);
}

TEST_CASE("a writer destroyed before ending its callbacks ends them with loop_exited") {
    auto recorded_results = std::make_shared<request_results>();
    {
        result_writer writer = make_writer(recorded_results);
        writer.append(4);
    }

    REQUIRE(recorded_results->tokens == token_list{4});
    REQUIRE(ended_with(*recorded_results, request_error::loop_exited));
    REQUIRE(recorded_results->end_count == 1);
}

TEST_CASE("a moved-from writer does not end the callbacks") {
    auto recorded_results = std::make_shared<request_results>();
    std::optional<result_writer> source(make_writer(recorded_results));
    result_writer target(std::move(*source));
    source.reset();
    REQUIRE(!recorded_results->end);

    target.end(std::unexpected(request_error::miss));
    REQUIRE(ended_with(*recorded_results, request_error::miss));
}

TEST_CASE("cancelled reads the callbacks") {
    auto recorded_results = std::make_shared<request_results>();
    result_writer writer = make_writer(recorded_results);
    REQUIRE(!writer.cancelled());
    recorded_results->cancelled = true;
    REQUIRE(writer.cancelled());
}

}  // namespace
