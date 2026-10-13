#include "pulsarlm/digest.hpp"
#include "pulsarlm/request_callbacks.hpp"

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
    std::shared_ptr<request_results> recorded;

  public:
    explicit recording_callbacks(std::shared_ptr<request_results> recorded)
    : recorded(std::move(recorded)) {}

    ~recording_callbacks() override {
        recorded->callbacks_destroyed = true;
    }

    void on_token(std::int64_t token) noexcept override {
        recorded->tokens.push_back(token);
    }

    void on_cycle_end(pulsarlm::cycle_end&& ended) noexcept override {
        recorded->cycle_ends.push_back(std::move(ended));
    }

    void on_end(std::expected<pulsarlm::state, request_error>&& result) noexcept override {
        recorded->end.emplace(std::move(result));
        recorded->end_count++;
    }

    bool cancelled() const noexcept override {
        return recorded->cancelled;
    }
};

result_writer make_writer(const std::shared_ptr<request_results>& recorded) {
    return result_writer(std::make_unique<recording_callbacks>(recorded));
}

bool ended_with(const request_results& recorded, request_error error) {
    return recorded.end && !recorded.end->has_value() && recorded.end->error() == error;
}

TEST_CASE("the writer passes tokens and cycle ends to its callbacks") {
    auto recorded = std::make_shared<request_results>();
    result_writer writer = make_writer(recorded);
    pulsarlm::digest page_hash{.hi = 3, .lo = 4};
    writer.append(1);
    writer.append(
        pulsarlm::cycle_end{.reached = pulsarlm::state{.page_hashes = {page_hash}, .tail = {1}}, .token_count = 5}
    );
    writer.append(2);

    REQUIRE(recorded->tokens == token_list{1, 2});
    REQUIRE(recorded->cycle_ends.size() == 1);
    REQUIRE(recorded->cycle_ends[0].reached.page_hashes == std::vector{page_hash});
    REQUIRE(recorded->cycle_ends[0].reached.tail == token_list{1});
    REQUIRE(recorded->cycle_ends[0].token_count == 5);
    REQUIRE(!recorded->end);
}

TEST_CASE("the writer ends its callbacks once, then destroys them") {
    auto recorded = std::make_shared<request_results>();
    {
        result_writer writer = make_writer(recorded);
        pulsarlm::digest page_hash{.hi = 1, .lo = 2};
        writer.end(pulsarlm::state{.page_hashes = {page_hash}, .tail = {5}});

        REQUIRE(recorded->callbacks_destroyed);
        REQUIRE(recorded->end->has_value());
        REQUIRE((*recorded->end)->page_hashes == std::vector{page_hash});
        REQUIRE((*recorded->end)->tail == token_list{5});
    }

    REQUIRE(recorded->end_count == 1);
}

TEST_CASE("a writer destroyed before ending its callbacks ends them with loop_exited") {
    auto recorded = std::make_shared<request_results>();
    {
        result_writer writer = make_writer(recorded);
        writer.append(4);
    }

    REQUIRE(recorded->tokens == token_list{4});
    REQUIRE(ended_with(*recorded, request_error::loop_exited));
    REQUIRE(recorded->end_count == 1);
}

TEST_CASE("a moved-from writer does not end the callbacks") {
    auto recorded = std::make_shared<request_results>();
    std::optional<result_writer> source(make_writer(recorded));
    result_writer target(std::move(*source));
    source.reset();
    REQUIRE(!recorded->end);

    target.end(std::unexpected(request_error::miss));
    REQUIRE(ended_with(*recorded, request_error::miss));
}

TEST_CASE("cancelled reads the callbacks") {
    auto recorded = std::make_shared<request_results>();
    result_writer writer = make_writer(recorded);
    REQUIRE(!writer.cancelled());
    recorded->cancelled = true;
    REQUIRE(writer.cancelled());
}

}  // namespace
