#include "pulsar/digest.hpp"
#include "pulsar/result_stream.hpp"

#undef CHECK
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace {

using pulsar::request_error;
using pulsar::result_stream;
using pulsar::result_writer;

using token_list = std::vector<std::int64_t>;

bool ended_with(const result_stream::taken& taken, request_error error) {
    return taken.end && !taken.end->has_value() && taken.end->error() == error;
}

TEST_CASE("take returns the tokens appended since the last take") {
    auto [stream, writer] = result_stream::make();
    writer.append(1);
    writer.append(2);
    result_stream::taken first = stream->take();
    writer.append(3);
    result_stream::taken second = stream->take();

    REQUIRE(first.tokens == token_list{1, 2});
    REQUIRE(!first.end);
    REQUIRE(second.tokens == token_list{3});
    REQUIRE(!second.end);
}

TEST_CASE("take returns the end once") {
    auto [stream, writer] = result_stream::make();
    pulsar::digest page_hash{.hi = 1, .lo = 2};
    writer.append(7);
    writer.end(pulsar::state{.page_hashes = {page_hash}, .tail = {5}});
    result_stream::taken first = stream->take();
    result_stream::taken second = stream->take();

    REQUIRE(first.tokens == token_list{7});
    REQUIRE(first.end.has_value());
    REQUIRE(first.end->has_value());
    REQUIRE((*first.end)->page_hashes == std::vector{page_hash});
    REQUIRE((*first.end)->tail == token_list{5});
    REQUIRE(!second.end);
}

TEST_CASE("a writer destroyed before ending its stream ends it with loop_exited") {
    std::shared_ptr<result_stream> stream;
    {
        auto [made, writer] = result_stream::make();
        stream = made;
        writer.append(4);
    }
    result_stream::taken taken = stream->take();

    REQUIRE(taken.tokens == token_list{4});
    REQUIRE(ended_with(taken, request_error::loop_exited));
}

TEST_CASE("a moved-from writer does not end the stream") {
    auto [stream, writer] = result_stream::make();
    std::optional<result_writer> source(std::move(writer));
    result_writer target(std::move(*source));
    source.reset();
    REQUIRE(!stream->take().end);

    target.end(std::unexpected(request_error::miss));
    REQUIRE(ended_with(stream->take(), request_error::miss));
}

TEST_CASE("cancel is visible to the writer") {
    auto [stream, writer] = result_stream::make();
    REQUIRE(!writer.cancelled());
    stream->cancel();
    REQUIRE(writer.cancelled());
}

}  // namespace
