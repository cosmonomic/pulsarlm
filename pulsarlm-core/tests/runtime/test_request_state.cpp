#include "pulsarlm/engine/request_state.hpp"
#include "pulsarlm/digest.hpp"
#include "pulsarlm/engine/event_loop.hpp"
#include "pulsarlm/engine/page_store.hpp"
#include "pulsarlm/engine/request.hpp"
#include "pulsarlm/engine/state.hpp"

#undef CHECK
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace {

using pulsarlm::digest;
using pulsarlm::page_error;
using pulsarlm::request_state;
using pulsarlm::state;
using pulsarlm::task;
using pulsarlm::view_layout;

using token_list = std::vector<std::int64_t>;

constexpr digest root{.hi = 1, .lo = 2};
constexpr state::lease_id_type lease_id = 7;

view_layout make_layout(std::size_t page_size, bool has_recurrent_layer) {
    view_layout layout{.layers = {}, .page_size = page_size};
    layout.layers.push_back(
        view_layout::layer{
            .is_recurrent = false,
            .mirrors_held_pages = false,
            .sliding_window = std::nullopt,
            .state_slots = 0,
            .mass_heads = 0,
        }
    );
    if (has_recurrent_layer) {
        layout.layers.push_back(
            view_layout::layer{
                .is_recurrent = true,
                .mirrors_held_pages = false,
                .sliding_window = std::nullopt,
                .state_slots = 1,
                .mass_heads = 0,
            }
        );
    }
    return layout;
}

// The request state of a request that resumes the empty state, with what it refers to.
class started_request {
    view_layout layout;

  public:
    request_state req;

    started_request(const token_list& tokens, std::optional<std::size_t> max_tokens, view_layout&& layout)
    : layout(std::move(layout)),
      req(pulsarlm::request{.tokens = tokens, .max_tokens = max_tokens},
          std::make_shared<const state>(),
          lease_id,
          pulsarlm::make_page_digests(root, tokens, this->layout.page_size),
          root,
          this->layout) {}
};

token_list to_list(std::span<const std::int64_t> tokens) {
    return token_list(tokens.begin(), tokens.end());
}

TEST_CASE("a prefill step and decode steps hand out their tokens by counts and apply them at their write-back") {
    const token_list prompt{10, 11, 12, 13, 14, 15};
    started_request started(prompt, 3, make_layout(4, false));
    request_state& req = started.req;
    REQUIRE(to_list(req.prompt_tokens()) == prompt);
    REQUIRE(req.unprepared_token_count() == 6);
    REQUIRE(req.history_position() == 0);
    REQUIRE(req.prompt_token_count() == 6);
    REQUIRE(req.pending_token() == 10);

    req.prepare_step(6, true, 0, false);
    REQUIRE(req.unprepared_token_count() == 1);
    REQUIRE(req.prepared_prompt_token_count() == 6);
    REQUIRE(req.prepared_position_bound() == 6);
    REQUIRE(req.prepared_sample_count() == 1);
    REQUIRE(req.tail_tokens().empty());
    REQUIRE(req.history_position() == 0);
    REQUIRE(req.next_draw().count == 0);
    REQUIRE(req.unfinished_step_count() == 1);
    REQUIRE_FALSE(req.has_prepared_last_step());

    req.apply_step(6, token_list{20}, {}, 0);
    req.finish_step();
    REQUIRE(to_list(req.tail_tokens()) == prompt);
    REQUIRE(req.pending_token() == 20);
    REQUIRE(req.next_draw().count == 1);
    REQUIRE(req.unfinished_step_count() == 0);

    req.prepare_step(1, true, 0, false);
    REQUIRE_FALSE(req.has_prepared_last_step());
    req.prepare_step(1, true, 0, false);
    REQUIRE(req.prepared_sample_count() == 3);
    REQUIRE(req.prepared_position_bound() == 8);
    REQUIRE(req.delivered_token_count() == 1);
    REQUIRE_FALSE(req.may_sample());
    REQUIRE(req.has_prepared_last_step());

    req.apply_step(1, token_list{21}, {}, 0);
    req.finish_step();
    REQUIRE(to_list(req.tail_tokens()) == token_list{10, 11, 12, 13, 14, 15, 20});
    REQUIRE(req.pending_token() == 21);
    req.apply_step(1, token_list{22}, {}, 0);
    req.finish_step();
    REQUIRE(to_list(req.tail_tokens()) == token_list{10, 11, 12, 13, 14, 15, 20, 21});
    REQUIRE(req.pending_token() == 22);
    REQUIRE(req.delivered_token_count() == 3);
    REQUIRE(req.prepared_position_bound() == req.history_position());
}

TEST_CASE("a prefill step that samples no token leaves the next prompt token pending") {
    started_request started({1, 2, 3, 4, 5}, std::nullopt, make_layout(4, false));
    request_state& req = started.req;

    req.prepare_step(4, false, 0, false);
    REQUIRE(req.unprepared_token_count() == 1);
    REQUIRE(req.prepared_sample_count() == 0);
    REQUIRE_FALSE(req.has_prepared_last_step());

    req.apply_step(4, {}, {}, 0);
    req.finish_step();
    REQUIRE(to_list(req.tail_tokens()) == token_list{1, 2, 3, 4});
    REQUIRE(req.pending_token() == 5);
    REQUIRE(req.next_draw().count == 0);
}

TEST_CASE("completing pages moves the full pages of the tail to the views") {
    const token_list prompt{1, 2, 3, 4, 5, 6, 7};
    started_request started(prompt, std::nullopt, make_layout(4, false));
    request_state& req = started.req;
    req.prepare_step(7, true, 0, false);
    REQUIRE(req.complete_pages() == 0);
    req.apply_step(7, token_list{8}, {}, 0);
    req.finish_step();
    REQUIRE(req.complete_pages() == 1);

    const std::vector<digest> prompt_digests = pulsarlm::make_page_digests(root, prompt, 4);
    REQUIRE(req.history_page_count() == 1);
    REQUIRE((req.head_digest() == prompt_digests.front()));
    REQUIRE((req.views()[0].pages == prompt_digests));
    REQUIRE(req.views()[0].indices == std::vector<std::size_t>{0});
    REQUIRE(to_list(req.tail_tokens()) == token_list{5, 6, 7});
    REQUIRE(req.history_position() == 7);
    REQUIRE(req.next_position() == 7);
    REQUIRE(req.next_index(0) == 7);
    REQUIRE_FALSE(req.has_pending_state());

    req.prepare_step(1, true, 0, false);
    req.apply_step(1, token_list{9}, {}, 0);
    req.finish_step();
    REQUIRE(req.complete_pages() == 1);
    const token_list second_page{5, 6, 7, 8};
    REQUIRE((req.head_digest() == pulsarlm::make_digest(prompt_digests.front(), second_page)));
    REQUIRE(req.views()[0].indices == std::vector<std::size_t>{0, 1});
    REQUIRE(req.tail_tokens().empty());
    REQUIRE(req.pending_token() == 9);
}

TEST_CASE("completing a page leaves the state of a recurrent layer pending until the store has it") {
    started_request started({1, 2, 3, 4, 5}, std::nullopt, make_layout(4, true));
    request_state& req = started.req;
    req.prepare_step(4, false, 0, false);
    req.apply_step(4, {}, {}, 0);
    req.finish_step();
    REQUIRE_FALSE(req.has_pending_state());

    REQUIRE(req.complete_pages() == 1);
    REQUIRE(req.has_pending_state());
    REQUIRE(req.views()[0].pages.size() == 1);
    REQUIRE(req.views()[1].pages.empty());

    req.store_head_state();
    REQUIRE_FALSE(req.has_pending_state());
}

// Each step forwards a page, whose state the store then takes; layer 1 is recurrent, with one state slot.
TEST_CASE("a stored head state line leaves a place that a step filling a page of its parity takes a line for") {
    started_request started(token_list(17, 1), std::nullopt, make_layout(4, true));
    request_state& req = started.req;
    req.add_lines(1, std::vector<std::uint32_t>{10, 11, 12});
    auto forward_page_and_store = [&req] {
        req.prepare_step(4, false, 0, false);
        req.apply_step(4, {}, {}, 0);
        REQUIRE(req.complete_pages() == 1);
        req.finish_step();
        req.store_head_state();
        return req.release_head_state_line(1);
    };

    REQUIRE(forward_page_and_store() == 11);
    REQUIRE(req.views()[1].lines == std::vector<std::uint32_t>{10, pulsarlm::layer_view::no_line, 12});
    REQUIRE(req.views()[1].count_held_lines() == 2);
    REQUIRE(req.missing_lines(std::vector<std::size_t>{4, 4})[1] == 0);
    REQUIRE(req.missing_lines(std::vector<std::size_t>{8, 8})[1] == 1);
    REQUIRE(forward_page_and_store() == 12);
    REQUIRE(req.missing_lines(std::vector<std::size_t>{4, 4})[1] == 1);
    req.add_step_lines(std::vector<std::size_t>{4, 4}, std::vector<std::vector<std::uint32_t>>{{}, {20}});
    REQUIRE(req.views()[1].lines == std::vector<std::uint32_t>{10, 20, pulsarlm::layer_view::no_line});

    REQUIRE(forward_page_and_store() == 20);
    REQUIRE(req.missing_lines(std::vector<std::size_t>{4, 4})[1] == 1);
    req.add_step_lines(std::vector<std::size_t>{4, 4}, std::vector<std::vector<std::uint32_t>>{{}, {21}});
    REQUIRE(req.views()[1].lines == std::vector<std::uint32_t>{10, pulsarlm::layer_view::no_line, 21});
}

TEST_CASE("a verify step forwards the tokens it accepts and counts the drafts it rejects") {
    started_request started({1, 2, 3}, std::nullopt, make_layout(8, false));
    request_state& req = started.req;
    REQUIRE(req.reads_last_hidden());
    req.prepare_step(3, true, 2, true);
    REQUIRE_FALSE(req.reads_last_hidden());
    REQUIRE(req.prepared_draft_count() == 2);
    req.apply_step(3, token_list{30}, token_list{31, 32}, 0);
    req.finish_step();
    REQUIRE(to_list(req.draft_tokens()) == token_list{31, 32});
    REQUIRE(req.rejected_token_count() == 0);

    req.prepare_verify_step(2);
    REQUIRE(req.prepared_position_bound() == 6);
    REQUIRE(req.prepared_sample_count() == 4);
    REQUIRE(req.history_position() == 3);
    REQUIRE(req.next_draw().count == 1);

    req.apply_step(2, token_list{31, 40}, token_list{41, 42}, 1);
    req.record_verified_drafts(2, 1);
    req.finish_step();
    REQUIRE(to_list(req.tail_tokens()) == token_list{1, 2, 3, 30, 31});
    REQUIRE(req.pending_token() == 40);
    REQUIRE(to_list(req.draft_tokens()) == token_list{41, 42});
    REQUIRE(req.rejected_token_count() == 1);
    REQUIRE_FALSE(req.reads_last_hidden());
    REQUIRE(req.prepared_position_bound() == 5);
    REQUIRE(req.prepared_sample_count() == 3);
    REQUIRE(req.next_draw().count == 3);
    REQUIRE(req.drafted_token_count() == 2);
    REQUIRE(req.accepted_draft_count() == 1);
}

TEST_CASE("a recorded step counts as a prefill step until the prefill ends") {
    started_request started({1, 2, 3, 4, 5}, std::nullopt, make_layout(4, false));
    request_state& req = started.req;
    REQUIRE_FALSE(req.has_ended_prefill());

    req.record_step(std::chrono::milliseconds(3));
    req.record_step(std::chrono::milliseconds(4));
    req.end_prefill();
    req.record_step(std::chrono::milliseconds(5));

    REQUIRE(req.has_ended_prefill());
    REQUIRE(req.prefill_step_count() == 2);
    REQUIRE(req.prefill_step_duration() == std::chrono::milliseconds(7));
    REQUIRE(req.step_duration() == std::chrono::milliseconds(12));
}

TEST_CASE("a verify step prepared before the previous one is written back counts both as accepting every token") {
    started_request started({1, 2, 3}, std::nullopt, make_layout(8, false));
    request_state& req = started.req;
    req.prepare_step(3, true, 2, true);
    req.apply_step(3, token_list{30}, token_list{31, 32}, 0);
    req.finish_step();
    req.prepare_verify_step(2);
    req.prepare_verify_step(2);
    REQUIRE(req.prepared_position_bound() == 9);
    REQUIRE(req.prepared_sample_count() == 7);

    req.apply_step(3, token_list{31, 32, 33}, token_list{34, 35}, 0);
    req.finish_step();
    REQUIRE(to_list(req.tail_tokens()) == token_list{1, 2, 3, 30, 31, 32});
    REQUIRE(req.pending_token() == 33);
    REQUIRE(req.prepared_position_bound() == 9);

    req.apply_step(1, token_list{50}, {}, 2);
    req.finish_step();
    REQUIRE(to_list(req.tail_tokens()) == token_list{1, 2, 3, 30, 31, 32, 33});
    REQUIRE(req.pending_token() == 50);
    REQUIRE(req.rejected_token_count() == 2);
    REQUIRE(req.prepared_position_bound() == 7);
    REQUIRE(req.prepared_sample_count() == 5);
    REQUIRE(req.delivered_token_count() == 5);
}

TEST_CASE("a step prepared before the stop token's write-back leaves the state of a step-by-step run") {
    started_request started({1, 2}, std::nullopt, make_layout(4, false));
    request_state& req = started.req;
    req.prepare_step(2, true, 0, false);
    req.prepare_step(1, true, 0, false);
    REQUIRE(req.prepared_sample_count() == 2);
    REQUIRE(req.unfinished_step_count() == 2);

    req.apply_step(2, token_list{99}, {}, 0);
    req.reach_end();
    req.finish_step();
    REQUIRE(req.discards_steps());
    req.finish_step();
    REQUIRE(to_list(req.tail_tokens()) == token_list{1, 2});
    REQUIRE(req.pending_token() == 99);
    REQUIRE(req.delivered_token_count() == 1);
    REQUIRE(req.next_draw().count == 1);
    REQUIRE(req.unfinished_step_count() == 0);
    REQUIRE(req.to_state().tail == token_list{1, 2});
}

TEST_CASE("a request state cancelled in no prepared step exits as cancelled and keeps the id of its lease") {
    started_request started({1, 2}, std::nullopt, make_layout(4, false));
    request_state& req = started.req;
    REQUIRE_FALSE(req.has_exit_cause());
    REQUIRE_FALSE(req.has_exited());
    REQUIRE_FALSE(req.reason_of_exit().has_value());
    REQUIRE(req.lease_id() == lease_id);

    req.cancel();
    REQUIRE(req.has_exit_cause());
    REQUIRE_FALSE(req.discards_steps());
    req.exit();
    REQUIRE(req.has_exited());
    REQUIRE((req.reason_of_exit() == request_state::exit_reason::cancelled));
}

TEST_CASE("a request state that reached its end exits as finished whether it ran out of lines or was cancelled") {
    started_request started({1, 2}, std::nullopt, make_layout(4, false));
    request_state& req = started.req;
    req.prepare_step(2, true, 0, false);
    req.cancel();
    req.run_out_of_lines();
    req.apply_step(2, token_list{9}, {}, 0);
    req.reach_end();
    req.finish_step();
    REQUIRE(req.discards_steps());
    req.exit();
    REQUIRE((req.reason_of_exit() == request_state::exit_reason::finished));
}

TEST_CASE("a request state that ran out of lines exits with oom even when cancelled") {
    started_request started({1, 2}, std::nullopt, make_layout(4, false));
    request_state& req = started.req;
    req.cancel();
    req.run_out_of_lines();
    req.exit();
    REQUIRE((req.reason_of_exit() == request_state::exit_reason::oom));
}

TEST_CASE("a request state whose store write failed exits with oom even when it reached its end") {
    started_request started({1, 2}, std::nullopt, make_layout(4, false));
    request_state& req = started.req;
    req.reach_end();
    req.fail_store_write();
    REQUIRE(req.has_failed_store_write());
    req.exit();
    REQUIRE((req.reason_of_exit() == request_state::exit_reason::oom));
}

}  // namespace
