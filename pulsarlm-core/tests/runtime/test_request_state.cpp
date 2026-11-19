#include "pulsarlm/engine/request_state.hpp"
#include "pulsarlm/digest.hpp"
#include "pulsarlm/engine/event_loop.hpp"
#include "pulsarlm/engine/page_store.hpp"
#include "pulsarlm/engine/request.hpp"
#include "pulsarlm/engine/state.hpp"

#include <ATen/ATen.h>

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
    std::vector<at::Device> devices{at::Device(at::kCUDA, 0)};

  public:
    request_state req;

    started_request(const token_list& tokens, std::optional<std::size_t> max_tokens, view_layout&& layout)
    : layout(std::move(layout)),
      req(pulsarlm::request{.tokens = tokens, .max_tokens = max_tokens},
          std::make_shared<const state>(),
          lease_id,
          pulsarlm::make_page_digests(root, tokens, this->layout.page_size),
          root,
          this->layout,
          devices) {}
};

token_list to_list(std::span<const std::int64_t> tokens) {
    return token_list(tokens.begin(), tokens.end());
}

TEST_CASE("a prefill step and a decode step move their tokens to the tail and apply their samples") {
    const token_list prompt{10, 11, 12, 13, 14, 15};
    started_request started(prompt, 3, make_layout(4, false));
    request_state& req = started.req;
    REQUIRE(to_list(req.pending_token_list()) == prompt);
    REQUIRE(req.history_position() == 0);
    REQUIRE(req.prompt_token_count() == 6);

    req.launch_step(6, 0, 0, 0);
    REQUIRE(to_list(req.tail_tokens()) == prompt);
    REQUIRE(to_list(req.pending_token_list()) == token_list{request_state::awaited_token});
    REQUIRE(req.awaited_sample_row() == 0);
    REQUIRE(req.history_position() == 6);
    REQUIRE(req.generated_token_count() == 1);
    REQUIRE(req.next_draw().count == 1);
    REQUIRE(req.launched_step_count() == 1);
    REQUIRE_FALSE(req.has_launched_last_step());

    req.apply_sample(6, 20);
    req.finish_step();
    REQUIRE(to_list(req.pending_token_list()) == token_list{20});
    REQUIRE(req.launched_step_count() == 0);

    req.launch_step(1, 5, 0, 0);
    REQUIRE(to_list(req.tail_tokens()) == token_list{10, 11, 12, 13, 14, 15, 20});
    REQUIRE(to_list(req.pending_token_list()) == token_list{request_state::awaited_token});
    REQUIRE(req.awaited_sample_row() == 5);
    REQUIRE(req.generated_token_count() == 2);
    REQUIRE_FALSE(req.has_launched_last_step());

    req.apply_sample(7, 21);
    req.finish_step();
    REQUIRE(to_list(req.pending_token_list()) == token_list{21});

    req.launch_step(1, 0, 0, 0);
    REQUIRE(req.generated_token_count() == 3);
    REQUIRE(req.delivered_token_count() == 2);
    REQUIRE_FALSE(req.may_sample());
    REQUIRE(req.has_launched_last_step());
}

TEST_CASE("a prefill step that samples no token leaves no awaited token") {
    started_request started({1, 2, 3, 4, 5}, std::nullopt, make_layout(4, false));
    request_state& req = started.req;

    req.launch_step(4, std::nullopt, 0, 0);
    REQUIRE(to_list(req.tail_tokens()) == token_list{1, 2, 3, 4});
    REQUIRE(to_list(req.pending_token_list()) == token_list{5});
    REQUIRE(req.generated_token_count() == 0);
    REQUIRE(req.next_draw().count == 0);
    REQUIRE_FALSE(req.has_launched_last_step());
}

TEST_CASE("completing pages moves the full pages of the tail to the views") {
    const token_list prompt{1, 2, 3, 4, 5, 6, 7};
    started_request started(prompt, std::nullopt, make_layout(4, false));
    request_state& req = started.req;
    req.launch_step(7, 0, 0, 0);
    REQUIRE(req.complete_pages(3) == 0);
    REQUIRE(req.complete_pages(7) == 1);

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

    req.apply_sample(7, 8);
    req.finish_step();
    req.launch_step(1, 0, 0, 0);
    REQUIRE(req.complete_pages(8) == 1);
    const token_list second_page{5, 6, 7, 8};
    REQUIRE((req.head_digest() == pulsarlm::make_digest(prompt_digests.front(), second_page)));
    REQUIRE(req.views()[0].indices == std::vector<std::size_t>{0, 1});
    REQUIRE(req.tail_tokens().empty());
    REQUIRE(to_list(req.pending_token_list()) == token_list{request_state::awaited_token});
}

TEST_CASE("completing a page leaves the state of a recurrent layer pending until the store has it") {
    started_request started({1, 2, 3, 4, 5}, std::nullopt, make_layout(4, true));
    request_state& req = started.req;
    req.launch_step(4, std::nullopt, 0, 0);
    REQUIRE_FALSE(req.has_pending_state());

    REQUIRE(req.complete_pages(4) == 1);
    REQUIRE(req.has_pending_state());
    REQUIRE(req.views()[0].pages.size() == 1);
    REQUIRE(req.views()[1].pages.empty());

    req.store_head_state();
    REQUIRE_FALSE(req.has_pending_state());
}

TEST_CASE("a verify step inserts the tokens it accepts and counts the drafts it rejects") {
    started_request started({1, 2, 3}, std::nullopt, make_layout(8, false));
    request_state& req = started.req;
    req.launch_step(3, 0, 0, 2);
    REQUIRE(to_list(req.draft_tokens()) == token_list{request_state::awaited_token, request_state::awaited_token});
    req.apply_sample(3, 30);
    const token_list drafts{31, 32};
    req.set_drafts(drafts);
    req.finish_step();
    REQUIRE(req.rejected_token_count() == 0);

    req.launch_verify_step(1, 2);
    REQUIRE(to_list(req.tail_tokens()) == token_list{1, 2, 3, 30});
    REQUIRE(to_list(req.pending_token_list()) == token_list{request_state::awaited_token});
    REQUIRE(req.rejected_token_count() == 2);
    REQUIRE(req.generated_token_count() == 1);
    REQUIRE(req.next_draw().count == 1);

    const token_list accepted_tokens{31, 40};
    req.apply_accepted_tokens(3, accepted_tokens, 3);
    req.finish_step();
    REQUIRE(to_list(req.tail_tokens()) == token_list{1, 2, 3, 30, 31});
    REQUIRE(to_list(req.pending_token_list()) == token_list{40});
    REQUIRE(req.rejected_token_count() == 1);
    REQUIRE(req.end_of_draft_rows() == 6);
    REQUIRE(req.unapplied_position_shift() == 0);
    REQUIRE(req.generated_token_count() == 3);
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

TEST_CASE("a verify step written back before a later one leaves its accepted count for that step") {
    started_request started({1, 2, 3}, std::nullopt, make_layout(8, false));
    request_state& req = started.req;
    req.launch_step(3, 0, 0, 2);
    req.apply_sample(3, 30);
    const token_list drafts{31, 32};
    req.set_drafts(drafts);
    req.finish_step();
    req.launch_verify_step(1, 2);
    req.launch_verify_step(1, 2);
    REQUIRE(to_list(req.tail_tokens()) == token_list{1, 2, 3, 30, request_state::awaited_token});

    const token_list accepted_tokens{31, 32, 33};
    req.apply_accepted_tokens(3, accepted_tokens, 3);
    req.finish_step();
    REQUIRE(to_list(req.tail_tokens()) == token_list{1, 2, 3, 30, 31, 32, 33});
    REQUIRE(req.unapplied_position_shift() == 2);
    REQUIRE(req.rejected_token_count() == 2);

    req.exit(request_state::exit_reason::finished);
    req.withdraw_latest_verify_step();
    REQUIRE(to_list(req.tail_tokens()) == token_list{1, 2, 3, 30, 31, 32});
    REQUIRE(to_list(req.pending_token_list()) == token_list{33});
    REQUIRE(req.draft_tokens().empty());
    REQUIRE(req.unapplied_position_shift() == 0);
}

TEST_CASE("a stop token withdraws the step launched after the one that sampled it") {
    started_request started({1, 2}, std::nullopt, make_layout(4, false));
    request_state& req = started.req;
    req.launch_step(2, 0, 0, 0);
    req.launch_step(1, 0, 0, 0);
    REQUIRE(to_list(req.tail_tokens()) == token_list{1, 2, request_state::awaited_token});
    REQUIRE(req.generated_token_count() == 2);
    REQUIRE(req.launched_step_count() == 2);

    req.apply_sample(2, 99);
    req.exit(request_state::exit_reason::finished);
    req.withdraw_latest_step();
    req.finish_step();
    REQUIRE(to_list(req.tail_tokens()) == token_list{1, 2});
    REQUIRE(to_list(req.pending_token_list()) == token_list{99});
    REQUIRE(req.generated_token_count() == 1);
    REQUIRE(req.next_draw().count == 1);
    REQUIRE(req.launched_step_count() == 1);
    REQUIRE(req.to_state().tail == token_list{1, 2});
}

TEST_CASE("a request state exits with the last reason given and keeps the id of its lease") {
    started_request started({1, 2}, std::nullopt, make_layout(4, false));
    request_state& req = started.req;
    REQUIRE_FALSE(req.has_exited());
    REQUIRE_FALSE(req.reason_of_exit().has_value());
    REQUIRE(req.lease_id() == lease_id);

    req.exit(request_state::exit_reason::cancelled);
    REQUIRE(req.has_exited());
    REQUIRE((req.reason_of_exit() == request_state::exit_reason::cancelled));

    req.exit(request_state::exit_reason::oom);
    REQUIRE((req.reason_of_exit() == request_state::exit_reason::oom));
}

}  // namespace
