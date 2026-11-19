#include "pulsarlm/engine/sequence.hpp"
#include "pulsarlm/digest.hpp"
#include "pulsarlm/engine/event_loop.hpp"
#include "pulsarlm/engine/page_store.hpp"
#include "pulsarlm/engine/request.hpp"
#include "pulsarlm/engine/request_callbacks.hpp"
#include "pulsarlm/engine/state.hpp"

#include <ATen/ATen.h>

#undef CHECK
#include <catch2/catch_test_macros.hpp>

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
using pulsarlm::sequence;
using pulsarlm::state;
using pulsarlm::task;
using pulsarlm::view_layout;

using token_list = std::vector<std::int64_t>;

constexpr digest root{.hi = 1, .lo = 2};
constexpr state::lease_id_type lease_id = 7;

using lease_list = std::vector<state::lease_id_type>;

class recording_store final : public pulsarlm::page_store {
    std::shared_ptr<lease_list> released_leases;

  public:
    explicit recording_store(std::shared_ptr<lease_list> released_leases)
    : released_leases(std::move(released_leases)) {}

    task<std::vector<bool>> read_pages(
        std::span<const digest> hashes,
        std::vector<pulsarlm::sharded_page>&&,
        std::span<const cudaStream_t>
    ) override {
        co_return std::vector<bool>(hashes.size(), false);
    }

    task<std::expected<void, page_error>> write_pages(
        std::vector<digest>&&,
        std::vector<pulsarlm::sharded_page>&&,
        std::vector<std::pair<state::lease_id_type, page_index_type>>&&,
        std::span<const cudaStream_t>
    ) override {
        co_return std::expected<void, page_error>{};
    }

    task<std::expected<std::pair<std::shared_ptr<const state>, state::lease_id_type>, page_error>>
    read_state(std::span<const std::int64_t>, std::span<const digest>) override {
        co_return std::unexpected(page_error::miss);
    }

    task<void> write_state(state::lease_id_type, std::shared_ptr<const state>&&) override {
        co_return;
    }

    void release(state::lease_id_type released_lease) noexcept override {
        released_leases->push_back(released_lease);
    }
};

class silent_callbacks final : public pulsarlm::request_callbacks {
  public:
    void on_token(std::int64_t) noexcept override {}

    void on_cycle_end(pulsarlm::cycle_end&&) noexcept override {}

    void on_end(std::expected<pulsarlm::request_result, pulsarlm::request_error>&&) noexcept override {}

    bool cancelled() const noexcept override {
        return false;
    }
};

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

// A sequence of a request that resumes the empty state, with what it refers to.
class started_sequence {
    view_layout layout;
    std::vector<at::Device> devices{at::Device(at::kCUDA, 0)};

  public:
    std::shared_ptr<lease_list> released_leases = std::make_shared<lease_list>();

  private:
    recording_store store;
    pulsarlm::result_writer writer;

  public:
    sequence seq;

    started_sequence(const token_list& tokens, std::optional<std::size_t> max_tokens, view_layout&& layout)
    : layout(std::move(layout)),
      store(released_leases),
      writer(std::make_unique<silent_callbacks>()),
      seq(pulsarlm::request{.tokens = tokens, .max_tokens = max_tokens},
          std::make_shared<const state>(),
          store,
          lease_id,
          pulsarlm::make_page_digests(root, tokens, this->layout.page_size),
          root,
          this->layout,
          writer,
          devices) {}
};

token_list to_list(std::span<const std::int64_t> tokens) {
    return token_list(tokens.begin(), tokens.end());
}

TEST_CASE("a prefill step and a decode step move their tokens to the tail and apply their samples") {
    const token_list prompt{10, 11, 12, 13, 14, 15};
    started_sequence started(prompt, 3, make_layout(4, false));
    sequence& seq = started.seq;
    REQUIRE(to_list(seq.pending_token_list()) == prompt);
    REQUIRE(seq.history_position() == 0);
    REQUIRE(seq.prompt_token_count() == 6);

    seq.launch_step(6, 0, 0, 0);
    REQUIRE(to_list(seq.tail_tokens()) == prompt);
    REQUIRE(to_list(seq.pending_token_list()) == token_list{sequence::awaited_token});
    REQUIRE(seq.awaited_sample_row() == 0);
    REQUIRE(seq.history_position() == 6);
    REQUIRE(seq.generated_token_count() == 1);
    REQUIRE(seq.next_draw().count == 1);
    REQUIRE(seq.launched_step_count() == 1);
    REQUIRE_FALSE(seq.has_launched_last_step());

    seq.apply_sample(6, 20);
    seq.finish_step();
    REQUIRE(to_list(seq.pending_token_list()) == token_list{20});
    REQUIRE(seq.launched_step_count() == 0);

    seq.launch_step(1, 5, 0, 0);
    REQUIRE(to_list(seq.tail_tokens()) == token_list{10, 11, 12, 13, 14, 15, 20});
    REQUIRE(to_list(seq.pending_token_list()) == token_list{sequence::awaited_token});
    REQUIRE(seq.awaited_sample_row() == 5);
    REQUIRE(seq.generated_token_count() == 2);
    REQUIRE_FALSE(seq.has_launched_last_step());

    seq.apply_sample(7, 21);
    seq.finish_step();
    REQUIRE(to_list(seq.pending_token_list()) == token_list{21});

    seq.launch_step(1, 0, 0, 0);
    REQUIRE(seq.generated_token_count() == 3);
    REQUIRE_FALSE(seq.may_sample());
    REQUIRE(seq.has_launched_last_step());
}

TEST_CASE("a prefill step that samples no token leaves no awaited token") {
    started_sequence started({1, 2, 3, 4, 5}, std::nullopt, make_layout(4, false));
    sequence& seq = started.seq;

    seq.launch_step(4, std::nullopt, 0, 0);
    REQUIRE(to_list(seq.tail_tokens()) == token_list{1, 2, 3, 4});
    REQUIRE(to_list(seq.pending_token_list()) == token_list{5});
    REQUIRE(seq.generated_token_count() == 0);
    REQUIRE(seq.next_draw().count == 0);
    REQUIRE_FALSE(seq.has_launched_last_step());
}

TEST_CASE("completing pages moves the full pages of the tail to the views") {
    const token_list prompt{1, 2, 3, 4, 5, 6, 7};
    started_sequence started(prompt, std::nullopt, make_layout(4, false));
    sequence& seq = started.seq;
    seq.launch_step(7, 0, 0, 0);
    REQUIRE(seq.complete_pages(3) == 0);
    REQUIRE(seq.complete_pages(7) == 1);

    const std::vector<digest> prompt_digests = pulsarlm::make_page_digests(root, prompt, 4);
    REQUIRE(seq.history_page_count() == 1);
    REQUIRE((seq.head_digest() == prompt_digests.front()));
    REQUIRE((seq.views()[0].pages == prompt_digests));
    REQUIRE(seq.views()[0].indices == std::vector<std::size_t>{0});
    REQUIRE(to_list(seq.tail_tokens()) == token_list{5, 6, 7});
    REQUIRE(seq.history_position() == 7);
    REQUIRE(seq.next_position() == 7);
    REQUIRE(seq.next_index(0) == 7);
    REQUIRE_FALSE(seq.has_pending_state());

    seq.apply_sample(7, 8);
    seq.finish_step();
    seq.launch_step(1, 0, 0, 0);
    REQUIRE(seq.complete_pages(8) == 1);
    const token_list second_page{5, 6, 7, 8};
    REQUIRE((seq.head_digest() == pulsarlm::make_digest(prompt_digests.front(), second_page)));
    REQUIRE(seq.views()[0].indices == std::vector<std::size_t>{0, 1});
    REQUIRE(seq.tail_tokens().empty());
    REQUIRE(to_list(seq.pending_token_list()) == token_list{sequence::awaited_token});
}

TEST_CASE("completing a page leaves the state of a recurrent layer pending until the store has it") {
    started_sequence started({1, 2, 3, 4, 5}, std::nullopt, make_layout(4, true));
    sequence& seq = started.seq;
    seq.launch_step(4, std::nullopt, 0, 0);
    REQUIRE_FALSE(seq.has_pending_state());

    REQUIRE(seq.complete_pages(4) == 1);
    REQUIRE(seq.has_pending_state());
    REQUIRE(seq.views()[0].pages.size() == 1);
    REQUIRE(seq.views()[1].pages.empty());

    seq.store_head_state();
    REQUIRE_FALSE(seq.has_pending_state());
}

TEST_CASE("a verify step inserts the tokens it accepts and counts the drafts it rejects") {
    started_sequence started({1, 2, 3}, std::nullopt, make_layout(8, false));
    sequence& seq = started.seq;
    seq.launch_step(3, 0, 0, 2);
    REQUIRE(to_list(seq.draft_tokens()) == token_list{sequence::awaited_token, sequence::awaited_token});
    seq.apply_sample(3, 30);
    const token_list drafts{31, 32};
    seq.set_drafts(drafts);
    seq.finish_step();
    REQUIRE(seq.rejected_token_count() == 0);

    seq.launch_verify_step(1, 2);
    REQUIRE(to_list(seq.tail_tokens()) == token_list{1, 2, 3, 30});
    REQUIRE(to_list(seq.pending_token_list()) == token_list{sequence::awaited_token});
    REQUIRE(seq.rejected_token_count() == 2);
    REQUIRE(seq.generated_token_count() == 1);
    REQUIRE(seq.next_draw().count == 1);

    const token_list accepted_tokens{31, 40};
    seq.apply_accepted_tokens(3, accepted_tokens, 3);
    seq.finish_step();
    REQUIRE(to_list(seq.tail_tokens()) == token_list{1, 2, 3, 30, 31});
    REQUIRE(to_list(seq.pending_token_list()) == token_list{40});
    REQUIRE(seq.rejected_token_count() == 1);
    REQUIRE(seq.end_of_draft_rows() == 6);
    REQUIRE(seq.unapplied_position_shift() == 0);
    REQUIRE(seq.generated_token_count() == 3);
    REQUIRE(seq.next_draw().count == 3);
}

TEST_CASE("a verify step written back before a later one leaves its accepted count for that step") {
    started_sequence started({1, 2, 3}, std::nullopt, make_layout(8, false));
    sequence& seq = started.seq;
    seq.launch_step(3, 0, 0, 2);
    seq.apply_sample(3, 30);
    const token_list drafts{31, 32};
    seq.set_drafts(drafts);
    seq.finish_step();
    seq.launch_verify_step(1, 2);
    seq.launch_verify_step(1, 2);
    REQUIRE(to_list(seq.tail_tokens()) == token_list{1, 2, 3, 30, sequence::awaited_token});

    const token_list accepted_tokens{31, 32, 33};
    seq.apply_accepted_tokens(3, accepted_tokens, 3);
    seq.finish_step();
    REQUIRE(to_list(seq.tail_tokens()) == token_list{1, 2, 3, 30, 31, 32, 33});
    REQUIRE(seq.unapplied_position_shift() == 2);
    REQUIRE(seq.rejected_token_count() == 2);

    seq.exit(sequence::exit_reason::finished);
    seq.withdraw_latest_verify_step();
    REQUIRE(to_list(seq.tail_tokens()) == token_list{1, 2, 3, 30, 31, 32});
    REQUIRE(to_list(seq.pending_token_list()) == token_list{33});
    REQUIRE(seq.draft_tokens().empty());
    REQUIRE(seq.unapplied_position_shift() == 0);
}

TEST_CASE("a stop token withdraws the step launched after the one that sampled it") {
    started_sequence started({1, 2}, std::nullopt, make_layout(4, false));
    sequence& seq = started.seq;
    seq.launch_step(2, 0, 0, 0);
    seq.launch_step(1, 0, 0, 0);
    REQUIRE(to_list(seq.tail_tokens()) == token_list{1, 2, sequence::awaited_token});
    REQUIRE(seq.generated_token_count() == 2);
    REQUIRE(seq.launched_step_count() == 2);

    seq.apply_sample(2, 99);
    seq.exit(sequence::exit_reason::finished);
    seq.withdraw_latest_step();
    seq.finish_step();
    REQUIRE(to_list(seq.tail_tokens()) == token_list{1, 2});
    REQUIRE(to_list(seq.pending_token_list()) == token_list{99});
    REQUIRE(seq.generated_token_count() == 1);
    REQUIRE(seq.next_draw().count == 1);
    REQUIRE(seq.launched_step_count() == 1);
    REQUIRE(seq.to_state().tail == token_list{1, 2});
}

TEST_CASE("a sequence exits with the last reason given and releases its lease when destroyed") {
    std::shared_ptr<lease_list> released_leases;
    {
        started_sequence started({1, 2}, std::nullopt, make_layout(4, false));
        sequence& seq = started.seq;
        released_leases = started.released_leases;
        REQUIRE_FALSE(seq.has_exited());
        REQUIRE_FALSE(seq.reason_of_exit().has_value());
        REQUIRE(seq.lease_id() == lease_id);

        seq.exit(sequence::exit_reason::cancelled);
        REQUIRE(seq.has_exited());
        REQUIRE((seq.reason_of_exit() == sequence::exit_reason::cancelled));

        seq.exit(sequence::exit_reason::oom);
        REQUIRE((seq.reason_of_exit() == sequence::exit_reason::oom));
        REQUIRE(released_leases->empty());
    }
    REQUIRE(*released_leases == lease_list{lease_id});
}

}  // namespace
