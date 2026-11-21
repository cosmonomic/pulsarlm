#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "pulsarlm/digest.hpp"
#include "pulsarlm/engine/eviction.hpp"
#include "pulsarlm/engine/page_store.hpp"
#include "pulsarlm/engine/pending_writes.hpp"
#include "pulsarlm/engine/request.hpp"
#include "pulsarlm/engine/sampling.hpp"
#include "pulsarlm/engine/state.hpp"

namespace pulsarlm {

// What a request state needs to know about each KV layer of its executor.
struct view_layout {
    struct layer {
        bool is_recurrent;
        bool mirrors_held_pages;
        std::optional<std::size_t> sliding_window;
        std::size_t state_slots;
        std::size_t mass_heads;
    };
    std::vector<layer> layers;
    std::size_t page_size;
};

// One KV layer's view of a request. A layer with pages has one line per view page: its full pages, each with its index
// in the history, then the pages of tail and of the tokens after it. A layer that measures mass keeps, per line,
// mass_heads mass_sum values and one weight_sum (add_step_mass), as of the last step that measured them, 0 for a page
// never measured, and the moments of the pages it sampled (add_sampled_page). A recurrent layer has no pages: its
// lines are its state slots, then the lines holding the state at the end of its even and of its odd pages (state_rows),
// either of the last two no_line once the store has the state it held and until a step's prologue gives it a line.
class layer_view {
  public:
    // A place in lines that holds no line; the request table holds -1 there.
    static constexpr std::uint32_t no_line = std::numeric_limits<std::uint32_t>::max();

    std::vector<digest> pages;
    std::vector<std::size_t> indices;
    std::vector<std::uint32_t> lines;
    // The leading pages and lines of a sliding layer's view that the request table no longer holds: they passed the
    // window of every query after the prepared steps (request_state::pass_pages), and leave the view once their pages
    // are complete (request_state::drop_passed_pages). A view index is the table column passed_count after it.
    std::size_t passed_count = 0;
    std::size_t mass_heads = 0;
    std::vector<float> mass_sum;
    std::vector<float> weight_sum;
    std::vector<moments> density_moments;
    moments score_moments;

    // Drops the full pages at the ascending view indices and returns their lines.
    std::vector<std::uint32_t> remove(std::span<const std::size_t> view_indices);
    void append_lines(std::span<const std::uint32_t> added);
    // The lines other than no_line.
    std::size_t count_held_lines() const noexcept;
};

// A request being served, whose state each phase of a step changes apart. A step's prologue hands out the request's
// tokens by counts alone (prepare_step, prepare_verify_step) and gives it lines; its write-back applies what the step
// computed: the tokens it forwarded join tail and the pages they fill join every layer's view, the last of them head.
class request_state {
  public:
    enum class exit_reason { finished, cancelled, oom, context_exceeded };

  private:
    state::lease_id_type held_lease_id;
    sampling_params sampling;
    std::uint64_t seed;
    std::optional<std::size_t> max_tokens;
    // The request's tokens, each at its history position.
    std::vector<std::int64_t> prompt;
    // The tokens the prepared steps sampled, each verify step counted as accepting every token it verifies, and the
    // sampled tokens the steps written back applied.
    std::size_t prepared_samples = 0;
    std::size_t delivered_tokens = 0;
    // The prompt tokens before the first one no prepared step forwards: those of the resumed state's full pages and
    // those the prepared steps forwarded.
    std::size_t prepared_prompt_tokens = 0;
    // The digest the page chain starts from.
    digest root;
    std::optional<digest> head;
    // The full pages of the history, head the last.
    std::size_t history_pages = 0;
    // The tokens after head that the steps written back forwarded.
    std::vector<std::int64_t> tail;
    // The token the last step written back sampled, when no step written back forwarded it.
    std::optional<std::int64_t> pending_sample;
    // The digests of the full pages of the request's tokens after head, in order; the first next_prompt_digest of them
    // joined the history.
    std::vector<digest> prompt_digests;
    std::size_t next_prompt_digest = 0;
    const view_layout& layout;
    // One per KV layer.
    std::vector<layer_view> layers;
    // The ascending history indices of the full pages some layer with pages, other than a mirroring one, holds in its
    // request table view, its passed pages aside. A full page's RoPE position is its rank here times the page size, in
    // every layer; the held pages after the full ones follow the last (find_held_incomplete_pages).
    std::vector<std::size_t> held_pages;
    // Whether the recurrent layers' head_state_line hold the state at head, which the store does not have yet.
    bool state_pending = false;
    // The writes of the request's pages to the store not waited for; shared with the tasks writing them.
    std::shared_ptr<pending_writes> page_writes = std::make_shared<pending_writes>();
    // Set by exit only, once no step of the request is unfinished.
    std::optional<exit_reason> end_reason;
    // Whether a step written back sampled a stop token or reached the end that max_tokens or the prompt sets.
    bool has_reached_end = false;
    // Whether a write of the request's pages or state to the store failed.
    bool has_failed_write = false;
    // Whether a line allocation for a step of the request failed.
    bool is_out_of_lines = false;
    // Whether the request's view would grow past the columns of its page tables.
    bool has_exceeded_context = false;
    bool is_cancelled = false;
    // The steps of the request prepared, and those of them written back.
    std::size_t prepared_step_count = 0;
    std::size_t written_back_step_count = 0;
    // Whether a prepared step forwards the request's last tokens: it samples the last token max_tokens allows, known
    // from exact prepared counts (has_exact_prepared_counts), or it forwards the last prompt tokens and does not
    // sample.
    bool last_step_prepared = false;
    // The prepared steps through the latest that is a verify step or was prepared while a verify step was unfinished.
    // Until these are written back, the prepared counts take each such verify step as accepting every token it
    // verifies.
    std::size_t bounded_count_steps = 0;
    // Per unfinished step, oldest first: the fewest and the most tokens it forwards, a verify step at least its pending
    // token and at most every token it verifies, and the cycle end it may reach.
    using forwarded_count = std::size_t;
    using reachable_cycle_end = std::optional<std::size_t>;
    std::deque<std::tuple<forwarded_count, forwarded_count, reachable_cycle_end>> unfinished_steps;
    // Per recurrent KV layer and page parity, a line that held the state at the end of a page the unfinished step may
    // end its cycle at, taken out of the view by the prologue of the step after it so that its write-back can copy
    // the line while that step writes the state of a later page of the same parity; no_line otherwise.
    std::vector<std::array<std::uint32_t, 2>> detached_head_state_lines;
    // The tokens the latest prepared step drafts, which the next step verifies.
    std::size_t prepared_drafts = 0;
    // The tokens the latest step written back drafted; empty when it drafted none.
    std::vector<std::int64_t> drafts;
    // The tokens the latest step written back forwarded and did not accept (model::batch::rejected_tokens).
    std::size_t rejected_tokens = 0;
    // Whether the draft head's KV layer holds no key for the next step's first token, whose key then reads the
    // last_hidden of the position before it.
    bool next_step_reads_last_hidden = true;
    page_eviction_policy eviction;
    // The history position at the end of the resumed state: its full pages and its tail. No cycle ends at or before
    // it, so that forwarding its tail again never reaches a cycle end that state already passed.
    std::size_t resumed_end_position = 0;
    // The history position up to which the layers' mass_sum and weight_sum hold the queries' mass, and its value at
    // the start, which no step starts before.
    std::size_t mass_end_position = 0;
    std::size_t resumed_mass_end_position = 0;
    // The tokens of the resumed state's full pages, and of those every layer reading them had resident at the start.
    std::size_t cached_count = 0;
    std::size_t device_cached_count = 0;
    std::size_t peak_device_memory = 0;
    // Whether the step that computed the last of the request's tokens was written back.
    bool has_prefill_ended = false;
    // The steps written back before the prefill ended, and their durations summed.
    std::size_t prefill_steps = 0;
    std::chrono::steady_clock::duration prefill_duration{};
    // The durations of every step written back, summed.
    std::chrono::steady_clock::duration total_step_duration{};
    // The drafts the verify steps written back verified, and those of them they accepted.
    std::size_t drafted_tokens = 0;
    std::size_t accepted_drafts = 0;

    // The view index of recurrent KV layer layer's line holding the state at the end of head's page.
    std::size_t head_state_line_index(std::size_t layer) const;
    // Sets held_pages from the layers' indices, the passed ones aside (layer_view::passed_count).
    void update_held();
    // The history indices [first, end) of the pages after the full pages that some layer with pages, other than a
    // mirroring one, holds in its request table view. A mirroring layer may hold lines past end, for the keys of
    // drafts.
    std::pair<std::size_t, std::size_t> find_held_incomplete_pages() const;
    // The view indices of recurrent KV layer layer's lines that the next key_count keys after the prepared steps write
    // the state at the end of a page into and that hold no line, or hold a line that the unfinished step's write-back
    // may copy as the state at the cycle end it may reach (detached_head_state_lines).
    std::vector<std::size_t> find_unheld_state_lines(std::size_t layer, std::size_t key_count) const;

  public:
    // Resumes initial_state, the newest stored state reaching a prefix of r's tokens, leased by lease_id.
    // digests are the digests of the full pages of r's tokens, chained from root. layout outlives the request state.
    request_state(
        request&& r,
        std::shared_ptr<const state>&& initial_state,
        state::lease_id_type lease_id,
        std::vector<digest>&& digests,
        const digest& root,
        const view_layout& layout
    );

    // One per KV layer.
    std::span<const layer_view> views() const noexcept;
    std::span<const std::int64_t> prompt_tokens() const noexcept;
    std::span<const std::int64_t> tail_tokens() const noexcept;
    // The token at the history position, which no step written back forwarded: a prompt token or the last sample.
    std::optional<std::int64_t> pending_token() const noexcept;
    std::span<const std::int64_t> draft_tokens() const noexcept;
    std::size_t prepared_draft_count() const noexcept;
    const sampling_params& sampling_parameters() const noexcept;
    // The draw of the next token sampled after the steps written back, each token sampled before it having taken its
    // own draw of the seed's noise.
    sampling_draw next_draw() const noexcept;
    const std::optional<std::size_t>& token_limit() const noexcept;
    // The tokens the prepared steps sampled, each unfinished verify step counted as accepting every token it verifies.
    std::size_t prepared_sample_count() const noexcept;
    // The sampled tokens that steps written back applied.
    std::size_t delivered_token_count() const noexcept;
    // The history position after the prepared steps, each unfinished verify step counted as accepting every token it
    // verifies.
    std::size_t prepared_position_bound() const noexcept;
    // The history position after the prepared steps, each unfinished verify step counted as accepting only its pending
    // token.
    std::size_t prepared_position_lower_bound() const noexcept;
    std::size_t prepared_prompt_token_count() const noexcept;
    // The tokens the next step can forward: the prompt tokens no prepared step forwards, or else the token the latest
    // prepared step samples.
    std::size_t unprepared_token_count() const noexcept;
    std::size_t prompt_token_count() const noexcept;
    std::size_t cached_token_count() const noexcept;
    std::size_t device_cached_token_count() const noexcept;
    std::size_t device_memory_peak() const noexcept;
    bool has_ended_prefill() const noexcept;
    std::size_t prefill_step_count() const noexcept;
    std::chrono::steady_clock::duration prefill_step_duration() const noexcept;
    // The durations of every step written back, summed.
    std::chrono::steady_clock::duration step_duration() const noexcept;
    std::size_t drafted_token_count() const noexcept;
    std::size_t accepted_draft_count() const noexcept;
    std::size_t history_page_count() const noexcept;
    const std::optional<digest>& head_digest() const noexcept;
    std::span<const std::size_t> held_page_indices() const noexcept;
    // The history pages some layer with pages, other than a mirroring one, holds in its request table view: the full
    // pages of held_page_indices and the held pages after them.
    std::size_t count_held_pages() const noexcept;
    bool has_pending_state() const noexcept;
    const std::shared_ptr<pending_writes>& pending_page_writes() const noexcept;
    bool has_exited() const noexcept;
    std::optional<exit_reason> reason_of_exit() const noexcept;
    // Whether the request exits once no step of it is unfinished: a step written back reached the end, a store write
    // failed, its view would grow past its page tables, it ran out of lines or it was cancelled.
    bool has_exit_cause() const noexcept;
    // Whether the write-back of a later step of the request applies nothing: a step written back reached the end or a
    // store write failed.
    bool discards_steps() const noexcept;
    bool has_failed_store_write() const noexcept;
    // The prepared steps of the request not written back yet.
    std::size_t unfinished_step_count() const noexcept;
    bool has_prepared_last_step() const noexcept;
    std::size_t rejected_token_count() const noexcept;
    bool reads_last_hidden() const noexcept;
    const page_eviction_policy& eviction_policy_in_pages() const noexcept;
    state::lease_id_type lease_id() const noexcept;
    // Recurrent KV layer layer's line holding the state at the end of head's page, the detached one when a prologue
    // detached it (detached_head_state_lines).
    std::uint32_t head_state_line(std::size_t layer) const;
    // The view index of the next token in KV layer layer's view.
    std::size_t next_index(std::size_t layer) const noexcept;
    // The RoPE position of the next token, the same in every layer.
    std::size_t next_position() const noexcept;
    // The full pages of KV layer layer's view wholly before the window of a query at view index query.
    std::size_t passed_pages(std::size_t layer, std::size_t query) const noexcept;
    // The state the request reached: per KV layer, its view's full pages within the window of the next token.
    state to_state() const;
    // Whether the prepared sample count and position bound are exact: every step prepared while a verify step was
    // unfinished, and every verify step, is written back.
    bool has_exact_prepared_counts() const noexcept;
    // Whether a prepared step may sample another token under max_tokens: the prepared counts are not exact, or they
    // leave a token.
    bool may_sample() const noexcept;
    // The tokens of the history up to the next token: its full pages, then tail.
    std::size_t history_position() const noexcept;
    // The history position at which the first cycle ending after position ends: the next multiple of cycle_tokens
    // after position and after the resumed state's end.
    std::size_t find_cycle_end(std::size_t position) const noexcept;
    // The history position at which the eviction cycle ends after the prepared steps: the cycle end after the lower
    // bound (prepared_position_lower_bound), never later than the one the steps reach.
    std::size_t cycle_end_position() const noexcept;
    // The history position from which the cycle's queries add mass: the cycle's last mass_tokens tokens, none before
    // the resumed state's end.
    std::size_t mass_start_position() const noexcept;
    // The weight of the queries of weights.size() tokens from history position first (query_weights), 0 for those
    // before the mass start of the cycle that first lies in.
    void fill_query_weights(std::size_t first, std::span<float> weights) const noexcept;
    // Whether forwarding token_count more tokens after the prepared steps may end the eviction cycle.
    bool ends_cycle(std::size_t token_count) const noexcept;
    // The tokens the cycle can still forward after the prepared steps: no more than its length allows, nor than the
    // request will forward.
    std::size_t cycle_tokens_left() const noexcept;
    // The columns of a page table the lines take: the most lines of a KV layer the request table holds.
    std::size_t table_width() const noexcept;
    // The RoPE position of the first key of view page page in KV layer layer, which has pages.
    std::size_t page_position(std::size_t layer, std::size_t page) const;
    // Per KV layer, the lines the request lacks for the next key_counts[layer] keys after the prepared steps in that
    // layer; in a recurrent layer, for its state slots and for the state at the end of each page those keys fill.
    std::vector<std::size_t> missing_lines(std::span<const std::size_t> key_counts) const;
    // The view indices of the full pages a cycle's end evicts from KV layer layer, which measures mass, ascending.
    std::vector<std::size_t> pages_to_evict(std::size_t layer) const;

    // Prepares a step forwarding the next token_count tokens, which samples a token with samples, drafts draft_count
    // tokens, and with continues_drafting adds the key of the token after its own to the draft head's KV layer.
    void prepare_step(std::size_t token_count, bool samples, std::size_t draft_count, bool continues_drafting);
    // Prepares a step verifying the drafts of the latest prepared step after the next token, which drafts draft_count
    // tokens and adds the key of the token after those it accepts to the draft head's KV layer.
    void prepare_verify_step(std::size_t draft_count);
    // Writes back a step that forwarded forwarded_count tokens from the history position, sampled emitted, drafted
    // step_drafts and forwarded rejected_count tokens it did not accept. The tokens it forwarded are the prompt's,
    // then the pending sample, then those of emitted but the last; the token of emitted it did not forward is the
    // pending sample.
    void apply_step(
        std::size_t forwarded_count,
        std::span<const std::int64_t> emitted,
        std::span<const std::int64_t> step_drafts,
        std::size_t rejected_count
    );
    // Counts the drafts a verify step written back verified, and those of them it accepted.
    void record_verified_drafts(std::size_t drafted_count, std::size_t accepted_count) noexcept;
    // One prepared step of the request was written back.
    void finish_step() noexcept;
    // A step written back sampled a stop token or reached the end that max_tokens or the prompt sets.
    void reach_end() noexcept;
    void fail_store_write() noexcept;
    void run_out_of_lines() noexcept;
    void exceed_context() noexcept;
    void cancel() noexcept;
    // Adds a step written back, which took duration from its launch to its write-back, to the prefill's steps until
    // end_prefill and to the summed duration.
    void record_step(std::chrono::steady_clock::duration duration) noexcept;
    void end_prefill() noexcept;
    // Moves the full pages of tail to every layer with pages, and returns how many. With a recurrent layer, the state
    // at the new head is then pending.
    std::size_t complete_pages();
    // The store has the state at head.
    void store_head_state() noexcept;
    // Appends added to KV layer layer's lines.
    void add_lines(std::size_t layer, std::span<const std::uint32_t> added);
    // Adds added, per KV layer the lines missing_lines(key_counts) counts: a layer with pages appends them, a recurrent
    // layer puts them in the places the keys write the state at the end of a page into, detaching a line held there
    // (find_unheld_state_lines).
    void add_step_lines(std::span<const std::size_t> key_counts, std::span<const std::vector<std::uint32_t>> added);
    // Drops the full pages at the ascending view_indices from KV layer layer's view and returns their lines.
    std::vector<std::uint32_t> remove_pages(std::size_t layer, std::span<const std::size_t> view_indices);
    // Marks the pages of sliding KV layer layer's view wholly before the window of a query at history position
    // position, which no query after the prepared steps precedes, as passed (layer_view::passed_count), and returns
    // how many it newly marked.
    std::size_t pass_pages(std::size_t layer, std::size_t position);
    // Drops the passed pages of KV layer layer's view whose pages are complete and returns their lines.
    std::vector<std::uint32_t> drop_passed_pages(std::size_t layer);
    // Removes every line of every KV layer, detached ones included, and returns them, per layer.
    std::vector<std::vector<std::uint32_t>> take_lines();
    // Takes recurrent KV layer layer's line holding the state at the end of head's page, which the store has, out of
    // the view, or the detached one (head_state_line), and returns it. No unfinished step writes that line.
    std::uint32_t release_head_state_line(std::size_t layer);
    // Takes the detached lines out and returns them, per KV layer.
    std::vector<std::vector<std::uint32_t>> take_detached_lines();
    // Adds the mass a step ending at history position end measured: per KV layer that measures mass, the mass_sum and
    // weight_sum it gave each line of the view as of the step's preparation, which may hold more lines than the view;
    // empty for other layers. The lines added to the view after that preparation hold no mass yet.
    void add_step_mass(
        std::span<const std::span<const float>> step_mass_sums,
        std::span<const std::span<const float>> step_weight_sums,
        std::size_t end
    );
    // At the end of a cycle, the history position, adds to the moments of each layer that measures mass its full pages
    // outside the sinks whose last token is at least cycle_tokens and less than twice cycle_tokens before that end.
    void sample_pages();
    // Sets the exit reason of a request with an exit cause and no unfinished step: oom when a store write failed,
    // otherwise finished when a step written back reached the end, otherwise context_exceeded when its view would grow
    // past its page tables, otherwise oom when it ran out of lines, otherwise cancelled.
    void exit() noexcept;
    void record_device_cached_tokens(std::size_t count) noexcept;
    // Raises the peak of the bytes of each device's pool the lines take to bytes.
    void record_device_memory(std::size_t bytes) noexcept;
};

}  // namespace pulsarlm
