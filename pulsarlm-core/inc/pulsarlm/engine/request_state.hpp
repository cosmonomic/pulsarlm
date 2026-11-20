#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
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
// in the history, then the pages of tail and pending. A layer that measures mass keeps, per line, mass_heads mass_sum
// values and one weight_sum (add_step_mass), as of the last step that measured them, 0 for a page never measured, and
// the moments of the pages it sampled (add_sampled_page). A recurrent layer has no pages: its lines are its state
// slots, then the lines holding the state at the end of its even and of its odd pages (state_rows).
class layer_view {
  public:
    std::vector<digest> pages;
    std::vector<std::size_t> indices;
    std::vector<std::uint32_t> lines;
    std::size_t mass_heads = 0;
    std::vector<float> mass_sum;
    std::vector<float> weight_sum;
    std::vector<moments> density_moments;
    moments score_moments;

    // Drops the full pages at the ascending view indices and returns their lines.
    std::vector<std::uint32_t> remove(std::span<const std::size_t> view_indices);
    void append_lines(std::span<const std::uint32_t> added);
};

// A request being served. Its tokens move from pending to tail as steps forward them, and the pages the tail fills
// join every layer's view, the last of them head. A step moves its tokens to tail when it is launched, and its
// write-back completes the pages they fill; a token it samples is awaited_token on the host until the step is written
// back.
class request_state {
  public:
    enum class exit_reason { finished, cancelled, oom, context_exceeded };
    static constexpr std::int64_t awaited_token = -1;

  private:
    state::lease_id_type held_lease_id;
    sampling_params sampling;
    std::uint64_t seed;
    std::optional<std::size_t> max_tokens;
    // The tokens the launched steps sampled. A verify step's are counted at its write-back, as the tokens it accepted.
    std::size_t prepared_samples = 0;
    // The sampled tokens that steps written back applied.
    std::size_t delivered_tokens = 0;
    // The digest the page chain starts from.
    digest root;
    std::optional<digest> head;
    // The full pages of the history, head the last.
    std::size_t history_pages = 0;
    std::vector<std::int64_t> tail;
    std::vector<std::int64_t> pending_tokens;
    // The digests of the full pages of the request's tokens after head, in order; the first next_prompt_digest of them
    // joined the history.
    std::vector<digest> prompt_digests;
    std::size_t next_prompt_digest = 0;
    const view_layout& layout;
    // One per KV layer.
    std::vector<layer_view> layers;
    // The ascending history indices of the full pages some layer with pages, other than a mirroring one, holds. A full page's RoPE position is its
    // rank here times the page size, in every layer; the pages of tail and pending follow the last.
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
    // Whether a line allocation for the request's next tokens failed.
    bool is_out_of_lines = false;
    // Whether the request's view would grow past the columns of its page tables.
    bool has_exceeded_context = false;
    bool is_cancelled = false;
    // The steps of the request launched, and those of them written back.
    std::size_t prepared_step_count = 0;
    std::size_t written_back_step_count = 0;
    // Whether a launched step forwards the request's last tokens: it samples the last token max_tokens allows, or it
    // forwards the last pending tokens and does not sample.
    bool last_step_launched = false;
    // The tokens the draft head drafted after the pending token, which the next step verifies; empty when it does not.
    std::vector<std::int64_t> drafts;
    // The tokens the last step forwarded and did not accept (model::batch::rejected_tokens).
    std::size_t rejected_tokens = 0;
    // Whether the draft head's KV layer holds no row for the next step's first token, whose row then reads the
    // last_hidden of the position before it.
    bool next_step_reads_last_hidden = true;
    page_eviction_policy eviction;
    // The history position, in tokens, at which the eviction cycle ends: a multiple of cycle_tokens, and past the end of
    // the resumed state, so that forwarding its tail again never reaches a cycle end that state already passed.
    std::size_t next_cycle_end = 0;
    // The history position up to which the layers' mass_sum and weight_sum hold the queries' mass.
    std::size_t mass_end_position = 0;
    // The request's tokens.
    std::size_t prompt_count = 0;
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
    // Sets held_pages from the layers' indices.
    void update_held();
    // Moves the first count pending tokens to tail.
    void advance(std::size_t count);
    // After a verify step that forwarded its pending token at history position start, in tail, and its drafts: inserts
    // after it the first accepted_tokens.size() - 1 of accepted_tokens, the drafts they equal, and sets the next token,
    // awaited after them in tail or pending, to the last.
    void accept_tokens(std::size_t start, std::span<const std::int64_t> accepted_tokens);
    // Sets the token at history position index, in tail or pending, which was awaited_token.
    void resolve_token(std::size_t index, std::int64_t token);

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
    std::span<const std::int64_t> tail_tokens() const noexcept;
    std::span<const std::int64_t> pending_token_list() const noexcept;
    std::span<const std::int64_t> draft_tokens() const noexcept;
    const sampling_params& sampling_parameters() const noexcept;
    // The draw of the next token sampled after the steps written back, each token sampled before it having taken its
    // own draw of the seed's noise.
    sampling_draw next_draw() const noexcept;
    const std::optional<std::size_t>& token_limit() const noexcept;
    std::size_t prepared_sample_count() const noexcept;
    // The sampled tokens that steps written back applied, without those of launched steps.
    std::size_t delivered_token_count() const noexcept;
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
    // The launched steps of the request not written back yet.
    std::size_t unfinished_step_count() const noexcept;
    bool has_launched_last_step() const noexcept;
    std::size_t rejected_token_count() const noexcept;
    bool reads_last_hidden() const noexcept;
    const page_eviction_policy& eviction_policy_in_pages() const noexcept;
    state::lease_id_type lease_id() const noexcept;
    // Recurrent KV layer layer's line holding the state at the end of head's page.
    std::uint32_t head_state_line(std::size_t layer) const;
    // The view index of the next token in KV layer layer's view.
    std::size_t next_index(std::size_t layer) const noexcept;
    // The RoPE position of the next token, the same in every layer.
    std::size_t next_position() const noexcept;
    // The full pages of KV layer layer's view wholly before the window of a query at view index query.
    std::size_t passed_pages(std::size_t layer, std::size_t query) const noexcept;
    // The state the request reached: per KV layer, its view's full pages within the window of the next token.
    state to_state() const;
    bool may_sample() const noexcept;
    // The tokens of the history up to the next token: its full pages, then tail.
    std::size_t history_position() const noexcept;
    // The history position at which the eviction cycle ends.
    std::size_t cycle_end_position() const noexcept;
    // The history position from which the cycle's queries add mass: the cycle's last mass_tokens tokens, less those
    // the masses already hold.
    std::size_t mass_start_position() const noexcept;
    // The weight of the queries of weights.size() tokens from history position first (query_weights), 0 for those
    // before mass_start_position.
    void fill_query_weights(std::size_t first, std::span<float> weights) const noexcept;
    // Whether forwarding token_count more tokens ends the eviction cycle.
    bool ends_cycle(std::size_t token_count) const noexcept;
    // The tokens the cycle can still forward: no more than its length allows, nor than the request will forward.
    std::size_t cycle_tokens_left() const noexcept;
    // The columns of a page table the lines take: the most lines of a KV layer.
    std::size_t table_width() const noexcept;
    // The RoPE position of the first key of view page page in KV layer layer, which has pages.
    std::size_t page_position(std::size_t layer, std::size_t page) const;
    // One line per page the cycle's tokens fill, in each KV layer with pages. Called at the start of a cycle, with an
    // empty tail.
    std::size_t cycle_lines() const noexcept;
    // Per KV layer, the lines the request lacks to forward every token left in its cycle.
    std::vector<std::size_t> missing_lines() const;
    // The view indices of the full pages a cycle's end evicts from KV layer layer, which measures mass, ascending.
    std::vector<std::size_t> pages_to_evict(std::size_t layer) const;

    // Launches a step forwarding the first token_count pending tokens, which move to tail. With samples, the step
    // samples a token, awaited_token in pending until apply_sample. The step drafts draft_count tokens, awaited until
    // set_drafts, and with continues_drafting adds the row of the token after its own to the draft head's KV layer.
    void launch_step(std::size_t token_count, bool samples, std::size_t draft_count, bool continues_drafting);
    // Launches a step verifying the drafts after the first token_count pending tokens, which move to tail. The token
    // after those it accepts is awaited_token in pending until apply_accepted_tokens, as if it accepted only its
    // pending token. The step drafts draft_count tokens, awaited until set_drafts, and adds the row of the token after
    // those it accepts to the draft head's KV layer.
    void launch_verify_step(std::size_t token_count, std::size_t draft_count);
    // Sets the awaited token at history position position to token, which a step written back sampled.
    void apply_sample(std::size_t position, std::int64_t token);
    // A verify step written back, whose pending token is at history position start, verified verified_count tokens
    // and accepted accepted_tokens, its samples after the pending token and the drafts they equal: inserts them and
    // sets the awaited token after them to the last.
    void
    apply_accepted_tokens(std::size_t start, std::span<const std::int64_t> accepted_tokens, std::size_t verified_count);
    // One launched step of the request was written back.
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
    // Undoes the advance by one token of the latest launched step, whose pages are not completed yet, and its sample if
    // it took one: the token it forwarded returns to pending.
    void withdraw_latest_step();
    // Undoes the launch of the latest step, a verify step whose pending token and drafts the GPU took and which is not
    // written back: its pending token, in tail, returns to pending.
    void withdraw_latest_verify_step();
    void set_drafts(std::span<const std::int64_t> tokens);
    // Moves the full pages of tail before history position end to every layer with pages, and returns how many. The
    // tokens of those pages are not awaited_token. With a recurrent layer, the state at the new head is then pending.
    std::size_t complete_pages(std::size_t end);
    // The store has the state at head.
    void store_head_state() noexcept;
    // Appends added to KV layer layer's lines.
    void add_lines(std::size_t layer, std::span<const std::uint32_t> added);
    // Drops the full pages at the ascending view_indices from KV layer layer's view and returns their lines.
    std::vector<std::uint32_t> remove_pages(std::size_t layer, std::span<const std::size_t> view_indices);
    // Removes every line of every KV layer and returns them, per layer.
    std::vector<std::vector<std::uint32_t>> take_lines();
    // Sets recurrent KV layer layer's line holding the state at the end of head's page, and returns the previous one.
    std::uint32_t replace_head_state_line(std::size_t layer, std::uint32_t line);
    // Adds the mass a step ending at history position end measured: per KV layer that measures mass, the mass_sum and
    // weight_sum it gave each line of the view, which may hold more lines than the view; empty for other layers.
    void add_step_mass(
        std::span<const std::span<const float>> step_mass_sums,
        std::span<const std::span<const float>> step_weight_sums,
        std::size_t end
    );
    // At the end of the cycle, adds to the moments of each layer that measures mass its full pages outside the sinks
    // whose last token is at least cycle_tokens and less than twice cycle_tokens before that end.
    void sample_pages();
    void start_next_cycle() noexcept;
    // Sets the exit reason of a request with an exit cause and no unfinished step: oom when a store write failed,
    // otherwise finished when a step written back reached the end, otherwise context_exceeded when its view would grow
    // past its page tables, otherwise oom when it ran out of lines, otherwise cancelled.
    void exit() noexcept;
    void record_device_cached_tokens(std::size_t count) noexcept;
    // Raises the peak of the bytes of each device's pool the lines take to bytes.
    void record_device_memory(std::size_t bytes) noexcept;
};

}  // namespace pulsarlm
