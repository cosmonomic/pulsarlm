#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "pulsarlm/digest.hpp"
#include "pulsarlm/engine/event_loop.hpp"
#include "pulsarlm/engine/eviction.hpp"
#include "pulsarlm/engine/executor.hpp"
#include "pulsarlm/engine/page_store.hpp"
#include "pulsarlm/engine/request.hpp"
#include "pulsarlm/engine/request_callbacks.hpp"
#include "pulsarlm/engine/sampling.hpp"
#include "pulsarlm/engine/state.hpp"

namespace pulsarlm {

// Holds a lease of a store and releases it once: on release, or when destroyed before it, as with a frame destroyed
// after a panic, since the lease is the store's.
class lease_guard {
    page_store& store;
    state::lease_id_type lease_id;
    bool held = true;

  public:
    lease_guard(page_store& store, state::lease_id_type lease_id) noexcept;
    lease_guard(const lease_guard&) = delete;
    lease_guard& operator=(const lease_guard&) = delete;
    ~lease_guard();

    state::lease_id_type id() const noexcept;
    void release() noexcept;
};

// One KV layer's view of a sequence. A layer with pages has one line per view page: its full pages, each with its index
// in the history, then the pages of tail and pending. A layer that measures mass keeps, per line, mass_heads mass_sum
// values and one weight_sum (add_step_mass), as of the last step that measured them, 0 for a page never measured, and
// the moments of the pages it sampled (add_sampled_page). A recurrent layer has no pages: its lines are the one its
// state is updated in, then the one a step writes the state at its newest page end into.
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
// join every layer's view, the last of them head.
class sequence {
  public:
    enum class exit_reason { finished, cancelled, oom };

  private:
    result_writer& writer;
    lease_guard lease;
    // The state the request resumed, then the last state written to the store.
    std::shared_ptr<const state> last_state;
    sampling_params sampling;
    std::uint64_t seed;
    // Tokens sampled so far, each with its own draw of the seed's noise.
    std::uint64_t draw_count = 0;
    std::optional<std::size_t> max_tokens;
    std::size_t generated_count = 0;
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
    // Per KV layer, whether it is recurrent and its sliding window; the executor's.
    const std::vector<bool>& recurrent;
    const std::vector<std::optional<std::size_t>>& sliding_windows;
    std::size_t page_size;
    // One per KV layer.
    std::vector<layer_view> layers;
    // The ascending history indices of the full pages some layer with pages holds. A full page's RoPE position is its
    // rank here times the page size, in every layer; the pages of tail and pending follow the last.
    std::vector<std::size_t> held_pages;
    // Whether the recurrent layers' second lines hold the state at head, which the store does not have yet.
    bool state_pending = false;
    // Per shard, int32 [n_kv_layers, capacity] on its device: each view page's line, and the RoPE position of its
    // first key. Columns past those of a layer's lines are unused.
    std::vector<at::Tensor> tables;
    std::vector<at::Tensor> position_tables;
    std::optional<exit_reason> end_reason;
    // Set once the sequence leaves the running set.
    std::optional<event> exit_event;
    page_eviction_policy eviction;
    // covered_count when the eviction cycle began.
    std::size_t cycle_start = 0;
    // The tokens forwarded so far: the resumed state's tail, then the request's tokens after it, then sampled tokens.
    std::size_t covered_count = 0;
    // The request's tokens.
    std::size_t prompt_count = 0;
    // The tokens of the resumed state's full pages, and of those every layer reading them had resident at the start.
    std::size_t cached_count = 0;
    std::size_t device_cached_count = 0;

    // Resumes initial_state, the newest stored state reaching a prefix of r's tokens, leased from store by lease_id.
    // digests are the digests of the full pages of r's tokens, chained from root.
    sequence(
        request&& r,
        std::shared_ptr<const state>&& initial_state,
        page_store& store,
        state::lease_id_type lease_id,
        std::vector<digest>&& digests,
        const digest& root,
        const executor& owner,
        result_writer& writer,
        std::span<const at::Device> devices
    );

    // The view index of the next token in KV layer layer's view.
    std::size_t next_index(std::size_t layer) const noexcept;
    // The RoPE position of the next token, the same in every layer.
    std::size_t next_position() const noexcept;
    // The RoPE position of the first key of view page page in KV layer layer.
    std::size_t page_position(std::size_t layer, std::size_t page) const;
    // The full pages of KV layer layer's view wholly before the window of a query at view index query.
    std::size_t passed_pages(std::size_t layer, std::size_t query) const noexcept;
    // The state the sequence reached: per KV layer, its view's full pages within the window of the next token.
    state to_state() const;
    // Sets held from the layers' indices, after pages left them.
    void update_held();
    bool may_sample() const noexcept;
    // Moves the first count pending tokens to tail, then the pages they complete to every layer with pages, and returns
    // how many pages they complete.
    std::size_t advance(std::size_t count);
    void start_cycle() noexcept;
    // Whether forwarding token_count more tokens ends the eviction cycle.
    bool ends_cycle(std::size_t token_count) const noexcept;
    // The tokens the cycle can still forward: no more than its length allows, nor than the request will forward.
    std::size_t cycle_tokens_left() const noexcept;
    // The columns of table the lines take.
    std::size_t table_width() const noexcept;
    // Queues on each shard's stream the upload of the lines and positions from column first_page on into its tables,
    // all of them when the tables grow.
    void upload_table(std::size_t first_page, std::span<const cudaStream_t> streams);
    // One line per page the cycle's tokens fill, in each KV layer with pages. Called at the start of a cycle, with an
    // empty tail.
    std::size_t cycle_lines() const noexcept;
    // Per KV layer, the lines the sequence lacks to forward every token left in its cycle.
    std::vector<std::size_t> missing_lines() const;

    friend class executor;
    friend class executor::batch;
    friend class scheduler;
};

// One sequence's part of a step: the tokens it forwards, whether it samples after them, and whether they end its
// eviction cycle.
struct batch_entry {
    sequence* seq;
    std::size_t token_count;
    bool samples;
    bool ends_cycle;
};

}  // namespace pulsarlm
