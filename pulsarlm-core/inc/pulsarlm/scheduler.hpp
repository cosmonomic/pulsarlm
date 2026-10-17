#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "pulsarlm/digest.hpp"
#include "pulsarlm/event_loop.hpp"
#include "pulsarlm/eviction.hpp"
#include "pulsarlm/kv_cache.hpp"
#include "pulsarlm/model/interface.hpp"
#include "pulsarlm/request_callbacks.hpp"
#include "pulsarlm/sampling.hpp"

namespace pulsarlm {

// The tail of initial_state is forwarded again, then tokens; together they hold at least one token. Without
// max_tokens, the request samples until one of the engine's stop tokens. The pages its steps put are owned by id.
struct request {
    state initial_state;
    std::vector<std::int64_t> tokens;
    sampling_params sampling;
    std::optional<std::size_t> max_tokens;
    eviction_policy eviction;
    std::uint64_t id = 0;
};

class scheduler;

// One KV layer's view of a sequence. A layer with pages has one line per view page: its full pages, then the pages of
// tail and pending, each with the EMA of the attention mass it received, as of the last step that measured it, 0 for a
// page never measured. A recurrent layer has no pages: its lines are the one its state is updated in, then the one a
// step writes the state at its newest page end into.
struct layer_view {
    std::vector<digest> pages;
    std::vector<std::uint32_t> lines;
    std::vector<float> density;

    // Drops the full pages at the ascending view indices and returns their lines.
    std::vector<std::uint32_t> remove(std::span<const std::size_t> indices);
    void append_lines(std::span<const std::uint32_t> added);
};

// A request being served. Its tokens move from pending to tail as steps forward them, and the pages the tail fills
// join every layer's view, the last of them head.
class sequence {
    enum class exit_reason { finished, cancelled, oom };

    result_writer& writer;
    std::uint64_t id;
    sampling_params sampling;
    std::uint64_t seed;
    // Tokens sampled so far, each with its own draw of the seed's noise.
    std::uint64_t draw_count = 0;
    std::optional<std::size_t> max_tokens;
    std::size_t generated = 0;
    std::optional<digest> head;
    std::vector<std::int64_t> tail;
    std::vector<std::int64_t> pending;
    // Per KV layer, whether it is recurrent; the scheduler's.
    const std::vector<bool>& recurrent;
    // One per KV layer.
    std::vector<layer_view> layers;
    // Whether the recurrent layers' second lines hold the state at head, which the store does not have yet.
    bool state_pending = false;
    // int32 [n_kv_layers, capacity] on the device. Columns past those of a layer's lines are unused.
    at::Tensor table;
    std::optional<exit_reason> exited;
    // Set once the sequence leaves the running set.
    std::optional<event> exit_event;
    eviction_policy eviction;
    // covered_count when the eviction cycle began.
    std::size_t cycle_start = 0;
    // The request's tokens forwarded so far, which the state covers: the initial state's tail, then the request's
    // tokens, then sampled tokens.
    std::size_t covered_count = 0;

    sequence(request&& r, const std::vector<bool>& recurrent, result_writer& writer, at::Device device);

    // The view index of the next token in KV layer layer's view.
    std::size_t next_index(std::size_t layer, std::size_t page_size) const noexcept;
    bool may_sample() const noexcept;
    // Moves the first count pending tokens to tail, then the pages they complete to every layer with pages, and returns
    // how many pages they complete.
    std::size_t advance(std::size_t count, std::size_t page_size, const digest& root);
    void start_cycle() noexcept;
    // Whether forwarding token_count more tokens ends the eviction cycle.
    bool ends_cycle(std::size_t token_count) const noexcept;
    // The tokens the cycle can still forward: no more than its length allows, nor than the request will forward.
    std::size_t cycle_tokens_left() const noexcept;
    // The columns of table the lines take.
    std::size_t table_width() const noexcept;
    // Queues on stream the upload of the lines from column first_page on into table, all of them when table grows.
    void upload_table(std::size_t first_page, cudaStream_t stream);
    // One line per page the cycle's tokens fill, in each KV layer with pages. Called at the start of a cycle, with an
    // empty tail.
    std::size_t cycle_lines(std::size_t page_size) const noexcept;
    // Per KV layer, the lines the sequence lacks to forward every token left in its cycle.
    std::vector<std::size_t> missing_lines(std::size_t page_size) const;

    friend class scheduler;
    friend class batch;
};

// The tokens each running sequence forwards in one step, the model inputs for them, and the tokens sampled from the
// step's logits.
class batch {
    struct entry {
        sequence* seq;
        std::size_t token_count;
        bool samples;
        bool ends_cycle;
    };

    // decode_count decode entries first, each forwarding its sequence's only pending token.
    std::vector<entry> entries;
    std::size_t decode_count = 0;
    // Pinned host tensors.
    StepBatch inputs;
    // Pinned host int64, one token per entry that samples, in entry order.
    at::Tensor sampled;
    // Pinned host fp32, defined for a group that measures attention mass: the page mass of each evictable KV layer in
    // turn, laid out as the layer's row of the group's cu_view_pages.
    at::Tensor decode_mass;
    at::Tensor prefill_mass;

    // With recurrent, the inputs carry each sequence's page end tokens.
    batch(
        std::span<sequence* const> running,
        std::size_t page_size,
        std::size_t prefill_tokens_per_step,
        bool recurrent
    );

    static GroupBatch make_group(std::span<const entry> group, std::size_t page_size, bool prefill, bool recurrent);
    static bool measures_mass(std::span<const entry> group) noexcept;
    static at::Tensor page_tables(std::span<const entry> group, std::size_t n_kv_layers);
    // Queues on stream the inputs' copies, the forward, the copies of the page mass of the evictable KV layers to the
    // host, the sampling and the copy of the sampled tokens to the host.
    void enqueue(Model& model, std::span<const kv> lines, const std::vector<bool>& evictable, cudaStream_t stream);
    // Per KV layer, the attention mass the step gave each of entry index's view pages; empty for a layer that is not
    // evictable. The entry's group measures mass.
    std::vector<std::span<const float>> step_mass(std::size_t index, const std::vector<bool>& evictable) const;

    friend class scheduler;
};

class scheduler {
    using exit_reason = sequence::exit_reason;

    Model& model;
    // One layer per KV layer, in the model's KV layer order.
    kv_cache cache;
    std::vector<kv> lines;
    std::vector<std::optional<std::size_t>> sliding_windows;
    std::vector<bool> recurrent;
    // Per KV layer, whether it evicts pages by its own attention mass: a full-attention layer.
    std::vector<bool> evictable;
    cudaStream_t stream;
    std::size_t page_size;
    digest root;
    std::vector<std::int64_t> stop_ids;
    std::size_t prefill_tokens_per_step;
    std::vector<sequence*> running;
    // Set when a sequence joins the running set.
    std::optional<event> idle;

    static event& next(std::optional<event>& waited);
    static void wake(std::optional<event>& waited) noexcept;
    // The full pages of KV layer layer's view wholly before the window of a query at view index query.
    std::size_t passed_pages(std::size_t layer, std::size_t query) const noexcept;
    state reached(const sequence& seq) const;
    // Per KV layer with pages, lines for its pages, then line_count empty lines; per recurrent layer, a copy of the
    // state at head, or zeros without head, and an empty line. Holds nothing on failure.
    task<std::expected<void, page_error>> take_lines(sequence& seq, std::size_t line_count);
    void release_lines(sequence& seq) noexcept;
    // Drops from each sliding layer's view the pages before the window of the tail's first token, and returns whether
    // any view changed.
    bool release_passed_pages(sequence& seq);
    task<batch> build_batch();
    task<void> launch(batch& b);
    task<void> write_back(batch& b);
    // Returns whether a view changed.
    bool end_cycle(sequence& seq);

  public:
    // cache holds one layer per entry of model.kv_layers(). root is the digest the page chains of a state with no
    // pages start from. A step forwards every decoding sequence and at most prefill_tokens_per_step tokens of the
    // others.
    scheduler(
        Model& model,
        kv_cache cache,
        cudaStream_t stream,
        digest root,
        std::vector<std::int64_t>&& stop_ids,
        std::size_t prefill_tokens_per_step
    );

    // Serves r, then ends writer with its final state, which leaves out the last sampled token, or an error. Appends
    // the state to writer at the end of each eviction cycle. Ends with oom, never waiting, when the free lines do not
    // cover the state's pages and one cycle's tokens at the start, or the next cycle's tokens while running; the
    // caller resubmits.
    task<void> submit(request r, result_writer writer);
    // Steps the running sequences until the loop stops.
    task<void> run();
    // Throws std::invalid_argument unless s is a state the scheduler reaches: with one sequence per KV layer, a
    // full-attention layer's ending with head, a sliding layer's ending with head and holding no more pages than its
    // window reaches, a recurrent layer's head alone, and every density finite and non-negative. Safe to call from any
    // thread.
    void validate_state(const state& s) const;
};

}  // namespace pulsarlm
