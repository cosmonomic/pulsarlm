#pragma once

#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
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

// A request being served. Its tokens move from pending to tail as steps forward them, and the pages the tail fills
// move to page_hashes, the view of the full pages, the last of them head.
class sequence {
    enum class exit_reason { finished, cancelled, oom };

    static constexpr std::uint32_t no_line = std::numeric_limits<std::uint32_t>::max();

    result_writer& writer;
    std::uint64_t id;
    sampling_params sampling;
    std::uint64_t seed;
    // Tokens sampled so far, each with its own draw of the seed's noise.
    std::uint64_t draw_count = 0;
    std::optional<std::size_t> max_tokens;
    std::size_t generated = 0;
    std::optional<digest> head;
    std::vector<digest> page_hashes;
    std::vector<std::int64_t> tail;
    // The density of each full page in the states the sequence reaches: as of the last step that measured mass, 0 for a
    // page never measured.
    std::vector<float> state_density;
    std::vector<std::int64_t> pending;
    // Per KV layer, whether it is recurrent; the scheduler's.
    const std::vector<bool>& recurrent;
    // The KV layer whose lines are one per view page.
    std::size_t view_layer;
    // Per KV layer, the line of each page in view order: the full pages, then the pages of tail and pending. A sliding
    // layer's pages before first_held[layer] hold no_line. A recurrent layer's lines are the one its state is updated
    // in, then the one a step writes the state at its newest page end into.
    std::vector<std::vector<std::uint32_t>> lines;
    // Whether the recurrent layers' second lines hold the state at head, which the store does not have yet.
    bool state_pending = false;
    std::vector<std::size_t> first_held;
    // int32 [n_kv_layers, capacity] on the device, lines with no_line as 0. Columns past those of lines are unused.
    at::Tensor table;
    std::optional<exit_reason> exited;
    // Set once the sequence leaves the running set.
    std::optional<event> exit_event;
    eviction_policy eviction;
    // forwarded_count when the eviction cycle began.
    std::size_t cycle_start = 0;
    // Defined when the sequence can evict.
    page_density density;
    // The request's tokens the state covers: the initial state's tail, then the request's tokens, then sampled tokens.
    std::size_t covered_count = 0;

    // The view comes from the initial state's sequence of KV layer view_layer. Queues on stream the upload of its
    // density.
    sequence(
        request&& r,
        const std::vector<bool>& recurrent,
        std::size_t view_layer,
        result_writer& writer,
        at::Device device,
        cudaStream_t stream
    );

    std::size_t forwarded_count(std::size_t page_size) const noexcept;
    bool may_sample() const noexcept;
    // Moves the first count pending tokens to tail, then the pages they complete to page_hashes. host_density, when
    // defined, is the density after the step that forwarded them.
    void advance(std::size_t count, std::size_t page_size, const digest& root, const at::Tensor& host_density);
    void start_cycle(std::size_t page_size) noexcept;
    // Whether forwarding token_count more tokens ends the eviction cycle.
    bool ends_cycle(std::size_t token_count, std::size_t page_size) const noexcept;
    // The tokens the cycle can still forward: no more than its length allows, nor than the request will forward.
    std::size_t cycle_tokens_left(std::size_t page_size) const noexcept;
    std::size_t view_page_count() const noexcept;
    // The columns of table the lines take.
    std::size_t table_width() const noexcept;
    // Drops the full pages at the ascending view indices pages, queuing on stream the drop of their density, and returns
    // their held lines per KV layer.
    std::vector<std::vector<std::uint32_t>> remove_pages(std::span<const std::size_t> pages, cudaStream_t stream);
    // Queues on stream the upload of the view pages from first_page on into table, all of them when table grows.
    void upload_table(std::size_t first_page, cudaStream_t stream);
    // One line per page the cycle's tokens fill, in each KV layer of pages. Called at the start of a cycle, with an
    // empty tail.
    std::size_t cycle_lines(std::size_t page_size) const noexcept;
    // Queue on stream: ages the density by token_count tokens and adds step_mass.
    void add_step_mass(const at::Tensor& step_mass, std::size_t token_count, cudaStream_t stream);
    // A pinned host copy of the density, complete once stream reaches this point.
    at::Tensor density_on_host(cudaStream_t stream) const;
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
        // fp32 [view pages] on the device, the attention mass the step gave each of the sequence's view pages.
        // Defined when the sequence can evict.
        at::Tensor step_mass;
        // Pinned host fp32, the sequence's density after the step. Set by the scheduler with step_mass.
        at::Tensor host_density;
    };

    // decode_count decode entries first, each forwarding its sequence's only pending token.
    std::vector<entry> entries;
    std::size_t decode_count = 0;
    // Pinned host tensors.
    StepBatch inputs;
    // Pinned host int64, one token per entry that samples, in entry order.
    at::Tensor sampled;

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
    // Queues on stream the inputs' copies, the forward, the sampling and the copy of the sampled tokens to the host,
    // and sets each entry's step_mass.
    void enqueue(Model& model, std::span<const kv> lines, cudaStream_t stream);
    static void set_step_mass(std::span<entry> group, const GroupBatch& inputs);

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
    // A full-attention KV layer, whose sequence of a state is the view every other layer's sequence is a part of.
    std::size_t view_layer;
    // The full pages eviction keeps last so that no sliding window loses a key.
    std::size_t window_pages;
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
    // The view page of the first key KV layer layer reads for a query at view index query.
    std::size_t first_held_page(std::size_t layer, std::size_t query) const noexcept;
    // Per KV layer, the pages of view a request resuming from it reads: every page for a full-attention layer, those
    // its window reaches for a sliding layer, the last for a recurrent layer.
    std::vector<layer_sequence> layer_sequences(std::span<const active_page> view) const;
    state reached(const sequence& seq) const;
    // Per KV layer of pages, lines for the page_hashes from its first held page for a query at their end, in their
    // order, then line_count empty lines; per recurrent layer, a copy of the state at head, or zeros without head, and
    // an empty line. Holds nothing on failure.
    task<std::expected<void, page_error>> take_lines(sequence& seq, std::size_t line_count);
    void release_lines(sequence& seq) noexcept;
    // Releases the lines of the sliding layers' pages before the window of the next query.
    void release_passed_lines(sequence& seq);
    task<batch> build_batch();
    task<void> launch(batch& b);
    task<void> write_back(batch& b);
    void end_cycle(sequence& seq);

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
    // Throws std::invalid_argument unless s is a state the scheduler reaches: with one sequence per KV layer, each
    // the pages of the view layer's sequence its layer reads, ending with head, and every density finite and
    // non-negative. Safe to call from any thread.
    void validate_state(const state& s) const;
};

}  // namespace pulsarlm
