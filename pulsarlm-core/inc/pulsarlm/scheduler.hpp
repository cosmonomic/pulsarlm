#pragma once

#include <ATen/core/Generator.h>
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
// max_tokens, the request samples until one of the engine's stop tokens.
struct request {
    state initial_state;
    std::vector<std::int64_t> tokens;
    sampling_params sampling;
    std::optional<std::size_t> max_tokens;
    eviction_policy eviction;
};

class scheduler;

// A request being served. Its tokens move from pending to tail as steps forward them, and the pages the tail fills
// move to page_hashes.
class sequence {
    enum class exit_reason { finished, requeued, cancelled, oom };

    static constexpr std::uint32_t no_line = std::numeric_limits<std::uint32_t>::max();

    result_writer& writer;
    sampling_params sampling;
    at::Generator generator;
    std::optional<std::size_t> max_tokens;
    std::size_t generated = 0;
    std::vector<digest> page_hashes;
    std::vector<std::int64_t> tail;
    std::vector<std::int64_t> pending;
    // Per KV layer, the line of each page in view order: the full pages, then the pages of tail and pending. A sliding
    // layer's pages before first_held[layer] hold no_line.
    std::vector<std::vector<std::uint32_t>> lines;
    std::vector<std::size_t> first_held;
    // int32 [n_kv_layers, capacity] on the device, lines with no_line as 0. Columns past those of lines are unused.
    at::Tensor table;
    std::optional<exit_reason> exited;
    // Set once the sequence leaves the running set.
    std::optional<event> exit_event;
    eviction_policy eviction;
    // forwarded_count when the eviction cycle began.
    std::size_t cycle_start = 0;
    cycle_density density;
    // The request's tokens the state covers: the initial state's tail, then the request's tokens, then sampled tokens.
    std::size_t covered_count = 0;

    sequence(request&& r, result_writer& writer, std::size_t n_kv_layers, at::Device device);

    std::size_t forwarded_count(std::size_t page_size) const noexcept;
    bool may_sample() const noexcept;
    // Moves the first count pending tokens to tail, then the pages they complete to page_hashes.
    void advance(std::size_t count, std::size_t page_size, const digest& root);
    void start_cycle(std::size_t page_size) noexcept;
    // Whether forwarding token_count more tokens ends the eviction cycle.
    bool ends_cycle(std::size_t token_count, std::size_t page_size) const noexcept;
    // The tokens the cycle can still forward: no more than its length allows, nor than the request will forward.
    std::size_t cycle_tokens_left(std::size_t page_size) const noexcept;
    std::size_t view_page_count() const noexcept;
    // Drops the full pages at the ascending view indices pages, and returns their held lines per KV layer.
    std::vector<std::vector<std::uint32_t>> remove_pages(std::span<const std::size_t> pages);
    // Queues on stream the upload of the view pages from first_page on into table, all of them when table grows.
    void upload_table(std::size_t first_page, cudaStream_t stream);
    // One line per page the cycle's tokens fill, in each KV layer. Called at the start of a cycle, with an empty tail.
    std::size_t cycle_lines(std::size_t page_size) const noexcept;
    // Queue on stream: ages the cycle density by token_count tokens and adds step_mass.
    void add_step_mass(const at::Tensor& step_mass, std::size_t token_count, cudaStream_t stream);
    // A pinned host copy of the cycle density, complete once stream reaches this point.
    at::Tensor density_on_host(cudaStream_t stream) const;
    // The lines the sequence lacks in each KV layer to forward every token left in its cycle.
    std::size_t missing_lines(std::size_t page_size) const noexcept;
    // Moves tail back to pending, to be forwarded again into new lines.
    void requeue();

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
        // Pinned host fp32, the sequence's cycle density after the step. Set by the scheduler when the step ends a
        // cycle of a sequence that can evict.
        at::Tensor host_density;
    };

    // decode_count decode entries first, each forwarding its sequence's only pending token.
    std::vector<entry> entries;
    std::size_t decode_count = 0;
    // Pinned host tensors.
    StepBatch inputs;
    // Pinned host int64, one token per entry that samples, in entry order.
    at::Tensor sampled;

    batch(std::span<sequence* const> running, std::size_t page_size, std::size_t prefill_tokens_per_step);

    static GroupBatch make_group(std::span<const entry> group, std::size_t page_size, bool prefill);
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
    // Set when lines are released.
    std::optional<event> released;

    static event& next(std::optional<event>& waited);
    static void wake(std::optional<event>& waited) noexcept;
    // The view page of the first key KV layer layer reads for a query at view index query.
    std::size_t first_held_page(std::size_t layer, std::size_t query) const noexcept;
    // The lines of KV layer layer the sequence needs to run its cycle: one per page from its first held page, and the
    // cycle's lines.
    std::size_t lines_needed(const sequence& seq, std::size_t layer) const noexcept;
    // Per KV layer, lines for the page_hashes from its first held page for a query at their end, in their order, then
    // line_count empty lines. Holds nothing on failure.
    task<std::expected<void, page_error>> take_lines(sequence& seq, std::size_t line_count);
    void release_lines(sequence& seq) noexcept;
    // Releases the lines of the sliding layers' pages before the window of the next query.
    void release_passed_lines(sequence& seq);
    task<batch> build_batch();
    task<void> launch(batch& b);
    task<void> write_back(batch& b);
    void end_cycle(sequence& seq, const at::Tensor& host_density);

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
    // the state to writer at the end of each eviction cycle. Waits without holding lines until free lines cover the
    // state's pages and one cycle's tokens. One whose pages and cycle cannot fit in the cache ends with oom; one that
    // cannot get its next cycle's lines while running releases its lines, waits again and starts a new cycle.
    task<void> submit(request r, result_writer writer);
    // Steps the running sequences until the loop stops.
    task<void> run();
    // The store keys of the pages a request resuming from s reads: every page of a full-attention layer, and those a
    // sliding layer's window reaches. Safe to call from any thread.
    std::vector<digest> page_keys(const state& s) const;
};

}  // namespace pulsarlm
