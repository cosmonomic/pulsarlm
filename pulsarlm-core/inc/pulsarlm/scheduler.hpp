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

class scheduler {
    enum class exit_reason { finished, cancelled, oom };
    class layer_view;
    class sequence;
    class batch;

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
