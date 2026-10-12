#pragma once

#include <ATen/core/Generator.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "pulsar/digest.hpp"
#include "pulsar/event_loop.hpp"
#include "pulsar/eviction.hpp"
#include "pulsar/kv_cache.hpp"
#include "pulsar/model/interface.hpp"
#include "pulsar/result_stream.hpp"
#include "pulsar/sampling.hpp"

namespace pulsar {

// The tail of initial_state is forwarded again, then tokens; together they hold at least one token. Without
// max_tokens, the request samples until the engine's eos token.
struct request {
    state initial_state;
    std::vector<std::int64_t> tokens;
    sampling_params sampling;
    std::optional<std::size_t> max_tokens;
    eviction_policy eviction;
};

template <page_store t_store> class scheduler;

// A request being served. Its tokens move from pending to tail as steps forward them, and the pages the tail fills
// move to page_hashes.
class sequence {
    enum class exit_reason { finished, requeued, cancelled, oom };

    result_writer& writer;
    sampling_params sampling;
    at::Generator generator;
    std::optional<std::size_t> max_tokens;
    std::size_t generated = 0;
    std::vector<digest> page_hashes;
    std::vector<std::int64_t> tail;
    std::vector<std::int64_t> pending;
    // The line of each page in view order: the full pages, then the pages of tail and pending.
    std::vector<std::uint32_t> lines;
    std::optional<exit_reason> exited;
    // Set once the sequence leaves the running set.
    std::optional<event> exit_event;
    eviction_policy eviction;
    // forwarded_count when the eviction cycle began.
    std::size_t cycle_start = 0;
    cycle_density density;
    // The request's tokens the state covers: the initial state's tail, then the request's tokens, then sampled tokens.
    std::size_t covered_count = 0;

    sequence(request r, result_writer& writer, at::Device device);

    std::size_t forwarded_count(std::size_t page_size) const noexcept;
    bool may_sample() const noexcept;
    // Moves the first count pending tokens to tail, then the pages they complete to page_hashes.
    void advance(std::size_t count, std::size_t page_size, const digest& root);
    void start_cycle(std::size_t page_size) noexcept;
    // Whether forwarding token_count more tokens ends the eviction cycle.
    bool ends_cycle(std::size_t token_count, std::size_t page_size) const noexcept;
    // The tokens the cycle can still forward: no more than its length allows, nor than the request will forward.
    std::size_t cycle_tokens_left(std::size_t page_size) const noexcept;
    // Drops the full pages at the ascending view indices pages, and returns their lines.
    std::vector<std::uint32_t> remove_pages(std::span<const std::size_t> pages);
    // One line per distinct page hash and one per page the cycle's tokens fill. Called at the start of a cycle, with
    // an empty tail.
    std::size_t admission_lines(std::size_t page_size) const;
    // The lines the sequence lacks to forward every token left in its cycle.
    std::size_t missing_lines(std::size_t page_size) const noexcept;
    // Moves tail back to pending, to be forwarded again into new lines.
    void requeue();

    template <page_store> friend class scheduler;
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
        // Pinned host fp32, the sequence's cycle density after the step. Defined when the step ends a cycle of a
        // sequence that can evict.
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
    // Queues on stream the inputs' copies, the forward, the sampling, the density updates, and the copies of the
    // sampled tokens and of the densities that end a cycle to the host.
    void enqueue(Model& model, const kv_lines& kv, cudaStream_t stream);
    void add_page_mass(std::span<entry> group, const GroupBatch& inputs);

    template <page_store> friend class scheduler;
};

template <page_store t_store> class scheduler {
    using exit_reason = sequence::exit_reason;

    Model& model;
    kv_cache<t_store> cache;
    kv_lines kv;
    cudaStream_t stream;
    std::size_t page_size;
    std::size_t n_lines;
    digest root;
    std::int64_t eos_id;
    std::size_t prefill_tokens_per_step;
    std::vector<sequence*> running;
    // Set when a sequence joins the running set.
    std::optional<event> idle;
    // Set when lines are released.
    std::optional<event> released;

    static event& next(std::optional<event>& waited);
    static void wake(std::optional<event>& waited) noexcept;
    task<std::expected<void, page_error>> admit(sequence& seq);
    task<batch> build_batch();
    task<void> launch(batch& b);
    task<void> write_back(batch& b);
    void end_cycle(sequence& seq, const at::Tensor& host_density);

  public:
    // root is the digest the page chains of a state with no pages start from. A step forwards every decoding sequence
    // and at most prefill_tokens_per_step tokens of the others.
    scheduler(
        Model& model,
        kv_cache<t_store> cache,
        cudaStream_t stream,
        digest root,
        std::int64_t eos_id,
        std::size_t prefill_tokens_per_step
    );

    // Serves r, then ends writer with its final state, which leaves out the last sampled token, or an error. Appends
    // the state to writer at the end of each eviction cycle. Waits without holding lines until free lines cover the
    // state's pages and one cycle's tokens. One whose pages and cycle cannot fit in the cache ends with oom; one that
    // cannot get its next cycle's lines while running releases its lines, waits again and starts a new cycle.
    task<void> submit(request r, result_writer writer);
    // Steps the running sequences until the loop stops.
    task<void> run();
};

template <page_store t_store>
scheduler<t_store>::scheduler(
    Model& model,
    kv_cache<t_store> cache,
    cudaStream_t stream,
    digest root,
    std::int64_t eos_id,
    std::size_t prefill_tokens_per_step
)
: model(model),
  cache(std::move(cache)),
  kv{.k = this->cache.k_lines(), .v = this->cache.v_lines()},
  stream(stream),
  page_size(static_cast<std::size_t>(kv.k.size(2))),
  n_lines(static_cast<std::size_t>(kv.k.size(1))),
  root(root),
  eos_id(eos_id),
  prefill_tokens_per_step(prefill_tokens_per_step) {
    assert(prefill_tokens_per_step > 0);
}

template <page_store t_store> event& scheduler<t_store>::next(std::optional<event>& waited) {
    if (!waited) {
        waited.emplace();
    }
    return *waited;
}

// The awaiters a set event scheduled no longer refer to it, so it is destroyed right away.
template <page_store t_store> void scheduler<t_store>::wake(std::optional<event>& waited) noexcept {
    if (waited) {
        waited->set();
        waited.reset();
    }
}

template <page_store t_store> task<void> scheduler<t_store>::submit(request r, result_writer writer) {
    assert(r.initial_state.tail.size() < page_size);
    assert(!r.initial_state.tail.empty() || !r.tokens.empty());
    assert(r.eviction.n_working >= 1 && r.eviction.cycle_tokens >= 1);
    assert(r.eviction.attention_mass_decay > 0.0 && r.eviction.attention_mass_decay <= 1.0);
    sequence seq(std::move(r), writer, kv.k.device());
    for (;;) {
        seq.start_cycle(page_size);
        if (writer.cancelled()) {
            writer.end(std::unexpected(request_error::cancelled));
            co_return;
        }
        std::size_t needed = seq.admission_lines(page_size);
        if (needed > n_lines) {
            writer.end(std::unexpected(request_error::oom));
            co_return;
        }
        // Admitting without enough free lines can fail after get took lines, and releasing them wakes the other
        // waiters into the same failure.
        if (needed > cache.free_line_count()) {
            co_await next(released);
            continue;
        }
        std::expected<void, page_error> admitted = co_await admit(seq);
        if (!admitted) {
            if (admitted.error() == page_error::miss) {
                writer.end(std::unexpected(request_error::miss));
                co_return;
            }
            co_await next(released);
            continue;
        }

        seq.exited.reset();
        seq.exit_event.emplace();
        running.push_back(&seq);
        wake(idle);
        co_await *seq.exit_event;
        cache.release(seq.lines);
        seq.lines.clear();
        wake(released);

        switch (*seq.exited) {
        case exit_reason::requeued:
            seq.requeue();
            continue;
        case exit_reason::finished:
            writer.end(state{.page_hashes = std::move(seq.page_hashes), .tail = std::move(seq.tail)});
            co_return;
        case exit_reason::cancelled:
            writer.end(std::unexpected(request_error::cancelled));
            co_return;
        case exit_reason::oom:
            writer.end(std::unexpected(request_error::oom));
            co_return;
        }
    }
}

template <page_store t_store> task<std::expected<void, page_error>> scheduler<t_store>::admit(sequence& seq) {
    if (!seq.page_hashes.empty()) {
        std::expected<std::vector<std::uint32_t>, page_error> got = co_await cache.get(seq.page_hashes, stream);
        if (!got) {
            if (got.error() == page_error::miss) {
                wake(released);
            }
            co_return std::unexpected(got.error());
        }
        seq.lines = std::move(*got);
    }
    std::expected<std::vector<std::uint32_t>, page_error> allocated = cache.allocate(seq.missing_lines(page_size));
    if (!allocated) {
        cache.release(seq.lines);
        seq.lines.clear();
        wake(released);
        co_return std::unexpected(allocated.error());
    }
    seq.lines.insert(seq.lines.end(), allocated->begin(), allocated->end());
    co_return {};
}

template <page_store t_store> task<void> scheduler<t_store>::run() {
    for (;;) {
        batch b = co_await build_batch();
        co_await launch(b);
        co_await write_back(b);
    }
}

template <page_store t_store> task<batch> scheduler<t_store>::build_batch() {
    while (running.empty()) {
        co_await next(idle);
    }
    co_return batch(running, page_size, prefill_tokens_per_step);
}

template <page_store t_store> task<void> scheduler<t_store>::launch(batch& b) {
    b.enqueue(model, kv, stream);
    co_await synchronize(stream);
}

// A sequence ends with oom when the put of its new pages fails, as a state names only pages the store has.
template <page_store t_store> task<void> scheduler<t_store>::write_back(batch& b) {
    std::vector<digest> new_hashes;
    std::vector<std::uint32_t> new_lines;
    std::vector<sequence*> putting;
    const std::int64_t* next_sampled = b.sampled.defined() ? b.sampled.const_data_ptr<std::int64_t>() : nullptr;
    for (const batch::entry& e : b.entries) {
        sequence& seq = *e.seq;
        std::size_t first_new_page = seq.page_hashes.size();
        seq.advance(e.token_count, page_size, root);
        if (seq.page_hashes.size() > first_new_page) {
            new_hashes.insert(new_hashes.end(), seq.page_hashes.begin() + first_new_page, seq.page_hashes.end());
            new_lines.insert(
                new_lines.end(),
                seq.lines.begin() + first_new_page,
                seq.lines.begin() + seq.page_hashes.size()
            );
            putting.push_back(&seq);
        }
        if (e.samples) {
            std::int64_t token = *next_sampled++;
            seq.writer.append(token);
            seq.generated++;
            seq.pending.push_back(token);
            if (token == eos_id || !seq.may_sample()) {
                seq.exited = exit_reason::finished;
            }
        } else if (seq.pending.empty()) {
            seq.exited = exit_reason::finished;
        }
    }

    if (!new_hashes.empty()) {
        std::expected<void, page_error> stored = co_await cache.put(std::move(new_hashes), std::move(new_lines), stream);
        if (!stored) {
            for (sequence* seq : putting) {
                seq->exited = exit_reason::oom;
            }
        }
    }

    for (const batch::entry& e : b.entries) {
        sequence& seq = *e.seq;
        if (seq.exited) {
            continue;
        }
        if (seq.writer.cancelled()) {
            seq.exited = exit_reason::cancelled;
            continue;
        }
        if (e.ends_cycle) {
            end_cycle(seq, e.host_density);
        }
        if (std::size_t missing = seq.missing_lines(page_size); missing > 0) {
            std::expected<std::vector<std::uint32_t>, page_error> allocated = cache.allocate(missing);
            if (!allocated) {
                seq.exited = exit_reason::requeued;
                continue;
            }
            seq.lines.insert(seq.lines.end(), allocated->begin(), allocated->end());
        }
    }

    std::erase_if(running, [](const sequence* seq) { return seq->exited.has_value(); });
    for (const batch::entry& e : b.entries) {
        if (e.seq->exited) {
            e.seq->exit_event->set();
        }
    }
}

// The step that last used the evicted lines has completed, so they are released right away.
template <page_store t_store> void scheduler<t_store>::end_cycle(sequence& seq, const at::Tensor& host_density) {
    if (host_density.defined()) {
        std::span<const float> density(host_density.const_data_ptr<float>(), seq.page_hashes.size());
        std::vector<std::size_t> evicted =
            pages_to_evict(density, page_size, seq.cycle_start, seq.forwarded_count(page_size), seq.eviction);
        if (!evicted.empty()) {
            cache.release(seq.remove_pages(evicted));
            wake(released);
        }
    }
    seq.writer.append(
        cycle_end{.reached = state{.page_hashes = seq.page_hashes, .tail = seq.tail}, .token_count = seq.covered_count}
    );
    seq.start_cycle(page_size);
}

}  // namespace pulsar
