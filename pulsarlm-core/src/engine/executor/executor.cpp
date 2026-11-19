#include "batch.hpp"

#include "pulsarlm/engine/executor.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace pulsarlm {

std::size_t executor::slots_of(const kv_layer& layer) const noexcept {
    if (!layer.state) {
        return 0;
    }
    return layer.state->input_window ? 1 : draft_state_slots;
}

executor::executor(
    std::vector<model*>&& shards,
    kv_cache cache,
    page_store& store,
    std::vector<cudaStream_t>&& streams,
    std::vector<cudaStream_t>&& copy_streams,
    std::vector<event_loop*>&& worker_loops,
    digest root,
    std::vector<std::int64_t>&& stop_ids,
    std::optional<std::size_t> graph_pages,
    std::size_t draft_tokens,
    std::size_t sampled_row_limit
)
: shards(std::move(shards)),
  cache(std::move(cache)),
  store(store),
  streams(std::move(streams)),
  copy_streams(std::move(copy_streams)),
  worker_loops(std::move(worker_loops)),
  layout{.layers = {}, .page_size = 0},
  root(root),
  stop_ids(std::move(stop_ids)),
  graph_pages(graph_pages),
  draft_state_slots(draft_tokens + 1),
  sampled_row_capacity(sampled_row_limit) {
    if (sampled_row_limit == 0) {
        throw std::invalid_argument("executor: sampled_row_limit must be at least 1");
    }
    if (sampled_row_limit < draft_tokens + 1) {
        throw std::invalid_argument("executor: sampled_row_limit must cover one verify entry, draft_tokens + 1 rows");
    }
    require(
        !this->shards.empty() && this->shards.size() == this->cache.shard_count() &&
            this->streams.size() == this->shards.size() && this->copy_streams.size() == this->shards.size() &&
            this->worker_loops.size() + 1 == this->shards.size(),
        "executor: the cache, the streams, the copy streams and the worker loops must match the model shards"
    );
    shard_mass_heads.resize(this->shards.size());
    lines.resize(this->shards.size());
    for (std::size_t shard = 0; shard < this->shards.size(); shard++) {
        std::vector<kv_layer> layers = this->shards[shard]->kv_layers();
        require(layers.size() == this->cache.layer_count(), "executor: the cache must hold one layer per KV layer");
        for (std::size_t layer = 0; layer < layers.size(); layer++) {
            lines[shard].push_back(this->cache.tensors(layer, shard));
            bool layer_recurrent = layers[layer].state.has_value();
            bool evictable = !layers[layer].sliding_window && !layer_recurrent && !layers[layer].mirrors_held_pages;
            require(
                !evictable || layers[layer].n_q_heads > 0,
                "executor: a full-attention KV layer must state its query heads"
            );
            shard_mass_heads[shard].push_back(evictable ? layers[layer].n_readers * layers[layer].n_q_heads : 0);
            if (shard == 0) {
                layout.layers.push_back(
                    view_layout::layer{
                        .is_recurrent = layer_recurrent,
                        .mirrors_held_pages = layers[layer].mirrors_held_pages,
                        .sliding_window = layers[layer].sliding_window,
                        .state_slots = slots_of(layers[layer]),
                        .mass_heads = 0,
                    }
                );
                if (layout.page_size == 0 && !layer_recurrent) {
                    layout.page_size = static_cast<std::size_t>(lines[0].back().k.size(1));
                }
            }
            view_layout::layer& layer_layout = layout.layers[layer];
            require(
                layer_layout.sliding_window == layers[layer].sliding_window &&
                    layer_layout.is_recurrent == layer_recurrent &&
                    layer_layout.mirrors_held_pages == layers[layer].mirrors_held_pages &&
                    layer_layout.state_slots == slots_of(layers[layer]),
                "executor: the model shards must agree on the kind of each KV layer"
            );
            layer_layout.mass_heads += shard_mass_heads[shard].back();
            if (shard == 0 && draft_tokens > 0 && layers[layer].last_hidden_dim > 0) {
                draft_layer = layer;
            }
        }
    }
    require(layout.page_size > 0, "executor: the model must have a KV layer with pages");
    for (std::size_t shard = 0; shard < this->shards.size(); shard++) {
        samples.push_back(
            at::zeros(
                {static_cast<std::int64_t>(sampled_row_capacity)},
                lines[shard].front().k.options().dtype(at::kLong)
            )
        );
        stop_tokens.push_back(
            at::tensor(this->stop_ids, at::TensorOptions().dtype(at::kLong)).to(lines[shard].front().k.device())
        );
    }
    eager_counters.resize(this->shards.size());
    if (this->graph_pages) {
        for (std::size_t shard = 0; shard < this->shards.size(); shard++) {
            c10::cuda::CUDAGuard device_guard(lines[shard].front().k.device());
            graphs.push_back(
                std::make_unique<decode_graphs>(
                    *this->shards[shard],
                    lines[shard],
                    *this->graph_pages,
                    layout.page_size,
                    draft_state_slots,
                    shard_mass_heads[shard],
                    samples[shard],
                    stop_tokens[shard]
                )
            );
        }
        capture_decode_graphs();
    }
}

executor::~executor() {
    for (const sequence& unstopped : sequences) {
        store.release(unstopped.lease_id());
    }
}

executor::sequence_ref::sequence_ref(executor& owner, sequence_id id) noexcept
: owner(&owner),
  id(id) {}

sequence& executor::sequence_ref::operator*() const {
    return owner->sequences.at(id);
}

sequence* executor::sequence_ref::operator->() const {
    return &**this;
}

executor::sequence_id executor::sequence_ref::key() const noexcept {
    return id;
}

bool executor::is_steppable(sequence_id id) const {
    const sequence& seq = sequences.at(id);
    return !seq.has_launched_last_step() && !seq.has_exited();
}

std::size_t executor::pending_token_count(sequence_id id) const {
    return sequences.at(id).pending_token_list().size();
}

std::size_t executor::cycle_tokens_left(sequence_id id) const {
    return sequences.at(id).cycle_tokens_left();
}

bool executor::has_exited(sequence_id id) const {
    return sequences.at(id).has_exited();
}

bool executor::is_in_launched_batch(sequence_id id) const {
    return sequences.at(id).launched_step_count() > 0;
}

bool executor::is_cancelled(sequence_id id) const {
    return sequences.at(id).writer().cancelled();
}

void executor::cancel(sequence_id id) {
    sequence& seq = sequences.at(id);
    if (!seq.has_exited()) {
        seq.exit(sequence::exit_reason::cancelled);
    }
}

std::size_t executor::kv_cache_bytes() const noexcept {
    return cache.used_bytes();
}

std::uint64_t executor::drafted_tokens() const noexcept {
    return drafted_count.load(std::memory_order_relaxed);
}

std::uint64_t executor::accepted_drafts() const noexcept {
    return accepted_draft_count.load(std::memory_order_relaxed);
}

bool executor::completes_page(const sequence& seq, std::size_t token_count) const noexcept {
    std::size_t start = seq.tail_tokens().size();
    return (start + token_count) / layout.page_size * layout.page_size > start;
}

// A drafting sequence's verify step ends neither its cycle nor max_tokens (executor::advance).
bool executor::entries_allow_overlap(std::span<const batch_tokens> entries) const {
    if (draft_layer) {
        return std::ranges::all_of(entries, [&](const batch_tokens& e) {
            return !sequences.at(e.first).draft_tokens().empty();
        });
    }
    const bool sliding = std::ranges::any_of(layout.layers, [](const view_layout::layer& layer) {
        return layer.sliding_window.has_value();
    });
    return std::ranges::none_of(entries, [&](const batch_tokens& e) {
        const auto& [id, token_count] = e;
        const sequence& seq = sequences.at(id);
        return seq.ends_cycle(token_count) || (sliding && completes_page(seq, token_count));
    });
}

bool executor::allows_overlap(std::span<const batch_tokens> entries) const {
    return (launched_batches.empty() || launched_batches.back()->allows_overlap) && entries_allow_overlap(entries);
}

std::size_t executor::launched_batch_count() const noexcept {
    return launched_batches.size();
}

std::size_t executor::sampled_row_limit() const noexcept {
    return sampled_row_capacity;
}

std::size_t executor::sampled_row_count(sequence_id id, std::size_t token_count) const {
    const sequence& seq = sequences.at(id);
    if (token_count != seq.pending_token_list().size() || !seq.may_sample()) {
        return 0;
    }
    return seq.draft_tokens().empty() ? 1 : token_count + seq.draft_tokens().size();
}

void executor::check_launch(std::span<const batch_tokens> entries) const {
    if (entries.empty()) {
        throw std::logic_error("executor::launch: a step has at least one entry");
    }
    if (launched_batches.size() >= 2) {
        throw std::logic_error("executor::launch: two steps are launched already");
    }
    std::unordered_set<sequence_id> entered;
    for (const auto& [id, token_count] : entries) {
        if (!entered.insert(id).second) {
            throw std::logic_error("executor::launch: a sequence has two entries");
        }
        if (token_count == 0 || token_count > pending_token_count(id)) {
            throw std::logic_error("executor::launch: an entry computes no token or more than its sequence's pending");
        }
        if (!is_steppable(id)) {
            throw std::logic_error("executor::launch: a sequence is not steppable");
        }
    }
    if (!launched_batches.empty() && !allows_overlap(entries)) {
        throw std::logic_error("executor::launch: the launched step and the entries do not allow overlap");
    }
}

// A verify entry's tokens are accepted, and its samples taken, at its write-back. Its pending token and drafts are
// awaited_token until then, and a sequence whose previous step is launched lags it by up to that step's drafts.
void executor::advance(batch& b) {
    const std::size_t draft_count = draft_state_slots - 1;
    for (batch::group* g : b.groups()) {
        for (batch::entry& e : g->entries) {
            sequence& seq = *e.seq;
            if (draft_layer && seq.end_of_draft_rows() == seq.history_position() && seq.history_position() > 0) {
                const layer_view& view = seq.views()[*draft_layer];
                e.prior_row = b.prior_lines.size();
                b.prior_lines.push_back(view.lines[(seq.next_index(*draft_layer) - 1) / layout.page_size]);
            }
        }
    }
    std::size_t first_hidden_row = 1 + b.prior_lines.size();
    for (batch::group* g : b.groups()) {
        const bool verifies = g == &b.verify_group;
        for (batch::entry& e : g->entries) {
            sequence& seq = *e.seq;
            const std::size_t lag = seq.launched_step_count() > 0 ? draft_count : 0;
            e.is_measured = batch::measures_mass(e);
            e.start_position = seq.history_position();
            bool drafts = false;
            if (draft_layer) {
                const std::optional<std::size_t>& token_limit = seq.token_limit();
                const bool ends_sequence = e.is_sampling
                    ? token_limit && seq.generated_token_count() + 1 >= *token_limit
                    : seq.pending_token_list().size() == e.token_count;
                const bool next_row = verifies || (!e.ends_cycle && !ends_sequence);
                // The next step's verify group then ends neither the cycle nor max_tokens, whatever this one accepts.
                drafts = next_row && (verifies || e.is_sampling) &&
                    seq.cycle_tokens_left() >= batch::forwarded(e) + lag + draft_count + 2;
                b.plan_draft_rows(e, verifies, first_hidden_row, next_row, drafts, *draft_layer, draft_count);
                first_hidden_row += batch::forwarded(e);
                if (verifies) {
                    const std::size_t remaining = token_limit ? *token_limit - seq.generated_token_count()
                                                              : b.verify_rows;
                    e.max_accepted_token_count = std::clamp<std::size_t>(remaining, 1, b.verify_rows);
                    e.measured_start = seq.measured_start();
                }
            }
            if (verifies) {
                e.end_position = seq.history_position();
                e.sampled_position = seq.history_position() + e.token_count;
                seq.launch_verify_step(e.token_count, drafts ? draft_count : 0);
            } else {
                e.end_position = seq.history_position() + e.token_count;
                e.sampled_position = seq.history_position() + seq.pending_token_list().size();
                seq.launch_step(
                    e.token_count,
                    e.is_sampling ? std::optional(e.first_sample_row) : std::nullopt,
                    e.draft ? e.draft->row_count : 0,
                    drafts ? draft_count : 0
                );
            }
            e.ends_sequence = !verifies && seq.has_launched_last_step();
        }
    }
}

// Shard 0 queues on the executor's thread while the others queue on their own, so every shard's kernels launch at
// once. Each shard's completion is taken once every shard has queued, before any later step queues.
task<void> executor::launch(std::span<const batch_tokens> entries) {
    check_launch(entries);
    std::unique_ptr<batch> b(new batch(*this, entries));
    if (b->sample_params.size() > sampled_row_capacity) {
        throw std::logic_error("executor::launch: the step samples more than sampled_row_limit rows");
    }
    b->allows_overlap = entries_allow_overlap(entries);
    link_previous(*b);
    advance(*b);
    for (batch::group* g : b->groups()) {
        g->mass_by_shard.assign(shards.size(), at::Tensor());
        g->hidden_by_shard.assign(shards.size(), at::Tensor());
    }
    b->shard_prior_hidden.assign(shards.size(), at::Tensor());
    b->verify_tokens.assign(shards.size(), at::Tensor());
    b->accepted_token_counts_by_shard.assign(shards.size(), at::Tensor());
    b->shard_next_tokens.assign(shards.size(), at::Tensor());
    std::vector<external_task<>> other_shards;
    for (std::size_t shard = 1; shard < shards.size(); shard++) {
        other_shards.push_back(run_on(*worker_loops[shard - 1], enqueue_on_shard(*b, shard)));
    }
    co_await enqueue_on_shard(*b, 0);
    for (external_task<>& other : other_shards) {
        co_await std::move(other);
    }
    for (std::size_t shard = 0; shard < shards.size(); shard++) {
        at::Device device = lines[shard].front().k.device();
        c10::cuda::CUDAGuard device_guard(device);
        b->ends.emplace_back().record(c10::cuda::getStreamFromExternal(streams[shard], device.index()));
        b->completions.push_back(completion_of(streams[shard]));
    }
    launched_batches.push_back(std::move(b));
}

void executor::link_previous(batch& b) const {
    if (launched_batches.empty() || b.verify_group.entries.empty()) {
        return;
    }
    const batch& previous = *launched_batches.back();
    b.previous_batch = &previous;
    const std::vector<batch::entry> previous_drafting_entries = previous.drafting_entries();
    for (batch::entry& e : b.verify_group.entries) {
        if (e.seq->launched_step_count() == 0) {
            continue;
        }
        for (std::size_t pv = 0; pv < previous.verify_group.entries.size(); pv++) {
            if (previous.verify_group.entries[pv].seq == e.seq) {
                e.previous_verify_row = pv;
            }
        }
        for (std::size_t d = 0; d < previous_drafting_entries.size(); d++) {
            if (previous_drafting_entries[d].seq == e.seq) {
                e.previous_drafting_row = d;
            }
        }
        if (!e.previous_verify_row || !e.previous_drafting_row) {
            throw std::logic_error(
                "executor::launch: a sequence with a launched step is not a verify and drafting entry of that step"
            );
        }
    }
}

task<void> executor::enqueue_on_shard(batch& b, std::size_t shard) {
    c10::cuda::CUDAGuard device_guard(lines[shard].front().k.device());
    b.enqueue(
        *shards[shard],
        lines[shard],
        shard_mass_heads[shard],
        draft_layer,
        stop_tokens[shard],
        streams[shard],
        shard,
        graphs.empty() ? nullptr : graphs[shard].get(),
        samples[shard],
        eager_counters[shard]
    );
    co_return;
}

// Each shard on its own thread, as the eager forwards' collectives need every shard at once.
void executor::capture_decode_graphs() {
    std::vector<std::exception_ptr> failures(shards.size());
    std::vector<std::thread> threads;
    for (std::size_t shard = 0; shard < shards.size(); shard++) {
        threads.emplace_back([this, shard, &failures] {
            try {
                at::Device device = lines[shard].front().k.device();
                c10::cuda::CUDAGuard device_guard(device);
                c10::cuda::CUDAStreamGuard stream_guard(
                    c10::cuda::getStreamFromExternal(streams[shard], device.index())
                );
                graphs[shard]->capture_startup();
                C10_CUDA_CHECK(cudaStreamSynchronize(streams[shard]));
            } catch (...) {
                failures[shard] = std::current_exception();
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    for (const std::exception_ptr& failure : failures) {
        if (failure) {
            std::rethrow_exception(failure);
        }
    }
}

}  // namespace pulsarlm
