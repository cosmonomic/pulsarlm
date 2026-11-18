#include "pulsarlm/engine/executor.hpp"
#include "pulsarlm/engine/sequence.hpp"

#include "pulsarlm/kernels/ops.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <bit>
#include <exception>
#include <thread>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <unordered_map>
#include <utility>
#include <string>
#include <stdexcept>
#include <vector>

namespace pulsarlm {

namespace {

std::size_t ceil_div(std::size_t n, std::size_t d) {
    return (n + d - 1) / d;
}

at::Tensor pinned(at::IntArrayRef sizes, at::ScalarType dtype) {
    return at::empty(sizes, at::TensorOptions().dtype(dtype).pinned_memory(true));
}

at::Tensor to_device(const at::Tensor& host, at::Device device) {
    return host.defined() ? host.to(host.options().device(device), /*non_blocking=*/true) : host;
}

group_batch to_device(const group_batch& host, at::Device device) {
    return group_batch{
        .tokens = to_device(host.tokens, device),
        .pos = to_device(host.pos, device),
        .history_positions = to_device(host.history_positions, device),
        .rejected_tokens = to_device(host.rejected_tokens, device),
        .page_size = host.page_size,
        .state_slots = host.state_slots,
        .cu_seqlens_q = to_device(host.cu_seqlens_q, device),
        .key_counts = to_device(host.key_counts, device),
        .cu_view_pages = to_device(host.cu_view_pages, device),
        .query_weights = to_device(host.query_weights, device),
    };
}

// Sets on_device's page mass_sum and weight_sum for the KV layers that measure mass, those with mass_heads above 0, to
// zeroed parts of one tensor, which it returns: per such layer in turn, its mass_sum, then its weight_sum.
// host_cu_view_pages is on the host.
at::Tensor zero_page_mass(
    group_batch& on_device,
    const at::Tensor& host_cu_view_pages,
    const std::vector<std::size_t>& mass_heads
) {
    std::int64_t row_size = host_cu_view_pages.size(1);
    const auto* cu_view_pages = host_cu_view_pages.const_data_ptr<std::int32_t>();
    auto layer_pages = [&](std::size_t layer) {
        return cu_view_pages[static_cast<std::int64_t>(layer) * row_size + row_size - 1];
    };
    std::int64_t total = 0;
    for (std::size_t layer = 0; layer < mass_heads.size(); layer++) {
        if (mass_heads[layer] > 0) {
            total += layer_pages(layer) * static_cast<std::int64_t>(mass_heads[layer] + 1);
        }
    }
    at::Tensor mass = at::zeros({total}, on_device.tokens.options().dtype(at::kFloat));
    on_device.page_mass_sum.resize(mass_heads.size());
    on_device.page_weight_sum.resize(mass_heads.size());
    std::int64_t first = 0;
    for (std::size_t layer = 0; layer < mass_heads.size(); layer++) {
        if (mass_heads[layer] > 0) {
            auto heads = static_cast<std::int64_t>(mass_heads[layer]);
            std::int64_t pages = layer_pages(layer);
            on_device.page_mass_sum[layer] = mass.narrow(0, first, pages * heads).view({pages, heads});
            first += pages * heads;
            on_device.page_weight_sum[layer] = mass.narrow(0, first, pages);
            first += pages;
        }
    }
    return mass;
}

}  // namespace

// The tokens each running sequence forwards in one step, the model inputs for them, and the tokens sampled from the
// step's logits.
class executor::batch {
    enum class group_kind { decode, verify, prefill };

    // decode_count decode entries first, each forwarding its sequence's only pending token, the last verify_count of
    // them with their drafts after it.
    std::vector<batch_entry> entries;
    std::size_t decode_count = 0;
    std::size_t verify_count = 0;
    // The tokens a verify entry forwards and samples after: its pending token and its drafts.
    std::size_t verify_rows = 0;
    // Pinned host tensors.
    step_batch inputs;
    // Whether a decode entry's token is awaited, -(r + 1) in the decode group's tokens for row r of the previous
    // launched step's samples.
    bool awaits_tokens = false;
    // Per entry, its first row among the rows sampled.
    std::vector<std::size_t> sample_rows;
    // Per row sampled, in entry order: a row of each entry that samples but a verify entry's verify_rows, the params
    // and draw.
    std::vector<sampling_params> sample_params;
    std::vector<sampling_draw> sample_draws;
    // Pinned host int64, one token per row sampled.
    at::Tensor sampled_tokens;
    // Per entry, as of its launch: whether it measures mass, the history position its tokens end at, the history
    // position of the token it samples, and whether it forwards its sequence's last tokens.
    std::vector<bool> measured;
    std::vector<std::size_t> step_starts;
    std::vector<std::size_t> step_ends;
    // Per entry, whether its tokens end its sequence's eviction cycle; a verify entry's is set at its write-back.
    std::vector<bool> ends_cycle;
    std::vector<std::size_t> sampled_positions;
    std::vector<bool> ends_sequence;
    // Per shard, completing once the step's work on its stream has, and an event recorded after that work.
    std::vector<external_task<cudaError_t>> completions;
    std::vector<at::cuda::CUDAEvent> ends;
    // Pinned host fp32, defined for a group that measures mass: the page mass of each KV layer that measures it in
    // turn, laid out as the layer's row of the group's cu_view_pages, its heads innermost; per shard, its heads, then
    // joined over the shards.
    std::vector<at::Tensor> shard_decode_mass;
    std::vector<at::Tensor> shard_verify_mass;
    std::vector<at::Tensor> shard_prefill_mass;
    at::Tensor decode_mass;
    at::Tensor verify_mass;
    at::Tensor prefill_mass;
    // Per entry whose draft rows start at its first token after an earlier step forwarded the position before it,
    // the line of the draft head's KV layer holding that position, and per shard, those lines' last_hidden as of the
    // step's launch; prior_rows gives each entry's row, if it has one.
    std::vector<std::int64_t> prior_lines;
    std::vector<std::optional<std::size_t>> prior_rows;
    std::vector<at::Tensor> shard_prior_hidden;
    // Per shard, the step's hidden rows, which the draft head's pass reads.
    std::vector<step_logits> shard_hidden;
    // The draft head's pass, queued after the step's sampling: per sequence writing draft rows, its entry, its rows'
    // tokens, -(r + 1) for the step's sampled row r, their RoPE positions and its key count after them in the draft
    // head's KV layer, and the verify entry it is, -1 for another; per row, its hidden row (draft_head::draft) as an
    // index into a zeros row, then the prior hidden rows, then the step's prefill, decode and verify hidden rows. A
    // verify entry's rows are the verify_rows positions after its pending token's, the row after its kept tokens
    // drafting: its first row's index plus kept - 1. Another sequence drafts from its last row.
    std::vector<batch_entry> draft_entries;
    std::vector<std::int64_t> draft_tokens;
    std::vector<std::int64_t> draft_positions;
    std::vector<std::int32_t> draft_key_counts;
    std::vector<std::int64_t> draft_sources;
    std::vector<std::int64_t> draft_logit_rows;
    std::vector<std::int64_t> draft_verify_index;
    // The sequences that draft, as indices among draft_entries, and per draft after the first, each one's RoPE
    // position and key count before a verify entry's kept count is added.
    std::vector<std::int64_t> drafting;
    std::vector<std::vector<std::int64_t>> later_positions;
    std::vector<std::vector<std::int32_t>> later_key_counts;
    // Per verify entry, the most tokens it may keep (max_tokens), and its sequence's measured_start.
    std::vector<std::int64_t> max_kept;
    std::vector<std::int64_t> verify_measured_starts;
    // Per verify entry whose sequence's previous step is launched and not written back, that step and the entry's
    // index among its verify entries and among its drafting sequences, -1 for the others: its tokens are that
    // step's next tokens, and its positions lie that step's kept count - 1 after those its inputs hold.
    const batch* previous = nullptr;
    std::vector<std::int64_t> previous_verify_rows;
    std::vector<std::int64_t> previous_drafting_rows;
    // Per shard, the verify group's tokens on its device.
    std::vector<at::Tensor> verify_tokens;
    // Per shard, each verify entry's kept tokens (int64 [verify_count]), and each drafting sequence's next token then
    // its drafts (int64 [drafting, verify_rows]); shard 0's copies to the host.
    std::vector<at::Tensor> shard_kept;
    std::vector<at::Tensor> shard_next_tokens;
    at::Tensor kept;
    at::Tensor next_tokens;

    // The decode and verify groups' page-table widths, and whether their forwards replay graphs.
    std::size_t decode_width = 0;
    bool decode_replays = false;
    std::size_t verify_width = 0;
    bool verify_replays = false;

    // With graph_pages, a decode group whose tables fit graph_pages pages takes tables that wide and replays a graph.
    batch(
        std::span<const batch_entry> step_entries,
        std::size_t page_size,
        std::size_t state_slots,
        std::optional<std::size_t> graph_pages
    );

    // A verify group measures each row's mass apart (group_batch::cu_view_pages), its queries weighing 1 from
    // measured_start on: the rows its write-back keeps are then weighed as one step's.
    static group_batch
    make_group(std::span<const batch_entry> group, std::size_t page_size, std::size_t state_slots, group_kind kind);
    // The tokens e forwards: its pending tokens', and its drafts.
    static std::size_t forwarded(const batch_entry& e) noexcept;
    // Whether entry index is a plain decode, a verify or a prefill entry.
    group_kind kind_of(std::size_t index) const noexcept;
    // The first row of entry index in its group.
    std::size_t first_row(std::size_t index) const noexcept;
    // The widest table among the group's sequences.
    static std::size_t table_width(std::span<const batch_entry> group) noexcept;
    static bool measures_mass(std::span<const batch_entry> group) noexcept;
    // Whether e forwards a query that adds mass.
    static bool measures_mass(const batch_entry& e) noexcept;
    // The group's tables of lines, or of positions with positions, on shard's device, as group_batch lays them out.
    static at::Tensor page_tables(
        std::span<const batch_entry> group,
        std::size_t n_kv_layers,
        bool positions,
        std::size_t shard,
        std::size_t width
    );
    // Queues on stream shard's copies of the inputs, the filling of the awaited tokens from samples, the copy of the
    // prior hidden rows from draft_layer's lines, its forward, the copies of its page mass to the host and the
    // sampling into samples; shard 0 also queues the copy of the sampled tokens to the host. mass_heads is the shard's
    // part of the executor's. graphs is the shard's when decode_replays. Every shard's call may run at once, each on
    // its own thread.
    void enqueue(
        model& shard_model,
        std::span<const kv> lines,
        const std::vector<std::size_t>& mass_heads,
        std::optional<std::size_t> draft_layer,
        const at::Tensor& stop_tokens,
        cudaStream_t stream,
        std::size_t shard,
        decode_graphs* graphs,
        const at::Tensor& samples,
        at::Tensor& counters
    );
    // Adds to the draft head's pass the rows of entry index's sequence, at launch before it advances: a verify entry's
    // verify_rows rows, another's from its first position without one through its tokens and, with next_row, its
    // next token, the sequence drafting draft_count tokens after them with drafts.
    void
    plan_draft_rows(std::size_t index, bool next_row, bool drafts, std::size_t draft_layer, std::size_t draft_count);
    // Queues on stream shard, after the step's sampling into samples, the verify entries' acceptance of their samples,
    // stop_tokens ending a run (int64), and the draft head's pass, replayed from graphs when every entry verifies and
    // drafts and its verify group replays; shard 0 also queues the copies of the kept counts and next tokens to the
    // host.
    void enqueue_drafts(
        model& shard_model,
        std::span<const kv> lines,
        const at::Tensor& samples,
        const at::Tensor& stop_tokens,
        std::size_t shard,
        decode_graphs* graphs,
        at::Tensor& counters
    );
    // Whether the step replays a decode_graphs verify step: every entry verifies and drafts, and the verify group
    // replays.
    bool steps_replay() const noexcept;
    // Queues on device, shard's, the replay of its verify step from graphs, on_device holding the verify group and
    // verify_mass its page mass, then the copies of its outputs: into samples, and on shard 0 to the host.
    void enqueue_step(
        step_batch& on_device,
        const at::Tensor& verify_mass,
        std::size_t shard,
        at::Device device,
        decode_graphs* graphs,
        const at::Tensor& samples
    );
    // Takes into g, the verify group on shard's device, the previous step's next tokens of the entries whose previous
    // step is launched, and moves their positions by its kept count - 1.
    void take_previous_tokens(group_batch& g, std::size_t shard, at::Device device) const;
    // On device, shard's, per row sampled, the tokens the previous step of a verify entry's sequence keeps, by which its
    // draw counts lag; 0 for the other rows.
    at::Tensor previous_draw_shifts(std::size_t shard, at::Device device) const;
    // On device, shard's, each verify entry's kept count - 1 of its previous step, 0 without one, int64 [verify_count].
    at::Tensor previous_shift(std::size_t shard, at::Device device) const;
    // Joins the shards' page mass, each shard's heads after the previous shards', once their copies completed.
    void join_mass(
        const std::vector<std::vector<std::size_t>>& shard_mass_heads,
        const std::vector<std::size_t>& mass_heads
    );
    // Per KV layer, the mass_sum the step gave each of entry index's view pages, mass_heads[layer] values per page, and
    // their weight_sum; empty for a layer that measures none. The entry's group measures mass.
    struct layer_step_mass {
        std::span<const float> mass_sum;
        std::span<const float> weight_sum;
    };
    std::vector<layer_step_mass> step_mass(std::size_t index, const std::vector<std::size_t>& mass_heads) const;
    // step_mass of row row of verify entry index, its query weighing 1.
    std::vector<layer_step_mass>
    verify_row_mass(std::size_t index, std::size_t row, const std::vector<std::size_t>& mass_heads) const;
    // The page mass of the view view of group from mass, laid out as join_mass leaves it.
    static std::vector<layer_step_mass> view_mass(
        const group_batch& group,
        const at::Tensor& mass,
        std::size_t view,
        const std::vector<std::size_t>& mass_heads
    );

    friend class executor;
};

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
    std::size_t draft_tokens
)
: shards(std::move(shards)),
  cache(std::move(cache)),
  store(store),
  streams(std::move(streams)),
  copy_streams(std::move(copy_streams)),
  worker_loops(std::move(worker_loops)),
  page_size(0),
  root(root),
  stop_ids(std::move(stop_ids)),
  graph_pages(graph_pages),
  draft_state_slots(draft_tokens + 1) {
    assert(this->shards.size() == this->cache.shard_count() && this->streams.size() == this->shards.size());
    assert(this->worker_loops.size() + 1 == this->shards.size());
    shard_mass_heads.resize(this->shards.size());
    lines.resize(this->shards.size());
    for (std::size_t shard = 0; shard < this->shards.size(); shard++) {
        std::vector<kv_layer> layers = this->shards[shard]->kv_layers();
        assert(layers.size() == this->cache.layer_count());
        for (std::size_t layer = 0; layer < layers.size(); layer++) {
            lines[shard].push_back(this->cache.tensors(layer, shard));
            bool layer_recurrent = layers[layer].state.has_value();
            bool evictable = !layers[layer].sliding_window && !layer_recurrent && !layers[layer].mirrors_held_pages;
            assert(!evictable || layers[layer].n_q_heads > 0);
            shard_mass_heads[shard].push_back(evictable ? layers[layer].n_readers * layers[layer].n_q_heads : 0);
            if (shard == 0) {
                sliding_windows.push_back(layers[layer].sliding_window);
                recurrent.push_back(layer_recurrent);
                state_slots.push_back(slots_of(layers[layer]));
                mirrors.push_back(layers[layer].mirrors_held_pages);
                mass_heads.push_back(0);
                if (page_size == 0 && !layer_recurrent) {
                    page_size = static_cast<std::size_t>(lines[0].back().k.size(1));
                }
            }
            assert(sliding_windows[layer] == layers[layer].sliding_window && recurrent[layer] == layer_recurrent);
            assert(mirrors[layer] == layers[layer].mirrors_held_pages);
            assert(state_slots[layer] == slots_of(layers[layer]));
            mass_heads[layer] += shard_mass_heads[shard].back();
            if (shard == 0 && draft_tokens > 0 && layers[layer].last_hidden_dim > 0) {
                draft_layer = layer;
            }
        }
    }
    assert(draft_tokens == 0 || (draft_layer && this->shards.front()->drafter() != nullptr));
    assert(page_size > 0);
    for (std::size_t shard = 0; shard < this->shards.size(); shard++) {
        samples.push_back(at::zeros({max_sampled_rows}, lines[shard].front().k.options().dtype(at::kLong)));
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
                    page_size,
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

executor::batch::batch(
    std::span<const batch_entry> step_entries,
    std::size_t page_size,
    std::size_t state_slots,
    std::optional<std::size_t> graph_pages
)
: entries(step_entries.begin(), step_entries.end()) {
    auto first_prefill = std::ranges::find_if(entries, [](const batch_entry& e) {
        return e.seq->pending_tokens.size() > 1;
    });
    decode_count = static_cast<std::size_t>(first_prefill - entries.begin());
    auto first_verify = std::stable_partition(entries.begin(), first_prefill, [](const batch_entry& e) {
        return e.seq->drafts.empty();
    });
    verify_count = static_cast<std::size_t>(first_prefill - first_verify);
    const std::size_t plain_count = decode_count - verify_count;
    std::span<const batch_entry> entry_span(entries);
    if (verify_count > 0) {
        verify_rows = forwarded(entries[plain_count]);
        assert(std::ranges::all_of(entry_span.subspan(plain_count, verify_count), [&](const batch_entry& e) {
            return e.samples && forwarded(e) == verify_rows;
        }));
    }
    if (plain_count > 0) {
        inputs.decode_group = make_group(entry_span.first(plain_count), page_size, state_slots, group_kind::decode);
        decode_width = table_width(entry_span.first(plain_count));
        decode_replays = graph_pages && decode_width <= *graph_pages;
        if (decode_replays) {
            decode_width = *graph_pages;
        }
    }
    if (verify_count > 0) {
        inputs.verify_group = make_group(
            entry_span.subspan(plain_count, verify_count),
            page_size,
            state_slots,
            group_kind::verify
        );
        verify_width = table_width(entry_span.subspan(plain_count, verify_count));
        verify_replays = graph_pages && verify_width <= *graph_pages;
        if (verify_replays) {
            verify_width = *graph_pages;
        }
    }
    if (entries.size() > decode_count) {
        inputs
            .prefill_group = make_group(entry_span.subspan(decode_count), page_size, state_slots, group_kind::prefill);
    }
    awaits_tokens = std::ranges::any_of(entry_span.first(plain_count), [](const batch_entry& e) {
        return e.seq->pending_tokens.front() == sequence::awaited_token;
    });
    // A verify entry's draws advance by the tokens its write-back keeps.
    for (std::size_t i = 0; i < entries.size(); i++) {
        const batch_entry& e = entries[i];
        sample_rows.push_back(sample_params.size());
        if (!e.samples) {
            continue;
        }
        if (kind_of(i) == group_kind::verify) {
            for (std::size_t row = 0; row < verify_rows; row++) {
                sample_params.push_back(e.seq->sampling);
                sample_draws.push_back(sampling_draw{.seed = e.seq->seed, .count = e.seq->draw_count + row});
            }
        } else {
            sample_params.push_back(e.seq->sampling);
            sample_draws.push_back(sampling_draw{.seed = e.seq->seed, .count = e.seq->draw_count++});
        }
    }
}

std::size_t executor::batch::forwarded(const batch_entry& e) noexcept {
    return e.token_count + e.seq->drafts.size();
}

executor::batch::group_kind executor::batch::kind_of(std::size_t index) const noexcept {
    if (index >= decode_count) {
        return group_kind::prefill;
    }
    return index >= decode_count - verify_count ? group_kind::verify : group_kind::decode;
}

std::size_t executor::batch::first_row(std::size_t index) const noexcept {
    switch (kind_of(index)) {
    case group_kind::decode:
        return index;
    case group_kind::verify:
        return (index - (decode_count - verify_count)) * verify_rows;
    case group_kind::prefill:
        return static_cast<std::size_t>(
            inputs.prefill_group->cu_seqlens_q.const_data_ptr<std::int32_t>()[index - decode_count]
        );
    }
    return 0;
}

group_batch executor::batch::make_group(
    std::span<const batch_entry> group,
    std::size_t page_size,
    std::size_t state_slots,
    group_kind kind
) {
    std::size_t token_count = 0;
    for (const batch_entry& e : group) {
        token_count += forwarded(e);
    }
    std::size_t n_layers = group.front().seq->layers.size();
    auto n_tokens = static_cast<std::int64_t>(token_count);
    auto n_seqs = static_cast<std::int64_t>(group.size());
    auto n_kv_layers = static_cast<std::int64_t>(n_layers);
    group_batch g{
        .tokens = pinned({n_tokens}, at::kLong),
        .pos = pinned({n_tokens}, at::kLong),
    };
    at::Tensor key_counts = pinned({n_kv_layers, n_seqs}, at::kInt);
    at::Tensor query_starts = pinned({n_seqs + 1}, at::kInt);

    auto* tokens = g.tokens.mutable_data_ptr<std::int64_t>();
    auto* positions = g.pos.mutable_data_ptr<std::int64_t>();
    auto* key_count = key_counts.mutable_data_ptr<std::int32_t>();
    auto* query_start = query_starts.mutable_data_ptr<std::int32_t>();
    std::size_t first_row = 0;
    query_start[0] = 0;
    for (std::size_t s = 0; s < group.size(); s++) {
        const sequence& seq = *group[s].seq;
        std::size_t first_position = seq.next_position();
        const std::size_t count = forwarded(group[s]);
        for (std::size_t i = 0; i < count; i++) {
            const std::int64_t token = i < group[s].token_count ? seq.pending_tokens[i]
                                                                : seq.drafts[i - group[s].token_count];
            tokens[first_row + i] = token == sequence::awaited_token ? -static_cast<std::int64_t>(seq.awaited_row) - 1
                                                                     : token;
            positions[first_row + i] = static_cast<std::int64_t>(first_position + i);
        }
        for (std::size_t layer = 0; layer < n_layers; layer++) {
            std::size_t start = seq.next_index(layer);
            key_count[layer * group.size() + s] = static_cast<std::int32_t>(start + count);
        }
        first_row += count;
        query_start[s + 1] = static_cast<std::int32_t>(first_row);
    }
    g.key_counts = key_counts;
    if (kind != group_kind::decode) {
        g.cu_seqlens_q = query_starts;
    }
    g.history_positions = pinned({n_seqs}, at::kLong);
    auto* history_position = g.history_positions.mutable_data_ptr<std::int64_t>();
    for (std::size_t s = 0; s < group.size(); s++) {
        history_position[s] = static_cast<std::int64_t>(group[s].seq->history_position());
    }
    g.rejected_tokens = pinned({n_seqs}, at::kInt);
    auto* rejected = g.rejected_tokens.mutable_data_ptr<std::int32_t>();
    for (std::size_t s = 0; s < group.size(); s++) {
        rejected[s] = static_cast<std::int32_t>(group[s].seq->rejected_tokens);
    }
    g.page_size = static_cast<std::int64_t>(page_size);
    g.state_slots = static_cast<std::int64_t>(state_slots);
    if (measures_mass(group)) {
        // A verify group's views are its rows'.
        const bool per_row = kind == group_kind::verify;
        const std::int64_t views = per_row ? n_tokens : n_seqs;
        g.cu_view_pages = pinned({n_kv_layers, views + 1}, at::kInt);
        g.query_weights = pinned({n_tokens}, at::kFloat);
        auto* cu_view_pages = g.cu_view_pages.mutable_data_ptr<std::int32_t>();
        for (std::size_t layer = 0; layer < n_layers; layer++) {
            std::int32_t* row = cu_view_pages + layer * static_cast<std::size_t>(views + 1);
            row[0] = 0;
            std::size_t view = 0;
            for (const batch_entry& e : group) {
                const auto pages = static_cast<std::int32_t>(e.seq->layers[layer].lines.size());
                for (std::size_t i = 0; i < (per_row ? forwarded(e) : 1); i++, view++) {
                    row[view + 1] = row[view] + pages;
                }
            }
        }
        std::span<float> weights(g.query_weights.mutable_data_ptr<float>(), token_count);
        std::size_t first = 0;
        for (const batch_entry& e : group) {
            const std::size_t count = forwarded(e);
            const std::size_t start = e.seq->history_position();
            if (per_row) {
                for (std::size_t i = 0; i < count; i++) {
                    weights[first + i] = start + i >= e.seq->measured_start() ? 1.0f : 0.0f;
                }
            } else {
                e.seq->fill_query_weights(start, weights.subspan(first, count));
            }
            first += count;
        }
    }
    return g;
}

std::size_t executor::batch::table_width(std::span<const batch_entry> group) noexcept {
    std::size_t width = 0;
    for (const batch_entry& e : group) {
        width = std::max(width, e.seq->table_width());
    }
    return width;
}

bool executor::batch::measures_mass(std::span<const batch_entry> group) noexcept {
    return std::ranges::any_of(group, [](const batch_entry& e) { return measures_mass(e); });
}

// A sequence whose previous step is launched may lie up to that step's drafts further on.
bool executor::batch::measures_mass(const batch_entry& e) noexcept {
    const std::size_t lag = e.seq->steps_in_flight > 0 ? e.seq->drafts.size() : 0;
    return e.seq->eviction.can_evict() && e.seq->history_position() + forwarded(e) + lag > e.seq->measured_start();
}

at::Tensor executor::batch::page_tables(
    std::span<const batch_entry> group,
    std::size_t n_kv_layers,
    bool positions,
    std::size_t shard,
    std::size_t width
) {
    std::vector<at::Tensor> tables;
    std::vector<std::size_t> page_counts;
    for (const batch_entry& e : group) {
        tables.push_back(positions ? e.seq->position_tables[shard] : e.seq->tables[shard]);
        page_counts.push_back(e.seq->table_width());
    }
    return gather_page_tables_cuda(
        tables,
        page_counts,
        static_cast<std::int64_t>(n_kv_layers),
        static_cast<std::int64_t>(width)
    );
}

void executor::batch::enqueue(
    model& shard_model,
    std::span<const kv> lines,
    const std::vector<std::size_t>& mass_heads,
    std::optional<std::size_t> draft_layer,
    const at::Tensor& stop_tokens,
    cudaStream_t stream,
    std::size_t shard,
    decode_graphs* graphs,
    const at::Tensor& samples,
    at::Tensor& counters
) {
    at::Device device = lines.front().k.device();
    c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(stream, device.index()));
    step_batch on_device;
    std::span<batch_entry> entry_span(entries);
    const std::size_t plain_count = decode_count - verify_count;
    if (draft_layer && !prior_lines.empty()) {
        at::Tensor line_ids = at::tensor(prior_lines, at::TensorOptions().dtype(at::kLong)).to(device, true);
        shard_prior_hidden[shard] = lines[*draft_layer].last_hidden.index_select(0, line_ids);
    }
    at::Tensor decode_on_device, verify_on_device, prefill_on_device;
    if (inputs.decode_group) {
        on_device.decode_group = to_device(*inputs.decode_group, device);
        if (awaits_tokens && !decode_replays) {
            fill_awaited_tokens_cuda(on_device.decode_group->tokens, samples);
        }
        std::span<batch_entry> group = entry_span.first(plain_count);
        on_device.decode_group->page_tables = page_tables(group, lines.size(), false, shard, decode_width);
        on_device.decode_group->page_positions = page_tables(group, lines.size(), true, shard, decode_width);
        if (inputs.decode_group->cu_view_pages.defined()) {
            decode_on_device = zero_page_mass(*on_device.decode_group, inputs.decode_group->cu_view_pages, mass_heads);
        }
    }
    if (inputs.verify_group) {
        group_batch& g = on_device.verify_group.emplace(to_device(*inputs.verify_group, device));
        std::span<batch_entry> group = entry_span.subspan(plain_count, verify_count);
        g.page_tables = page_tables(group, lines.size(), false, shard, verify_width);
        g.page_positions = page_tables(group, lines.size(), true, shard, verify_width);
        if (previous) {
            take_previous_tokens(g, shard, device);
        }
        if (inputs.verify_group->cu_view_pages.defined()) {
            verify_on_device = zero_page_mass(g, inputs.verify_group->cu_view_pages, mass_heads);
        }
        verify_tokens[shard] = g.tokens;
    }
    if (inputs.prefill_group) {
        on_device.prefill_group = to_device(*inputs.prefill_group, device);
        std::span<batch_entry> group = entry_span.subspan(decode_count);
        std::size_t width = table_width(group);
        on_device.prefill_group->page_tables = page_tables(group, lines.size(), false, shard, width);
        on_device.prefill_group->page_positions = page_tables(group, lines.size(), true, shard, width);
        const at::Tensor& cu_seqlens_q = on_device.prefill_group->cu_seqlens_q;
        on_device.prefill_group->logit_rows = cu_seqlens_q.narrow(0, 1, cu_seqlens_q.size(0) - 1).to(at::kLong) - 1;
        if (inputs.prefill_group->cu_view_pages.defined()) {
            prefill_on_device = zero_page_mass(
                *on_device.prefill_group,
                inputs.prefill_group->cu_view_pages,
                mass_heads
            );
        }
    }
    if (steps_replay()) {
        enqueue_step(on_device, verify_on_device, shard, device, graphs, samples);
        return;
    }
    // A replayed group reads its graph's own counters.
    std::int64_t needed = 0;
    if (on_device.decode_group && !decode_replays) {
        needed = shard_model.arrival_counter_count(
            static_cast<std::int64_t>(plain_count),
            on_device.decode_group->tokens.size(0)
        );
    }
    if (on_device.verify_group && !verify_replays) {
        needed = std::max(
            needed,
            shard_model
                .arrival_counter_count(static_cast<std::int64_t>(verify_count), on_device.verify_group->tokens.size(0))
        );
    }
    if (on_device.prefill_group) {
        needed = std::max(
            needed,
            shard_model.arrival_counter_count(
                static_cast<std::int64_t>(entries.size() - decode_count),
                on_device.prefill_group->tokens.size(0)
            )
        );
    }
    if (!counters.defined() || counters.numel() < needed) {
        counters = at::zeros({needed}, at::device(device).dtype(at::kInt));
    }
    for (std::optional<group_batch>* group :
         {&on_device.decode_group, &on_device.verify_group, &on_device.prefill_group}) {
        if (*group) {
            (*group)->arrival_counters = counters;
        }
    }
    std::optional<group_batch> replayed_decode;
    std::optional<group_batch> replayed_verify;
    if (decode_replays) {
        replayed_decode = std::move(on_device.decode_group);
        on_device.decode_group.reset();
    }
    if (verify_replays) {
        replayed_verify = std::move(on_device.verify_group);
        on_device.verify_group.reset();
    }
    step_logits logits = shard_model.forward(on_device, lines);
    if (replayed_decode) {
        auto [decode_logits, decode_hidden] = graphs->decode(*replayed_decode);
        logits.decode_logits = decode_logits;
        if (decode_hidden.defined()) {
            logits.decode_hidden = decode_hidden;
        }
    }
    if (replayed_verify) {
        auto [verify_logits, verify_hidden] = graphs->verify(*replayed_verify);
        logits.verify_logits = verify_logits;
        logits.verify_hidden = verify_hidden;
    }
    for (auto [by_shard, on_shard] : {
             std::pair{&shard_decode_mass, &decode_on_device},
             std::pair{&shard_verify_mass, &verify_on_device},
             std::pair{&shard_prefill_mass, &prefill_on_device},
         }) {
        if (on_shard->defined()) {
            (*by_shard)[shard] = pinned(on_shard->sizes(), at::kFloat);
            (*by_shard)[shard].copy_(*on_shard, /*non_blocking=*/true);
        }
    }
    std::vector<at::Tensor> rows;
    for (std::size_t i = 0; i < entries.size(); i++) {
        if (!entries[i].samples) {
            continue;
        }
        switch (kind_of(i)) {
        case group_kind::decode:
            rows.push_back(logits.decode_logits->narrow(0, static_cast<std::int64_t>(i), 1));
            break;
        case group_kind::verify:
            rows.push_back(logits.verify_logits->narrow(
                0,
                static_cast<std::int64_t>(first_row(i)),
                static_cast<std::int64_t>(verify_rows)
            ));
            break;
        case group_kind::prefill:
            rows.push_back(logits.prefill_logits->narrow(0, static_cast<std::int64_t>(i - decode_count), 1));
            break;
        }
    }
    shard_hidden[shard] = step_logits{
        .prefill_hidden = logits.prefill_hidden,
        .decode_hidden = logits.decode_hidden,
        .verify_hidden = logits.verify_hidden,
    };
    if (!rows.empty()) {
        at::Tensor tokens = previous
            ? sample(at::cat(rows), sample_params, sample_draws, previous_draw_shifts(shard, device))
            : sample(at::cat(rows), sample_params, sample_draws);
        samples.narrow(0, 0, tokens.size(0)).copy_(tokens);
        if (shard == 0) {
            sampled_tokens = pinned({tokens.size(0)}, at::kLong);
            sampled_tokens.copy_(tokens, /*non_blocking=*/true);
        }
    }
    if (draft_layer && !draft_entries.empty()) {
        enqueue_drafts(shard_model, lines, samples, stop_tokens, shard, graphs, counters);
    }
}

at::Tensor executor::batch::previous_draw_shifts(std::size_t shard, at::Device device) const {
    const auto longs = at::TensorOptions().dtype(at::kLong);
    std::vector<std::int64_t> rows;
    std::vector<std::int64_t> verify_entries;
    for (std::size_t v = 0; v < previous_verify_rows.size(); v++) {
        if (previous_verify_rows[v] < 0) {
            continue;
        }
        for (std::size_t j = 0; j < verify_rows; j++) {
            rows.push_back(static_cast<std::int64_t>(sample_rows[decode_count - verify_count + v] + j));
            verify_entries.push_back(static_cast<std::int64_t>(v));
        }
    }
    at::Tensor shifts = at::zeros(
        {static_cast<std::int64_t>(sample_params.size())},
        at::device(device).dtype(at::kLong)
    );
    if (rows.empty()) {
        return shifts;
    }
    at::Tensor kept_before = previous_shift(shard, device) + 1;
    return shifts.index_copy(
        0,
        at::tensor(rows, longs).to(device, /*non_blocking=*/true),
        kept_before.index_select(0, at::tensor(verify_entries, longs).to(device, /*non_blocking=*/true))
    );
}

bool executor::batch::steps_replay() const noexcept {
    return verify_replays && verify_count == entries.size() && drafting.size() == entries.size();
}

void executor::batch::enqueue_step(
    step_batch& on_device,
    const at::Tensor& verify_mass,
    std::size_t shard,
    at::Device device,
    decode_graphs* graphs,
    const at::Tensor& samples
) {
    const auto longs = at::TensorOptions().dtype(at::kLong);
    auto on_device_tensor = [&](const auto& values) {
        return at::tensor(values, longs).to(device, /*non_blocking=*/true);
    };
    const auto rows = static_cast<std::int64_t>(verify_rows);
    const auto sequences = static_cast<std::int64_t>(verify_count);
    sampling_inputs sampling = sampling_inputs_of(sample_params, sample_draws, device);
    at::Tensor shift = at::zeros({sequences}, at::device(device).dtype(at::kLong));
    if (previous) {
        sampling.draws.select(0, 1).add_(previous_draw_shifts(shard, device));
        shift = previous_shift(shard, device);
    }
    std::vector<std::int64_t> later_positions_flat;
    std::vector<std::int64_t> later_key_counts_flat;
    for (std::size_t j = 0; j < later_positions.size(); j++) {
        later_positions_flat.insert(later_positions_flat.end(), later_positions[j].begin(), later_positions[j].end());
        later_key_counts_flat
            .insert(later_key_counts_flat.end(), later_key_counts[j].begin(), later_key_counts[j].end());
    }
    const auto later = static_cast<std::int64_t>(later_positions.size());
    std::vector<std::int64_t> key_counts(draft_key_counts.begin(), draft_key_counts.end());
    auto [sampled, kept_counts, next] = graphs->verify_step(
        *on_device.verify_group,
        sampling,
        on_device_tensor(max_kept),
        on_device_tensor(draft_positions) + at::repeat_interleave(shift, rows),
        on_device_tensor(key_counts) + shift,
        on_device_tensor(later_positions_flat).view({later, sequences}) + shift,
        on_device_tensor(later_key_counts_flat).view({later, sequences}) + shift
    );
    shard_kept[shard] = kept_counts;
    shard_next_tokens[shard] = next;
    samples.narrow(0, 0, sampled.size(0)).copy_(sampled);
    if (verify_mass.defined()) {
        shard_verify_mass[shard] = pinned(verify_mass.sizes(), at::kFloat);
        shard_verify_mass[shard].copy_(verify_mass, /*non_blocking=*/true);
    }
    if (shard == 0) {
        sampled_tokens = pinned(sampled.sizes(), at::kLong);
        sampled_tokens.copy_(sampled, /*non_blocking=*/true);
        kept = pinned(kept_counts.sizes(), at::kLong);
        kept.copy_(kept_counts, /*non_blocking=*/true);
        next_tokens = pinned(next.sizes(), at::kLong);
        next_tokens.copy_(next, /*non_blocking=*/true);
    }
}

void executor::batch::take_previous_tokens(group_batch& g, std::size_t shard, at::Device device) const {
    const auto longs = at::TensorOptions().dtype(at::kLong);
    const auto rows = static_cast<std::int64_t>(verify_rows);
    std::vector<std::int64_t> entries_with_previous;
    std::vector<std::int64_t> drafting_rows;
    for (std::size_t v = 0; v < previous_drafting_rows.size(); v++) {
        if (previous_drafting_rows[v] >= 0) {
            entries_with_previous.push_back(static_cast<std::int64_t>(v));
            drafting_rows.push_back(previous_drafting_rows[v]);
        }
    }
    g.tokens.view({static_cast<std::int64_t>(verify_count), rows})
        .index_copy_(
            0,
            at::tensor(entries_with_previous, longs).to(device, /*non_blocking=*/true),
            previous->shard_next_tokens[shard]
                .index_select(0, at::tensor(drafting_rows, longs).to(device, /*non_blocking=*/true))
        );
    at::Tensor shift = previous_shift(shard, device);
    g.pos += at::repeat_interleave(shift, rows);
    g.key_counts += shift.to(at::kInt).unsqueeze(0);
    g.history_positions += shift;
    g.rejected_tokens -= shift.to(at::kInt);
    if (g.query_weights.defined()) {
        at::Tensor row_positions = at::repeat_interleave(g.history_positions, rows) +
            at::arange(rows, g.history_positions.options()).repeat({static_cast<std::int64_t>(verify_count)});
        at::Tensor measured_starts = at::tensor(verify_measured_starts, longs).to(device, /*non_blocking=*/true);
        g.query_weights = row_positions.ge(at::repeat_interleave(measured_starts, rows)).to(at::kFloat);
    }
}

std::vector<executor::batch::layer_step_mass>
executor::batch::step_mass(std::size_t index, const std::vector<std::size_t>& mass_heads) const {
    if (kind_of(index) == group_kind::decode) {
        return view_mass(*inputs.decode_group, decode_mass, index, mass_heads);
    }
    assert(kind_of(index) == group_kind::prefill);
    return view_mass(*inputs.prefill_group, prefill_mass, index - decode_count, mass_heads);
}

std::vector<executor::batch::layer_step_mass>
executor::batch::verify_row_mass(std::size_t index, std::size_t row, const std::vector<std::size_t>& mass_heads) const {
    return view_mass(*inputs.verify_group, verify_mass, first_row(index) + row, mass_heads);
}

std::vector<executor::batch::layer_step_mass> executor::batch::view_mass(
    const group_batch& group,
    const at::Tensor& mass,
    std::size_t view,
    const std::vector<std::size_t>& mass_heads
) {
    const float* values = mass.const_data_ptr<float>();
    auto row_size = static_cast<std::size_t>(group.cu_view_pages.size(1));
    const auto* cu_view_pages = group.cu_view_pages.const_data_ptr<std::int32_t>();
    std::vector<layer_step_mass> by_layer(mass_heads.size());
    std::size_t first = 0;
    for (std::size_t layer = 0; layer < mass_heads.size(); layer++) {
        std::size_t heads = mass_heads[layer];
        if (heads == 0) {
            continue;
        }
        const std::int32_t* row = cu_view_pages + layer * row_size;
        auto first_page = static_cast<std::size_t>(row[view]);
        auto pages = static_cast<std::size_t>(row[view + 1] - row[view]);
        auto layer_pages = static_cast<std::size_t>(row[row_size - 1]);
        by_layer[layer] = layer_step_mass{
            .mass_sum = std::span(values + first + first_page * heads, pages * heads),
            .weight_sum = std::span(values + first + layer_pages * heads + first_page, pages),
        };
        first += layer_pages * (heads + 1);
    }
    return by_layer;
}

void executor::batch::plan_draft_rows(
    std::size_t index,
    bool next_row,
    bool drafts,
    std::size_t draft_layer,
    std::size_t draft_count
) {
    const batch_entry& e = entries[index];
    sequence& seq = *e.seq;
    const std::size_t start = seq.history_position();
    const std::size_t next_index = seq.next_index(draft_layer);
    const std::size_t next_position = seq.next_position();
    auto rows_of = [](const std::optional<group_batch>& group) {
        return group ? static_cast<std::size_t>(group->tokens.size(0)) : 0;
    };
    std::size_t step_source = 1 + prior_lines.size() + first_row(index);
    if (kind_of(index) != group_kind::prefill) {
        step_source += rows_of(inputs.prefill_group);
    }
    if (kind_of(index) == group_kind::verify) {
        step_source += rows_of(inputs.decode_group);
    }
    const auto sampled = static_cast<std::int64_t>(sample_rows[index]);
    const auto first_draft_row = static_cast<std::int64_t>(draft_tokens.size());
    std::size_t rows = 0;
    // The position after the row that drafts, before a verify entry's kept count is added.
    std::size_t drafted_after = 0;
    if (kind_of(index) == group_kind::verify) {
        rows = verify_rows;
        for (std::size_t j = 0; j < rows; j++) {
            draft_tokens.push_back(-(sampled + static_cast<std::int64_t>(j)) - 1);
            draft_positions.push_back(static_cast<std::int64_t>(next_position + j + 1));
            draft_sources.push_back(static_cast<std::int64_t>(step_source + j));
        }
        draft_key_counts.push_back(static_cast<std::int32_t>(next_index + rows + 1));
        draft_logit_rows.push_back(first_draft_row);
        draft_verify_index.push_back(static_cast<std::int64_t>(index - (decode_count - verify_count)));
    } else {
        const std::size_t first = seq.draft_rows_end;
        const std::size_t next = start + e.token_count;
        const std::size_t end = next_row ? next + 1 : next;
        assert((first == start || first == start + 1) && (next_row || !drafts));
        for (std::size_t t = first; t < end; t++) {
            if (t < next) {
                draft_tokens.push_back(seq.pending_tokens[t - start]);
            } else {
                draft_tokens.push_back(e.samples ? -sampled - 1 : seq.pending_tokens[e.token_count]);
            }
            draft_positions.push_back(static_cast<std::int64_t>(next_position + t - start));
            if (t == 0) {
                draft_sources.push_back(0);
            } else if (t == start) {
                assert(prior_rows[index]);
                draft_sources.push_back(static_cast<std::int64_t>(1 + *prior_rows[index]));
            } else {
                draft_sources.push_back(static_cast<std::int64_t>(step_source + t - 1 - start));
            }
        }
        rows = end - first;
        seq.draft_rows_end = end;
        if (rows == 0) {
            return;
        }
        draft_key_counts.push_back(static_cast<std::int32_t>(next_index + end - start));
        draft_logit_rows.push_back(first_draft_row + static_cast<std::int64_t>(rows) - 1);
        draft_verify_index.push_back(-1);
        drafted_after = e.token_count;
    }
    if (drafts) {
        drafting.push_back(static_cast<std::int64_t>(draft_entries.size()));
        later_positions.resize(draft_count - 1);
        later_key_counts.resize(draft_count - 1);
        for (std::size_t j = 1; j < draft_count; j++) {
            later_positions[j - 1].push_back(static_cast<std::int64_t>(next_position + drafted_after + j));
            later_key_counts[j - 1].push_back(static_cast<std::int32_t>(next_index + drafted_after + j + 1));
        }
    }
    draft_entries.push_back(batch_entry{.seq = &seq, .token_count = rows, .samples = drafts});
}

at::Tensor executor::batch::previous_shift(std::size_t shard, at::Device device) const {
    at::Tensor shift = at::zeros({static_cast<std::int64_t>(verify_count)}, at::device(device).dtype(at::kLong));
    std::vector<std::int64_t> entries_with_previous;
    std::vector<std::int64_t> previous_rows;
    for (std::size_t v = 0; v < previous_verify_rows.size(); v++) {
        if (previous_verify_rows[v] >= 0) {
            entries_with_previous.push_back(static_cast<std::int64_t>(v));
            previous_rows.push_back(previous_verify_rows[v]);
        }
    }
    if (entries_with_previous.empty()) {
        return shift;
    }
    const auto longs = at::TensorOptions().dtype(at::kLong);
    at::Tensor kept_before = previous->shard_kept[shard].index_select(
        0,
        at::tensor(previous_rows, longs).to(device, /*non_blocking=*/true)
    );
    return shift
        .index_copy(0, at::tensor(entries_with_previous, longs).to(device, /*non_blocking=*/true), kept_before - 1);
}

void executor::batch::enqueue_drafts(
    model& shard_model,
    std::span<const kv> lines,
    const at::Tensor& samples,
    const at::Tensor& stop_tokens,
    std::size_t shard,
    decode_graphs* graphs,
    at::Tensor& counters
) {
    const bool replays = verify_replays && decode_count == entries.size() && verify_count == entries.size() &&
        drafting.size() == entries.size();
    at::Device device = lines.front().k.device();
    draft_head& head = *shard_model.drafter();
    const auto n_layers = lines.size();
    const auto longs = at::TensorOptions().dtype(at::kLong);
    const auto ints = at::TensorOptions().dtype(at::kInt);
    auto on_device = [&](const auto& values, at::TensorOptions options) {
        return at::tensor(values, options).to(device, /*non_blocking=*/true);
    };
    // Every KV layer's row of a group's key counts, as the draft head reads only its own.
    auto key_counts = [&](const at::Tensor& counts) {
        return counts.to(at::kInt).unsqueeze(0).expand({static_cast<std::int64_t>(n_layers), -1}).contiguous();
    };
    auto tables = [&](std::span<const batch_entry> group, group_batch& g) {
        const std::size_t width = replays ? verify_width : table_width(group);
        g.page_tables = page_tables(group, n_layers, false, shard, width);
        g.page_positions = page_tables(group, n_layers, true, shard, width);
    };
    auto with_counters = [&](group_batch& g, std::int64_t sequences) {
        const std::int64_t needed = shard_model.arrival_counter_count(sequences, g.tokens.size(0));
        if (!counters.defined() || counters.numel() < needed) {
            counters = at::zeros({needed}, at::device(device).dtype(at::kInt));
        }
        g.arrival_counters = counters;
    };

    // Each verify entry keeps its pending token and its drafts while sample t equals draft t + 1, then the sample
    // after them, up to and including a stop token, and at most max_kept.
    at::Tensor kept = at::empty({0}, at::device(device).dtype(at::kLong));
    at::Tensor shift = previous_shift(shard, device);
    if (verify_count > 0) {
        const auto rows = static_cast<std::int64_t>(verify_rows);
        const auto verifying = static_cast<std::int64_t>(verify_count);
        std::vector<std::int64_t> sample_indices;
        for (std::size_t v = 0; v < verify_count; v++) {
            for (std::size_t j = 0; j < verify_rows; j++) {
                sample_indices.push_back(static_cast<std::int64_t>(sample_rows[decode_count - verify_count + v] + j));
            }
        }
        at::Tensor verified = samples.index_select(0, on_device(sample_indices, longs)).view({verifying, rows});
        at::Tensor drafted = verify_tokens[shard].view({verifying, rows}).narrow(1, 1, rows - 1);
        kept = verified.narrow(1, 0, rows - 1).eq(drafted).to(at::kLong).cumprod(1).sum(1) + 1;
        if (stop_tokens.numel() > 0) {
            at::Tensor stops = at::isin(verified, stop_tokens);
            at::Tensor first_stop = at::where(stops.any(1), stops.to(at::kLong).argmax(1), rows);
            kept = at::minimum(kept, first_stop + 1);
        }
        kept = at::minimum(kept, on_device(max_kept, longs));
    }
    shard_kept[shard] = kept;

    const step_logits& step = shard_hidden[shard];
    std::vector<at::Tensor> step_rows;
    for (const std::optional<at::Tensor>& hidden : {step.prefill_hidden, step.decode_hidden, step.verify_hidden}) {
        if (hidden) {
            step_rows.push_back(*hidden);
        }
    }
    std::vector<at::Tensor> sources{at::zeros({1, head.hidden_dim()}, step_rows.front().options())};
    if (shard_prior_hidden[shard].defined()) {
        sources.push_back(shard_prior_hidden[shard]);
    }
    sources.insert(sources.end(), step_rows.begin(), step_rows.end());
    std::vector<std::int32_t> starts{0};
    std::vector<std::int64_t> entry_rows;
    for (const batch_entry& e : draft_entries) {
        starts.push_back(starts.back() + static_cast<std::int32_t>(e.token_count));
        entry_rows.push_back(static_cast<std::int64_t>(e.token_count));
    }
    // Per draft entry, the kept count - 1 of a verify entry, and the previous step's shift of its positions.
    const auto draft_count = static_cast<std::int64_t>(draft_entries.size());
    at::Tensor verify_index = on_device(draft_verify_index, longs);
    at::Tensor is_verify = verify_index.ge(0);
    at::Tensor clamped = verify_index.clamp_min(0);
    at::Tensor kept_after = verify_count > 0 ? at::where(is_verify, kept.index_select(0, clamped) - 1, 0)
                                             : at::zeros({draft_count}, at::device(device).dtype(at::kLong));
    at::Tensor entry_shift = verify_count > 0 ? at::where(is_verify, shift.index_select(0, clamped), 0)
                                              : at::zeros({draft_count}, at::device(device).dtype(at::kLong));
    at::Tensor row_shift = at::repeat_interleave(
        entry_shift,
        on_device(entry_rows, longs),
        /*dim=*/0,
        static_cast<std::int64_t>(draft_tokens.size())
    );
    group_batch rows{
        .tokens = on_device(draft_tokens, longs),
        .pos = on_device(draft_positions, longs) + row_shift,
        .cu_seqlens_q = on_device(starts, ints),
        .key_counts = key_counts(on_device(draft_key_counts, longs) + entry_shift),
        .logit_rows = on_device(draft_logit_rows, longs) + kept_after,
    };
    fill_awaited_tokens_cuda(rows.tokens, samples);
    tables(draft_entries, rows);
    at::Tensor hidden = at::cat(sources).index_select(0, on_device(draft_sources, longs));
    draft_output drafted_rows;
    if (replays) {
        drafted_rows = graphs->draft(rows, hidden, /*verify_rows=*/true);
    } else {
        with_counters(rows, draft_count);
        drafted_rows = head.draft(rows, hidden, lines);
    }
    if (!drafting.empty()) {
        at::Tensor drafting_entries_index = on_device(drafting, longs);
        std::vector<batch_entry> drafting_entries;
        for (std::int64_t d : drafting) {
            drafting_entries.push_back(draft_entries[static_cast<std::size_t>(d)]);
        }
        at::Tensor drafting_rows = rows.logit_rows.index_select(0, drafting_entries_index);
        at::Tensor later_shift = (kept_after + 1).mul(is_verify.to(at::kLong)) + entry_shift;
        later_shift = later_shift.index_select(0, drafting_entries_index);
        at::Tensor token = drafted_rows.logits.index_select(0, drafting_entries_index).argmax(-1);
        at::Tensor previous_hidden = drafted_rows.hidden.index_select(0, drafting_rows);
        std::vector<at::Tensor> tokens{rows.tokens.index_select(0, drafting_rows), token};
        for (std::size_t j = 0; j < later_positions.size(); j++) {
            group_batch one_row{
                .tokens = token,
                .pos = on_device(later_positions[j], longs) + later_shift,
                .key_counts = key_counts(on_device(later_key_counts[j], longs) + later_shift),
            };
            tables(drafting_entries, one_row);
            draft_output next;
            if (replays) {
                next = graphs->draft(one_row, previous_hidden, /*verify_rows=*/false);
            } else {
                with_counters(one_row, static_cast<std::int64_t>(drafting_entries.size()));
                next = head.draft(one_row, previous_hidden, lines);
            }
            token = next.logits.argmax(-1);
            previous_hidden = next.hidden;
            tokens.push_back(token);
        }
        shard_next_tokens[shard] = at::stack(tokens, 1);
    }
    if (shard == 0) {
        this->kept = pinned(kept.sizes(), at::kLong);
        this->kept.copy_(kept, /*non_blocking=*/true);
        if (!drafting.empty()) {
            next_tokens = pinned(shard_next_tokens[shard].sizes(), at::kLong);
            next_tokens.copy_(shard_next_tokens[shard], /*non_blocking=*/true);
        }
    }
}

void executor::batch::join_mass(
    const std::vector<std::vector<std::size_t>>& shard_mass_heads,
    const std::vector<std::size_t>& mass_heads
) {
    auto join = [&](const std::vector<at::Tensor>& by_shard, const std::optional<group_batch>& group) {
        if (!by_shard.front().defined() || by_shard.size() == 1) {
            return by_shard.front();
        }
        std::int64_t total = 0;
        auto row_size = static_cast<std::size_t>(group->cu_view_pages.size(1));
        const auto* cu_view_pages = group->cu_view_pages.const_data_ptr<std::int32_t>();
        for (std::size_t layer = 0; layer < mass_heads.size(); layer++) {
            if (mass_heads[layer] > 0) {
                auto pages = static_cast<std::int64_t>(cu_view_pages[layer * row_size + row_size - 1]);
                total += pages * static_cast<std::int64_t>(mass_heads[layer] + 1);
            }
        }
        at::Tensor joined = pinned({total}, at::kFloat);
        float* out = joined.mutable_data_ptr<float>();
        std::vector<const float*> in;
        for (const at::Tensor& mass : by_shard) {
            in.push_back(mass.const_data_ptr<float>());
        }
        for (std::size_t layer = 0; layer < mass_heads.size(); layer++) {
            if (mass_heads[layer] == 0) {
                continue;
            }
            auto pages = static_cast<std::size_t>(cu_view_pages[layer * row_size + row_size - 1]);
            for (std::size_t page = 0; page < pages; page++) {
                for (std::size_t shard = 0; shard < in.size(); shard++) {
                    std::size_t heads = shard_mass_heads[shard][layer];
                    out = std::copy_n(in[shard], heads, out);
                    in[shard] += heads;
                }
            }
            out = std::copy_n(in.front(), pages, out);
            for (const float*& shard_in : in) {
                shard_in += pages;
            }
        }
        return joined;
    };
    decode_mass = join(shard_decode_mass, inputs.decode_group);
    verify_mass = join(shard_verify_mass, inputs.verify_group);
    prefill_mass = join(shard_prefill_mass, inputs.prefill_group);
}

task<std::expected<std::unique_ptr<sequence>, request_error>> executor::start(request&& r, result_writer& writer) {
    assert(!r.tokens.empty());
    assert(r.eviction.n_working >= 1 && r.eviction.cycle_tokens >= 1);
    assert(r.eviction.attention_mass_decay > 0.0 && r.eviction.attention_mass_decay <= 1.0);
    assert(r.eviction.eviction_threshold >= 0.0 && r.eviction.eviction_threshold <= 1.0);
    digest chain_root = r.initial_digest.value_or(make_request_root_digest(root, to_pages(r.eviction, page_size)));
    std::vector<digest> digests = make_page_digests(chain_root, r.tokens, page_size);
    std::expected<std::pair<std::shared_ptr<const state>, state::lease_id_type>, page_error>
        lookup = co_await store.read_state(r.tokens, digests);
    if (!lookup) {
        co_return std::unexpected(request_error::miss);
    }
    auto& [stored_state, lease_id] = *lookup;
    std::size_t stored_pages = stored_state->head
        ? static_cast<std::size_t>(std::ranges::find(digests, *stored_state->head) - digests.begin()) + 1
        : 0;
    std::shared_ptr<const state> initial_state = extend_with_hot_pages(std::move(stored_state), r, digests);
    std::vector<at::Device> devices;
    for (const std::vector<kv>& shard_lines : lines) {
        devices.push_back(shard_lines.front().k.device());
    }
    std::unique_ptr<sequence> seq(new sequence(
        std::move(r),
        std::move(initial_state),
        store,
        lease_id,
        std::move(digests),
        chain_root,
        *this,
        writer,
        devices
    ));
    // take_lines holds the resident pages before it first suspends, so they are the ones counted here.
    seq->device_cached_count = device_cached_tokens(*seq);
    std::expected<void, page_error> line_allocation = co_await take_lines(*seq, seq->cycle_lines());
    if (!line_allocation) {
        seq->lease.release();
        co_return std::unexpected(
            line_allocation.error() == page_error::miss ? request_error::miss : request_error::oom
        );
    }
    if (seq->history_pages > stored_pages && !co_await own_pages_from(*seq, stored_pages)) {
        release_lines(*seq);
        seq->lease.release();
        co_return std::unexpected(request_error::oom);
    }
    seq->upload_table(0, streams);
    co_return std::move(seq);
}

std::shared_ptr<const state> executor::extend_with_hot_pages(
    std::shared_ptr<const state>&& s,
    const request& r,
    std::span<const digest> digests
) const {
    if (std::ranges::contains(recurrent, true)) {
        return std::move(s);
    }
    std::size_t head_pages = 0;
    if (s->head) {
        auto head_digest = std::ranges::find(digests, *s->head);
        assert(head_digest != digests.end());
        head_pages = static_cast<std::size_t>(head_digest - digests.begin()) + 1;
    }
    page_eviction_policy policy = to_pages(r.eviction, page_size);
    std::size_t resumed_end = head_pages * page_size + s->tail.size();
    std::size_t cycle_end = (resumed_end / policy.cycle_tokens + 1) * policy.cycle_tokens;
    std::size_t window_start = cycle_end - policy.mass_tokens;
    if (resumed_end > window_start) {
        return std::move(s);
    }
    std::size_t page_limit = std::min(window_start, r.tokens.size() - 1) / page_size;
    std::size_t end_page = head_pages;
    while (end_page < page_limit &&
           std::ranges::all_of(
               std::views::iota(std::size_t{0}, cache.layer_count()),
               [&](std::size_t layer) { return cache.contains(layer, digests[end_page]); }
           )) {
        end_page++;
    }
    if (end_page == head_pages) {
        return std::move(s);
    }
    state extended{.head = digests[end_page - 1], .layers = s->layers, .tail = {}};
    extended.layers.resize(cache.layer_count());
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        std::vector<active_page>& pages = extended.layers[layer].pages;
        for (std::size_t index = head_pages; index < end_page; index++) {
            pages.push_back(active_page{.hash = digests[index], .index = index});
        }
        const std::optional<std::size_t>& window = sliding_windows[layer];
        std::size_t next_query = pages.size() * page_size;
        if (window && next_query + 1 > *window) {
            std::size_t passed = std::min((next_query + 1 - *window) / page_size, pages.size());
            pages.erase(pages.begin(), pages.begin() + static_cast<std::ptrdiff_t>(passed));
        }
    }
    return std::make_shared<const state>(std::move(extended));
}

task<std::expected<void, page_error>> executor::own_pages_from(sequence& seq, std::size_t first_index) {
    std::size_t n_layers = cache.layer_count();
    std::vector<std::vector<digest>> hashes(n_layers);
    std::vector<std::vector<state::lease_id_type>> owners(n_layers);
    std::vector<std::vector<std::uint32_t>> line_ids(n_layers);
    for (std::size_t layer = 0; layer < n_layers; layer++) {
        const layer_view& view = seq.layers[layer];
        for (std::size_t page = 0; page < view.pages.size(); page++) {
            if (view.indices[page] >= first_index) {
                hashes[layer].push_back(view.pages[page]);
                owners[layer].push_back(seq.lease.id());
                line_ids[layer].push_back(view.lines[page]);
            }
        }
    }
    co_return co_await cache.sync(hashes, owners, line_ids, streams);
}

task<std::expected<request_result, request_error>> executor::finish(sequence& seq) {
    release_lines(seq);
    std::expected<request_result, request_error> result = std::unexpected(request_error::cancelled);
    switch (*seq.end_reason) {
    case sequence::exit_reason::finished: {
        auto end_state = std::make_shared<const state>(seq.to_state());
        co_await store.write_state(seq.lease.id(), std::shared_ptr<const state>(end_state));
        result = request_result{
            .state = std::move(end_state),
            .total_tokens = seq.prompt_count + seq.generated_count,
            .cached_tokens = seq.cached_count,
            .device_cached_tokens = seq.device_cached_count,
            .peak_device_memory = seq.peak_device_memory,
        };
        break;
    }
    case sequence::exit_reason::cancelled:
        result = std::unexpected(request_error::cancelled);
        break;
    case sequence::exit_reason::oom:
        result = std::unexpected(request_error::oom);
        break;
    }
    seq.lease.release();
    co_return result;
}

std::size_t executor::device_cached_tokens(const sequence& seq) const {
    std::vector<bool> page_fetched(seq.history_pages, false);
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        const layer_view& view = seq.layers[layer];
        if (recurrent[layer]) {
            if (seq.head && !cache.contains(layer, *seq.head)) {
                page_fetched[seq.history_pages - 1] = true;
            }
            continue;
        }
        for (std::size_t i = 0; i < view.pages.size(); i++) {
            if (!cache.contains(layer, view.pages[i])) {
                page_fetched[view.indices[i]] = true;
            }
        }
    }
    return (seq.history_pages - static_cast<std::size_t>(std::ranges::count(page_fetched, true))) * page_size;
}

task<std::expected<void, page_error>> executor::take_lines(sequence& seq, std::size_t line_count) {
    std::vector<std::span<const digest>> held_hashes(cache.layer_count());
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        if (!recurrent[layer]) {
            held_hashes[layer] = seq.layers[layer].pages;
        } else if (seq.head) {
            held_hashes[layer] = std::span(&*seq.head, 1);
        }
    }
    // A get failing with miss releases the lines it held_hashes while it fetched.
    std::expected<std::vector<std::vector<std::uint32_t>>, page_error> page_lines = co_await cache.get(
        held_hashes,
        streams
    );
    if (!page_lines) {
        co_return std::unexpected(page_lines.error());
    }
    std::vector<std::size_t> counts(cache.layer_count(), line_count);
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        if (recurrent[layer]) {
            // The stored state's copy takes the last slot.
            counts[layer] = state_slots[layer] + 2 - ((*page_lines)[layer].empty() ? 0 : 1);
        }
    }
    std::expected<std::vector<std::vector<std::uint32_t>>, page_error> allocation = cache.allocate(counts);
    std::vector<std::vector<std::uint32_t>> states(cache.layer_count());
    bool copied = allocation.has_value();
    for (std::size_t layer = 0; layer < cache.layer_count() && copied; layer++) {
        if (recurrent[layer] && !(*page_lines)[layer].empty()) {
            std::expected<std::vector<std::uint32_t>, page_error>
                state_copy = cache.clone(layer, (*page_lines)[layer].front(), 1, streams);
            copied = state_copy.has_value();
            if (copied) {
                states[layer] = std::move(*state_copy);
            }
        }
    }
    if (!copied) {
        for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
            cache.release(layer, (*page_lines)[layer]);
            cache.release(layer, states[layer]);
            if (allocation) {
                cache.release(layer, (*allocation)[layer]);
            }
        }
        co_return std::unexpected(page_error::oom);
    }
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        layer_view& view = seq.layers[layer];
        std::span<const std::uint32_t> allocated = (*allocation)[layer];
        if (recurrent[layer]) {
            const std::size_t first_slots = state_slots[layer] - 1;
            view.append_lines(allocated.first(first_slots));
            allocated = allocated.subspan(first_slots);
            if (states[layer].empty()) {
                for (std::size_t shard = 0; shard < shards.size(); shard++) {
                    c10::cuda::CUDAStreamGuard guard(
                        c10::cuda::getStreamFromExternal(streams[shard], lines[shard][layer].k.device().index())
                    );
                    lines[shard][layer].k.select(0, allocated.front()).zero_();
                }
                view.append_lines(allocated.first(1));
                allocated = allocated.subspan(1);
            } else {
                view.append_lines(states[layer]);
            }
        } else {
            view.append_lines((*page_lines)[layer]);
        }
        view.append_lines(allocated);
    }
    // The copies of the states at head read the fetched lines.
    co_await synchronize_shards();
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        if (recurrent[layer]) {
            cache.release(layer, (*page_lines)[layer]);
        }
    }
    record_device_memory(seq);
    co_return std::expected<void, page_error>{};
}

void executor::record_device_memory(sequence& seq) const noexcept {
    std::size_t bytes = 0;
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        bytes += seq.layers[layer].lines.size() * cache.line_block_bytes(layer);
    }
    seq.peak_device_memory = std::max(seq.peak_device_memory, bytes);
}

void executor::release_lines(sequence& seq) noexcept {
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        cache.release(layer, seq.layers[layer].lines);
        seq.layers[layer].lines.clear();
    }
}

bool executor::release_passed_pages(sequence& seq) {
    bool changed = false;
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        layer_view& view = seq.layers[layer];
        std::vector<std::size_t> passed_indices(seq.passed_pages(layer, view.pages.size() * page_size));
        if (passed_indices.empty()) {
            continue;
        }
        std::iota(passed_indices.begin(), passed_indices.end(), 0);
        cache.release(layer, view.remove(passed_indices));
        changed = true;
    }
    return changed;
}

executor::~executor() = default;

void executor::update_views(sequence& seq) {
    seq.update_held();
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        if (!mirrors[layer]) {
            continue;
        }
        layer_view& view = seq.layers[layer];
        std::vector<std::size_t> unheld;
        for (std::size_t page = 0; page < view.pages.size(); page++) {
            if (!std::ranges::binary_search(seq.held_pages, view.indices[page])) {
                unheld.push_back(page);
            }
        }
        if (!unheld.empty()) {
            cache.release(layer, view.remove(unheld));
        }
    }
}

bool executor::completes_page(const batch_entry& e) const noexcept {
    std::size_t start = e.seq->tail.size();
    return (start + e.token_count) / page_size * page_size > start;
}

// A drafting sequence's verify step ends neither its cycle nor max_tokens (executor::advance).
bool executor::allows_overlap(std::span<const batch_entry> entries) const noexcept {
    if (draft_layer) {
        return std::ranges::all_of(entries, [](const batch_entry& e) { return !e.seq->drafts.empty(); });
    }
    const bool sliding = std::ranges::any_of(sliding_windows, [](const std::optional<std::size_t>& window) {
        return window.has_value();
    });
    return std::ranges::none_of(entries, [&](const batch_entry& e) {
        return e.ends_cycle || (sliding && completes_page(e));
    });
}

// A verify entry measured each row's mass with weight 1; its kept rows are weighed as one step of kept tokens.
void executor::add_measured_mass(const batch& b, std::size_t index, std::size_t kept) {
    sequence& seq = *b.entries[index].seq;
    std::vector<batch::layer_step_mass> step_mass;
    std::vector<std::vector<float>> verified_mass(mass_heads.size());
    std::vector<std::vector<float>> verified_weight(mass_heads.size());
    if (b.kind_of(index) == batch::group_kind::verify) {
        std::vector<float> weights(kept);
        seq.fill_query_weights(b.step_starts[index], weights);
        step_mass.resize(mass_heads.size());
        for (std::size_t row = 0; row < kept; row++) {
            std::vector<batch::layer_step_mass> row_mass = b.verify_row_mass(index, row, mass_heads);
            for (std::size_t layer = 0; layer < mass_heads.size(); layer++) {
                if (mass_heads[layer] == 0) {
                    continue;
                }
                verified_mass[layer].resize(row_mass[layer].mass_sum.size());
                verified_weight[layer].resize(row_mass[layer].weight_sum.size());
                for (std::size_t j = 0; j < verified_mass[layer].size(); j++) {
                    verified_mass[layer][j] += weights[row] * row_mass[layer].mass_sum[j];
                }
                for (std::size_t j = 0; j < verified_weight[layer].size(); j++) {
                    verified_weight[layer][j] += weights[row] * row_mass[layer].weight_sum[j];
                }
                step_mass[layer] = batch::layer_step_mass{
                    .mass_sum = verified_mass[layer],
                    .weight_sum = verified_weight[layer],
                };
            }
        }
    } else {
        step_mass = b.step_mass(index, mass_heads);
    }
    for (std::size_t layer = 0; layer < mass_heads.size(); layer++) {
        if (mass_heads[layer] > 0) {
            layer_view& view = seq.layers[layer];
            add_step_mass(
                view.mass_sum,
                view.weight_sum,
                step_mass[layer].mass_sum.first(view.mass_sum.size()),
                step_mass[layer].weight_sum.first(view.weight_sum.size()),
                b.step_ends[index] - seq.measured_end,
                seq.eviction.attention_mass_decay
            );
        }
    }
}

// A verify entry's tokens are kept, and its samples taken, at its write-back. Its pending token and drafts are
// awaited_token until then, and a sequence whose previous step is launched lags it by up to that step's drafts.
void executor::advance(batch& b) {
    const std::size_t draft_count = draft_state_slots - 1;
    for (const batch_entry& e : b.entries) {
        sequence& seq = *e.seq;
        if (draft_layer && seq.draft_rows_end == seq.history_position() && seq.history_position() > 0) {
            const layer_view& view = seq.layers[*draft_layer];
            b.prior_rows.push_back(b.prior_lines.size());
            b.prior_lines.push_back(view.lines[(seq.next_index(*draft_layer) - 1) / page_size]);
        } else {
            b.prior_rows.emplace_back();
        }
    }
    std::size_t sampled_row = 0;
    for (std::size_t i = 0; i < b.entries.size(); i++) {
        const batch_entry& e = b.entries[i];
        sequence& seq = *e.seq;
        const bool verifies = b.kind_of(i) == batch::group_kind::verify;
        const std::size_t lag = seq.steps_in_flight > 0 ? draft_count : 0;
        b.measured.push_back(batch::measures_mass(e));
        b.step_starts.push_back(seq.history_position());
        b.ends_cycle.push_back(e.ends_cycle);
        bool drafts = false;
        if (draft_layer) {
            const bool ends_sequence = e.samples ? seq.max_tokens && seq.generated_count + 1 >= *seq.max_tokens
                                                 : seq.pending_tokens.size() == e.token_count;
            const bool next_row = verifies || (!e.ends_cycle && !ends_sequence);
            // The next step's verify group then ends neither the cycle nor max_tokens, whatever this one keeps.
            drafts = next_row && (verifies || e.samples) &&
                seq.cycle_tokens_left() >= batch::forwarded(e) + lag + draft_count + 2;
            b.plan_draft_rows(i, next_row, drafts, *draft_layer, draft_count);
            if (verifies) {
                const std::size_t remaining = seq.max_tokens ? *seq.max_tokens - seq.generated_count : b.verify_rows;
                b.max_kept.push_back(static_cast<std::int64_t>(std::clamp<std::size_t>(remaining, 1, b.verify_rows)));
                b.verify_measured_starts.push_back(static_cast<std::int64_t>(seq.measured_start()));
            }
        }
        if (verifies) {
            b.step_ends.push_back(seq.history_position());
            seq.advance(e.token_count);
            b.sampled_positions.push_back(seq.history_position());
            seq.pending_tokens.push_back(sequence::awaited_token);
            // As if it kept only its pending token, until its write-back.
            seq.rejected_tokens = seq.drafts.size();
        } else {
            seq.rejected_tokens = 0;
            b.step_ends.push_back(seq.history_position() + e.token_count);
            seq.advance(e.token_count);
            b.sampled_positions.push_back(seq.history_position() + seq.pending_tokens.size());
            if (e.samples) {
                seq.pending_tokens.push_back(sequence::awaited_token);
                seq.awaited_row = sampled_row;
                seq.generated_count++;
            }
            seq.last_step_launched = e.samples ? !seq.may_sample() : seq.pending_tokens.empty();
        }
        if (e.samples) {
            sampled_row += verifies ? b.verify_rows : 1;
        }
        seq.drafts.assign(drafts ? draft_count : 0, sequence::awaited_token);
        b.ends_sequence.push_back(!verifies && seq.last_step_launched);
        seq.steps_in_flight++;
    }
}

// Shard 0 queues on the executor's thread while the others queue on their own, so every shard's kernels launch at
// once. Each shard's completion is taken once every shard has queued, before any later step queues.
task<void> executor::launch(std::vector<batch_entry> entries) {
    std::unique_ptr<batch> b(new batch(entries, page_size, draft_state_slots, graph_pages));
    link_previous(*b);
    advance(*b);
    b->shard_decode_mass.assign(shards.size(), at::Tensor());
    b->shard_verify_mass.assign(shards.size(), at::Tensor());
    b->shard_prefill_mass.assign(shards.size(), at::Tensor());
    b->shard_prior_hidden.assign(shards.size(), at::Tensor());
    b->shard_hidden.assign(shards.size(), step_logits());
    b->verify_tokens.assign(shards.size(), at::Tensor());
    b->shard_kept.assign(shards.size(), at::Tensor());
    b->shard_next_tokens.assign(shards.size(), at::Tensor());
    TORCH_CHECK(
        static_cast<std::int64_t>(b->sample_params.size()) <= max_sampled_rows,
        "executor: a step samples at most ",
        max_sampled_rows,
        " tokens"
    );
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
    launched.push_back(std::move(b));
}

void executor::link_previous(batch& b) const {
    if (launched.empty() || b.verify_count == 0) {
        return;
    }
    const batch& previous = *launched.back();
    b.previous = &previous;
    const std::size_t plain_count = b.decode_count - b.verify_count;
    const std::size_t previous_plain = previous.decode_count - previous.verify_count;
    for (std::size_t v = 0; v < b.verify_count; v++) {
        const sequence* seq = b.entries[plain_count + v].seq;
        std::int64_t verify_row = -1;
        std::int64_t drafting_row = -1;
        if (seq->steps_in_flight > 0) {
            for (std::size_t pv = 0; pv < previous.verify_count; pv++) {
                if (previous.entries[previous_plain + pv].seq == seq) {
                    verify_row = static_cast<std::int64_t>(pv);
                }
            }
            for (std::size_t d = 0; d < previous.drafting.size(); d++) {
                if (previous.draft_entries[static_cast<std::size_t>(previous.drafting[d])].seq == seq) {
                    drafting_row = static_cast<std::int64_t>(d);
                }
            }
            assert(verify_row >= 0 && drafting_row >= 0);
        }
        b.previous_verify_rows.push_back(verify_row);
        b.previous_drafting_rows.push_back(drafting_row);
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

task<void> executor::synchronize_shards() {
    for (cudaStream_t stream : streams) {
        co_await synchronize(stream);
    }
}

// A sequence ends with oom when the sync of its new pages fails, as a state names only pages the store has. The step
// that last read the lines of pages that left a view has completed, so they are released right after the sync. A
// recurrent layer's state at head reaches the store at the resume points, cycle ends and the end, that find it pending.
task<void> executor::write_back() {
    std::unique_ptr<batch> written = std::move(launched.front());
    launched.pop_front();
    batch& b = *written;
    for (external_task<cudaError_t>& completion : b.completions) {
        cudaError_t status = co_await std::move(completion);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("write_back: ") + cudaGetErrorString(status));
        }
    }
    b.join_mass(shard_mass_heads, mass_heads);
    std::size_t n_layers = cache.layer_count();
    const bool has_recurrent = std::ranges::contains(recurrent, true);
    std::vector<bool> exited;
    for (const batch_entry& e : b.entries) {
        exited.push_back(e.seq->end_reason.has_value());
    }
    // The pages of the sequences reaching a state, written now with state_writers' states, and those a spawned task
    // writes.
    std::vector<std::vector<digest>> new_hashes(n_layers);
    std::vector<std::vector<state::lease_id_type>> new_owners(n_layers);
    std::vector<std::vector<std::uint32_t>> new_lines(n_layers);
    std::vector<std::vector<digest>> spawned_hashes(n_layers);
    std::vector<std::vector<state::lease_id_type>> spawned_owners(n_layers);
    std::vector<std::vector<std::uint32_t>> spawned_lines(n_layers);
    std::vector<std::shared_ptr<pending_writes>> spawned_writers;
    std::vector<sequence*> page_writers;
    std::vector<sequence*> state_reachers;
    std::vector<sequence*> state_writers;
    const std::int64_t* next_sampled = b.sampled_tokens.defined() ? b.sampled_tokens.const_data_ptr<std::int64_t>()
                                                                  : nullptr;
    // Each drafting sequence's next token then its drafts.
    std::unordered_map<const sequence*, std::span<const std::int64_t>> drafted;
    for (std::size_t d = 0; d < b.drafting.size(); d++) {
        const auto row = static_cast<std::size_t>(b.next_tokens.size(1));
        drafted.emplace(
            b.draft_entries[static_cast<std::size_t>(b.drafting[d])].seq,
            std::span(b.next_tokens.const_data_ptr<std::int64_t>() + d * row, row)
        );
    }
    for (std::size_t i = 0; i < b.entries.size(); i++) {
        const batch_entry& e = b.entries[i];
        sequence& seq = *e.seq;
        const bool verifies = b.kind_of(i) == batch::group_kind::verify;
        std::span<const std::int64_t> samples;
        if (e.samples) {
            samples = std::span(next_sampled, verifies ? b.verify_rows : 1);
            next_sampled += samples.size();
        }
        if (exited[i]) {
            continue;
        }
        if (seq.page_writes->any_failed()) {
            seq.end_reason = sequence::exit_reason::oom;
            continue;
        }
        // The tokens the step keeps, and of its samples those it emits. A step launched after this one took its next
        // tokens on the GPU and lags its kept count - 1.
        const bool next_launched = seq.steps_in_flight > 1;
        std::size_t kept = batch::forwarded(e);
        std::span<const std::int64_t> emitted = samples;
        if (verifies) {
            kept = static_cast<std::size_t>(
                b.kept.const_data_ptr<std::int64_t>()[i - (b.decode_count - b.verify_count)]
            );
            emitted = samples.first(kept);
            const std::size_t start = b.step_starts[i] + seq.unapplied_shift;
            seq.keep_verified(start, emitted);
            drafted_count.fetch_add(b.verify_rows - 1, std::memory_order_relaxed);
            accepted_count.fetch_add(kept - 1, std::memory_order_relaxed);
            seq.draw_count += kept;
            seq.generated_count += kept;
            b.step_starts[i] = start;
            b.step_ends[i] = start + kept;
            seq.unapplied_shift = next_launched ? kept - 1 : 0;
            if (!next_launched) {
                seq.rejected_tokens = b.verify_rows - kept;
                seq.draft_rows_end = start + kept + 1;
            }
        }
        std::size_t completed_count = seq.complete_pages(b.step_ends[i]);
        if (completed_count > 0 && has_recurrent) {
            seq.state_pending = true;
        }
        if (b.measured[i]) {
            add_measured_mass(b, i, kept);
            seq.measured_end = b.step_ends[i];
        }
        if (verifies) {
            for (std::int64_t token : emitted) {
                seq.writer.append(token);
            }
            if (std::ranges::contains(stop_ids, emitted.back()) || !seq.may_sample()) {
                seq.end_reason = sequence::exit_reason::finished;
                if (next_launched) {
                    seq.withdraw_latest_verify_step();
                }
            }
        } else if (!samples.empty()) {
            const std::int64_t token = samples.front();
            seq.resolve_token(b.sampled_positions[i], token);
            seq.writer.append(token);
            if (std::ranges::contains(stop_ids, token)) {
                seq.end_reason = sequence::exit_reason::finished;
                if (seq.steps_in_flight > 1) {
                    seq.withdraw_latest_step();
                }
            }
        }
        if (b.ends_sequence[i]) {
            seq.end_reason = sequence::exit_reason::finished;
        }
        if (seq.end_reason) {
            seq.drafts.clear();
        } else if (auto found = drafted.find(&seq); found != drafted.end() && !next_launched) {
            seq.drafts.assign(found->second.begin() + 1, found->second.end());
        }
        const bool reaches_state = b.ends_cycle[i] || seq.end_reason == sequence::exit_reason::finished;
        if (reaches_state) {
            state_reachers.push_back(&seq);
            if (seq.state_pending) {
                state_writers.push_back(&seq);
            }
        }
        if (completed_count > 0) {
            auto& hashes = reaches_state ? new_hashes : spawned_hashes;
            auto& owners = reaches_state ? new_owners : spawned_owners;
            auto& line_ids = reaches_state ? new_lines : spawned_lines;
            for (std::size_t layer = 0; layer < n_layers; layer++) {
                if (recurrent[layer]) {
                    continue;
                }
                const layer_view& view = seq.layers[layer];
                std::size_t first_new_page = view.pages.size() - completed_count;
                hashes[layer].insert(hashes[layer].end(), view.pages.begin() + first_new_page, view.pages.end());
                owners[layer].resize(hashes[layer].size(), seq.lease.id());
                line_ids[layer].insert(
                    line_ids[layer].end(),
                    view.lines.begin() + first_new_page,
                    view.lines.begin() + view.pages.size()
                );
            }
            if (reaches_state) {
                page_writers.push_back(&seq);
            } else {
                spawned_writers.push_back(seq.page_writes);
            }
        }
    }
    for (sequence* seq : state_writers) {
        for (std::size_t layer = 0; layer < n_layers; layer++) {
            if (recurrent[layer]) {
                new_hashes[layer].push_back(*seq->head);
                new_owners[layer].push_back(seq->lease.id());
                new_lines[layer].push_back(seq->head_state_line(layer));
            }
        }
    }

    // The copies wait for this step only, its draft pass included, not for the step launched after it, which writes none
    // of the copied lines.
    for (std::size_t shard = 0; shard < shards.size(); shard++) {
        at::Device device = lines[shard].front().k.device();
        b.ends[shard].block(c10::cuda::getStreamFromExternal(copy_streams[shard], device.index()));
    }
    if (!spawned_writers.empty()) {
        while (store_writes.size() >= max_store_writes) {
            co_await store_writes.wait();
        }
        store_writes.start();
        for (const std::shared_ptr<pending_writes>& writer : spawned_writers) {
            writer->start();
        }
        event_loop::current()->spawn(async_write_back(
            std::move(spawned_hashes),
            std::move(spawned_owners),
            std::move(spawned_lines),
            std::move(spawned_writers)
        ));
    }
    // A state names only pages the store has.
    for (sequence* seq : state_reachers) {
        co_await seq->page_writes->wait();
        if (seq->page_writes->any_failed()) {
            seq->end_reason = sequence::exit_reason::oom;
        }
    }
    std::expected<void, page_error> sync_result = co_await cache.sync(new_hashes, new_owners, new_lines, copy_streams);
    if (!sync_result) {
        for (sequence* seq : page_writers) {
            seq->end_reason = sequence::exit_reason::oom;
        }
        for (sequence* seq : state_writers) {
            seq->end_reason = sequence::exit_reason::oom;
        }
    } else {
        for (sequence* seq : state_writers) {
            if (seq->end_reason == sequence::exit_reason::oom) {
                continue;
            }
            seq->state_pending = false;
            // The synced line keeps the stored state as a cached page; a new line takes its place.
            std::vector<std::size_t> counts(n_layers, 0);
            for (std::size_t layer = 0; layer < n_layers; layer++) {
                counts[layer] = recurrent[layer] ? 1 : 0;
            }
            std::expected<std::vector<std::vector<std::uint32_t>>, page_error> replacements = cache.allocate(counts);
            if (!replacements) {
                seq->end_reason = sequence::exit_reason::oom;
                continue;
            }
            for (std::size_t layer = 0; layer < n_layers; layer++) {
                if (recurrent[layer]) {
                    std::uint32_t& head_line = seq->head_state_line(layer);
                    cache.release(layer, std::span(&head_line, 1));
                    head_line = (*replacements)[layer].front();
                }
            }
        }
    }

    for (std::size_t i = 0; i < b.entries.size(); i++) {
        const batch_entry& e = b.entries[i];
        sequence& seq = *e.seq;
        seq.steps_in_flight--;
        if (seq.end_reason) {
            continue;
        }
        bool views_changed = release_passed_pages(seq);
        if (b.ends_cycle[i]) {
            add_sampled_pages(seq);
            views_changed = evict(seq) || views_changed;
        }
        if (views_changed) {
            update_views(seq);
        }
        if (b.ends_cycle[i]) {
            co_await write_cycle_end(seq);
            seq.start_next_cycle();
        }
        std::optional<std::size_t> first_stale_page;
        if (views_changed) {
            first_stale_page = 0;
        } else if (std::ranges::contains(state_writers, &seq)) {
            first_stale_page = 0;
        }
        std::vector<std::size_t> missing_counts = seq.missing_lines();
        if (std::ranges::any_of(missing_counts, [](std::size_t count) { return count > 0; })) {
            std::expected<std::vector<std::vector<std::uint32_t>>, page_error> allocation = cache.allocate(
                missing_counts
            );
            if (!allocation) {
                seq.end_reason = sequence::exit_reason::oom;
                continue;
            }
            for (std::size_t layer = 0; layer < n_layers; layer++) {
                layer_view& view = seq.layers[layer];
                if (missing_counts[layer] > 0) {
                    first_stale_page = std::min(first_stale_page.value_or(view.lines.size()), view.lines.size());
                }
                view.append_lines((*allocation)[layer]);
            }
            record_device_memory(seq);
        }
        if (first_stale_page) {
            seq.upload_table(*first_stale_page, streams);
        }
    }
}

task<void> executor::async_write_back(
    std::vector<std::vector<digest>> hashes,
    std::vector<std::vector<state::lease_id_type>> owners,
    std::vector<std::vector<std::uint32_t>> line_ids,
    std::vector<std::shared_ptr<pending_writes>> writers
) {
    std::expected<void, page_error> written = co_await cache.sync(hashes, owners, line_ids, copy_streams);
    for (const std::shared_ptr<pending_writes>& writer : writers) {
        writer->finish(written.has_value());
    }
    store_writes.finish(written.has_value());
}

void executor::add_sampled_pages(sequence& seq) {
    if (!seq.eviction.can_evict()) {
        return;
    }
    std::size_t cycle_end = seq.next_cycle_end;
    std::size_t cycle_tokens = seq.eviction.cycle_tokens;
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        std::size_t heads = mass_heads[layer];
        if (heads == 0) {
            continue;
        }
        layer_view& view = seq.layers[layer];
        for (std::size_t page = 0; page < view.pages.size(); page++) {
            std::size_t index = view.indices[page];
            std::size_t page_end = (index + 1) * page_size;
            if (index < seq.eviction.n_sink || page_end + cycle_tokens > cycle_end ||
                page_end + 2 * cycle_tokens <= cycle_end) {
                continue;
            }
            add_sampled_page(
                view.density_moments,
                view.score_moments,
                std::span(view.mass_sum).subspan(page * heads, heads),
                view.weight_sum[page],
                seq.eviction.eviction_warmup
            );
        }
    }
}

// The step that last used the evicted lines has completed, so they are released right away.
bool executor::evict(sequence& seq) {
    if (!seq.eviction.can_evict()) {
        return false;
    }
    bool evicted = false;
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        std::size_t heads = mass_heads[layer];
        if (heads == 0) {
            continue;
        }
        layer_view& view = seq.layers[layer];
        std::vector<std::size_t> pages = pages_to_evict(
            std::span(view.mass_sum).first(view.pages.size() * heads),
            std::span(view.weight_sum).first(view.pages.size()),
            heads,
            view.density_moments,
            view.score_moments,
            page_size,
            seq.eviction
        );
        if (!pages.empty()) {
            cache.release(layer, view.remove(pages));
            evicted = true;
        }
    }
    return evicted;
}

task<void> executor::write_cycle_end(sequence& seq) {
    auto cycle_end_state = std::make_shared<const state>(seq.to_state());
    co_await store.write_state(seq.lease.id(), std::shared_ptr<const state>(cycle_end_state));
    seq.last_state = cycle_end_state;
    seq.writer.append(cycle_end{.state = std::move(cycle_end_state)});
}

std::size_t executor::kv_cache_bytes() const noexcept {
    return cache.used_bytes();
}

std::uint64_t executor::drafted_tokens() const noexcept {
    return drafted_count.load(std::memory_order_relaxed);
}

std::uint64_t executor::accepted_tokens() const noexcept {
    return accepted_count.load(std::memory_order_relaxed);
}

}  // namespace pulsarlm
