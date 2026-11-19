#include "pulsarlm/engine/executor.hpp"

#include "pulsarlm/kernels/ops.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <array>
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

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

at::Tensor pinned(at::IntArrayRef sizes, at::ScalarType dtype) {
    return at::empty(sizes, at::TensorOptions().dtype(dtype).pinned_memory(true));
}

at::Tensor to_device(const at::Tensor& host, at::Device device) {
    return host.defined() ? host.to(host.options().device(device), /*non_blocking=*/true) : host;
}

model::batch to_device(const model::batch& host, at::Device device) {
    return model::batch{
        .kind = host.kind,
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
    model::batch& on_device,
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
    // One sequence's part of the batch, as recorded at its launch.
    struct entry {
        sequence_ref seq;
        std::size_t token_count;
        bool is_sampling;
        bool ends_cycle;
        bool ends_sequence = false;
        bool is_measured = false;
        std::size_t start_position = 0;
        std::size_t end_position = 0;
        std::size_t sampled_position = 0;
        std::size_t first_sample_row = 0;
        std::optional<std::size_t> prior_row;

        // The sequence's rows in the draft pass. logit_row is the row that drafts, before a verify entry's accepted
        // count - 1 is added, and key_count the sequence's keys after its rows in the draft head's KV layer.
        struct draft_rows {
            std::size_t row_count;
            std::size_t logit_row;
            std::size_t key_count;
            bool is_drafting;
        };
        std::optional<draft_rows> draft;

        // Of a verify entry, the most tokens it may accept (max_tokens), and its sequence's measured_start.
        std::size_t max_accepted_token_count = 0;
        std::size_t measured_start = 0;
        // Of a verify entry whose sequence's previous step is launched and not written back, its position among
        // that step's verify entries and among its drafting sequences: its tokens are that step's next tokens, and its
        // positions lie that step's accepted count - 1 after those its inputs hold.
        std::optional<std::size_t> previous_verify_row;
        std::optional<std::size_t> previous_drafting_row;
    };

    struct group {
        // Empty when the batch has no group of the kind.
        std::vector<entry> entries;
        // Pinned host tensors.
        model::batch inputs;
        // The page-table width, and whether the group's forward replays a graph.
        std::size_t table_width = 0;
        bool is_replayed = false;
        // Pinned host fp32, defined for a group that measures mass: the page mass of each KV layer that measures it in
        // turn, laid out as the layer's row of the group's cu_view_pages, its heads innermost; per shard, its heads,
        // then joined over the shards.
        std::vector<at::Tensor> mass_by_shard;
        at::Tensor mass;
        // Per shard, the group's hidden rows, which the draft head's pass reads.
        std::vector<at::Tensor> hidden_by_shard;
    };

    // Each decode entry forwards its sequence's only pending token, a verify entry with its drafts after it.
    group decode_group{.inputs = {.kind = model::batch_kind::decode}};
    group verify_group{.inputs = {.kind = model::batch_kind::verify}};
    group prefill_group{.inputs = {.kind = model::batch_kind::prefill}};
    // The tokens a verify entry forwards and samples after: its pending token and its drafts.
    std::size_t verify_rows = 0;
    // Whether a decode entry's token is awaited, -(r + 1) in the decode group's tokens for row r of the previous
    // launched step's samples.
    bool awaits_tokens = false;
    // Per row sampled, in entry order: a row of each entry that samples but a verify entry's verify_rows, the params
    // and draw.
    std::vector<sampling_params> sample_params;
    std::vector<sampling_draw> sample_draws;
    // Pinned host int64, one token per row sampled.
    at::Tensor sampled_tokens;
    // Per shard, completing once the step's work on its stream has, and an event recorded after that work.
    std::vector<external_task<cudaError_t>> completions;
    std::vector<at::cuda::CUDAEvent> ends;
    // Per entry whose draft rows start at its first token after an earlier step forwarded the position before it,
    // the line of the draft head's KV layer holding that position, and per shard, those lines' last_hidden as of the
    // step's launch; the entry's prior_row is its row.
    std::vector<std::int64_t> prior_lines;
    std::vector<at::Tensor> shard_prior_hidden;
    // The draft head's pass, queued after the step's sampling, over the entries with draft rows in entry order: per
    // row, its token, -(r + 1) for the step's sampled row r, its RoPE position, and its hidden row (draft_head::draft)
    // as an index into a zeros row, then the prior hidden rows, then the hidden rows of the decode, verify and prefill
    // groups. A verify entry's rows are the verify_rows positions after its pending token's, the row after its accepted
    // tokens drafting: its first row's index plus its accepted count - 1. Another sequence drafts from its last row.
    std::vector<std::int64_t> draft_tokens;
    std::vector<std::int64_t> draft_positions;
    std::vector<std::int64_t> draft_sources;
    // Per draft after the first, the RoPE position and key count of each sequence that drafts, in entry order, before a
    // verify entry's accepted count is added.
    std::vector<std::vector<std::int64_t>> later_positions;
    std::vector<std::vector<std::int32_t>> later_key_counts;
    // The step the verify entries' previous_verify_row and previous_drafting_row index.
    const batch* previous_batch = nullptr;
    // Per shard, the verify group's tokens on its device.
    std::vector<at::Tensor> verify_tokens;
    // Per shard, each verify entry's accepted tokens (int64 [verify entries]), and each drafting sequence's next token
    // then its drafts (int64 [drafting, verify_rows]); shard 0's copies to the host.
    std::vector<at::Tensor> accepted_token_counts_by_shard;
    std::vector<at::Tensor> shard_next_tokens;
    at::Tensor accepted_token_counts;
    at::Tensor next_tokens;

    // step_entries with their sequences found in owner, each recorded with whether it samples after its tokens and
    // whether they end its eviction cycle. With owner's graph_pages, a decode or verify group whose tables fit
    // graph_pages pages takes tables that wide and replays a graph.
    batch(executor& owner, std::span<const batch_tokens> step_entries);

    // The groups in entry order: decode, verify, prefill.
    std::array<group*, 3> groups() noexcept;
    std::array<const group*, 3> groups() const noexcept;
    // The entries of the sequences that draft, in entry order.
    std::vector<entry> drafting_entries() const;
    // A verify group measures each row's mass apart (model::batch::cu_view_pages), its queries weighing 1 from
    // measured_start on: the rows its write-back accepts are then weighed as one step's.
    static model::batch
    make_group(std::span<const entry> group, std::size_t page_size, std::size_t state_slots, model::batch_kind kind);
    // The tokens e forwards: its pending tokens', and its drafts.
    static std::size_t forwarded(const entry& e) noexcept;
    // The widest table among the group's sequences.
    static std::size_t table_width(std::span<const entry> group) noexcept;
    static bool measures_mass(std::span<const entry> group) noexcept;
    // Whether e forwards a query that adds mass.
    static bool measures_mass(const entry& e) noexcept;
    // The group's tables of lines, or of positions with positions, on shard's device, as model::batch lays them out.
    static at::Tensor page_tables(
        std::span<const entry> group,
        std::size_t n_kv_layers,
        bool positions,
        std::size_t shard,
        std::size_t width
    );
    // Queues on the current stream, shard's, the copies of g's inputs to its device, with its tables, the filling of a
    // decode group's awaited tokens from samples and a verify group's tokens of the previous step. Returns the inputs
    // on the device and their zeroed page mass, undefined for a group that measures none. mass_heads is the shard's
    // part of the executor's.
    std::pair<model::batch, at::Tensor> upload_inputs(
        const group& g,
        std::span<const kv> lines,
        const std::vector<std::size_t>& mass_heads,
        std::size_t shard,
        const at::Tensor& samples
    );
    // Queues on stream shard's copy of the prior hidden rows from draft_layer's lines, then per group its inputs
    // (upload_inputs), its forward and the copy of its page mass to the host, then the sampling into samples; shard 0
    // also queues the copy of the sampled tokens to the host. mass_heads is the shard's part of the executor's. graphs
    // is the shard's when a group is replayed. Every shard's call may run at once, each on its own thread.
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
    // Adds to the draft head's pass the rows of e's sequence, at launch before it advances: those of an entry of the
    // verify group, with verifies, its verify_rows rows, another's from its first position without one through its
    // tokens and, with next_row, its next token, the sequence drafting draft_count tokens after them with drafts. The
    // entry's hidden rows start at first_hidden_row of the pass's hidden rows.
    void plan_draft_rows(
        entry& e,
        bool verifies,
        std::size_t first_hidden_row,
        bool next_row,
        bool drafts,
        std::size_t draft_layer,
        std::size_t draft_count
    );
    // Queues on stream shard, after the step's sampling into samples, the verify entries' acceptance of their samples,
    // stop_tokens ending a run (int64), and the draft head's pass; shard 0 also queues the copies of the accepted
    // counts and next tokens to the host.
    void enqueue_drafts(
        model& shard_model,
        std::span<const kv> lines,
        const at::Tensor& samples,
        const at::Tensor& stop_tokens,
        std::size_t shard,
        at::Tensor& counters
    );
    // Whether the step replays a decode_graphs verify step: every entry verifies and drafts, and the verify group
    // replays.
    bool steps_replay() const noexcept;
    // Queues on device, shard's, the replay of its verify step from graphs, on_device being the verify group and
    // verify_mass its page mass, then the copies of its outputs: into samples, and on shard 0 to the host.
    void enqueue_step(
        const model::batch& on_device,
        const at::Tensor& verify_mass,
        std::size_t shard,
        at::Device device,
        decode_graphs* graphs,
        const at::Tensor& samples
    );
    // Takes into g, the verify group on shard's device, the previous step's next tokens of the entries whose previous
    // step is launched, and moves their positions by its accepted count - 1.
    void take_previous_tokens(model::batch& g, std::size_t shard, at::Device device) const;
    // On device, shard's, per row sampled, the tokens the previous step of a verify entry's sequence accepts, by which its
    // draw counts lag; 0 for the other rows.
    at::Tensor previous_draw_shifts(std::size_t shard, at::Device device) const;
    // On device, shard's, each verify entry's accepted count - 1 of its previous step, 0 without one, int64 [verify
    // entries].
    at::Tensor previous_shift(std::size_t shard, at::Device device) const;
    // Joins the shards' page mass, each shard's heads after the previous shards', once their copies completed.
    void join_mass(const std::vector<std::vector<std::size_t>>& shard_mass_heads, const view_layout& layout);
    // Per KV layer, the mass_sum the step gave each page of a view, the layer's mass_heads values per page, and their
    // weight_sum; empty for a layer that measures none.
    struct layer_step_mass {
        std::span<const float> mass_sum;
        std::span<const float> weight_sum;
    };
    // The page mass of the view view of group from mass, laid out as join_mass leaves it.
    static std::vector<layer_step_mass>
    view_mass(const model::batch& group, const at::Tensor& mass, std::size_t view, const view_layout& layout);
    // Adds the page mass of the entry at position of g, which measures mass and whose sequence accepted accepted_count
    // tokens, to its sequence's layers. layout is the executor's.
    void add_measured_mass(
        const group& g,
        std::size_t position,
        std::size_t accepted_count,
        const view_layout& layout
    ) const;

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
  layout{.layers = {}, .page_size = 0},
  root(root),
  stop_ids(std::move(stop_ids)),
  graph_pages(graph_pages),
  draft_state_slots(draft_tokens + 1) {
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

executor::batch::batch(executor& owner, std::span<const batch_tokens> step_entries) {
    bool has_prefill = false;
    for (const auto& [id, token_count] : step_entries) {
        sequence& seq = owner.sequences.at(id);
        has_prefill = has_prefill || seq.pending_token_list().size() > 1;
        group& g = has_prefill ? prefill_group : seq.draft_tokens().empty() ? decode_group : verify_group;
        g.entries.push_back(
            entry{
                .seq = sequence_ref(owner, id),
                .token_count = token_count,
                .is_sampling = token_count == seq.pending_token_list().size() && seq.may_sample(),
                .ends_cycle = seq.ends_cycle(token_count),
            }
        );
    }
    if (!verify_group.entries.empty()) {
        verify_rows = forwarded(verify_group.entries.front());
        assert(std::ranges::all_of(verify_group.entries, [&](const entry& e) {
            return e.is_sampling && forwarded(e) == verify_rows;
        }));
    }
    for (group* g : groups()) {
        if (g->entries.empty()) {
            continue;
        }
        g->inputs = make_group(g->entries, owner.layout.page_size, owner.draft_state_slots, g->inputs.kind);
        g->table_width = table_width(g->entries);
        if (g != &prefill_group && owner.graph_pages && g->table_width <= *owner.graph_pages) {
            g->is_replayed = true;
            g->table_width = *owner.graph_pages;
        }
    }
    awaits_tokens = std::ranges::any_of(decode_group.entries, [](const entry& e) {
        return e.seq->pending_token_list().front() == sequence::awaited_token;
    });
    // A verify entry's draws advance by the tokens its write-back accepts.
    for (group* g : groups()) {
        for (entry& e : g->entries) {
            e.first_sample_row = sample_params.size();
            if (!e.is_sampling) {
                continue;
            }
            const sampling_draw next_draw = e.seq->next_draw();
            for (std::size_t row = 0; row < (g == &verify_group ? verify_rows : 1); row++) {
                sample_params.push_back(e.seq->sampling_parameters());
                sample_draws.push_back(sampling_draw{.seed = next_draw.seed, .count = next_draw.count + row});
            }
        }
    }
}

std::array<executor::batch::group*, 3> executor::batch::groups() noexcept {
    return {&decode_group, &verify_group, &prefill_group};
}

std::array<const executor::batch::group*, 3> executor::batch::groups() const noexcept {
    return {&decode_group, &verify_group, &prefill_group};
}

std::vector<executor::batch::entry> executor::batch::drafting_entries() const {
    std::vector<entry> drafting;
    for (const group* g : groups()) {
        for (const entry& e : g->entries) {
            if (e.draft && e.draft->is_drafting) {
                drafting.push_back(e);
            }
        }
    }
    return drafting;
}

std::size_t executor::batch::forwarded(const entry& e) noexcept {
    return e.token_count + e.seq->draft_tokens().size();
}

model::batch executor::batch::make_group(
    std::span<const entry> group,
    std::size_t page_size,
    std::size_t state_slots,
    model::batch_kind kind
) {
    std::size_t token_count = 0;
    for (const entry& e : group) {
        token_count += forwarded(e);
    }
    std::size_t n_layers = group.front().seq->views().size();
    auto n_tokens = static_cast<std::int64_t>(token_count);
    auto n_seqs = static_cast<std::int64_t>(group.size());
    auto n_kv_layers = static_cast<std::int64_t>(n_layers);
    model::batch g{
        .kind = kind,
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
            const std::int64_t token = i < group[s].token_count ? seq.pending_token_list()[i]
                                                                : seq.draft_tokens()[i - group[s].token_count];
            tokens[first_row + i] = token == sequence::awaited_token
                ? -static_cast<std::int64_t>(seq.awaited_sample_row()) - 1
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
    if (kind != model::batch_kind::decode) {
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
        rejected[s] = static_cast<std::int32_t>(group[s].seq->rejected_token_count());
    }
    g.page_size = static_cast<std::int64_t>(page_size);
    g.state_slots = static_cast<std::int64_t>(state_slots);
    if (measures_mass(group)) {
        // A verify group's views are its rows'.
        const bool per_row = kind == model::batch_kind::verify;
        const std::int64_t views = per_row ? n_tokens : n_seqs;
        g.cu_view_pages = pinned({n_kv_layers, views + 1}, at::kInt);
        g.query_weights = pinned({n_tokens}, at::kFloat);
        auto* cu_view_pages = g.cu_view_pages.mutable_data_ptr<std::int32_t>();
        for (std::size_t layer = 0; layer < n_layers; layer++) {
            std::int32_t* row = cu_view_pages + layer * static_cast<std::size_t>(views + 1);
            row[0] = 0;
            std::size_t view = 0;
            for (const entry& e : group) {
                const auto pages = static_cast<std::int32_t>(e.seq->views()[layer].lines.size());
                for (std::size_t i = 0; i < (per_row ? forwarded(e) : 1); i++, view++) {
                    row[view + 1] = row[view] + pages;
                }
            }
        }
        std::span<float> weights(g.query_weights.mutable_data_ptr<float>(), token_count);
        std::size_t first = 0;
        for (const entry& e : group) {
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

std::size_t executor::batch::table_width(std::span<const entry> group) noexcept {
    std::size_t width = 0;
    for (const entry& e : group) {
        width = std::max(width, e.seq->table_width());
    }
    return width;
}

bool executor::batch::measures_mass(std::span<const entry> group) noexcept {
    return std::ranges::any_of(group, [](const entry& e) { return measures_mass(e); });
}

// A sequence whose previous step is launched may lie up to that step's drafts further on.
bool executor::batch::measures_mass(const entry& e) noexcept {
    const std::size_t lag = e.seq->launched_step_count() > 0 ? e.seq->draft_tokens().size() : 0;
    return e.seq->eviction_policy_in_pages().can_evict() &&
        e.seq->history_position() + forwarded(e) + lag > e.seq->measured_start();
}

at::Tensor executor::batch::page_tables(
    std::span<const entry> group,
    std::size_t n_kv_layers,
    bool positions,
    std::size_t shard,
    std::size_t width
) {
    std::vector<at::Tensor> tables;
    std::vector<std::size_t> page_counts;
    for (const entry& e : group) {
        tables.push_back(positions ? e.seq->position_table(shard) : e.seq->line_table(shard));
        page_counts.push_back(e.seq->table_width());
    }
    return gather_page_tables_cuda(
        tables,
        page_counts,
        static_cast<std::int64_t>(n_kv_layers),
        static_cast<std::int64_t>(width)
    );
}

std::pair<model::batch, at::Tensor> executor::batch::upload_inputs(
    const group& g,
    std::span<const kv> lines,
    const std::vector<std::size_t>& mass_heads,
    std::size_t shard,
    const at::Tensor& samples
) {
    at::Device device = lines.front().k.device();
    model::batch on_device = to_device(g.inputs, device);
    on_device.page_tables = page_tables(g.entries, lines.size(), false, shard, g.table_width);
    on_device.page_positions = page_tables(g.entries, lines.size(), true, shard, g.table_width);
    if (&g == &decode_group) {
        if (awaits_tokens && !g.is_replayed) {
            fill_awaited_tokens_cuda(on_device.tokens, samples);
        }
    } else if (&g == &verify_group) {
        if (previous_batch) {
            take_previous_tokens(on_device, shard, device);
        }
        verify_tokens[shard] = on_device.tokens;
    } else {
        const at::Tensor& cu_seqlens_q = on_device.cu_seqlens_q;
        on_device.logit_rows = cu_seqlens_q.narrow(0, 1, cu_seqlens_q.size(0) - 1).to(at::kLong) - 1;
    }
    at::Tensor mass;
    if (g.inputs.cu_view_pages.defined()) {
        mass = zero_page_mass(on_device, g.inputs.cu_view_pages, mass_heads);
    }
    return {std::move(on_device), std::move(mass)};
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
    if (draft_layer && !prior_lines.empty()) {
        at::Tensor line_ids = at::tensor(prior_lines, at::TensorOptions().dtype(at::kLong)).to(device, true);
        shard_prior_hidden[shard] = lines[*draft_layer].last_hidden.index_select(0, line_ids);
    }
    if (steps_replay()) {
        auto [on_device, mass] = upload_inputs(verify_group, lines, mass_heads, shard, samples);
        enqueue_step(on_device, mass, shard, device, graphs, samples);
        return;
    }
    // A replayed group reads its graph's own counters.
    std::int64_t needed = 0;
    for (const group* g : groups()) {
        if (!g->entries.empty() && !g->is_replayed) {
            needed = std::max(
                needed,
                shard_model
                    .arrival_counter_count(static_cast<std::int64_t>(g->entries.size()), g->inputs.tokens.size(0))
            );
        }
    }
    if (!counters.defined() || counters.numel() < needed) {
        counters = at::zeros({needed}, at::device(device).dtype(at::kInt));
    }
    std::vector<at::Tensor> rows;
    for (group* g : groups()) {
        if (g->entries.empty()) {
            continue;
        }
        auto [on_device, mass] = upload_inputs(*g, lines, mass_heads, shard, samples);
        on_device.arrival_counters = counters;
        model::output output;
        if (!g->is_replayed) {
            output = shard_model.forward(on_device, lines);
        } else if (g == &decode_group) {
            output = graphs->decode(on_device);
        } else {
            output = graphs->verify(on_device);
        }
        g->hidden_by_shard[shard] = output.hidden;
        if (mass.defined()) {
            g->mass_by_shard[shard] = pinned(mass.sizes(), at::kFloat);
            g->mass_by_shard[shard].copy_(mass, /*non_blocking=*/true);
        }
        const auto entry_rows = static_cast<std::int64_t>(g == &verify_group ? verify_rows : 1);
        for (std::size_t position = 0; position < g->entries.size(); position++) {
            if (g->entries[position].is_sampling) {
                rows.push_back(output.logits.narrow(0, static_cast<std::int64_t>(position) * entry_rows, entry_rows));
            }
        }
    }
    if (!rows.empty()) {
        at::Tensor tokens = previous_batch
            ? sample(at::cat(rows), sample_params, sample_draws, previous_draw_shifts(shard, device))
            : sample(at::cat(rows), sample_params, sample_draws);
        samples.narrow(0, 0, tokens.size(0)).copy_(tokens);
        if (shard == 0) {
            sampled_tokens = pinned({tokens.size(0)}, at::kLong);
            sampled_tokens.copy_(tokens, /*non_blocking=*/true);
        }
    }
    if (draft_layer && !draft_tokens.empty()) {
        enqueue_drafts(shard_model, lines, samples, stop_tokens, shard, counters);
    }
}

at::Tensor executor::batch::previous_draw_shifts(std::size_t shard, at::Device device) const {
    const auto longs = at::TensorOptions().dtype(at::kLong);
    std::vector<std::int64_t> rows;
    std::vector<std::int64_t> verify_entries;
    for (std::size_t v = 0; v < verify_group.entries.size(); v++) {
        const entry& e = verify_group.entries[v];
        if (!e.previous_verify_row) {
            continue;
        }
        for (std::size_t j = 0; j < verify_rows; j++) {
            rows.push_back(static_cast<std::int64_t>(e.first_sample_row + j));
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
    at::Tensor accepted_before = previous_shift(shard, device) + 1;
    return shifts.index_copy(
        0,
        at::tensor(rows, longs).to(device, /*non_blocking=*/true),
        accepted_before.index_select(0, at::tensor(verify_entries, longs).to(device, /*non_blocking=*/true))
    );
}

bool executor::batch::steps_replay() const noexcept {
    return verify_group.is_replayed && decode_group.entries.empty() && prefill_group.entries.empty() &&
        std::ranges::all_of(verify_group.entries, [](const entry& e) { return e.draft && e.draft->is_drafting; });
}

void executor::batch::enqueue_step(
    const model::batch& on_device,
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
    const auto sequences = static_cast<std::int64_t>(verify_group.entries.size());
    sampling_inputs sampling = sampling_inputs_of(sample_params, sample_draws, device);
    at::Tensor shift = at::zeros({sequences}, at::device(device).dtype(at::kLong));
    if (previous_batch) {
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
    std::vector<std::int64_t> max_accepted_token_counts;
    std::vector<std::int64_t> key_counts;
    for (const entry& e : verify_group.entries) {
        max_accepted_token_counts.push_back(static_cast<std::int64_t>(e.max_accepted_token_count));
        key_counts.push_back(static_cast<std::int64_t>(e.draft->key_count));
    }
    auto [sampled, verified_accepted_counts, next] = graphs->verify_step(
        on_device,
        sampling,
        on_device_tensor(max_accepted_token_counts),
        on_device_tensor(draft_positions) + at::repeat_interleave(shift, rows),
        on_device_tensor(key_counts) + shift,
        on_device_tensor(later_positions_flat).view({later, sequences}) + shift,
        on_device_tensor(later_key_counts_flat).view({later, sequences}) + shift
    );
    accepted_token_counts_by_shard[shard] = verified_accepted_counts;
    shard_next_tokens[shard] = next;
    samples.narrow(0, 0, sampled.size(0)).copy_(sampled);
    if (verify_mass.defined()) {
        verify_group.mass_by_shard[shard] = pinned(verify_mass.sizes(), at::kFloat);
        verify_group.mass_by_shard[shard].copy_(verify_mass, /*non_blocking=*/true);
    }
    if (shard == 0) {
        sampled_tokens = pinned(sampled.sizes(), at::kLong);
        sampled_tokens.copy_(sampled, /*non_blocking=*/true);
        accepted_token_counts = pinned(verified_accepted_counts.sizes(), at::kLong);
        accepted_token_counts.copy_(verified_accepted_counts, /*non_blocking=*/true);
        next_tokens = pinned(next.sizes(), at::kLong);
        next_tokens.copy_(next, /*non_blocking=*/true);
    }
}

void executor::batch::take_previous_tokens(model::batch& g, std::size_t shard, at::Device device) const {
    const auto longs = at::TensorOptions().dtype(at::kLong);
    const auto rows = static_cast<std::int64_t>(verify_rows);
    const auto verifying = static_cast<std::int64_t>(verify_group.entries.size());
    std::vector<std::int64_t> entries_with_previous;
    std::vector<std::int64_t> drafting_rows;
    std::vector<std::int64_t> verify_measured_starts;
    for (std::size_t v = 0; v < verify_group.entries.size(); v++) {
        const entry& e = verify_group.entries[v];
        if (e.previous_drafting_row) {
            entries_with_previous.push_back(static_cast<std::int64_t>(v));
            drafting_rows.push_back(static_cast<std::int64_t>(*e.previous_drafting_row));
        }
        verify_measured_starts.push_back(static_cast<std::int64_t>(e.measured_start));
    }
    g.tokens.view({verifying, rows})
        .index_copy_(
            0,
            at::tensor(entries_with_previous, longs).to(device, /*non_blocking=*/true),
            previous_batch->shard_next_tokens[shard]
                .index_select(0, at::tensor(drafting_rows, longs).to(device, /*non_blocking=*/true))
        );
    at::Tensor shift = previous_shift(shard, device);
    g.pos += at::repeat_interleave(shift, rows);
    g.key_counts += shift.to(at::kInt).unsqueeze(0);
    g.history_positions += shift;
    g.rejected_tokens -= shift.to(at::kInt);
    if (g.query_weights.defined()) {
        at::Tensor row_positions = at::repeat_interleave(g.history_positions, rows) +
            at::arange(rows, g.history_positions.options()).repeat({verifying});
        at::Tensor measured_starts = at::tensor(verify_measured_starts, longs).to(device, /*non_blocking=*/true);
        g.query_weights = row_positions.ge(at::repeat_interleave(measured_starts, rows)).to(at::kFloat);
    }
}

std::vector<executor::batch::layer_step_mass> executor::batch::view_mass(
    const model::batch& group,
    const at::Tensor& mass,
    std::size_t view,
    const view_layout& layout
) {
    const float* values = mass.const_data_ptr<float>();
    auto row_size = static_cast<std::size_t>(group.cu_view_pages.size(1));
    const auto* cu_view_pages = group.cu_view_pages.const_data_ptr<std::int32_t>();
    std::vector<layer_step_mass> by_layer(layout.layers.size());
    std::size_t first = 0;
    for (std::size_t layer = 0; layer < layout.layers.size(); layer++) {
        std::size_t heads = layout.layers[layer].mass_heads;
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
    entry& e,
    bool verifies,
    std::size_t first_hidden_row,
    bool next_row,
    bool drafts,
    std::size_t draft_layer,
    std::size_t draft_count
) {
    const sequence& seq = *e.seq;
    const std::size_t start = seq.history_position();
    const std::size_t next_index = seq.next_index(draft_layer);
    const std::size_t next_position = seq.next_position();
    const auto sampled = static_cast<std::int64_t>(e.first_sample_row);
    const std::size_t first_draft_row = draft_tokens.size();
    // The position after the row that drafts, before a verify entry's accepted count is added.
    std::size_t drafted_after = 0;
    if (verifies) {
        for (std::size_t j = 0; j < verify_rows; j++) {
            draft_tokens.push_back(-(sampled + static_cast<std::int64_t>(j)) - 1);
            draft_positions.push_back(static_cast<std::int64_t>(next_position + j + 1));
            draft_sources.push_back(static_cast<std::int64_t>(first_hidden_row + j));
        }
        e.draft = entry::draft_rows{
            .row_count = verify_rows,
            .logit_row = first_draft_row,
            .key_count = next_index + verify_rows + 1,
            .is_drafting = drafts,
        };
    } else {
        const std::size_t first = seq.end_of_draft_rows();
        const std::size_t next = start + e.token_count;
        const std::size_t end = next_row ? next + 1 : next;
        assert((first == start || first == start + 1) && (next_row || !drafts));
        for (std::size_t t = first; t < end; t++) {
            if (t < next) {
                draft_tokens.push_back(seq.pending_token_list()[t - start]);
            } else {
                draft_tokens.push_back(e.is_sampling ? -sampled - 1 : seq.pending_token_list()[e.token_count]);
            }
            draft_positions.push_back(static_cast<std::int64_t>(next_position + t - start));
            if (t == 0) {
                draft_sources.push_back(0);
            } else if (t == start) {
                assert(e.prior_row);
                draft_sources.push_back(static_cast<std::int64_t>(1 + *e.prior_row));
            } else {
                draft_sources.push_back(static_cast<std::int64_t>(first_hidden_row + t - 1 - start));
            }
        }
        if (end == first) {
            return;
        }
        e.draft = entry::draft_rows{
            .row_count = end - first,
            .logit_row = first_draft_row + end - first - 1,
            .key_count = next_index + end - start,
            .is_drafting = drafts,
        };
        drafted_after = e.token_count;
    }
    if (drafts) {
        later_positions.resize(draft_count - 1);
        later_key_counts.resize(draft_count - 1);
        for (std::size_t j = 1; j < draft_count; j++) {
            later_positions[j - 1].push_back(static_cast<std::int64_t>(next_position + drafted_after + j));
            later_key_counts[j - 1].push_back(static_cast<std::int32_t>(next_index + drafted_after + j + 1));
        }
    }
}

at::Tensor executor::batch::previous_shift(std::size_t shard, at::Device device) const {
    at::Tensor shift = at::zeros(
        {static_cast<std::int64_t>(verify_group.entries.size())},
        at::device(device).dtype(at::kLong)
    );
    std::vector<std::int64_t> entries_with_previous;
    std::vector<std::int64_t> previous_rows;
    for (std::size_t v = 0; v < verify_group.entries.size(); v++) {
        if (const std::optional<std::size_t>& previous_row = verify_group.entries[v].previous_verify_row) {
            entries_with_previous.push_back(static_cast<std::int64_t>(v));
            previous_rows.push_back(static_cast<std::int64_t>(*previous_row));
        }
    }
    if (entries_with_previous.empty()) {
        return shift;
    }
    const auto longs = at::TensorOptions().dtype(at::kLong);
    at::Tensor accepted_before = previous_batch->accepted_token_counts_by_shard[shard].index_select(
        0,
        at::tensor(previous_rows, longs).to(device, /*non_blocking=*/true)
    );
    return shift
        .index_copy(0, at::tensor(entries_with_previous, longs).to(device, /*non_blocking=*/true), accepted_before - 1);
}

void executor::batch::enqueue_drafts(
    model& shard_model,
    std::span<const kv> lines,
    const at::Tensor& samples,
    const at::Tensor& stop_tokens,
    std::size_t shard,
    at::Tensor& counters
) {
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
    auto tables = [&](std::span<const entry> group, model::batch& g) {
        const std::size_t width = table_width(group);
        g.page_tables = page_tables(group, n_layers, false, shard, width);
        g.page_positions = page_tables(group, n_layers, true, shard, width);
    };
    auto with_counters = [&](model::batch& g, std::int64_t sequences) {
        const std::int64_t needed = shard_model.arrival_counter_count(sequences, g.tokens.size(0));
        if (!counters.defined() || counters.numel() < needed) {
            counters = at::zeros({needed}, at::device(device).dtype(at::kInt));
        }
        g.arrival_counters = counters;
    };

    // Each verify entry accepts its pending token and its drafts while sample t equals draft t + 1, then the sample
    // after them, up to and including a stop token, and at most its max_accepted_token_count.
    at::Tensor accepted_counts = at::empty({0}, at::device(device).dtype(at::kLong));
    at::Tensor shift = previous_shift(shard, device);
    const auto verifying = static_cast<std::int64_t>(verify_group.entries.size());
    if (verifying > 0) {
        const auto rows = static_cast<std::int64_t>(verify_rows);
        std::vector<std::int64_t> sample_indices;
        std::vector<std::int64_t> max_accepted_token_counts;
        for (const entry& e : verify_group.entries) {
            for (std::size_t j = 0; j < verify_rows; j++) {
                sample_indices.push_back(static_cast<std::int64_t>(e.first_sample_row + j));
            }
            max_accepted_token_counts.push_back(static_cast<std::int64_t>(e.max_accepted_token_count));
        }
        at::Tensor verified = samples.index_select(0, on_device(sample_indices, longs)).view({verifying, rows});
        at::Tensor drafted = verify_tokens[shard].view({verifying, rows}).narrow(1, 1, rows - 1);
        accepted_counts = verified.narrow(1, 0, rows - 1).eq(drafted).to(at::kLong).cumprod(1).sum(1) + 1;
        if (stop_tokens.numel() > 0) {
            at::Tensor stops = at::isin(verified, stop_tokens);
            at::Tensor first_stop = at::where(stops.any(1), stops.to(at::kLong).argmax(1), rows);
            accepted_counts = at::minimum(accepted_counts, first_stop + 1);
        }
        accepted_counts = at::minimum(accepted_counts, on_device(max_accepted_token_counts, longs));
    }
    accepted_token_counts_by_shard[shard] = accepted_counts;

    std::vector<at::Tensor> sources;
    if (shard_prior_hidden[shard].defined()) {
        sources.push_back(shard_prior_hidden[shard]);
    }
    for (const group* g : groups()) {
        if (g->hidden_by_shard[shard].defined()) {
            sources.push_back(g->hidden_by_shard[shard]);
        }
    }
    sources.insert(sources.begin(), at::zeros({1, head.hidden_dim()}, sources.back().options()));
    // Per entry with draft rows, its position among the verify entries, -1 for another.
    std::vector<entry> draft_entries;
    std::vector<std::int64_t> verify_indices;
    std::vector<std::int64_t> drafting_indices;
    std::vector<std::int32_t> starts{0};
    std::vector<std::int64_t> entry_rows;
    std::vector<std::int64_t> draft_key_counts;
    std::vector<std::int64_t> draft_logit_rows;
    for (const group* g : groups()) {
        for (std::size_t position = 0; position < g->entries.size(); position++) {
            const entry& e = g->entries[position];
            if (!e.draft) {
                continue;
            }
            if (e.draft->is_drafting) {
                drafting_indices.push_back(static_cast<std::int64_t>(draft_entries.size()));
            }
            draft_entries.push_back(e);
            verify_indices.push_back(g == &verify_group ? static_cast<std::int64_t>(position) : -1);
            starts.push_back(starts.back() + static_cast<std::int32_t>(e.draft->row_count));
            entry_rows.push_back(static_cast<std::int64_t>(e.draft->row_count));
            draft_key_counts.push_back(static_cast<std::int64_t>(e.draft->key_count));
            draft_logit_rows.push_back(static_cast<std::int64_t>(e.draft->logit_row));
        }
    }
    // Per draft entry, the accepted count - 1 of a verify entry, and the previous step's shift of its positions.
    const auto draft_count = static_cast<std::int64_t>(draft_entries.size());
    at::Tensor verify_index = on_device(verify_indices, longs);
    at::Tensor is_verify = verify_index.ge(0);
    at::Tensor clamped = verify_index.clamp_min(0);
    at::Tensor accepted_after = verifying > 0 ? at::where(is_verify, accepted_counts.index_select(0, clamped) - 1, 0)
                                              : at::zeros({draft_count}, at::device(device).dtype(at::kLong));
    at::Tensor entry_shift = verifying > 0 ? at::where(is_verify, shift.index_select(0, clamped), 0)
                                           : at::zeros({draft_count}, at::device(device).dtype(at::kLong));
    at::Tensor row_shift = at::repeat_interleave(
        entry_shift,
        on_device(entry_rows, longs),
        /*dim=*/0,
        static_cast<std::int64_t>(draft_tokens.size())
    );
    model::batch rows{
        .kind = model::batch_kind::prefill,
        .tokens = on_device(draft_tokens, longs),
        .pos = on_device(draft_positions, longs) + row_shift,
        .cu_seqlens_q = on_device(starts, ints),
        .key_counts = key_counts(on_device(draft_key_counts, longs) + entry_shift),
        .logit_rows = on_device(draft_logit_rows, longs) + accepted_after,
    };
    fill_awaited_tokens_cuda(rows.tokens, samples);
    tables(draft_entries, rows);
    at::Tensor hidden = at::cat(sources).index_select(0, on_device(draft_sources, longs));
    with_counters(rows, draft_count);
    model::output drafted_rows = head.draft(rows, hidden, lines);
    if (!drafting_indices.empty()) {
        at::Tensor drafting_entries_index = on_device(drafting_indices, longs);
        const std::vector<entry> drafting = drafting_entries();
        at::Tensor drafting_rows = rows.logit_rows.index_select(0, drafting_entries_index);
        at::Tensor later_shift = (accepted_after + 1).mul(is_verify.to(at::kLong)) + entry_shift;
        later_shift = later_shift.index_select(0, drafting_entries_index);
        at::Tensor token = drafted_rows.logits.index_select(0, drafting_entries_index).argmax(-1);
        at::Tensor previous_hidden = drafted_rows.hidden.index_select(0, drafting_rows);
        std::vector<at::Tensor> tokens{rows.tokens.index_select(0, drafting_rows), token};
        for (std::size_t j = 0; j < later_positions.size(); j++) {
            model::batch one_row{
                .kind = model::batch_kind::decode,
                .tokens = token,
                .pos = on_device(later_positions[j], longs) + later_shift,
                .key_counts = key_counts(on_device(later_key_counts[j], longs) + later_shift),
            };
            tables(drafting, one_row);
            with_counters(one_row, static_cast<std::int64_t>(drafting.size()));
            model::output next = head.draft(one_row, previous_hidden, lines);
            token = next.logits.argmax(-1);
            previous_hidden = next.hidden;
            tokens.push_back(token);
        }
        shard_next_tokens[shard] = at::stack(tokens, 1);
    }
    if (shard == 0) {
        accepted_token_counts = pinned(accepted_counts.sizes(), at::kLong);
        accepted_token_counts.copy_(accepted_counts, /*non_blocking=*/true);
        if (!drafting_indices.empty()) {
            next_tokens = pinned(shard_next_tokens[shard].sizes(), at::kLong);
            next_tokens.copy_(shard_next_tokens[shard], /*non_blocking=*/true);
        }
    }
}

void executor::batch::join_mass(
    const std::vector<std::vector<std::size_t>>& shard_mass_heads,
    const view_layout& layout
) {
    for (group* g : groups()) {
        if (!g->mass_by_shard.front().defined() || g->mass_by_shard.size() == 1) {
            g->mass = g->mass_by_shard.front();
            continue;
        }
        std::int64_t total = 0;
        auto row_size = static_cast<std::size_t>(g->inputs.cu_view_pages.size(1));
        const auto* cu_view_pages = g->inputs.cu_view_pages.const_data_ptr<std::int32_t>();
        for (std::size_t layer = 0; layer < layout.layers.size(); layer++) {
            if (layout.layers[layer].mass_heads > 0) {
                auto pages = static_cast<std::int64_t>(cu_view_pages[layer * row_size + row_size - 1]);
                total += pages * static_cast<std::int64_t>(layout.layers[layer].mass_heads + 1);
            }
        }
        g->mass = pinned({total}, at::kFloat);
        float* out = g->mass.mutable_data_ptr<float>();
        std::vector<const float*> in;
        for (const at::Tensor& mass : g->mass_by_shard) {
            in.push_back(mass.const_data_ptr<float>());
        }
        for (std::size_t layer = 0; layer < layout.layers.size(); layer++) {
            if (layout.layers[layer].mass_heads == 0) {
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
    }
}

// A verify entry measured each row's mass with weight 1; its accepted rows are weighed as one step of accepted tokens.
void executor::batch::add_measured_mass(
    const group& g,
    std::size_t position,
    std::size_t accepted_count,
    const view_layout& layout
) const {
    const entry& e = g.entries[position];
    sequence& seq = *e.seq;
    const std::size_t n_layers = layout.layers.size();
    std::vector<layer_step_mass> step_mass;
    std::vector<std::vector<float>> verified_mass(n_layers);
    std::vector<std::vector<float>> verified_weight(n_layers);
    if (&g == &verify_group) {
        std::vector<float> weights(accepted_count);
        seq.fill_query_weights(e.start_position, weights);
        step_mass.resize(n_layers);
        for (std::size_t row = 0; row < accepted_count; row++) {
            std::vector<layer_step_mass> row_mass = view_mass(g.inputs, g.mass, position * verify_rows + row, layout);
            for (std::size_t layer = 0; layer < n_layers; layer++) {
                if (layout.layers[layer].mass_heads == 0) {
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
                step_mass[layer] = layer_step_mass{
                    .mass_sum = verified_mass[layer],
                    .weight_sum = verified_weight[layer],
                };
            }
        }
    } else {
        step_mass = view_mass(g.inputs, g.mass, position, layout);
    }
    std::vector<std::span<const float>> step_mass_sums;
    std::vector<std::span<const float>> step_weight_sums;
    for (const layer_step_mass& layer_mass : step_mass) {
        step_mass_sums.push_back(layer_mass.mass_sum);
        step_weight_sums.push_back(layer_mass.weight_sum);
    }
    seq.add_step_mass(step_mass_sums, step_weight_sums, e.end_position);
}

task<std::expected<executor::sequence_id, request_error>> executor::start(request&& r, result_writer& writer) {
    require(
        !r.tokens.empty() && r.eviction.n_working >= 1 && r.eviction.cycle_tokens >= 1 &&
            r.eviction.attention_mass_decay > 0.0 && r.eviction.attention_mass_decay <= 1.0 &&
            r.eviction.eviction_threshold >= 0.0 && r.eviction.eviction_threshold <= 1.0,
        "executor::start: the request has no token or an eviction policy out of range"
    );
    digest chain_root = r.initial_digest.value_or(
        make_request_root_digest(root, to_pages(r.eviction, layout.page_size))
    );
    std::vector<digest> digests = make_page_digests(chain_root, r.tokens, layout.page_size);
    std::expected<std::pair<std::shared_ptr<const state>, state::lease_id_type>, page_error>
        lookup = co_await store.read_state(r.tokens, digests);
    if (!lookup) {
        co_return std::unexpected(request_error::miss);
    }
    auto& [stored_state, lease_id] = *lookup;
    std::size_t stored_pages = 0;
    if (stored_state->head) {
        auto head_digest = std::ranges::find(digests, *stored_state->head);
        if (head_digest == digests.end()) {
            store.release(lease_id);
            co_return std::unexpected(request_error::miss);
        }
        stored_pages = static_cast<std::size_t>(head_digest - digests.begin()) + 1;
    }
    std::span<const std::int64_t> unstored_tokens = std::span(r.tokens).subspan(stored_pages * layout.page_size);
    if (stored_state->tail.size() >= unstored_tokens.size() ||
        !std::ranges::equal(stored_state->tail, unstored_tokens.first(stored_state->tail.size()))) {
        store.release(lease_id);
        co_return std::unexpected(request_error::miss);
    }
    std::shared_ptr<const state> initial_state = extend_with_hot_pages(std::move(stored_state), r, digests);
    std::vector<at::Device> devices;
    for (const std::vector<kv>& shard_lines : lines) {
        devices.push_back(shard_lines.front().k.device());
    }
    const sequence_id id = sequences.emplace(
        std::move(r),
        std::move(initial_state),
        lease_id,
        std::move(digests),
        chain_root,
        layout,
        writer,
        std::span<const at::Device>(devices)
    );
    sequence& seq = sequences.at(id);
    // take_lines holds the resident pages before it first suspends, so they are the ones counted here.
    seq.record_device_cached_tokens(device_cached_tokens(seq));
    std::expected<void, page_error> line_allocation = co_await take_lines(seq, seq.cycle_lines());
    if (!line_allocation) {
        drop(id);
        co_return std::unexpected(
            line_allocation.error() == page_error::miss ? request_error::miss : request_error::oom
        );
    }
    if (seq.history_page_count() > stored_pages && !co_await own_pages_from(seq, stored_pages)) {
        drop(id);
        co_return std::unexpected(request_error::oom);
    }
    seq.upload_table(0, streams);
    co_return id;
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

void executor::drop(sequence_id id) noexcept {
    if (sequence* dropped = sequences.find(id)) {
        release_lines(*dropped);
        store.release(dropped->lease_id());
        sequences.erase(id);
    }
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

std::shared_ptr<const state> executor::extend_with_hot_pages(
    std::shared_ptr<const state>&& s,
    const request& r,
    std::span<const digest> digests
) const {
    if (std::ranges::contains(layout.layers, true, &view_layout::layer::is_recurrent)) {
        return std::move(s);
    }
    std::size_t head_pages = 0;
    if (s->head) {
        auto head_digest = std::ranges::find(digests, *s->head);
        assert(head_digest != digests.end());
        head_pages = static_cast<std::size_t>(head_digest - digests.begin()) + 1;
    }
    page_eviction_policy policy = to_pages(r.eviction, layout.page_size);
    std::size_t resumed_end = head_pages * layout.page_size + s->tail.size();
    std::size_t cycle_end = (resumed_end / policy.cycle_tokens + 1) * policy.cycle_tokens;
    std::size_t window_start = cycle_end - policy.mass_tokens;
    if (resumed_end > window_start) {
        return std::move(s);
    }
    std::size_t page_limit = std::min(window_start, r.tokens.size() - 1) / layout.page_size;
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
        const std::optional<std::size_t>& window = layout.layers[layer].sliding_window;
        std::size_t next_query = pages.size() * layout.page_size;
        if (window && next_query + 1 > *window) {
            std::size_t passed = std::min((next_query + 1 - *window) / layout.page_size, pages.size());
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
        const layer_view& view = seq.views()[layer];
        for (std::size_t page = 0; page < view.pages.size(); page++) {
            if (view.indices[page] >= first_index) {
                hashes[layer].push_back(view.pages[page]);
                owners[layer].push_back(seq.lease_id());
                line_ids[layer].push_back(view.lines[page]);
            }
        }
    }
    co_return co_await cache.sync(hashes, owners, line_ids, streams);
}

task<std::expected<request_result, request_error>> executor::stop(sequence_id id) {
    sequence& seq = sequences.at(id);
    if (!seq.has_exited() || seq.launched_step_count() > 0) {
        throw std::logic_error("executor::stop: the sequence has not exited or a launched step holds it");
    }
    release_lines(seq);
    std::expected<request_result, request_error> result = std::unexpected(request_error::cancelled);
    switch (*seq.reason_of_exit()) {
    case sequence::exit_reason::finished: {
        auto end_state = std::make_shared<const state>(seq.to_state());
        co_await store.write_state(seq.lease_id(), std::shared_ptr<const state>(end_state));
        result = request_result{
            .state = std::move(end_state),
            .total_tokens = seq.prompt_token_count() + seq.generated_token_count(),
            .cached_tokens = seq.cached_token_count(),
            .device_cached_tokens = seq.device_cached_token_count(),
            .peak_device_memory = seq.device_memory_peak(),
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
    drop(id);
    co_return result;
}

std::size_t executor::device_cached_tokens(const sequence& seq) const {
    std::vector<bool> page_fetched(seq.history_page_count(), false);
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        const layer_view& view = seq.views()[layer];
        if (layout.layers[layer].is_recurrent) {
            if (seq.head_digest() && !cache.contains(layer, *seq.head_digest())) {
                page_fetched[seq.history_page_count() - 1] = true;
            }
            continue;
        }
        for (std::size_t i = 0; i < view.pages.size(); i++) {
            if (!cache.contains(layer, view.pages[i])) {
                page_fetched[view.indices[i]] = true;
            }
        }
    }
    return (seq.history_page_count() - static_cast<std::size_t>(std::ranges::count(page_fetched, true))) *
        layout.page_size;
}

task<std::expected<void, page_error>> executor::take_lines(sequence& seq, std::size_t line_count) {
    std::vector<std::span<const digest>> held_hashes(cache.layer_count());
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        if (!layout.layers[layer].is_recurrent) {
            held_hashes[layer] = seq.views()[layer].pages;
        } else if (seq.head_digest()) {
            held_hashes[layer] = std::span(&*seq.head_digest(), 1);
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
        if (layout.layers[layer].is_recurrent) {
            // The stored state's copy takes the last slot.
            counts[layer] = layout.layers[layer].state_slots + 2 - ((*page_lines)[layer].empty() ? 0 : 1);
        }
    }
    std::expected<std::vector<std::vector<std::uint32_t>>, page_error> allocation = cache.allocate(counts);
    std::vector<std::vector<std::uint32_t>> states(cache.layer_count());
    bool copied = allocation.has_value();
    for (std::size_t layer = 0; layer < cache.layer_count() && copied; layer++) {
        if (layout.layers[layer].is_recurrent && !(*page_lines)[layer].empty()) {
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
        std::span<const std::uint32_t> allocated = (*allocation)[layer];
        if (layout.layers[layer].is_recurrent) {
            const std::size_t first_slots = layout.layers[layer].state_slots - 1;
            seq.add_lines(layer, allocated.first(first_slots));
            allocated = allocated.subspan(first_slots);
            if (states[layer].empty()) {
                for (std::size_t shard = 0; shard < shards.size(); shard++) {
                    c10::cuda::CUDAStreamGuard guard(
                        c10::cuda::getStreamFromExternal(streams[shard], lines[shard][layer].k.device().index())
                    );
                    lines[shard][layer].k.select(0, allocated.front()).zero_();
                }
                seq.add_lines(layer, allocated.first(1));
                allocated = allocated.subspan(1);
            } else {
                seq.add_lines(layer, states[layer]);
            }
        } else {
            seq.add_lines(layer, (*page_lines)[layer]);
        }
        seq.add_lines(layer, allocated);
    }
    // The copies of the states at head read the fetched lines.
    co_await synchronize_shards();
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        if (layout.layers[layer].is_recurrent) {
            cache.release(layer, (*page_lines)[layer]);
        }
    }
    record_device_memory(seq);
    co_return std::expected<void, page_error>{};
}

void executor::record_device_memory(sequence& seq) const noexcept {
    std::size_t bytes = 0;
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        bytes += seq.views()[layer].lines.size() * cache.line_block_bytes(layer);
    }
    seq.record_device_memory(bytes);
}

void executor::release_lines(sequence& seq) noexcept {
    const std::vector<std::vector<std::uint32_t>> taken_lines = seq.take_lines();
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        cache.release(layer, taken_lines[layer]);
    }
}

bool executor::release_passed_pages(sequence& seq) {
    bool changed = false;
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        std::vector<std::size_t> passed_indices(
            seq.passed_pages(layer, seq.views()[layer].pages.size() * layout.page_size)
        );
        if (passed_indices.empty()) {
            continue;
        }
        std::iota(passed_indices.begin(), passed_indices.end(), 0);
        cache.release(layer, seq.remove_pages(layer, passed_indices));
        changed = true;
    }
    return changed;
}

executor::~executor() {
    for (const sequence& unstopped : sequences) {
        store.release(unstopped.lease_id());
    }
}

void executor::update_views(sequence& seq) {
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        if (!layout.layers[layer].mirrors_held_pages) {
            continue;
        }
        const layer_view& view = seq.views()[layer];
        std::vector<std::size_t> unheld;
        for (std::size_t page = 0; page < view.pages.size(); page++) {
            if (!std::ranges::binary_search(seq.held_page_indices(), view.indices[page])) {
                unheld.push_back(page);
            }
        }
        if (!unheld.empty()) {
            cache.release(layer, seq.remove_pages(layer, unheld));
        }
    }
}

bool executor::completes_page(const sequence& seq, std::size_t token_count) const noexcept {
    std::size_t start = seq.tail_tokens().size();
    return (start + token_count) / layout.page_size * layout.page_size > start;
}

// A drafting sequence's verify step ends neither its cycle nor max_tokens (executor::advance).
bool executor::allows_overlap(std::span<const batch_tokens> entries) const {
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
    std::unique_ptr<batch> b(new batch(*this, entries));
    TORCH_CHECK(
        static_cast<std::int64_t>(b->sample_params.size()) <= max_sampled_rows,
        "executor: a step samples at most ",
        max_sampled_rows,
        " tokens"
    );
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
        assert(e.previous_verify_row && e.previous_drafting_row);
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
    std::unique_ptr<batch> written = std::move(launched_batches.front());
    launched_batches.pop_front();
    batch& b = *written;
    for (external_task<cudaError_t>& completion : b.completions) {
        cudaError_t status = co_await std::move(completion);
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("write_back: ") + cudaGetErrorString(status));
        }
    }
    b.join_mass(shard_mass_heads, layout);
    std::size_t n_layers = cache.layer_count();
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
    const std::vector<batch::entry> drafting_entries = b.drafting_entries();
    for (std::size_t d = 0; d < drafting_entries.size(); d++) {
        const auto row = static_cast<std::size_t>(b.next_tokens.size(1));
        drafted.emplace(
            &*drafting_entries[d].seq,
            std::span(b.next_tokens.const_data_ptr<std::int64_t>() + d * row, row)
        );
    }
    for (batch::group* g : b.groups()) {
        const bool verifies = g == &b.verify_group;
        for (std::size_t position = 0; position < g->entries.size(); position++) {
            batch::entry& e = g->entries[position];
            sequence& seq = *e.seq;
            std::span<const std::int64_t> samples;
            if (e.is_sampling) {
                samples = std::span(next_sampled, verifies ? b.verify_rows : 1);
                next_sampled += samples.size();
            }
            if (seq.has_exited()) {
                continue;
            }
            if (seq.pending_page_writes()->any_failed()) {
                seq.exit(sequence::exit_reason::oom);
                continue;
            }
            // The tokens the step accepts, and of its samples those it emits. A step launched after this one took its next
            // tokens on the GPU and lags its accepted count - 1.
            const bool next_launched = seq.launched_step_count() > 1;
            std::size_t accepted_count = batch::forwarded(e);
            std::span<const std::int64_t> emitted = samples;
            if (verifies) {
                accepted_count = static_cast<std::size_t>(
                    b.accepted_token_counts.const_data_ptr<std::int64_t>()[position]
                );
                emitted = samples.first(accepted_count);
                const std::size_t start = e.start_position + seq.unapplied_position_shift();
                seq.apply_accepted_tokens(start, emitted, b.verify_rows);
                drafted_count.fetch_add(b.verify_rows - 1, std::memory_order_relaxed);
                accepted_draft_count.fetch_add(accepted_count - 1, std::memory_order_relaxed);
                e.start_position = start;
                e.end_position = start + accepted_count;
            }
            std::size_t completed_count = seq.complete_pages(e.end_position);
            if (e.is_measured) {
                b.add_measured_mass(*g, position, accepted_count, layout);
            }
            if (verifies) {
                for (std::int64_t token : emitted) {
                    seq.writer().append(token);
                }
                if (std::ranges::contains(stop_ids, emitted.back()) || !seq.may_sample()) {
                    seq.exit(sequence::exit_reason::finished);
                    if (next_launched) {
                        seq.withdraw_latest_verify_step();
                    }
                }
            } else if (!samples.empty()) {
                const std::int64_t token = samples.front();
                seq.apply_sample(e.sampled_position, token);
                seq.writer().append(token);
                if (std::ranges::contains(stop_ids, token)) {
                    seq.exit(sequence::exit_reason::finished);
                    if (seq.launched_step_count() > 1) {
                        seq.withdraw_latest_step();
                    }
                }
            }
            if (e.ends_sequence) {
                seq.exit(sequence::exit_reason::finished);
            }
            if (seq.has_exited()) {
                seq.set_drafts({});
            } else if (auto found = drafted.find(&seq); found != drafted.end() && !next_launched) {
                seq.set_drafts(found->second.subspan(1));
            }
            const bool reaches_state = e.ends_cycle || seq.reason_of_exit() == sequence::exit_reason::finished;
            if (reaches_state) {
                state_reachers.push_back(&seq);
                if (seq.has_pending_state()) {
                    state_writers.push_back(&seq);
                }
            }
            if (completed_count > 0) {
                auto& hashes = reaches_state ? new_hashes : spawned_hashes;
                auto& owners = reaches_state ? new_owners : spawned_owners;
                auto& line_ids = reaches_state ? new_lines : spawned_lines;
                for (std::size_t layer = 0; layer < n_layers; layer++) {
                    if (layout.layers[layer].is_recurrent) {
                        continue;
                    }
                    const layer_view& view = seq.views()[layer];
                    std::size_t first_new_page = view.pages.size() - completed_count;
                    hashes[layer].insert(hashes[layer].end(), view.pages.begin() + first_new_page, view.pages.end());
                    owners[layer].resize(hashes[layer].size(), seq.lease_id());
                    line_ids[layer].insert(
                        line_ids[layer].end(),
                        view.lines.begin() + first_new_page,
                        view.lines.begin() + view.pages.size()
                    );
                }
                if (reaches_state) {
                    page_writers.push_back(&seq);
                } else {
                    spawned_writers.push_back(seq.pending_page_writes());
                }
            }
        }
    }
    for (sequence* seq : state_writers) {
        for (std::size_t layer = 0; layer < n_layers; layer++) {
            if (layout.layers[layer].is_recurrent) {
                new_hashes[layer].push_back(*seq->head_digest());
                new_owners[layer].push_back(seq->lease_id());
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
        // kv_cache::sync holds the lines before the write-back below releases the ones that left a view.
        event_loop::current()->spawn(async_write_back(
            cache.sync(spawned_hashes, spawned_owners, spawned_lines, copy_streams),
            std::move(spawned_writers)
        ));
    }
    // A state names only pages the store has.
    for (sequence* seq : state_reachers) {
        co_await seq->pending_page_writes()->wait();
        if (seq->pending_page_writes()->any_failed()) {
            seq->exit(sequence::exit_reason::oom);
        }
    }
    std::expected<void, page_error> sync_result = co_await cache.sync(new_hashes, new_owners, new_lines, copy_streams);
    if (!sync_result) {
        for (sequence* seq : page_writers) {
            seq->exit(sequence::exit_reason::oom);
        }
        for (sequence* seq : state_writers) {
            seq->exit(sequence::exit_reason::oom);
        }
    } else {
        for (sequence* seq : state_writers) {
            if (seq->reason_of_exit() == sequence::exit_reason::oom) {
                continue;
            }
            seq->store_head_state();
            // The synced line keeps the stored state as a cached page; a new line takes its place.
            std::vector<std::size_t> counts(n_layers, 0);
            for (std::size_t layer = 0; layer < n_layers; layer++) {
                counts[layer] = layout.layers[layer].is_recurrent ? 1 : 0;
            }
            std::expected<std::vector<std::vector<std::uint32_t>>, page_error> replacements = cache.allocate(counts);
            if (!replacements) {
                seq->exit(sequence::exit_reason::oom);
                continue;
            }
            for (std::size_t layer = 0; layer < n_layers; layer++) {
                if (layout.layers[layer].is_recurrent) {
                    const std::uint32_t stored_line = seq->replace_head_state_line(
                        layer,
                        (*replacements)[layer].front()
                    );
                    cache.release(layer, std::span(&stored_line, 1));
                }
            }
        }
    }

    for (batch::group* g : b.groups()) {
        for (const batch::entry& e : g->entries) {
            sequence& seq = *e.seq;
            seq.finish_step();
            if (seq.has_exited()) {
                continue;
            }
            bool views_changed = release_passed_pages(seq);
            if (e.ends_cycle) {
                seq.sample_pages();
                views_changed = evict(seq) || views_changed;
            }
            if (views_changed) {
                update_views(seq);
            }
            if (e.ends_cycle) {
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
                    seq.exit(sequence::exit_reason::oom);
                    continue;
                }
                for (std::size_t layer = 0; layer < n_layers; layer++) {
                    const std::size_t line_count = seq.views()[layer].lines.size();
                    if (missing_counts[layer] > 0) {
                        first_stale_page = std::min(first_stale_page.value_or(line_count), line_count);
                    }
                    seq.add_lines(layer, (*allocation)[layer]);
                }
                record_device_memory(seq);
            }
            if (first_stale_page) {
                seq.upload_table(*first_stale_page, streams);
            }
        }
    }
}

task<void> executor::async_write_back(
    task<std::expected<void, page_error>> syncing,
    std::vector<std::shared_ptr<pending_writes>> writers
) {
    std::expected<void, page_error> written = co_await std::move(syncing);
    for (const std::shared_ptr<pending_writes>& writer : writers) {
        writer->finish(written.has_value());
    }
    store_writes.finish(written.has_value());
}

// The step that last used the evicted lines has completed, so they are released right away.
bool executor::evict(sequence& seq) {
    if (!seq.eviction_policy_in_pages().can_evict()) {
        return false;
    }
    bool evicted = false;
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        if (layout.layers[layer].mass_heads == 0) {
            continue;
        }
        std::vector<std::size_t> pages = seq.pages_to_evict(layer);
        if (!pages.empty()) {
            cache.release(layer, seq.remove_pages(layer, pages));
            evicted = true;
        }
    }
    return evicted;
}

task<void> executor::write_cycle_end(sequence& seq) {
    auto cycle_end_state = std::make_shared<const state>(seq.to_state());
    co_await store.write_state(seq.lease_id(), std::shared_ptr<const state>(cycle_end_state));
    seq.writer().append(cycle_end{.state = std::move(cycle_end_state)});
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

}  // namespace pulsarlm
