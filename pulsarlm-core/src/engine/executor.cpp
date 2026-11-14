#include "pulsarlm/engine/executor.hpp"
#include "pulsarlm/engine/sequence.hpp"

#include "pulsarlm/kernels/ops.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <utility>
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
        .page_end_tokens = to_device(host.page_end_tokens, device),
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
    // decode_count decode entries first, each forwarding its sequence's only pending token.
    std::vector<batch_entry> entries;
    std::size_t decode_count = 0;
    // Pinned host tensors.
    step_batch inputs;
    // Pinned host int64, one token per entry that samples, in entry order.
    at::Tensor sampled_tokens;
    // Pinned host fp32, defined for a group that measures mass: the page mass of each KV layer that measures it in
    // turn, laid out as the layer's row of the group's cu_view_pages, its heads innermost; per shard, its heads, then
    // joined over the shards.
    std::vector<at::Tensor> shard_decode_mass;
    std::vector<at::Tensor> shard_prefill_mass;
    at::Tensor decode_mass;
    at::Tensor prefill_mass;

    batch(std::span<const batch_entry> step_entries, std::size_t page_size);

    static group_batch make_group(std::span<const batch_entry> group, std::size_t page_size, bool prefill);
    static bool measures_mass(std::span<const batch_entry> group) noexcept;
    // The group's tables of lines, or of positions with positions, on shard's device, as group_batch lays them out.
    static at::Tensor
    page_tables(std::span<const batch_entry> group, std::size_t n_kv_layers, bool positions, std::size_t shard);
    // Queues on stream shard's copies of the inputs, its forward and the copies of its page mass to the host; shard 0
    // also queues the sampling and the copy of the sampled tokens to the host. mass_heads is the shard's part of the
    // executor's. Every shard's call may run at once, each on its own thread.
    void enqueue(
        model& shard_model,
        std::span<const kv> lines,
        const std::vector<std::size_t>& mass_heads,
        cudaStream_t stream,
        std::size_t shard
    );
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

    friend class executor;
};

executor::executor(
    std::vector<model*>&& shards,
    kv_cache cache,
    page_store& store,
    std::vector<cudaStream_t>&& streams,
    std::vector<event_loop*>&& worker_loops,
    digest root,
    std::vector<std::int64_t>&& stop_ids
)
: shards(std::move(shards)),
  cache(std::move(cache)),
  store(store),
  streams(std::move(streams)),
  worker_loops(std::move(worker_loops)),
  page_size(0),
  root(root),
  stop_ids(std::move(stop_ids)) {
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
            bool evictable = !layers[layer].sliding_window && !layer_recurrent;
            assert(!evictable || layers[layer].n_q_heads > 0);
            shard_mass_heads[shard].push_back(evictable ? layers[layer].n_readers * layers[layer].n_q_heads : 0);
            if (shard == 0) {
                sliding_windows.push_back(layers[layer].sliding_window);
                recurrent.push_back(layer_recurrent);
                mass_heads.push_back(0);
                if (page_size == 0 && !layer_recurrent) {
                    page_size = static_cast<std::size_t>(lines[0].back().k.size(1));
                }
            }
            assert(sliding_windows[layer] == layers[layer].sliding_window && recurrent[layer] == layer_recurrent);
            mass_heads[layer] += shard_mass_heads[shard].back();
        }
    }
    assert(page_size > 0);
}

executor::batch::batch(std::span<const batch_entry> step_entries, std::size_t page_size)
: entries(step_entries.begin(), step_entries.end()) {
    auto first_prefill = std::ranges::find_if(entries, [](const batch_entry& e) {
        return e.seq->pending_tokens.size() > 1;
    });
    decode_count = static_cast<std::size_t>(first_prefill - entries.begin());
    std::span<const batch_entry> entry_span(entries);
    if (decode_count > 0) {
        inputs.decode_group = make_group(entry_span.first(decode_count), page_size, false);
    }
    if (entries.size() > decode_count) {
        inputs.prefill_group = make_group(entry_span.subspan(decode_count), page_size, true);
    }
}

group_batch executor::batch::make_group(std::span<const batch_entry> group, std::size_t page_size, bool prefill) {
    std::size_t token_count = 0;
    for (const batch_entry& e : group) {
        token_count += e.token_count;
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
        for (std::size_t i = 0; i < group[s].token_count; i++) {
            tokens[first_row + i] = seq.pending_tokens[i];
            positions[first_row + i] = static_cast<std::int64_t>(first_position + i);
        }
        for (std::size_t layer = 0; layer < n_layers; layer++) {
            std::size_t start = seq.next_index(layer);
            key_count[layer * group.size() + s] = static_cast<std::int32_t>(start + group[s].token_count);
        }
        first_row += group[s].token_count;
        query_start[s + 1] = static_cast<std::int32_t>(first_row);
    }
    g.key_counts = key_counts;
    if (prefill) {
        g.cu_seqlens_q = query_starts;
    }
    g.page_end_tokens = pinned({n_seqs}, at::kInt);
    auto* page_end_tokens = g.page_end_tokens.mutable_data_ptr<std::int32_t>();
    for (std::size_t s = 0; s < group.size(); s++) {
        std::size_t start = group[s].seq->tail.size();
        std::size_t page_end = (start + group[s].token_count) / page_size * page_size;
        page_end_tokens[s] = static_cast<std::int32_t>(page_end > start ? page_end - start : 0);
    }
    if (measures_mass(group)) {
        g.cu_view_pages = pinned({n_kv_layers, n_seqs + 1}, at::kInt);
        g.query_weights = pinned({n_tokens}, at::kFloat);
        auto* cu_view_pages = g.cu_view_pages.mutable_data_ptr<std::int32_t>();
        for (std::size_t layer = 0; layer < n_layers; layer++) {
            std::int32_t* row = cu_view_pages + layer * (group.size() + 1);
            row[0] = 0;
            for (std::size_t s = 0; s < group.size(); s++) {
                row[s + 1] = row[s] + static_cast<std::int32_t>(group[s].seq->layers[layer].lines.size());
            }
        }
        std::span<float> weights(g.query_weights.mutable_data_ptr<float>(), token_count);
        std::size_t first = 0;
        for (const batch_entry& e : group) {
            pulsarlm::query_weights(weights.subspan(first, e.token_count), e.seq->eviction.attention_mass_decay);
            first += e.token_count;
        }
    }
    return g;
}

bool executor::batch::measures_mass(std::span<const batch_entry> group) noexcept {
    return std::ranges::any_of(group, [](const batch_entry& e) { return e.seq->eviction.can_evict(); });
}

at::Tensor executor::batch::page_tables(
    std::span<const batch_entry> group,
    std::size_t n_kv_layers,
    bool positions,
    std::size_t shard
) {
    std::vector<at::Tensor> tables;
    std::size_t width = 0;
    for (const batch_entry& e : group) {
        tables.push_back(positions ? e.seq->position_tables[shard] : e.seq->tables[shard]);
        width = std::max(width, e.seq->table_width());
    }
    return gather_page_tables_cuda(tables, static_cast<std::int64_t>(n_kv_layers), static_cast<std::int64_t>(width));
}

void executor::batch::enqueue(
    model& shard_model,
    std::span<const kv> lines,
    const std::vector<std::size_t>& mass_heads,
    cudaStream_t stream,
    std::size_t shard
) {
    at::Device device = lines.front().k.device();
    c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(stream, device.index()));
    step_batch on_device;
    std::span<batch_entry> entry_span(entries);
    at::Tensor decode_on_device, prefill_on_device;
    if (inputs.decode_group) {
        on_device.decode_group = to_device(*inputs.decode_group, device);
        on_device.decode_group->page_tables = page_tables(entry_span.first(decode_count), lines.size(), false, shard);
        on_device.decode_group->page_positions = page_tables(entry_span.first(decode_count), lines.size(), true, shard);
        if (inputs.decode_group->cu_view_pages.defined()) {
            decode_on_device = zero_page_mass(*on_device.decode_group, inputs.decode_group->cu_view_pages, mass_heads);
        }
    }
    if (inputs.prefill_group) {
        on_device.prefill_group = to_device(*inputs.prefill_group, device);
        on_device.prefill_group
            ->page_tables = page_tables(entry_span.subspan(decode_count), lines.size(), false, shard);
        on_device.prefill_group
            ->page_positions = page_tables(entry_span.subspan(decode_count), lines.size(), true, shard);
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
    step_logits logits = shard_model.forward(on_device, lines);
    if (decode_on_device.defined()) {
        shard_decode_mass[shard] = pinned(decode_on_device.sizes(), at::kFloat);
        shard_decode_mass[shard].copy_(decode_on_device, /*non_blocking=*/true);
    }
    if (prefill_on_device.defined()) {
        shard_prefill_mass[shard] = pinned(prefill_on_device.sizes(), at::kFloat);
        shard_prefill_mass[shard].copy_(prefill_on_device, /*non_blocking=*/true);
    }
    if (shard != 0) {
        return;
    }

    std::vector<at::Tensor> rows;
    std::vector<sampling_params> params;
    std::vector<sampling_draw> draws;
    for (std::size_t i = 0; i < entries.size(); i++) {
        const batch_entry& e = entries[i];
        if (e.samples) {
            const bool decodes = i < decode_count;
            const auto row = static_cast<std::int64_t>(decodes ? i : i - decode_count);
            rows.push_back((decodes ? *logits.decode_logits : *logits.prefill_logits).select(0, row));
            params.push_back(e.seq->sampling);
            draws.push_back(sampling_draw{.seed = e.seq->seed, .count = e.seq->draw_count++});
        }
    }
    if (rows.empty()) {
        return;
    }
    at::Tensor tokens = sample(at::stack(rows), params, draws);
    sampled_tokens = pinned({static_cast<std::int64_t>(rows.size())}, at::kLong);
    sampled_tokens.copy_(tokens, /*non_blocking=*/true);
}

std::vector<executor::batch::layer_step_mass>
executor::batch::step_mass(std::size_t index, const std::vector<std::size_t>& mass_heads) const {
    const bool decodes = index < decode_count;
    const group_batch& group = decodes ? *inputs.decode_group : *inputs.prefill_group;
    const float* mass = (decodes ? decode_mass : prefill_mass).const_data_ptr<float>();
    std::size_t s = decodes ? index : index - decode_count;
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
        auto first_page = static_cast<std::size_t>(row[s]);
        auto pages = static_cast<std::size_t>(row[s + 1] - row[s]);
        auto layer_pages = static_cast<std::size_t>(row[row_size - 1]);
        by_layer[layer] = layer_step_mass{
            .mass_sum = std::span(mass + first + first_page * heads, pages * heads),
            .weight_sum = std::span(mass + first + layer_pages * heads + first_page, pages),
        };
        first += layer_pages * (heads + 1);
    }
    return by_layer;
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
    prefill_mass = join(shard_prefill_mass, inputs.prefill_group);
}

task<std::expected<std::unique_ptr<sequence>, request_error>> executor::start(request&& r, result_writer& writer) {
    assert(!r.tokens.empty());
    assert(r.eviction.n_working >= 1 && r.eviction.cycle_tokens >= 1);
    assert(r.eviction.attention_mass_decay > 0.0 && r.eviction.attention_mass_decay <= 1.0);
    assert(r.eviction.eviction_threshold >= 0.0 && r.eviction.eviction_threshold <= 1.0);
    digest chain_root = r.initial_digest.value_or(root);
    std::vector<digest> digests = make_page_digests(chain_root, r.tokens, page_size);
    std::expected<std::pair<std::shared_ptr<const state>, state::lease_id_type>, page_error>
        lookup = co_await store.read_state(r.tokens, digests);
    if (!lookup) {
        co_return std::unexpected(request_error::miss);
    }
    auto& [initial_state, lease_id] = *lookup;
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
    seq->start_cycle();
    // take_lines holds the resident pages before it first suspends, so they are the ones counted here.
    seq->device_cached_count = device_cached_tokens(*seq);
    std::expected<void, page_error> line_allocation = co_await take_lines(*seq, seq->cycle_lines());
    if (!line_allocation) {
        seq->lease.release();
        co_return std::unexpected(
            line_allocation.error() == page_error::miss ? request_error::miss : request_error::oom
        );
    }
    seq->upload_table(0, streams);
    co_return std::move(seq);
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

task<void> executor::step(std::span<const batch_entry> entries) {
    batch b(entries, page_size);
    co_await launch(b);
    co_await write_back(b);
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
            counts[layer] = (*page_lines)[layer].empty() ? 2 : 1;
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
        if (recurrent[layer]) {
            if (states[layer].empty()) {
                for (std::size_t shard = 0; shard < shards.size(); shard++) {
                    c10::cuda::CUDAStreamGuard guard(
                        c10::cuda::getStreamFromExternal(streams[shard], lines[shard][layer].k.device().index())
                    );
                    lines[shard][layer].k.select(0, (*allocation)[layer].front()).zero_();
                }
            }
            view.append_lines(states[layer]);
        } else {
            view.append_lines((*page_lines)[layer]);
        }
        view.append_lines((*allocation)[layer]);
    }
    // The copies of the states at head read the fetched lines.
    co_await synchronize_shards();
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        if (recurrent[layer]) {
            cache.release(layer, (*page_lines)[layer]);
        }
    }
    co_return std::expected<void, page_error>{};
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

// Shard 0 runs on the executor's thread while the others run on their own, so every shard's kernels launch at once.
task<void> executor::launch(batch& b) {
    b.shard_decode_mass.assign(shards.size(), at::Tensor());
    b.shard_prefill_mass.assign(shards.size(), at::Tensor());
    std::vector<external_task<>> other_shards;
    for (std::size_t shard = 1; shard < shards.size(); shard++) {
        other_shards.push_back(run_on(*worker_loops[shard - 1], forward_on_shard(b, shard)));
    }
    b.enqueue(*shards[0], lines[0], shard_mass_heads[0], streams[0], 0);
    co_await synchronize(streams[0]);
    for (external_task<>& other : other_shards) {
        co_await std::move(other);
    }
    b.join_mass(shard_mass_heads, mass_heads);
}

task<void> executor::forward_on_shard(batch& b, std::size_t shard) {
    {
        c10::cuda::CUDAGuard device_guard(lines[shard].front().k.device());
        b.enqueue(*shards[shard], lines[shard], shard_mass_heads[shard], streams[shard], shard);
    }
    co_await synchronize(streams[shard]);
}

task<void> executor::synchronize_shards() {
    for (cudaStream_t stream : streams) {
        co_await synchronize(stream);
    }
}

// A sequence ends with oom when the sync of its new pages fails, as a state names only pages the store has. The step
// that last read the lines of pages that left a view has completed, so they are released right after the sync. A
// recurrent layer's state at head reaches the store at the resume points, cycle ends and the end, that find it pending.
task<void> executor::write_back(batch& b) {
    std::size_t n_layers = cache.layer_count();
    bool has_recurrent = std::ranges::contains(recurrent, true);
    std::vector<std::vector<digest>> new_hashes(n_layers);
    std::vector<std::vector<state::lease_id_type>> new_owners(n_layers);
    std::vector<std::vector<std::uint32_t>> new_lines(n_layers);
    std::vector<sequence*> page_writers;
    std::vector<sequence*> state_writers;
    const std::int64_t* next_sampled = b.sampled_tokens.defined() ? b.sampled_tokens.const_data_ptr<std::int64_t>()
                                                                  : nullptr;
    for (std::size_t i = 0; i < b.entries.size(); i++) {
        const batch_entry& e = b.entries[i];
        sequence& seq = *e.seq;
        if (seq.eviction.can_evict()) {
            std::vector<batch::layer_step_mass> step_mass = b.step_mass(i, mass_heads);
            for (std::size_t layer = 0; layer < n_layers; layer++) {
                if (mass_heads[layer] > 0) {
                    layer_view& view = seq.layers[layer];
                    add_step_mass(
                        view.mass_sum,
                        view.weight_sum,
                        step_mass[layer].mass_sum,
                        step_mass[layer].weight_sum,
                        e.token_count,
                        seq.eviction.attention_mass_decay
                    );
                }
            }
        }
        std::size_t completed_count = seq.advance(e.token_count);
        if (seq.eviction.can_evict() && completed_count > 0) {
            add_sampled_pages(seq, completed_count);
        }
        if (completed_count > 0) {
            for (std::size_t layer = 0; layer < n_layers; layer++) {
                if (recurrent[layer]) {
                    continue;
                }
                const layer_view& view = seq.layers[layer];
                std::size_t first_new_page = view.pages.size() - completed_count;
                new_hashes[layer]
                    .insert(new_hashes[layer].end(), view.pages.begin() + first_new_page, view.pages.end());
                new_owners[layer].resize(new_hashes[layer].size(), seq.lease.id());
                new_lines[layer].insert(
                    new_lines[layer].end(),
                    view.lines.begin() + first_new_page,
                    view.lines.begin() + view.pages.size()
                );
            }
            seq.state_pending = has_recurrent;
            page_writers.push_back(&seq);
        }
        if (e.samples) {
            std::int64_t token = *next_sampled++;
            seq.writer.append(token);
            seq.generated_count++;
            seq.pending_tokens.push_back(token);
            if (std::ranges::contains(stop_ids, token) || !seq.may_sample()) {
                seq.end_reason = sequence::exit_reason::finished;
            }
        } else if (seq.pending_tokens.empty()) {
            seq.end_reason = sequence::exit_reason::finished;
        }
        if (seq.state_pending && (e.ends_cycle || seq.end_reason == sequence::exit_reason::finished)) {
            for (std::size_t layer = 0; layer < n_layers; layer++) {
                if (recurrent[layer]) {
                    new_hashes[layer].push_back(*seq.head);
                    new_owners[layer].push_back(seq.lease.id());
                    new_lines[layer].push_back(seq.layers[layer].lines[1]);
                }
            }
            state_writers.push_back(&seq);
        }
    }

    std::expected<void, page_error> sync_result = co_await cache.sync(new_hashes, new_owners, new_lines, streams);
    if (!sync_result) {
        for (sequence* seq : page_writers) {
            seq->end_reason = sequence::exit_reason::oom;
        }
        for (sequence* seq : state_writers) {
            seq->end_reason = sequence::exit_reason::oom;
        }
    } else {
        for (sequence* seq : state_writers) {
            seq->state_pending = false;
            for (std::size_t layer = 0; layer < n_layers; layer++) {
                if (recurrent[layer]) {
                    std::vector<std::uint32_t>& state_lines = seq->layers[layer].lines;
                    cache.release(layer, std::span(state_lines).last(1));
                    state_lines.pop_back();
                }
            }
        }
    }

    for (const batch_entry& e : b.entries) {
        sequence& seq = *e.seq;
        if (seq.end_reason) {
            continue;
        }
        bool views_changed = release_passed_pages(seq);
        if (e.ends_cycle) {
            views_changed = evict(seq) || views_changed;
            co_await write_cycle_end(seq);
            seq.start_cycle();
        }
        std::optional<std::size_t> first_stale_page;
        if (views_changed) {
            seq.update_held();
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
        }
        if (first_stale_page) {
            seq.upload_table(*first_stale_page, streams);
        }
    }
}

// The pages after a full page fill at least cycle_tokens tokens once cycle_pages full pages follow it in the view, while
// eviction drops only pages older than those, as it does when n_working is at least cycle_tokens.
void executor::add_sampled_pages(sequence& seq, std::size_t completed_count) {
    std::size_t n_sink = seq.eviction.n_sink;
    std::size_t cycle_pages = ceil_div(seq.eviction.cycle_tokens, page_size);
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        std::size_t heads = mass_heads[layer];
        if (heads == 0) {
            continue;
        }
        layer_view& view = seq.layers[layer];
        std::size_t after = view.pages.size();
        std::size_t before = after - completed_count;
        std::size_t first = std::max(n_sink, before > cycle_pages ? before - cycle_pages : 0);
        std::size_t last = after > cycle_pages ? after - cycle_pages : 0;
        for (std::size_t page = first; page < last; page++) {
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

}  // namespace pulsarlm
