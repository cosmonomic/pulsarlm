#include "pulsarlm/scheduler.hpp"

#include "pulsarlm/ops.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
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

GroupBatch to_device(const GroupBatch& host, at::Device device) {
    return GroupBatch{
        .tokens = to_device(host.tokens, device),
        .pos = to_device(host.pos, device),
        .page_end_tokens = to_device(host.page_end_tokens, device),
        .cu_seqlens_q = to_device(host.cu_seqlens_q, device),
        .key_counts = to_device(host.key_counts, device),
        .rope_layout = to_device(host.rope_layout, device),
        .cu_view_pages = to_device(host.cu_view_pages, device),
        .mass_weights = to_device(host.mass_weights, device),
    };
}

// Sets on_device's page mass for the KV layers that measure it, those with mass_heads above 0, to zeroed parts of one
// tensor, which it returns. host_cu_view_pages is on the host.
at::Tensor zero_page_mass(
    GroupBatch& on_device,
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
        total += layer_pages(layer) * static_cast<std::int64_t>(mass_heads[layer]);
    }
    at::Tensor mass = at::zeros({total}, on_device.tokens.options().dtype(at::kFloat));
    on_device.page_mass.resize(mass_heads.size());
    std::int64_t first = 0;
    for (std::size_t layer = 0; layer < mass_heads.size(); layer++) {
        if (mass_heads[layer] > 0) {
            auto heads = static_cast<std::int64_t>(mass_heads[layer]);
            std::int64_t pages = layer_pages(layer);
            on_device.page_mass[layer] = mass.narrow(0, first, pages * heads).view({pages, heads});
            first += pages * heads;
        }
    }
    return mass;
}

}  // namespace

// One KV layer's view of a sequence. A layer with pages has one line per view page: its full pages, then the pages of
// tail and pending. A layer that measures mass keeps mass_heads densities per line, each a query head's EMA of the
// shares the page received, as of the last step that measured it, 0 for a page never measured. A recurrent layer has
// no pages: its lines are the one its state is updated in, then the one a step writes the state at its newest page end
// into.
class scheduler::layer_view {
  public:
    std::vector<digest> pages;
    std::vector<std::uint32_t> lines;
    std::size_t mass_heads = 0;
    std::vector<float> density;

    // Drops the full pages at the ascending view indices and returns their lines.
    std::vector<std::uint32_t> remove(std::span<const std::size_t> indices);
    void append_lines(std::span<const std::uint32_t> added);
};

// A request being served. Its tokens move from pending to tail as steps forward them, and the pages the tail fills
// join every layer's view, the last of them head.
class scheduler::sequence {
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
    page_eviction_policy eviction;
    // covered_count when the eviction cycle began.
    std::size_t cycle_start = 0;
    // The request's tokens forwarded so far, which the state covers: the initial state's tail, then the request's
    // tokens, then sampled tokens.
    std::size_t covered_count = 0;

    sequence(
        request&& r,
        const std::vector<bool>& recurrent,
        const std::vector<std::size_t>& mass_heads,
        result_writer& writer,
        at::Device device,
        std::size_t page_size
    );

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
class scheduler::batch {
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
    // Pinned host fp32, defined for a group that measures mass: the page mass of each KV layer that measures it in
    // turn, laid out as the layer's row of the group's cu_view_pages, its heads innermost.
    at::Tensor decode_mass;
    at::Tensor prefill_mass;

    batch(std::span<sequence* const> running, std::size_t page_size, std::size_t prefill_tokens_per_step);

    static GroupBatch make_group(std::span<const entry> group, std::size_t page_size, bool prefill);
    static bool measures_mass(std::span<const entry> group) noexcept;
    static at::Tensor page_tables(std::span<const entry> group, std::size_t n_kv_layers);
    // Queues on stream the inputs' copies, the forward, the copies of the page mass to the host, the sampling and the
    // copy of the sampled tokens to the host. mass_heads is the scheduler's.
    void
    enqueue(Model& model, std::span<const kv> lines, const std::vector<std::size_t>& mass_heads, cudaStream_t stream);
    // Per KV layer, the mass the step gave each of entry index's view pages, mass_heads[layer] values per page; empty
    // for a layer that measures none. The entry's group measures mass.
    std::vector<std::span<const float>> step_mass(std::size_t index, const std::vector<std::size_t>& mass_heads) const;

    friend class scheduler;
};

std::vector<std::uint32_t> scheduler::layer_view::remove(std::span<const std::size_t> indices) {
    std::vector<std::uint32_t> removed;
    std::size_t kept = 0;
    std::size_t next_removed = 0;
    for (std::size_t page = 0; page < pages.size(); page++) {
        if (next_removed < indices.size() && indices[next_removed] == page) {
            removed.push_back(lines[page]);
            next_removed++;
            continue;
        }
        pages[kept] = pages[page];
        lines[kept] = lines[page];
        std::copy_n(density.begin() + page * mass_heads, mass_heads, density.begin() + kept * mass_heads);
        kept++;
    }
    assert(next_removed == indices.size());
    pages.erase(pages.begin() + kept, pages.end());
    lines.erase(lines.begin() + kept, lines.begin() + kept + indices.size());
    density.erase(density.begin() + kept * mass_heads, density.begin() + (kept + indices.size()) * mass_heads);
    return removed;
}

void scheduler::layer_view::append_lines(std::span<const std::uint32_t> added) {
    lines.insert(lines.end(), added.begin(), added.end());
    density.resize(lines.size() * mass_heads, 0.0f);
}

scheduler::sequence::sequence(
    request&& r,
    const std::vector<bool>& recurrent,
    const std::vector<std::size_t>& mass_heads,
    result_writer& writer,
    at::Device device,
    std::size_t page_size
)
: writer(writer),
  id(r.id),
  sampling(r.sampling),
  seed(sampling_seed(r.sampling.seed)),
  max_tokens(r.max_tokens),
  head(r.initial_state.head),
  pending(std::move(r.initial_state.tail)),
  recurrent(recurrent),
  layers(recurrent.size()),
  table(
      at::zeros({static_cast<std::int64_t>(recurrent.size()), 0}, at::TensorOptions().dtype(at::kInt).device(device))
  ),
  eviction(to_pages(r.eviction, page_size)) {
    pending.insert(pending.end(), r.tokens.begin(), r.tokens.end());
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        layers[layer].mass_heads = mass_heads[layer];
    }
    for (std::size_t layer = 0; layer < r.initial_state.layers.size(); layer++) {
        if (recurrent[layer]) {
            continue;
        }
        layer_view& view = layers[layer];
        for (const active_page& page : r.initial_state.layers[layer].pages) {
            view.pages.push_back(page.hash);
            if (page.density.empty()) {
                view.density.resize(view.density.size() + view.mass_heads, 0.0f);
            } else {
                view.density.insert(view.density.end(), page.density.begin(), page.density.end());
            }
        }
    }
}

std::size_t scheduler::sequence::next_index(std::size_t layer, std::size_t page_size) const noexcept {
    return layers[layer].pages.size() * page_size + tail.size();
}

bool scheduler::sequence::may_sample() const noexcept {
    return !max_tokens || generated < *max_tokens;
}

std::size_t scheduler::sequence::advance(std::size_t count, std::size_t page_size, const digest& root) {
    assert(count <= pending.size());
    tail.insert(tail.end(), pending.begin(), pending.begin() + count);
    pending.erase(pending.begin(), pending.begin() + count);
    covered_count += count;
    std::size_t completed_tokens = tail.size() - tail.size() % page_size;
    if (completed_tokens == 0) {
        return 0;
    }
    std::vector<digest> completed = make_page_digests(
        head.value_or(root),
        std::span(tail).first(completed_tokens),
        page_size
    );
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        if (!recurrent[layer]) {
            layers[layer].pages.insert(layers[layer].pages.end(), completed.begin(), completed.end());
        }
    }
    head = completed.back();
    tail.erase(tail.begin(), tail.begin() + completed_tokens);
    return completed.size();
}

void scheduler::sequence::start_cycle() noexcept {
    cycle_start = covered_count;
}

bool scheduler::sequence::ends_cycle(std::size_t token_count) const noexcept {
    return covered_count + token_count - cycle_start >= eviction.cycle_tokens;
}

std::size_t scheduler::sequence::cycle_tokens_left() const noexcept {
    std::size_t left = eviction.cycle_tokens - (covered_count - cycle_start);
    if (max_tokens) {
        // Every token sampled from here is forwarded but the last.
        std::size_t samples_left = *max_tokens - generated;
        left = std::min(left, pending.size() + (samples_left > 0 ? samples_left - 1 : 0));
    }
    return left;
}

std::size_t scheduler::sequence::table_width() const noexcept {
    std::size_t width = 0;
    for (const layer_view& view : layers) {
        width = std::max(width, view.lines.size());
    }
    return width;
}

void scheduler::sequence::upload_table(std::size_t first_page, cudaStream_t stream) {
    auto n_layers = static_cast<std::int64_t>(layers.size());
    auto page_count = static_cast<std::int64_t>(table_width());
    if (table.size(1) < page_count) {
        std::int64_t capacity = std::max<std::int64_t>(page_count, 2 * table.size(1));
        table = at::empty({n_layers, capacity}, table.options());
        first_page = 0;
    }
    auto first = static_cast<std::int64_t>(first_page);
    if (first == page_count) {
        return;
    }
    at::Tensor host = pinned({n_layers, page_count - first}, at::kInt);
    auto* entries = host.mutable_data_ptr<std::int32_t>();
    for (const layer_view& view : layers) {
        for (std::int64_t page = first; page < page_count; page++) {
            auto index = static_cast<std::size_t>(page);
            *entries++ = index < view.lines.size() ? static_cast<std::int32_t>(view.lines[index]) : 0;
        }
    }
    c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(stream, table.device().index()));
    table.narrow(1, first, page_count - first).copy_(host, /*non_blocking=*/true);
}

std::size_t scheduler::sequence::cycle_lines(std::size_t page_size) const noexcept {
    assert(tail.empty());
    return ceil_div(cycle_tokens_left(), page_size);
}

std::vector<std::size_t> scheduler::sequence::missing_lines(std::size_t page_size) const {
    std::size_t left = cycle_tokens_left();
    std::vector<std::size_t> missing;
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        std::size_t held = layers[layer].lines.size();
        std::size_t needed = recurrent[layer] ? 2 : ceil_div(next_index(layer, page_size) + left, page_size);
        missing.push_back(needed > held ? needed - held : 0);
    }
    return missing;
}

scheduler::batch::batch(
    std::span<sequence* const> running,
    std::size_t page_size,
    std::size_t prefill_tokens_per_step
) {
    for (sequence* seq : running) {
        assert(!seq->pending.empty());
        if (seq->pending.size() == 1) {
            entries.push_back(
                entry{
                    .seq = seq,
                    .token_count = 1,
                    .samples = seq->may_sample(),
                    .ends_cycle = seq->ends_cycle(1),
                }
            );
        }
    }
    decode_count = entries.size();
    std::size_t budget = prefill_tokens_per_step;
    for (sequence* seq : running) {
        if (budget == 0) {
            break;
        }
        if (seq->pending.size() > 1) {
            std::size_t count = std::min({seq->pending.size(), budget, seq->cycle_tokens_left()});
            budget -= count;
            entries.push_back(
                entry{
                    .seq = seq,
                    .token_count = count,
                    .samples = count == seq->pending.size() && seq->may_sample(),
                    .ends_cycle = seq->ends_cycle(count),
                }
            );
        }
    }
    std::span<const entry> all(entries);
    if (decode_count > 0) {
        inputs.decode = make_group(all.first(decode_count), page_size, false);
    }
    if (entries.size() > decode_count) {
        inputs.prefill = make_group(all.subspan(decode_count), page_size, true);
    }
}

GroupBatch scheduler::batch::make_group(std::span<const entry> group, std::size_t page_size, bool prefill) {
    std::size_t token_count = 0;
    for (const entry& e : group) {
        token_count += e.token_count;
    }
    std::size_t n_layers = group.front().seq->layers.size();
    auto n_tokens = static_cast<std::int64_t>(token_count);
    auto n_seqs = static_cast<std::int64_t>(group.size());
    auto n_kv_layers = static_cast<std::int64_t>(n_layers);
    GroupBatch g{
        .tokens = pinned({n_tokens}, at::kLong),
        .pos = pinned({n_kv_layers, n_tokens}, at::kLong),
        .rope_layout = pinned({n_seqs, 3}, at::kInt),
    };
    at::Tensor key_counts = pinned({n_kv_layers, n_seqs}, at::kInt);
    at::Tensor query_starts = pinned({n_seqs + 1}, at::kInt);

    auto* tokens = g.tokens.mutable_data_ptr<std::int64_t>();
    auto* positions = g.pos.mutable_data_ptr<std::int64_t>();
    auto* rope_layout = g.rope_layout.mutable_data_ptr<std::int32_t>();
    auto* key_count = key_counts.mutable_data_ptr<std::int32_t>();
    auto* query_start = query_starts.mutable_data_ptr<std::int32_t>();
    std::size_t first_row = 0;
    query_start[0] = 0;
    for (std::size_t s = 0; s < group.size(); s++) {
        const sequence& seq = *group[s].seq;
        for (std::size_t i = 0; i < group[s].token_count; i++) {
            tokens[first_row + i] = seq.pending[i];
        }
        for (std::size_t layer = 0; layer < n_layers; layer++) {
            std::size_t start = seq.next_index(layer, page_size);
            for (std::size_t i = 0; i < group[s].token_count; i++) {
                positions[layer * token_count + first_row + i] = static_cast<std::int64_t>(start + i);
            }
            key_count[layer * group.size() + s] = static_cast<std::int32_t>(start + group[s].token_count);
        }
        first_row += group[s].token_count;
        query_start[s + 1] = static_cast<std::int32_t>(first_row);
        rope_layout[3 * s] = 0;
        rope_layout[3 * s + 1] = 0;
        rope_layout[3 * s + 2] = -1;
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
        g.mass_weights = pinned({n_tokens}, at::kFloat);
        auto* cu_view_pages = g.cu_view_pages.mutable_data_ptr<std::int32_t>();
        for (std::size_t layer = 0; layer < n_layers; layer++) {
            std::int32_t* row = cu_view_pages + layer * (group.size() + 1);
            row[0] = 0;
            for (std::size_t s = 0; s < group.size(); s++) {
                row[s + 1] = row[s] + static_cast<std::int32_t>(group[s].seq->layers[layer].lines.size());
            }
        }
        std::span<float> weights(g.mass_weights.mutable_data_ptr<float>(), token_count);
        std::size_t first = 0;
        for (const entry& e : group) {
            pulsarlm::mass_weights(
                weights.subspan(first, e.token_count),
                e.seq->tail.size(),
                page_size,
                e.seq->eviction.attention_mass_decay
            );
            first += e.token_count;
        }
    }
    return g;
}

bool scheduler::batch::measures_mass(std::span<const entry> group) noexcept {
    return std::ranges::any_of(group, [](const entry& e) { return e.seq->eviction.can_evict(); });
}

at::Tensor scheduler::batch::page_tables(std::span<const entry> group, std::size_t n_kv_layers) {
    std::vector<at::Tensor> tables;
    std::size_t width = 0;
    for (const entry& e : group) {
        tables.push_back(e.seq->table);
        width = std::max(width, e.seq->table_width());
    }
    return gather_page_tables_cuda(tables, static_cast<std::int64_t>(n_kv_layers), static_cast<std::int64_t>(width));
}

void scheduler::batch::enqueue(
    Model& model,
    std::span<const kv> lines,
    const std::vector<std::size_t>& mass_heads,
    cudaStream_t stream
) {
    at::Device device = lines.front().k.device();
    c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(stream, device.index()));
    StepBatch on_device;
    std::span<entry> all(entries);
    at::Tensor decode_on_device, prefill_on_device;
    if (inputs.decode) {
        on_device.decode = to_device(*inputs.decode, device);
        on_device.decode->page_tables = page_tables(all.first(decode_count), lines.size());
        if (inputs.decode->cu_view_pages.defined()) {
            decode_on_device = zero_page_mass(*on_device.decode, inputs.decode->cu_view_pages, mass_heads);
        }
    }
    if (inputs.prefill) {
        on_device.prefill = to_device(*inputs.prefill, device);
        on_device.prefill->page_tables = page_tables(all.subspan(decode_count), lines.size());
        const at::Tensor& cu_seqlens_q = on_device.prefill->cu_seqlens_q;
        on_device.prefill->logit_rows = cu_seqlens_q.narrow(0, 1, cu_seqlens_q.size(0) - 1).to(at::kLong) - 1;
        if (inputs.prefill->cu_view_pages.defined()) {
            prefill_on_device = zero_page_mass(*on_device.prefill, inputs.prefill->cu_view_pages, mass_heads);
        }
    }
    StepLogits logits = model.forward(on_device, lines);
    if (decode_on_device.defined()) {
        decode_mass = pinned(decode_on_device.sizes(), at::kFloat);
        decode_mass.copy_(decode_on_device, /*non_blocking=*/true);
    }
    if (prefill_on_device.defined()) {
        prefill_mass = pinned(prefill_on_device.sizes(), at::kFloat);
        prefill_mass.copy_(prefill_on_device, /*non_blocking=*/true);
    }

    std::vector<at::Tensor> rows;
    std::vector<sampling_params> params;
    std::vector<sampling_draw> draws;
    for (std::size_t i = 0; i < entries.size(); i++) {
        const entry& e = entries[i];
        if (e.samples) {
            const bool decodes = i < decode_count;
            const auto row = static_cast<std::int64_t>(decodes ? i : i - decode_count);
            rows.push_back((decodes ? *logits.decode : *logits.prefill).select(0, row));
            params.push_back(e.seq->sampling);
            draws.push_back(sampling_draw{.seed = e.seq->seed, .count = e.seq->draw_count++});
        }
    }
    if (rows.empty()) {
        return;
    }
    at::Tensor tokens = sample(at::stack(rows), params, draws);
    sampled = pinned({static_cast<std::int64_t>(rows.size())}, at::kLong);
    sampled.copy_(tokens, /*non_blocking=*/true);
}

std::vector<std::span<const float>>
scheduler::batch::step_mass(std::size_t index, const std::vector<std::size_t>& mass_heads) const {
    const bool decodes = index < decode_count;
    const GroupBatch& group = decodes ? *inputs.decode : *inputs.prefill;
    const float* mass = (decodes ? decode_mass : prefill_mass).const_data_ptr<float>();
    std::size_t s = decodes ? index : index - decode_count;
    auto row_size = static_cast<std::size_t>(group.cu_view_pages.size(1));
    const auto* cu_view_pages = group.cu_view_pages.const_data_ptr<std::int32_t>();
    std::vector<std::span<const float>> by_layer(mass_heads.size());
    std::size_t first = 0;
    for (std::size_t layer = 0; layer < mass_heads.size(); layer++) {
        std::size_t heads = mass_heads[layer];
        if (heads == 0) {
            continue;
        }
        const std::int32_t* row = cu_view_pages + layer * row_size;
        by_layer[layer] = std::span(
            mass + first + row[s] * heads,
            static_cast<std::size_t>(row[s + 1] - row[s]) * heads
        );
        first += static_cast<std::size_t>(row[row_size - 1]) * heads;
    }
    return by_layer;
}

scheduler::scheduler(
    Model& model,
    kv_cache cache,
    cudaStream_t stream,
    digest root,
    std::vector<std::int64_t>&& stop_ids,
    std::size_t prefill_tokens_per_step,
    std::optional<std::size_t> max_context
)
: model(model),
  cache(std::move(cache)),
  stream(stream),
  page_size(0),
  root(root),
  stop_ids(std::move(stop_ids)),
  prefill_tokens_per_step(prefill_tokens_per_step) {
    assert(prefill_tokens_per_step > 0);
    std::vector<kv_layer> layers = model.kv_layers();
    assert(layers.size() == this->cache.layer_count());
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        lines.push_back(this->cache.tensors(layer));
        sliding_windows.push_back(layers[layer].sliding_window);
        recurrent.push_back(layers[layer].state.has_value());
        bool evictable = !sliding_windows.back() && !recurrent.back();
        assert(!evictable || layers[layer].n_q_heads > 0);
        mass_heads.push_back(evictable ? layers[layer].n_q_heads : 0);
        if (page_size == 0 && !recurrent.back()) {
            page_size = static_cast<std::size_t>(lines.back().k.size(1));
        }
    }
    assert(page_size > 0);
    if (max_context) {
        max_context_pages = std::max<std::size_t>(*max_context / page_size, 1);
    }
}

event& scheduler::next(std::optional<event>& waited) {
    if (!waited) {
        waited.emplace();
    }
    return *waited;
}

// The awaiters a set event scheduled no longer refer to it, so it is destroyed right away.
void scheduler::wake(std::optional<event>& waited) noexcept {
    if (waited) {
        waited->set();
        waited.reset();
    }
}

std::size_t scheduler::passed_pages(std::size_t layer, std::size_t query) const noexcept {
    const std::optional<std::size_t>& window = sliding_windows[layer];
    if (!window || query + 1 <= *window) {
        return 0;
    }
    return (query + 1 - *window) / page_size;
}

task<void> scheduler::submit(request r, result_writer writer) {
    assert(r.initial_state.tail.size() < page_size);
    assert(!r.initial_state.tail.empty() || !r.tokens.empty());
    assert(r.eviction.n_working >= 1 && r.eviction.cycle_tokens >= 1);
    assert(r.eviction.attention_mass_decay > 0.0 && r.eviction.attention_mass_decay <= 1.0);
    assert(r.eviction.eviction_target > 0.0 && r.eviction.eviction_target <= 1.0);
    assert(r.eviction.eviction_target == 1.0 || max_context_pages);
    sequence seq(std::move(r), recurrent, mass_heads, writer, lines.front().k.device(), page_size);
    seq.start_cycle();
    if (writer.cancelled()) {
        writer.end(std::unexpected(request_error::cancelled));
        co_return;
    }
    std::expected<void, page_error> taken = co_await take_lines(seq, seq.cycle_lines(page_size));
    if (!taken) {
        writer.end(std::unexpected(taken.error() == page_error::miss ? request_error::miss : request_error::oom));
        co_return;
    }
    seq.upload_table(0, stream);

    seq.exit_event.emplace();
    running.push_back(&seq);
    wake(idle);
    co_await *seq.exit_event;
    release_lines(seq);

    switch (*seq.exited) {
    case exit_reason::finished:
        writer.end(reached(seq));
        co_return;
    case exit_reason::cancelled:
        writer.end(std::unexpected(request_error::cancelled));
        co_return;
    case exit_reason::oom:
        writer.end(std::unexpected(request_error::oom));
        co_return;
    }
}

task<std::expected<void, page_error>> scheduler::take_lines(sequence& seq, std::size_t line_count) {
    std::vector<std::span<const digest>> held(cache.layer_count());
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        if (!recurrent[layer]) {
            held[layer] = seq.layers[layer].pages;
        } else if (seq.head) {
            held[layer] = std::span(&*seq.head, 1);
        }
    }
    // A get failing with miss releases the lines it held while it fetched.
    std::expected<std::vector<std::vector<std::uint32_t>>, page_error> got = co_await cache.get(held, stream);
    if (!got) {
        co_return std::unexpected(got.error());
    }
    std::vector<std::size_t> counts(cache.layer_count(), line_count);
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        if (recurrent[layer]) {
            counts[layer] = (*got)[layer].empty() ? 2 : 1;
        }
    }
    std::expected<std::vector<std::vector<std::uint32_t>>, page_error> allocated = cache.allocate(counts);
    std::vector<std::vector<std::uint32_t>> states(cache.layer_count());
    bool copied = allocated.has_value();
    for (std::size_t layer = 0; layer < cache.layer_count() && copied; layer++) {
        if (recurrent[layer] && !(*got)[layer].empty()) {
            std::expected<std::vector<std::uint32_t>, page_error>
                copy = cache.clone(layer, (*got)[layer].front(), 1, stream);
            copied = copy.has_value();
            if (copied) {
                states[layer] = std::move(*copy);
            }
        }
    }
    if (!copied) {
        for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
            cache.release(layer, (*got)[layer]);
            cache.release(layer, states[layer]);
            if (allocated) {
                cache.release(layer, (*allocated)[layer]);
            }
        }
        co_return std::unexpected(page_error::oom);
    }
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        layer_view& view = seq.layers[layer];
        if (recurrent[layer]) {
            if (states[layer].empty()) {
                c10::cuda::CUDAStreamGuard guard(
                    c10::cuda::getStreamFromExternal(stream, lines[layer].k.device().index())
                );
                lines[layer].k.select(0, (*allocated)[layer].front()).zero_();
            }
            view.append_lines(states[layer]);
        } else {
            view.append_lines((*got)[layer]);
        }
        view.append_lines((*allocated)[layer]);
    }
    // The copies of the states at head read the fetched lines.
    co_await synchronize(stream);
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        if (recurrent[layer]) {
            cache.release(layer, (*got)[layer]);
        }
    }
    co_return std::expected<void, page_error>{};
}

void scheduler::release_lines(sequence& seq) noexcept {
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        cache.release(layer, seq.layers[layer].lines);
        seq.layers[layer].lines.clear();
    }
}

bool scheduler::release_passed_pages(sequence& seq) {
    bool changed = false;
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        layer_view& view = seq.layers[layer];
        std::vector<std::size_t> passed(passed_pages(layer, view.pages.size() * page_size));
        if (passed.empty()) {
            continue;
        }
        std::iota(passed.begin(), passed.end(), 0);
        cache.release(layer, view.remove(passed));
        changed = true;
    }
    return changed;
}

task<void> scheduler::run() {
    for (;;) {
        batch b = co_await build_batch();
        co_await launch(b);
        co_await write_back(b);
    }
}

task<scheduler::batch> scheduler::build_batch() {
    while (running.empty()) {
        co_await next(idle);
    }
    co_return batch(running, page_size, prefill_tokens_per_step);
}

task<void> scheduler::launch(batch& b) {
    b.enqueue(model, lines, mass_heads, stream);
    co_await synchronize(stream);
}

// A sequence ends with oom when the sync of its new pages fails, as a state names only pages the store has. The step
// that last read the lines of pages that left a view has completed, so they are released right after the sync. A
// recurrent layer's state at head reaches the store at the resume points, cycle ends and the end, that find it pending.
task<void> scheduler::write_back(batch& b) {
    std::size_t n_layers = cache.layer_count();
    bool has_recurrent = std::ranges::contains(recurrent, true);
    std::vector<std::vector<digest>> new_hashes(n_layers);
    std::vector<std::vector<std::uint64_t>> new_owners(n_layers);
    std::vector<std::vector<std::uint32_t>> new_lines(n_layers);
    std::vector<sequence*> putting;
    std::vector<sequence*> putting_states;
    const std::int64_t* next_sampled = b.sampled.defined() ? b.sampled.const_data_ptr<std::int64_t>() : nullptr;
    for (std::size_t i = 0; i < b.entries.size(); i++) {
        const batch::entry& e = b.entries[i];
        sequence& seq = *e.seq;
        if (seq.eviction.can_evict()) {
            std::vector<std::span<const float>> step_mass = b.step_mass(i, mass_heads);
            std::size_t page_advance = latest_page_advance(seq.tail.size(), e.token_count, page_size);
            for (std::size_t layer = 0; layer < n_layers; layer++) {
                if (mass_heads[layer] > 0) {
                    add_step_mass(
                        seq.layers[layer].density,
                        step_mass[layer],
                        page_advance,
                        seq.eviction.attention_mass_decay
                    );
                }
            }
        }
        if (std::size_t completed = seq.advance(e.token_count, page_size, root); completed > 0) {
            for (std::size_t layer = 0; layer < n_layers; layer++) {
                if (recurrent[layer]) {
                    continue;
                }
                const layer_view& view = seq.layers[layer];
                std::size_t first_new_page = view.pages.size() - completed;
                new_hashes[layer]
                    .insert(new_hashes[layer].end(), view.pages.begin() + first_new_page, view.pages.end());
                new_owners[layer].resize(new_hashes[layer].size(), seq.id);
                new_lines[layer].insert(
                    new_lines[layer].end(),
                    view.lines.begin() + first_new_page,
                    view.lines.begin() + view.pages.size()
                );
            }
            seq.state_pending = has_recurrent;
            putting.push_back(&seq);
        }
        if (e.samples) {
            std::int64_t token = *next_sampled++;
            seq.writer.append(token);
            seq.generated++;
            seq.pending.push_back(token);
            if (std::ranges::contains(stop_ids, token) || !seq.may_sample()) {
                seq.exited = exit_reason::finished;
            }
        } else if (seq.pending.empty()) {
            seq.exited = exit_reason::finished;
        }
        if (seq.state_pending && (e.ends_cycle || seq.exited == exit_reason::finished)) {
            for (std::size_t layer = 0; layer < n_layers; layer++) {
                if (recurrent[layer]) {
                    new_hashes[layer].push_back(*seq.head);
                    new_owners[layer].push_back(seq.id);
                    new_lines[layer].push_back(seq.layers[layer].lines[1]);
                }
            }
            putting_states.push_back(&seq);
        }
    }

    std::expected<void, page_error> synced = co_await cache.sync(new_hashes, new_owners, new_lines, stream);
    if (!synced) {
        for (sequence* seq : putting) {
            seq->exited = exit_reason::oom;
        }
        for (sequence* seq : putting_states) {
            seq->exited = exit_reason::oom;
        }
    } else {
        for (sequence* seq : putting_states) {
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

    for (const batch::entry& e : b.entries) {
        sequence& seq = *e.seq;
        if (seq.exited) {
            continue;
        }
        if (seq.writer.cancelled()) {
            seq.exited = exit_reason::cancelled;
            continue;
        }
        bool views_changed = release_passed_pages(seq);
        if (e.ends_cycle) {
            views_changed = end_cycle(seq) || views_changed;
        }
        std::optional<std::size_t> first_stale_page;
        if (views_changed) {
            first_stale_page = 0;
        }
        std::vector<std::size_t> missing = seq.missing_lines(page_size);
        if (std::ranges::any_of(missing, [](std::size_t count) { return count > 0; })) {
            std::expected<std::vector<std::vector<std::uint32_t>>, page_error> allocated = cache.allocate(missing);
            if (!allocated) {
                seq.exited = exit_reason::oom;
                continue;
            }
            for (std::size_t layer = 0; layer < n_layers; layer++) {
                layer_view& view = seq.layers[layer];
                if (missing[layer] > 0) {
                    first_stale_page = std::min(first_stale_page.value_or(view.lines.size()), view.lines.size());
                }
                view.append_lines((*allocated)[layer]);
            }
        }
        if (first_stale_page) {
            seq.upload_table(*first_stale_page, stream);
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
bool scheduler::end_cycle(sequence& seq) {
    bool evicted = false;
    if (seq.eviction.can_evict()) {
        for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
            std::size_t heads = mass_heads[layer];
            if (heads == 0) {
                continue;
            }
            layer_view& view = seq.layers[layer];
            double threshold = max_context_pages ? density_threshold(seq.eviction.eviction_target, heads, *max_context_pages) : 0.0;
            std::vector<std::size_t> pages = pages_to_evict(
                std::span(view.density).first(view.pages.size() * heads),
                heads,
                threshold,
                page_size,
                seq.eviction
            );
            if (!pages.empty()) {
                cache.release(layer, view.remove(pages));
                evicted = true;
            }
        }
    }
    seq.writer.append(cycle_end{.reached = reached(seq), .token_count = seq.covered_count});
    seq.start_cycle();
    return evicted;
}

state scheduler::reached(const sequence& seq) const {
    state s{.head = seq.head, .layers = std::vector<layer_sequence>(cache.layer_count()), .tail = seq.tail};
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        std::vector<active_page>& pages = s.layers[layer].pages;
        if (recurrent[layer]) {
            if (seq.head) {
                pages.push_back(active_page{.hash = *seq.head});
            }
            continue;
        }
        const layer_view& view = seq.layers[layer];
        for (std::size_t page = passed_pages(layer, view.pages.size() * page_size); page < view.pages.size(); page++) {
            auto density = std::span(view.density).subspan(page * view.mass_heads, view.mass_heads);
            pages.push_back(active_page{.hash = view.pages[page], .density = {density.begin(), density.end()}});
        }
    }
    return s;
}

void scheduler::validate_state(const state& s) const {
    if (s.layers.empty() && !s.head) {
        return;
    }
    if (s.layers.size() != cache.layer_count()) {
        throw std::invalid_argument(
            std::format("the state holds {} sequences for {} KV layers", s.layers.size(), cache.layer_count())
        );
    }
    for (std::size_t layer = 0; layer < s.layers.size(); layer++) {
        std::span<const active_page> pages = s.layers[layer].pages;
        std::size_t heads = mass_heads[layer];
        for (const active_page& page : pages) {
            if (!page.density.empty() && page.density.size() != heads) {
                throw std::invalid_argument(
                    std::format(
                        "a page of KV layer {} holds {} densities; it must hold none or one per query head ({})",
                        layer,
                        page.density.size(),
                        heads
                    )
                );
            }
            if (!std::ranges::all_of(page.density, [](float d) { return std::isfinite(d) && d >= 0.0f; })) {
                throw std::invalid_argument("the state's densities must be finite and non-negative");
            }
        }
        if (!pages.empty() && (!s.head || pages.back().hash != *s.head)) {
            throw std::invalid_argument(
                std::format("the sequence of KV layer {} must end with the state's head", layer)
            );
        }
        std::size_t max_pages = std::numeric_limits<std::size_t>::max();
        if (recurrent[layer]) {
            max_pages = 1;
        } else if (sliding_windows[layer]) {
            max_pages = ceil_div(*sliding_windows[layer] - 1, page_size);
        }
        if (pages.size() > max_pages) {
            throw std::invalid_argument(
                std::format(
                    "the sequence of KV layer {} holds {} pages, more than the {} it reads",
                    layer,
                    pages.size(),
                    max_pages
                )
            );
        }
        if (s.head && max_pages > 0 && pages.empty()) {
            throw std::invalid_argument(std::format("the sequence of KV layer {} must hold the state's head", layer));
        }
    }
}

}  // namespace pulsarlm
