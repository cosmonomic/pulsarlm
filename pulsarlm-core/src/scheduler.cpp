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

// cu_view_pages is on the host; its last element counts the view pages.
at::Tensor zero_page_mass(const at::Tensor& cu_view_pages, at::Device device) {
    std::int32_t total_view_pages = cu_view_pages.const_data_ptr<std::int32_t>()[cu_view_pages.size(0) - 1];
    return at::zeros({total_view_pages}, at::TensorOptions().dtype(at::kFloat).device(device));
}

GroupBatch to_device(const GroupBatch& host, at::Device device) {
    return GroupBatch{
        .tokens = to_device(host.tokens, device),
        .pos = to_device(host.pos, device),
        .page_end_tokens = to_device(host.page_end_tokens, device),
        .cu_seqlens_q = to_device(host.cu_seqlens_q, device),
        .seqlens_k = to_device(host.seqlens_k, device),
        .context_lens = to_device(host.context_lens, device),
        .rope_layout = to_device(host.rope_layout, device),
        .cu_view_pages = to_device(host.cu_view_pages, device),
        .page_mass = host.cu_view_pages.defined() ? zero_page_mass(host.cu_view_pages, device) : at::Tensor(),
        .attention_mass_decay = to_device(host.attention_mass_decay, device),
        .mass_length_gain = host.mass_length_gain,
    };
}

}  // namespace

sequence::sequence(
    request&& r,
    const std::vector<bool>& recurrent,
    std::size_t view_layer,
    result_writer& writer,
    at::Device device,
    cudaStream_t stream
)
: writer(writer),
  id(r.id),
  sampling(r.sampling),
  seed(sampling_seed(r.sampling.seed)),
  max_tokens(r.max_tokens),
  head(r.initial_state.head),
  pending(std::move(r.initial_state.tail)),
  recurrent(recurrent),
  view_layer(view_layer),
  lines(recurrent.size()),
  first_held(recurrent.size(), 0),
  table(
      at::zeros({static_cast<std::int64_t>(recurrent.size()), 0}, at::TensorOptions().dtype(at::kInt).device(device))
  ),
  eviction(r.eviction) {
    pending.insert(pending.end(), r.tokens.begin(), r.tokens.end());
    if (!r.initial_state.layers.empty()) {
        for (const active_page& page : r.initial_state.layers[view_layer].pages) {
            page_hashes.push_back(page.hash);
            state_density.push_back(page.density);
        }
    }
    if (eviction.can_evict()) {
        at::Tensor host = pinned({static_cast<std::int64_t>(state_density.size())}, at::kFloat);
        std::ranges::copy(state_density, host.mutable_data_ptr<float>());
        c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(stream, device.index()));
        density = page_density(host.to(device, /*non_blocking=*/true));
    }
}

std::size_t sequence::forwarded_count(std::size_t page_size) const noexcept {
    return page_hashes.size() * page_size + tail.size();
}

bool sequence::may_sample() const noexcept {
    return !max_tokens || generated < *max_tokens;
}

void sequence::advance(std::size_t count, std::size_t page_size, const digest& root, const at::Tensor& host_density) {
    assert(count <= pending.size());
    tail.insert(tail.end(), pending.begin(), pending.begin() + count);
    pending.erase(pending.begin(), pending.begin() + count);
    covered_count += count;
    if (std::size_t completed_tokens = tail.size() - tail.size() % page_size; completed_tokens > 0) {
        std::vector<digest> completed = make_page_digests(
            head.value_or(root),
            std::span(tail).first(completed_tokens),
            page_size
        );
        page_hashes.insert(page_hashes.end(), completed.begin(), completed.end());
        head = completed.back();
        tail.erase(tail.begin(), tail.begin() + completed_tokens);
    }
    if (host_density.defined()) {
        assert(static_cast<std::size_t>(host_density.size(0)) >= page_hashes.size());
        const float* measured = host_density.const_data_ptr<float>();
        state_density.assign(measured, measured + page_hashes.size());
    } else {
        state_density.resize(page_hashes.size(), 0.0f);
    }
}

void sequence::start_cycle(std::size_t page_size) noexcept {
    cycle_start = forwarded_count(page_size);
}

bool sequence::ends_cycle(std::size_t token_count, std::size_t page_size) const noexcept {
    return forwarded_count(page_size) + token_count - cycle_start >= eviction.cycle_tokens;
}

std::size_t sequence::cycle_tokens_left(std::size_t page_size) const noexcept {
    std::size_t left = eviction.cycle_tokens - (forwarded_count(page_size) - cycle_start);
    if (max_tokens) {
        // Every token sampled from here is forwarded but the last.
        std::size_t samples_left = *max_tokens - generated;
        left = std::min(left, pending.size() + (samples_left > 0 ? samples_left - 1 : 0));
    }
    return left;
}

std::size_t sequence::view_page_count() const noexcept {
    return lines[view_layer].size();
}

std::size_t sequence::table_width() const noexcept {
    std::size_t width = 0;
    for (const std::vector<std::uint32_t>& layer_lines : lines) {
        width = std::max(width, layer_lines.size());
    }
    return width;
}

// Evicted pages precede every sliding window, so a sliding layer's evicted entries hold no_line.
std::vector<std::vector<std::uint32_t>>
sequence::remove_pages(std::span<const std::size_t> pages, cudaStream_t stream) {
    std::vector<std::vector<std::uint32_t>> removed(lines.size());
    std::size_t kept = 0;
    std::size_t next_removed = 0;
    for (std::size_t page = 0; page < page_hashes.size(); page++) {
        if (next_removed < pages.size() && pages[next_removed] == page) {
            for (std::size_t layer = 0; layer < lines.size(); layer++) {
                if (!recurrent[layer] && lines[layer][page] != no_line) {
                    removed[layer].push_back(lines[layer][page]);
                }
            }
            next_removed++;
            continue;
        }
        page_hashes[kept] = page_hashes[page];
        state_density[kept] = state_density[page];
        for (std::size_t layer = 0; layer < lines.size(); layer++) {
            if (!recurrent[layer]) {
                lines[layer][kept] = lines[layer][page];
            }
        }
        kept++;
    }
    assert(next_removed == pages.size());
    std::size_t full_pages = page_hashes.size();
    page_hashes.resize(kept);
    state_density.resize(kept);
    for (std::size_t layer = 0; layer < lines.size(); layer++) {
        if (recurrent[layer]) {
            continue;
        }
        lines[layer].erase(lines[layer].begin() + kept, lines[layer].begin() + full_pages);
        assert(first_held[layer] == 0 || first_held[layer] >= pages.size());
        first_held[layer] -= std::min(first_held[layer], pages.size());
    }
    if (density.by_page().defined()) {
        const at::Tensor& on_device = density.by_page();
        c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(stream, on_device.device().index()));
        density.remove(pages);
    }
    return removed;
}

void sequence::upload_table(std::size_t first_page, cudaStream_t stream) {
    auto n_layers = static_cast<std::int64_t>(lines.size());
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
    for (std::size_t layer = 0; layer < lines.size(); layer++) {
        for (std::int64_t page = first; page < page_count; page++) {
            std::uint32_t line = static_cast<std::size_t>(page) < lines[layer].size() ? lines[layer][page] : no_line;
            *entries++ = line == no_line ? 0 : static_cast<std::int32_t>(line);
        }
    }
    c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(stream, table.device().index()));
    table.narrow(1, first, page_count - first).copy_(host, /*non_blocking=*/true);
}

std::size_t sequence::cycle_lines(std::size_t page_size) const noexcept {
    assert(tail.empty());
    return ceil_div(cycle_tokens_left(page_size), page_size);
}

void sequence::add_step_mass(const at::Tensor& step_mass, std::size_t token_count, cudaStream_t stream) {
    c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(stream, step_mass.device().index()));
    density.add(step_mass, token_count, eviction.attention_mass_decay);
}

at::Tensor sequence::density_on_host(cudaStream_t stream) const {
    const at::Tensor& on_device = density.by_page();
    c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(stream, on_device.device().index()));
    at::Tensor host = pinned(on_device.sizes(), at::kFloat);
    host.copy_(on_device, /*non_blocking=*/true);
    return host;
}

std::vector<std::size_t> sequence::missing_lines(std::size_t page_size) const {
    std::size_t covering = ceil_div(forwarded_count(page_size) + cycle_tokens_left(page_size), page_size);
    std::size_t missing_pages = covering > view_page_count() ? covering - view_page_count() : 0;
    std::vector<std::size_t> missing;
    for (std::size_t layer = 0; layer < lines.size(); layer++) {
        missing.push_back(recurrent[layer] ? 2 - lines[layer].size() : missing_pages);
    }
    return missing;
}

batch::batch(
    std::span<sequence* const> running,
    std::size_t page_size,
    std::size_t prefill_tokens_per_step,
    bool recurrent
) {
    for (sequence* seq : running) {
        assert(!seq->pending.empty());
        if (seq->pending.size() == 1) {
            entries.push_back(
                entry{
                    .seq = seq,
                    .token_count = 1,
                    .samples = seq->may_sample(),
                    .ends_cycle = seq->ends_cycle(1, page_size),
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
            std::size_t count = std::min({seq->pending.size(), budget, seq->cycle_tokens_left(page_size)});
            budget -= count;
            entries.push_back(
                entry{
                    .seq = seq,
                    .token_count = count,
                    .samples = count == seq->pending.size() && seq->may_sample(),
                    .ends_cycle = seq->ends_cycle(count, page_size),
                }
            );
        }
    }
    std::span<const entry> all(entries);
    if (decode_count > 0) {
        inputs.decode = make_group(all.first(decode_count), page_size, false, recurrent);
    }
    if (entries.size() > decode_count) {
        inputs.prefill = make_group(all.subspan(decode_count), page_size, true, recurrent);
    }
}

GroupBatch batch::make_group(std::span<const entry> group, std::size_t page_size, bool prefill, bool recurrent) {
    std::size_t token_count = 0;
    for (const entry& e : group) {
        token_count += e.token_count;
    }
    auto n_tokens = static_cast<std::int64_t>(token_count);
    auto n_seqs = static_cast<std::int64_t>(group.size());
    GroupBatch g{
        .tokens = pinned({n_tokens}, at::kLong),
        .pos = pinned({n_tokens}, at::kLong),
        .rope_layout = pinned({n_seqs, 3}, at::kInt),
    };
    at::Tensor key_counts = pinned({n_seqs}, at::kInt);
    at::Tensor query_starts = pinned({n_seqs + 1}, at::kInt);

    auto* tokens = g.tokens.mutable_data_ptr<std::int64_t>();
    auto* positions = g.pos.mutable_data_ptr<std::int64_t>();
    auto* rope_layout = g.rope_layout.mutable_data_ptr<std::int32_t>();
    auto* key_count = key_counts.mutable_data_ptr<std::int32_t>();
    auto* query_start = query_starts.mutable_data_ptr<std::int32_t>();
    std::size_t row = 0;
    query_start[0] = 0;
    for (std::size_t s = 0; s < group.size(); s++) {
        const sequence& seq = *group[s].seq;
        std::size_t start = seq.forwarded_count(page_size);
        for (std::size_t i = 0; i < group[s].token_count; i++, row++) {
            std::size_t index = start + i;
            tokens[row] = seq.pending[i];
            positions[row] = static_cast<std::int64_t>(index);
        }
        key_count[s] = static_cast<std::int32_t>(start + group[s].token_count);
        query_start[s + 1] = static_cast<std::int32_t>(row);
        rope_layout[3 * s] = 0;
        rope_layout[3 * s + 1] = 0;
        rope_layout[3 * s + 2] = -1;
    }
    if (prefill) {
        g.cu_seqlens_q = query_starts;
        g.seqlens_k = key_counts;
    } else {
        g.context_lens = key_counts;
    }
    if (recurrent) {
        g.page_end_tokens = pinned({n_seqs}, at::kInt);
        auto* page_end_tokens = g.page_end_tokens.mutable_data_ptr<std::int32_t>();
        for (std::size_t s = 0; s < group.size(); s++) {
            std::size_t start = group[s].seq->forwarded_count(page_size);
            std::size_t page_end = (start + group[s].token_count) / page_size * page_size;
            page_end_tokens[s] = static_cast<std::int32_t>(page_end > start ? page_end - start : 0);
        }
    }
    if (measures_mass(group)) {
        g.cu_view_pages = pinned({n_seqs + 1}, at::kInt);
        g.attention_mass_decay = pinned({n_seqs}, at::kFloat);
        auto* cu_view_pages = g.cu_view_pages.mutable_data_ptr<std::int32_t>();
        auto* attention_mass_decay = g.attention_mass_decay.mutable_data_ptr<float>();
        cu_view_pages[0] = 0;
        for (std::size_t s = 0; s < group.size(); s++) {
            const sequence& seq = *group[s].seq;
            cu_view_pages[s + 1] = cu_view_pages[s] + static_cast<std::int32_t>(seq.view_page_count());
            attention_mass_decay[s] = static_cast<float>(seq.eviction.attention_mass_decay);
        }
    }
    return g;
}

bool batch::measures_mass(std::span<const entry> group) noexcept {
    return std::ranges::any_of(group, [](const entry& e) { return e.seq->eviction.can_evict(); });
}

void batch::set_step_mass(std::span<entry> group, const GroupBatch& inputs) {
    std::int64_t first_page = 0;
    for (entry& e : group) {
        const sequence& seq = *e.seq;
        auto pages = static_cast<std::int64_t>(seq.view_page_count());
        if (seq.eviction.can_evict()) {
            e.step_mass = inputs.page_mass.narrow(0, first_page, pages);
        }
        first_page += pages;
    }
}

at::Tensor batch::page_tables(std::span<const entry> group, std::size_t n_kv_layers) {
    std::vector<at::Tensor> tables;
    std::size_t width = 0;
    for (const entry& e : group) {
        tables.push_back(e.seq->table);
        width = std::max(width, e.seq->table_width());
    }
    return gather_page_tables_cuda(tables, static_cast<std::int64_t>(n_kv_layers), static_cast<std::int64_t>(width));
}

void batch::enqueue(Model& model, std::span<const kv> lines, cudaStream_t stream) {
    at::Device device = lines.front().k.device();
    c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(stream, device.index()));
    StepBatch on_device;
    std::span<entry> all(entries);
    if (inputs.decode) {
        on_device.decode = to_device(*inputs.decode, device);
        on_device.decode->page_tables = page_tables(all.first(decode_count), lines.size());
    }
    if (inputs.prefill) {
        on_device.prefill = to_device(*inputs.prefill, device);
        on_device.prefill->page_tables = page_tables(all.subspan(decode_count), lines.size());
        const at::Tensor& cu_seqlens_q = on_device.prefill->cu_seqlens_q;
        on_device.prefill->logit_rows = cu_seqlens_q.narrow(0, 1, cu_seqlens_q.size(0) - 1).to(at::kLong) - 1;
    }
    StepLogits logits = model.forward(on_device, lines);
    if (on_device.decode && on_device.decode->cu_view_pages.defined()) {
        set_step_mass(all.first(decode_count), *on_device.decode);
    }
    if (on_device.prefill && on_device.prefill->cu_view_pages.defined()) {
        set_step_mass(all.subspan(decode_count), *on_device.prefill);
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

scheduler::scheduler(
    Model& model,
    kv_cache cache,
    cudaStream_t stream,
    digest root,
    std::vector<std::int64_t>&& stop_ids,
    std::size_t prefill_tokens_per_step
)
: model(model),
  cache(std::move(cache)),
  window_pages(0),
  view_layer(0),
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
    }
    while (sliding_windows[view_layer] || recurrent[view_layer]) {
        view_layer++;
        assert(view_layer < layers.size());
    }
    page_size = static_cast<std::size_t>(lines[view_layer].k.size(1));
    for (const std::optional<std::size_t>& window : sliding_windows) {
        if (window) {
            window_pages = std::max(window_pages, ceil_div(*window - 1, page_size));
        }
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

std::size_t scheduler::first_held_page(std::size_t layer, std::size_t query) const noexcept {
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
    r.eviction.n_working = std::max(r.eviction.n_working, window_pages);
    sequence seq(std::move(r), recurrent, view_layer, writer, cache.tensors(view_layer).k.device(), stream);
    seq.start_cycle(page_size);
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
    std::size_t pages = seq.page_hashes.size();
    std::vector<std::size_t> first_held(cache.layer_count());
    std::vector<std::span<const digest>> held(cache.layer_count());
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        if (recurrent[layer]) {
            held[layer] = std::span(seq.page_hashes).last(std::min<std::size_t>(pages, 1));
            continue;
        }
        first_held[layer] = first_held_page(layer, pages * page_size);
        held[layer] = std::span(seq.page_hashes).subspan(first_held[layer]);
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
        std::vector<std::uint32_t>& layer_lines = seq.lines[layer];
        if (recurrent[layer]) {
            if (states[layer].empty()) {
                c10::cuda::CUDAStreamGuard guard(
                    c10::cuda::getStreamFromExternal(stream, lines[layer].k.device().index())
                );
                lines[layer].k.select(0, (*allocated)[layer].front()).zero_();
            }
            layer_lines = std::move(states[layer]);
            layer_lines.insert(layer_lines.end(), (*allocated)[layer].begin(), (*allocated)[layer].end());
            continue;
        }
        layer_lines.assign(first_held[layer], sequence::no_line);
        layer_lines.insert(layer_lines.end(), (*got)[layer].begin(), (*got)[layer].end());
        layer_lines.insert(layer_lines.end(), (*allocated)[layer].begin(), (*allocated)[layer].end());
        seq.first_held[layer] = first_held[layer];
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
        std::vector<std::uint32_t>& layer_lines = seq.lines[layer];
        std::span<const std::uint32_t> held(layer_lines.begin() + seq.first_held[layer], layer_lines.end());
        cache.release(layer, held);
        layer_lines.clear();
        seq.first_held[layer] = 0;
    }
}

void scheduler::release_passed_lines(sequence& seq) {
    std::size_t query = seq.forwarded_count(page_size);
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        std::size_t first = first_held_page(layer, query);
        std::vector<std::uint32_t>& layer_lines = seq.lines[layer];
        if (first <= seq.first_held[layer]) {
            continue;
        }
        cache.release(layer, std::span(layer_lines).subspan(seq.first_held[layer], first - seq.first_held[layer]));
        std::fill(layer_lines.begin() + seq.first_held[layer], layer_lines.begin() + first, sequence::no_line);
        seq.first_held[layer] = first;
    }
}

task<void> scheduler::run() {
    for (;;) {
        batch b = co_await build_batch();
        co_await launch(b);
        co_await write_back(b);
    }
}

task<batch> scheduler::build_batch() {
    while (running.empty()) {
        co_await next(idle);
    }
    co_return batch(running, page_size, prefill_tokens_per_step, std::ranges::contains(recurrent, true));
}

task<void> scheduler::launch(batch& b) {
    b.enqueue(model, lines, stream);
    for (batch::entry& e : b.entries) {
        if (e.step_mass.defined()) {
            e.seq->add_step_mass(e.step_mass, e.token_count, stream);
            e.host_density = e.seq->density_on_host(stream);
        }
    }
    co_await synchronize(stream);
}

// A sequence ends with oom when the sync of its new pages fails, as a state names only pages the store has. The step
// that last read the lines of pages that left a window has completed, so they are released right after the sync. A
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
    for (const batch::entry& e : b.entries) {
        sequence& seq = *e.seq;
        std::size_t first_new_page = seq.page_hashes.size();
        seq.advance(e.token_count, page_size, root, e.host_density);
        if (seq.page_hashes.size() > first_new_page) {
            for (std::size_t layer = 0; layer < n_layers; layer++) {
                if (recurrent[layer]) {
                    continue;
                }
                new_hashes[layer]
                    .insert(new_hashes[layer].end(), seq.page_hashes.begin() + first_new_page, seq.page_hashes.end());
                new_owners[layer].resize(new_hashes[layer].size(), seq.id);
                new_lines[layer].insert(
                    new_lines[layer].end(),
                    seq.lines[layer].begin() + first_new_page,
                    seq.lines[layer].begin() + seq.page_hashes.size()
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
                    new_lines[layer].push_back(seq.lines[layer][1]);
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
                    cache.release(layer, std::span(seq->lines[layer]).last(1));
                    seq->lines[layer].pop_back();
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
        release_passed_lines(seq);
        if (e.ends_cycle) {
            end_cycle(seq);
        }
        std::vector<std::size_t> missing = seq.missing_lines(page_size);
        if (std::ranges::any_of(missing, [](std::size_t count) { return count > 0; })) {
            std::size_t first_new_page = seq.view_page_count();
            bool states_replaced = false;
            for (std::size_t layer = 0; layer < n_layers; layer++) {
                states_replaced = states_replaced || (recurrent[layer] && missing[layer] > 0);
            }
            std::expected<std::vector<std::vector<std::uint32_t>>, page_error> allocated = cache.allocate(missing);
            if (!allocated) {
                seq.exited = exit_reason::oom;
                continue;
            }
            for (std::size_t layer = 0; layer < n_layers; layer++) {
                seq.lines[layer].insert(seq.lines[layer].end(), (*allocated)[layer].begin(), (*allocated)[layer].end());
            }
            seq.upload_table(states_replaced ? 0 : first_new_page, stream);
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
void scheduler::end_cycle(sequence& seq) {
    if (seq.eviction.can_evict()) {
        std::vector<std::size_t> evicted = pages_to_evict(seq.state_density, seq.eviction);
        if (!evicted.empty()) {
            std::vector<std::vector<std::uint32_t>> removed = seq.remove_pages(evicted, stream);
            for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
                cache.release(layer, removed[layer]);
            }
            seq.upload_table(0, stream);
        }
    }
    seq.writer.append(cycle_end{.reached = reached(seq), .token_count = seq.covered_count});
    seq.start_cycle(page_size);
}

std::vector<layer_sequence> scheduler::layer_sequences(std::span<const active_page> view) const {
    std::vector<layer_sequence> layers(cache.layer_count());
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        std::size_t first = recurrent[layer] ? view.size() - std::min<std::size_t>(view.size(), 1)
                                             : first_held_page(layer, view.size() * page_size);
        std::span<const active_page> read = view.subspan(first);
        layers[layer].pages.assign(read.begin(), read.end());
    }
    return layers;
}

state scheduler::reached(const sequence& seq) const {
    std::vector<active_page> view;
    for (std::size_t page = 0; page < seq.page_hashes.size(); page++) {
        view.push_back(active_page{.hash = seq.page_hashes[page], .density = seq.state_density[page]});
    }
    return state{.head = seq.head, .layers = layer_sequences(view), .tail = seq.tail};
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
    std::span<const active_page> view = s.layers[view_layer].pages;
    if (view.empty() != !s.head || (s.head && view.back().hash != *s.head)) {
        throw std::invalid_argument(
            std::format("the sequence of KV layer {} must end with the state's head, and only it", view_layer)
        );
    }
    if (!std::ranges::all_of(view, [](const active_page& p) {
            return std::isfinite(p.density) && p.density >= 0.0f;
        })) {
        throw std::invalid_argument("the state's densities must be finite and non-negative");
    }
    std::vector<layer_sequence> expected = layer_sequences(view);
    for (std::size_t layer = 0; layer < expected.size(); layer++) {
        if (s.layers[layer] != expected[layer]) {
            throw std::invalid_argument(
                std::format(
                    "the sequence of KV layer {} must be the pages it reads of KV layer {}'s sequence",
                    layer,
                    view_layer
                )
            );
        }
    }
}

}  // namespace pulsarlm
