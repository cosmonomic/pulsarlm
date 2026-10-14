#include "pulsarlm/scheduler.hpp"

#include "pulsarlm/ops.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <expected>
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

// cu_view_pages is on the host; its last element counts the view pages.
at::Tensor zero_page_mass(const at::Tensor& cu_view_pages, at::Device device) {
    std::int32_t total_view_pages = cu_view_pages.const_data_ptr<std::int32_t>()[cu_view_pages.size(0) - 1];
    return at::zeros({total_view_pages}, at::TensorOptions().dtype(at::kFloat).device(device));
}

GroupBatch to_device(const GroupBatch& host, at::Device device) {
    return GroupBatch{
        .tokens = to_device(host.tokens, device),
        .pos = to_device(host.pos, device),
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

sequence::sequence(request&& r, result_writer& writer, std::size_t n_kv_layers, at::Device device)
: writer(writer),
  sampling(r.sampling),
  seed(sampling_seed(r.sampling.seed)),
  max_tokens(r.max_tokens),
  page_hashes(std::move(r.initial_state.page_hashes)),
  pending(std::move(r.initial_state.tail)),
  lines(n_kv_layers),
  first_held(n_kv_layers, 0),
  table(at::zeros({static_cast<std::int64_t>(n_kv_layers), 0}, at::TensorOptions().dtype(at::kInt).device(device))),
  eviction(r.eviction) {
    pending.insert(pending.end(), r.tokens.begin(), r.tokens.end());
}

std::size_t sequence::forwarded_count(std::size_t page_size) const noexcept {
    return page_hashes.size() * page_size + tail.size();
}

bool sequence::may_sample() const noexcept {
    return !max_tokens || generated < *max_tokens;
}

void sequence::advance(std::size_t count, std::size_t page_size, const digest& root) {
    assert(count <= pending.size());
    tail.insert(tail.end(), pending.begin(), pending.begin() + count);
    pending.erase(pending.begin(), pending.begin() + count);
    covered_count += count;
    std::size_t completed_tokens = tail.size() - tail.size() % page_size;
    if (completed_tokens == 0) {
        return;
    }
    digest parent = page_hashes.empty() ? root : page_hashes.back();
    std::vector<digest> completed = make_page_digests(parent, std::span(tail).first(completed_tokens), page_size);
    page_hashes.insert(page_hashes.end(), completed.begin(), completed.end());
    tail.erase(tail.begin(), tail.begin() + completed_tokens);
}

void sequence::start_cycle(std::size_t page_size) noexcept {
    cycle_start = forwarded_count(page_size);
    density.reset();
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
    return lines.front().size();
}

// Evicted pages precede every sliding window, so a sliding layer's evicted entries hold no_line.
std::vector<std::vector<std::uint32_t>> sequence::remove_pages(std::span<const std::size_t> pages) {
    std::vector<std::vector<std::uint32_t>> removed(lines.size());
    std::size_t kept = 0;
    std::size_t next_removed = 0;
    for (std::size_t page = 0; page < page_hashes.size(); page++) {
        if (next_removed < pages.size() && pages[next_removed] == page) {
            for (std::size_t layer = 0; layer < lines.size(); layer++) {
                if (lines[layer][page] != no_line) {
                    removed[layer].push_back(lines[layer][page]);
                }
            }
            next_removed++;
            continue;
        }
        page_hashes[kept] = page_hashes[page];
        for (std::vector<std::uint32_t>& layer_lines : lines) {
            layer_lines[kept] = layer_lines[page];
        }
        kept++;
    }
    assert(next_removed == pages.size());
    std::size_t full_pages = page_hashes.size();
    page_hashes.resize(kept);
    for (std::size_t layer = 0; layer < lines.size(); layer++) {
        lines[layer].erase(lines[layer].begin() + kept, lines[layer].begin() + full_pages);
        assert(first_held[layer] == 0 || first_held[layer] >= pages.size());
        first_held[layer] -= std::min(first_held[layer], pages.size());
    }
    return removed;
}

void sequence::upload_table(std::size_t first_page, cudaStream_t stream) {
    auto n_layers = static_cast<std::int64_t>(lines.size());
    auto page_count = static_cast<std::int64_t>(view_page_count());
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
            std::uint32_t line = lines[layer][page];
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

std::size_t sequence::missing_lines(std::size_t page_size) const noexcept {
    std::size_t covering = ceil_div(forwarded_count(page_size) + cycle_tokens_left(page_size), page_size);
    return covering > view_page_count() ? covering - view_page_count() : 0;
}

batch::batch(std::span<sequence* const> running, std::size_t page_size, std::size_t prefill_tokens_per_step) {
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
        inputs.decode = make_group(all.first(decode_count), page_size, false);
    }
    if (entries.size() > decode_count) {
        inputs.prefill = make_group(all.subspan(decode_count), page_size, true);
    }
}

GroupBatch batch::make_group(std::span<const entry> group, std::size_t page_size, bool prefill) {
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
        width = std::max(width, e.seq->view_page_count());
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
  stream(stream),
  page_size(static_cast<std::size_t>(this->cache.tensors(0).k.size(1))),
  root(root),
  stop_ids(std::move(stop_ids)),
  prefill_tokens_per_step(prefill_tokens_per_step) {
    assert(prefill_tokens_per_step > 0);
    std::vector<kv_layer> layers = model.kv_layers();
    assert(layers.size() == this->cache.layer_count());
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        lines.push_back(this->cache.tensors(layer));
        sliding_windows.push_back(layers[layer].sliding_window);
        if (layers[layer].sliding_window) {
            window_pages = std::max(window_pages, ceil_div(*layers[layer].sliding_window - 1, page_size));
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
    sequence seq(std::move(r), writer, cache.layer_count(), cache.tensors(0).k.device());
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

task<std::expected<void, page_error>> scheduler::take_lines(sequence& seq, std::size_t line_count) {
    std::size_t pages = seq.page_hashes.size();
    std::vector<std::size_t> first_held(cache.layer_count());
    std::vector<std::span<const digest>> held(cache.layer_count());
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        first_held[layer] = first_held_page(layer, pages * page_size);
        held[layer] = std::span(seq.page_hashes).subspan(first_held[layer]);
    }
    // A get failing with miss releases the lines it held while it fetched.
    std::expected<std::vector<std::vector<std::uint32_t>>, page_error> got = co_await cache.get(held, stream);
    if (!got) {
        co_return std::unexpected(got.error());
    }
    std::expected<std::vector<std::vector<std::uint32_t>>, page_error> allocated = cache.allocate(line_count);
    if (!allocated) {
        for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
            cache.release(layer, (*got)[layer]);
        }
        co_return std::unexpected(allocated.error());
    }
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        std::vector<std::uint32_t>& layer_lines = seq.lines[layer];
        layer_lines.assign(first_held[layer], sequence::no_line);
        layer_lines.insert(layer_lines.end(), (*got)[layer].begin(), (*got)[layer].end());
        layer_lines.insert(layer_lines.end(), (*allocated)[layer].begin(), (*allocated)[layer].end());
        seq.first_held[layer] = first_held[layer];
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
    co_return batch(running, page_size, prefill_tokens_per_step);
}

task<void> scheduler::launch(batch& b) {
    b.enqueue(model, lines, stream);
    for (batch::entry& e : b.entries) {
        if (e.step_mass.defined()) {
            e.seq->add_step_mass(e.step_mass, e.token_count, stream);
            if (e.ends_cycle) {
                e.host_density = e.seq->density_on_host(stream);
            }
        }
    }
    co_await synchronize(stream);
}

// A sequence ends with oom when the sync of its new pages fails, as a state names only pages the store has. The step
// that last read the lines of pages that left a window has completed, so they are released right after the sync.
task<void> scheduler::write_back(batch& b) {
    std::vector<digest> new_hashes;
    std::vector<std::vector<std::uint32_t>> new_lines(cache.layer_count());
    std::vector<sequence*> putting;
    const std::int64_t* next_sampled = b.sampled.defined() ? b.sampled.const_data_ptr<std::int64_t>() : nullptr;
    for (const batch::entry& e : b.entries) {
        sequence& seq = *e.seq;
        std::size_t first_new_page = seq.page_hashes.size();
        seq.advance(e.token_count, page_size, root);
        if (seq.page_hashes.size() > first_new_page) {
            new_hashes.insert(new_hashes.end(), seq.page_hashes.begin() + first_new_page, seq.page_hashes.end());
            for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
                new_lines[layer].insert(
                    new_lines[layer].end(),
                    seq.lines[layer].begin() + first_new_page,
                    seq.lines[layer].begin() + seq.page_hashes.size()
                );
            }
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
    }

    if (!new_hashes.empty()) {
        std::expected<void, page_error> synced = co_await cache.sync(new_hashes, new_lines, stream);
        if (!synced) {
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
        release_passed_lines(seq);
        if (e.ends_cycle) {
            end_cycle(seq, e.host_density);
        }
        if (std::size_t missing = seq.missing_lines(page_size); missing > 0) {
            std::size_t first_new_page = seq.view_page_count();
            std::expected<std::vector<std::vector<std::uint32_t>>, page_error> allocated = cache.allocate(missing);
            if (!allocated) {
                seq.exited = exit_reason::oom;
                continue;
            }
            for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
                seq.lines[layer].insert(seq.lines[layer].end(), (*allocated)[layer].begin(), (*allocated)[layer].end());
            }
            seq.upload_table(first_new_page, stream);
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
void scheduler::end_cycle(sequence& seq, const at::Tensor& host_density) {
    if (host_density.defined()) {
        std::span<const float> density(host_density.const_data_ptr<float>(), seq.page_hashes.size());
        std::vector<std::size_t>
            evicted = pages_to_evict(density, page_size, seq.cycle_start, seq.forwarded_count(page_size), seq.eviction);
        if (!evicted.empty()) {
            std::vector<std::vector<std::uint32_t>> removed = seq.remove_pages(evicted);
            for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
                cache.release(layer, removed[layer]);
            }
            seq.upload_table(0, stream);
        }
    }
    seq.writer.append(
        cycle_end{.reached = state{.page_hashes = seq.page_hashes, .tail = seq.tail}, .token_count = seq.covered_count}
    );
    seq.start_cycle(page_size);
}

std::vector<digest> scheduler::page_keys(const state& s) const {
    std::vector<digest> keys;
    std::size_t pages = s.page_hashes.size();
    for (std::size_t layer = 0; layer < cache.layer_count(); layer++) {
        for (std::size_t page = first_held_page(layer, pages * page_size); page < pages; page++) {
            keys.push_back(store_key(s.page_hashes[page], layer));
        }
    }
    return keys;
}

}  // namespace pulsarlm
