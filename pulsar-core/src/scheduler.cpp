#include "pulsar/scheduler.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

namespace pulsar {

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
        .slot_mapping = to_device(host.slot_mapping, device),
        .page_tables = to_device(host.page_tables, device),
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

sequence::sequence(request r, result_writer& writer, at::Device device)
: writer(writer),
  sampling(r.sampling),
  generator(make_sampling_generator(device, r.sampling.seed)),
  max_tokens(r.max_tokens),
  page_hashes(std::move(r.initial_state.page_hashes)),
  pending(std::move(r.initial_state.tail)),
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

std::vector<std::uint32_t> sequence::remove_pages(std::span<const std::size_t> pages) {
    std::vector<std::uint32_t> removed;
    std::size_t kept = 0;
    std::size_t next_removed = 0;
    for (std::size_t page = 0; page < page_hashes.size(); page++) {
        if (next_removed < pages.size() && pages[next_removed] == page) {
            removed.push_back(lines[page]);
            next_removed++;
            continue;
        }
        page_hashes[kept] = page_hashes[page];
        lines[kept] = lines[page];
        kept++;
    }
    assert(next_removed == pages.size());
    std::size_t full_pages = page_hashes.size();
    page_hashes.resize(kept);
    lines.erase(lines.begin() + kept, lines.begin() + full_pages);
    return removed;
}

std::size_t sequence::admission_lines(std::size_t page_size) const {
    assert(tail.empty());
    std::unordered_set<digest, digest_hash> distinct(page_hashes.begin(), page_hashes.end());
    return distinct.size() + ceil_div(cycle_tokens_left(page_size), page_size);
}

std::size_t sequence::missing_lines(std::size_t page_size) const noexcept {
    std::size_t covering = ceil_div(forwarded_count(page_size) + cycle_tokens_left(page_size), page_size);
    return covering > lines.size() ? covering - lines.size() : 0;
}

void sequence::requeue() {
    covered_count -= tail.size();
    pending.insert(pending.begin(), tail.begin(), tail.end());
    tail.clear();
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
    std::size_t width = 0;
    for (const entry& e : group) {
        token_count += e.token_count;
        width = std::max(width, e.seq->lines.size());
    }
    auto n_tokens = static_cast<std::int64_t>(token_count);
    auto n_seqs = static_cast<std::int64_t>(group.size());
    GroupBatch g{
        .tokens = pinned({n_tokens}, at::kLong),
        .pos = pinned({n_tokens}, at::kLong),
        .slot_mapping = pinned({n_tokens}, at::kInt),
        .page_tables = pinned({n_seqs, static_cast<std::int64_t>(width)}, at::kInt).zero_(),
        .rope_layout = pinned({n_seqs, 3}, at::kInt),
    };
    at::Tensor key_counts = pinned({n_seqs}, at::kInt);
    at::Tensor query_starts = pinned({n_seqs + 1}, at::kInt);

    auto* tokens = g.tokens.mutable_data_ptr<std::int64_t>();
    auto* positions = g.pos.mutable_data_ptr<std::int64_t>();
    auto* slots = g.slot_mapping.mutable_data_ptr<std::int32_t>();
    auto* page_tables = g.page_tables.mutable_data_ptr<std::int32_t>();
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
            slots[row] = static_cast<std::int32_t>(seq.lines[index / page_size] * page_size + index % page_size);
        }
        std::copy(seq.lines.begin(), seq.lines.end(), page_tables + s * width);
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
            cu_view_pages[s + 1] = cu_view_pages[s] + static_cast<std::int32_t>(seq.lines.size());
            attention_mass_decay[s] = static_cast<float>(seq.eviction.attention_mass_decay);
        }
    }
    return g;
}

bool batch::measures_mass(std::span<const entry> group) noexcept {
    return std::ranges::any_of(group, [](const entry& e) { return e.seq->eviction.can_evict(); });
}

void batch::add_page_mass(std::span<entry> group, const GroupBatch& inputs) {
    std::int64_t first_page = 0;
    for (entry& e : group) {
        sequence& seq = *e.seq;
        auto pages = static_cast<std::int64_t>(seq.lines.size());
        if (seq.eviction.can_evict()) {
            at::Tensor step_mass = inputs.page_mass.narrow(0, first_page, pages);
            seq.density.add(step_mass, e.token_count, seq.eviction.attention_mass_decay);
            if (e.ends_cycle) {
                e.host_density = pinned({pages}, at::kFloat);
                e.host_density.copy_(seq.density.by_page(), /*non_blocking=*/true);
            }
        }
        first_page += pages;
    }
}

void batch::enqueue(Model& model, const kv_lines& kv, cudaStream_t stream) {
    at::Device device = kv.k.device();
    c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(stream, device.index()));
    StepBatch on_device;
    if (inputs.decode) {
        on_device.decode = to_device(*inputs.decode, device);
    }
    if (inputs.prefill) {
        on_device.prefill = to_device(*inputs.prefill, device);
    }
    StepLogits logits = model.forward(on_device, kv);
    std::span<entry> all(entries);
    if (on_device.decode && on_device.decode->cu_view_pages.defined()) {
        add_page_mass(all.first(decode_count), *on_device.decode);
    }
    if (on_device.prefill && on_device.prefill->cu_view_pages.defined()) {
        add_page_mass(all.subspan(decode_count), *on_device.prefill);
    }

    std::vector<at::Tensor> rows;
    std::vector<sampling_params> params;
    std::vector<at::Generator> generators;
    std::int64_t prefill_rows = 0;
    for (std::size_t i = 0; i < entries.size(); i++) {
        const entry& e = entries[i];
        std::int64_t row = static_cast<std::int64_t>(i);
        if (i >= decode_count) {
            prefill_rows += static_cast<std::int64_t>(e.token_count);
            row = prefill_rows - 1;
        }
        if (e.samples) {
            rows.push_back((i < decode_count ? *logits.decode : *logits.prefill).select(0, row));
            params.push_back(e.seq->sampling);
            generators.push_back(e.seq->generator);
        }
    }
    if (rows.empty()) {
        return;
    }
    at::Tensor tokens = sample(at::stack(rows), params, generators);
    sampled = pinned({static_cast<std::int64_t>(rows.size())}, at::kLong);
    sampled.copy_(tokens, /*non_blocking=*/true);
}

}  // namespace pulsar
