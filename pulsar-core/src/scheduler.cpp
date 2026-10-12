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
  pending(std::move(r.initial_state.tail)) {
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
    std::size_t completed_tokens = tail.size() - tail.size() % page_size;
    if (completed_tokens == 0) {
        return;
    }
    digest parent = page_hashes.empty() ? root : page_hashes.back();
    std::vector<digest> completed = make_page_digests(parent, std::span(tail).first(completed_tokens), page_size);
    page_hashes.insert(page_hashes.end(), completed.begin(), completed.end());
    tail.erase(tail.begin(), tail.begin() + completed_tokens);
}

std::size_t sequence::admission_lines(std::size_t page_size) const {
    assert(tail.empty());
    std::unordered_set<digest, digest_hash> distinct(page_hashes.begin(), page_hashes.end());
    return distinct.size() + ceil_div(pending.size(), page_size);
}

std::size_t sequence::missing_lines(std::size_t page_size) const noexcept {
    std::size_t covering = ceil_div(forwarded_count(page_size) + pending.size(), page_size);
    return covering > lines.size() ? covering - lines.size() : 0;
}

void sequence::requeue() {
    pending.insert(pending.begin(), tail.begin(), tail.end());
    tail.clear();
}

batch::batch(std::span<sequence* const> running, std::size_t page_size, std::size_t prefill_tokens_per_step) {
    for (sequence* seq : running) {
        assert(!seq->pending.empty());
        if (seq->pending.size() == 1) {
            entries.push_back(entry{.seq = seq, .token_count = 1, .samples = seq->may_sample()});
        }
    }
    decode_count = entries.size();
    std::size_t budget = prefill_tokens_per_step;
    for (sequence* seq : running) {
        if (budget == 0) {
            break;
        }
        if (seq->pending.size() > 1) {
            std::size_t count = std::min(seq->pending.size(), budget);
            budget -= count;
            entries.push_back(
                entry{.seq = seq, .token_count = count, .samples = count == seq->pending.size() && seq->may_sample()}
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
    return g;
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
