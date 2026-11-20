#include "request_table_kernels.hpp"

#include "pulsarlm/kernels/device.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cassert>
#include <cstdint>

namespace pulsarlm {
namespace {

constexpr int cta_threads = 128;

struct table_entries {
    const std::int64_t* history_positions;
    const std::int64_t* rope_positions;
    const std::int32_t* key_counts;
    const std::int32_t* rejected_tokens;
    const std::int64_t* tokens;
    std::int64_t token_width;
    const std::int64_t* mass_start_positions;
    const double* attention_mass_decays;
    const std::int32_t* page_tables;
    const std::int32_t* page_positions;
    std::int64_t layers;
    std::int64_t capacity;
};

struct group_outputs {
    // Null for a prefill group, whose tokens the host sends.
    std::int64_t* tokens;
    std::int64_t* pos;
    std::int64_t* history_positions;
    std::int32_t* rejected_tokens;
    std::int32_t* key_counts;
    std::int32_t* page_tables;
    std::int32_t* page_positions;
    std::int64_t width;
    // Null for a group that measures no mass.
    float* query_weights;
    bool weighs_verified_rows;
};

// One CTA per request of the group. Without query_starts, request s computes token s alone.
__global__ void gather_group_inputs_kernel(
    const std::int64_t* __restrict__ slots,
    std::int64_t requests,
    const std::int32_t* __restrict__ query_starts,
    table_entries table,
    group_outputs out
) {
    const std::int64_t s = blockIdx.x;
    const std::int64_t slot = slots[s];
    const std::int64_t first = query_starts == nullptr ? s : query_starts[s];
    const std::int64_t count = query_starts == nullptr ? 1 : query_starts[s + 1] - first;
    const std::int64_t history_position = table.history_positions[slot];
    if (threadIdx.x == 0) {
        out.history_positions[s] = history_position;
        out.rejected_tokens[s] = table.rejected_tokens[slot];
    }
    for (std::int64_t layer = threadIdx.x; layer < table.layers; layer += blockDim.x) {
        out.key_counts[layer * requests + s] = table.key_counts[slot * table.layers + layer] +
            static_cast<std::int32_t>(count);
    }
    const std::int64_t mass_start = table.mass_start_positions[slot];
    const double decay = table.attention_mass_decays[slot];
    for (std::int64_t i = threadIdx.x; i < count; i += blockDim.x) {
        out.pos[first + i] = table.rope_positions[slot] + i;
        if (out.tokens != nullptr) {
            out.tokens[first + i] = table.tokens[slot * table.token_width + i];
        }
        if (out.query_weights != nullptr) {
            const bool measured = history_position + i >= mass_start;
            if (out.weighs_verified_rows) {
                out.query_weights[first + i] = measured ? 1.0f : 0.0f;
            } else {
                const auto age = static_cast<double>(count - 1 - i);
                out.query_weights[first + i] = measured ? static_cast<float>(decay * pow(1.0 - decay, age)) : 0.0f;
            }
        }
    }
    for (std::int64_t cell = threadIdx.x; cell < table.layers * out.width; cell += blockDim.x) {
        const std::int64_t layer = cell / out.width;
        const std::int64_t column = cell % out.width;
        const std::int64_t source = (slot * table.layers + layer) * table.capacity + column;
        const std::int64_t target = (layer * requests + s) * out.width + column;
        out.page_tables[target] = table.page_tables[source];
        out.page_positions[target] = table.page_positions[source];
    }
}

__global__ void advance_requests_kernel(
    const std::int64_t* __restrict__ slots,
    std::int64_t entries,
    const std::int64_t* __restrict__ computed_token_counts,
    const std::int64_t* __restrict__ rejected_token_counts,
    const std::int64_t* __restrict__ next_tokens,
    std::int64_t* history_positions,
    std::int64_t* rope_positions,
    std::int32_t* key_counts,
    std::int64_t layers,
    const bool* __restrict__ recurrent_layers,
    std::int64_t page_size,
    std::int32_t* rejected_tokens,
    std::int64_t* tokens,
    std::int64_t token_width,
    const std::int64_t* __restrict__ cycle_end_positions,
    bool* __restrict__ ends_cycle
) {
    const std::int64_t e = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= entries) {
        return;
    }
    const std::int64_t slot = slots[e];
    const std::int64_t count = computed_token_counts[e];
    const std::int64_t history_position = history_positions[slot] + count;
    history_positions[slot] = history_position;
    rope_positions[slot] += count;
    for (std::int64_t layer = 0; layer < layers; layer++) {
        std::int32_t& key_count = key_counts[slot * layers + layer];
        key_count = recurrent_layers[layer] ? static_cast<std::int32_t>(history_position % page_size)
                                            : key_count + static_cast<std::int32_t>(count);
    }
    rejected_tokens[slot] = static_cast<std::int32_t>(rejected_token_counts[e]);
    for (std::int64_t column = 0; column < token_width; column++) {
        tokens[slot * token_width + column] = next_tokens[e * token_width + column];
    }
    ends_cycle[e] = history_position >= cycle_end_positions[slot];
}

}  // namespace

void gather_group_inputs(
    const request_table& table,
    const at::Tensor& request_slots,
    std::size_t table_width,
    model::batch& inputs
) {
    const at::TensorOptions options = table.history_positions.options();
    const std::int64_t requests = request_slots.size(0);
    const std::int64_t layers = table.key_counts.size(1);
    const auto width = static_cast<std::int64_t>(table_width);
    const bool verifies = inputs.kind == model::batch_kind::verify;
    std::int64_t tokens = requests;
    if (verifies) {
        tokens = requests * inputs.state_slots;
    } else if (inputs.kind == model::batch_kind::prefill) {
        tokens = inputs.tokens.size(0);
    }
    assert(request_slots.scalar_type() == at::kLong && request_slots.is_contiguous());
    assert(inputs.cu_seqlens_q.defined() == (inputs.kind != model::batch_kind::decode));
    assert(width <= table.page_tables.size(2) && (!verifies || inputs.state_slots <= table.tokens.size(1)));
    if (inputs.kind != model::batch_kind::prefill) {
        inputs.tokens = at::empty({tokens}, options.dtype(at::kLong));
    }
    inputs.pos = at::empty({tokens}, options.dtype(at::kLong));
    inputs.history_positions = at::empty({requests}, options.dtype(at::kLong));
    inputs.rejected_tokens = at::empty({requests}, options.dtype(at::kInt));
    inputs.key_counts = at::empty({layers, requests}, options.dtype(at::kInt));
    inputs.page_tables = at::empty({layers, requests, width}, options.dtype(at::kInt));
    inputs.page_positions = at::empty({layers, requests, width}, options.dtype(at::kInt));
    if (inputs.cu_view_pages.defined()) {
        inputs.query_weights = at::empty({tokens}, options.dtype(at::kFloat));
    }
    if (requests == 0) {
        return;
    }
    const table_entries entries{
        .history_positions = table.history_positions.const_data_ptr<std::int64_t>(),
        .rope_positions = table.rope_positions.const_data_ptr<std::int64_t>(),
        .key_counts = table.key_counts.const_data_ptr<std::int32_t>(),
        .rejected_tokens = table.rejected_tokens.const_data_ptr<std::int32_t>(),
        .tokens = table.tokens.const_data_ptr<std::int64_t>(),
        .token_width = table.tokens.size(1),
        .mass_start_positions = table.mass_start_positions.const_data_ptr<std::int64_t>(),
        .attention_mass_decays = table.attention_mass_decays.const_data_ptr<double>(),
        .page_tables = table.page_tables.const_data_ptr<std::int32_t>(),
        .page_positions = table.page_positions.const_data_ptr<std::int32_t>(),
        .layers = layers,
        .capacity = table.page_tables.size(2),
    };
    const group_outputs outputs{
        .tokens = inputs.kind == model::batch_kind::prefill ? nullptr : inputs.tokens.mutable_data_ptr<std::int64_t>(),
        .pos = inputs.pos.mutable_data_ptr<std::int64_t>(),
        .history_positions = inputs.history_positions.mutable_data_ptr<std::int64_t>(),
        .rejected_tokens = inputs.rejected_tokens.mutable_data_ptr<std::int32_t>(),
        .key_counts = inputs.key_counts.mutable_data_ptr<std::int32_t>(),
        .page_tables = inputs.page_tables.mutable_data_ptr<std::int32_t>(),
        .page_positions = inputs.page_positions.mutable_data_ptr<std::int32_t>(),
        .width = width,
        .query_weights = inputs.query_weights.defined() ? inputs.query_weights.mutable_data_ptr<float>() : nullptr,
        .weighs_verified_rows = verifies,
    };
    gather_group_inputs_kernel<<<static_cast<unsigned>(requests), cta_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        request_slots.const_data_ptr<std::int64_t>(),
        requests,
        inputs.cu_seqlens_q.defined() ? inputs.cu_seqlens_q.const_data_ptr<std::int32_t>() : nullptr,
        entries,
        outputs
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void advance_requests(
    request_table& table,
    const at::Tensor& request_slots,
    const at::Tensor& computed_token_counts,
    const at::Tensor& rejected_token_counts,
    const at::Tensor& next_tokens,
    at::Tensor& ends_cycle
) {
    const std::int64_t entries = request_slots.size(0);
    assert(
        request_slots.scalar_type() == at::kLong && computed_token_counts.scalar_type() == at::kLong &&
        rejected_token_counts.scalar_type() == at::kLong && next_tokens.scalar_type() == at::kLong &&
        ends_cycle.scalar_type() == at::kBool
    );
    assert(
        computed_token_counts.size(0) == entries && rejected_token_counts.size(0) == entries &&
        next_tokens.size(0) == entries && next_tokens.size(1) == table.tokens.size(1) && ends_cycle.size(0) == entries
    );
    assert(
        request_slots.is_contiguous() && computed_token_counts.is_contiguous() &&
        rejected_token_counts.is_contiguous() && next_tokens.is_contiguous() && ends_cycle.is_contiguous()
    );
    if (entries == 0) {
        return;
    }
    advance_requests_kernel<<<cta_count(entries, cta_threads), cta_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        request_slots.const_data_ptr<std::int64_t>(),
        entries,
        computed_token_counts.const_data_ptr<std::int64_t>(),
        rejected_token_counts.const_data_ptr<std::int64_t>(),
        next_tokens.const_data_ptr<std::int64_t>(),
        table.history_positions.mutable_data_ptr<std::int64_t>(),
        table.rope_positions.mutable_data_ptr<std::int64_t>(),
        table.key_counts.mutable_data_ptr<std::int32_t>(),
        table.key_counts.size(1),
        table.recurrent_layers.const_data_ptr<bool>(),
        table.page_size,
        table.rejected_tokens.mutable_data_ptr<std::int32_t>(),
        table.tokens.mutable_data_ptr<std::int64_t>(),
        table.tokens.size(1),
        table.cycle_end_positions.const_data_ptr<std::int64_t>(),
        ends_cycle.mutable_data_ptr<bool>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace pulsarlm
