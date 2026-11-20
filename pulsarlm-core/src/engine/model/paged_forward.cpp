#include "pulsarlm/engine/model/paged_forward.hpp"

#include "pulsarlm/kernels/gated_deltanet.hpp"
#include "pulsarlm/kernels/paged_attention.hpp"
#include "pulsarlm/kernels/rope.hpp"

#include <ATen/ATen.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace pulsarlm {

at::Tensor rope_q(
    const at::Tensor& q_in,
    const at::Tensor& pos,
    int64_t n_q_heads,
    int64_t head_dim,
    const rope_parameters& rope
) {
    const int64_t ntok = q_in.size(0);
    // rope_cuda requires [heads, seq, head_dim].
    at::Tensor by_head = q_in.view({ntok, n_q_heads, head_dim}).transpose(0, 1).contiguous();
    return rope_cuda(by_head, pos, rope).transpose(0, 1).contiguous();
}

namespace {

page_views narrow_views(const page_views& views, std::int64_t first, std::int64_t count) {
    return page_views{
        .page_tables = views.page_tables.narrow(0, first, count),
        .page_positions = views.page_positions.narrow(0, first, count),
        .key_counts = views.key_counts.narrow(0, first, count),
    };
}

// The page mass of the query tokens [first_token, first_token + token_count), whose mass views are [first_view,
// first_view + view_count).
std::optional<page_mass> narrow_mass(
    const std::optional<page_mass>& mass,
    std::int64_t first_token,
    std::int64_t token_count,
    std::int64_t first_view,
    std::int64_t view_count
) {
    if (!mass) {
        return std::nullopt;
    }
    return page_mass{
        .mass_sum = mass->mass_sum,
        .weight_sum = mass->weight_sum,
        .cu_view_pages = mass->cu_view_pages.narrow(0, first_view, view_count + 1),
        .query_weights = mass->query_weights.narrow(0, first_token, token_count),
    };
}

}  // namespace

at::Tensor paged_attention(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const kv& lines,
    std::size_t kv_layer,
    std::size_t reader,
    const model::batch& g,
    double scale,
    const rope_parameters& rope,
    std::optional<int64_t> sliding_window,
    const std::optional<at::Tensor>& k_norm_weight
) {
    const kv_pools pools{.keys = lines.k, .values = lines.v};
    const auto layer_row = static_cast<int64_t>(kv_layer);
    const page_views views{
        .page_tables = g.page_tables.select(0, layer_row),
        .page_positions = g.page_positions.select(0, layer_row),
        .key_counts = g.key_counts.select(0, layer_row),
    };
    if (k.defined()) {
        write_kv_cuda(pools, k, v, views, g.cu_seqlens_q);
    }
    const attention_layer layer{
        .rope = rope,
        .scale = scale,
        .sliding_window = sliding_window,
        .k_norm_weight = k_norm_weight,
    };
    std::optional<page_mass> mass;
    at::Tensor reader_columns;
    if (!g.page_mass_sum.empty() && g.page_mass_sum[kv_layer].defined()) {
        const int64_t heads = q.size(1);
        reader_columns = g.page_mass_sum[kv_layer].narrow(1, static_cast<int64_t>(reader) * heads, heads);
        const at::Tensor& page_weight_sum = g.page_weight_sum[kv_layer];
        mass = page_mass{
            .mass_sum = reader_columns.is_contiguous() ? reader_columns : at::zeros_like(reader_columns),
            .weight_sum = reader == 0 ? page_weight_sum : at::zeros_like(page_weight_sum),
            .cu_view_pages = g.cu_view_pages.select(0, layer_row),
            .query_weights = g.query_weights,
        };
    }
    const std::int64_t ntok = q.size(0);
    const std::int64_t requests = views.key_counts.size(0);
    const std::int64_t single_queries = g.single_query_request_count;
    at::Tensor o = at::empty_like(q);
    if (single_queries > 0) {
        attn_decode_cuda(
            q.narrow(0, 0, single_queries),
            pools,
            narrow_views(views, 0, single_queries),
            layer,
            narrow_mass(mass, 0, single_queries, 0, single_queries),
            0,
            o.narrow(0, 0, single_queries)
        );
    }
    if (requests > single_queries) {
        const std::int64_t later_tokens = ntok - single_queries;
        const std::int64_t later_views = mass ? mass->cu_view_pages.size(0) - 1 - single_queries : 0;
        attn_prefill_cuda(
            q.narrow(0, single_queries, later_tokens),
            pools,
            narrow_views(views, single_queries, requests - single_queries),
            g.cu_seqlens_q_from_second_run,
            layer,
            narrow_mass(mass, single_queries, later_tokens, single_queries, later_views),
            mass ? g.per_query_state_request_count : 0,
            o.narrow(0, single_queries, later_tokens)
        );
    }
    if (mass && !mass->mass_sum.is_same(reader_columns)) {
        reader_columns.copy_(mass->mass_sum);
    }
    return o;
}

// The rows that write nothing write their sequence's last row into its line, as that row does, so the writes need no
// host sync.
void keep_last_hidden(const model::batch& g, const at::Tensor& hidden, const kv& line, std::size_t kv_layer) {
    const auto layer_row = static_cast<std::int64_t>(kv_layer);
    const std::int64_t page_size = line.k.size(1);
    const std::int64_t ntok = hidden.size(0);
    at::Tensor key_counts = g.key_counts.select(0, layer_row).to(at::kLong);
    at::Tensor page_tables = g.page_tables.select(0, layer_row).to(at::kLong);
    const std::int64_t num_seqs = key_counts.size(0);
    at::Tensor starts = g.cu_seqlens_q.to(at::kLong);
    at::Tensor counts = starts.narrow(0, 1, num_seqs) - starts.narrow(0, 0, num_seqs);
    at::Tensor sequence_of = at::repeat_interleave(counts, std::optional<std::int64_t>(ntok));
    at::Tensor local = at::arange(ntok, starts.options()) - starts.index_select(0, sequence_of);
    at::Tensor row_counts = counts.index_select(0, sequence_of);
    at::Tensor view_index = key_counts.index_select(0, sequence_of) - row_counts + local;
    at::Tensor last_row = starts.narrow(0, 1, num_seqs).index_select(0, sequence_of) - 1;
    at::Tensor writes = (local == row_counts - 1).logical_or(at::remainder(view_index + 1, page_size) == 0);
    at::Tensor written_row = at::where(writes, at::arange(ntok, starts.options()), last_row);
    at::Tensor written_index = view_index.index_select(0, written_row);
    at::Tensor lines = page_tables.index({sequence_of, at::floor_divide(written_index, page_size)});
    line.last_hidden.index_copy_(0, lines, hidden.index_select(0, written_row));
}

namespace {

state_rows recurrent_rows(
    const model::batch& g,
    std::size_t kv_layer,
    std::int64_t slots,
    std::int64_t first,
    std::int64_t count
) {
    return state_rows{
        .lines = g.page_tables.select(0, static_cast<std::int64_t>(kv_layer)).narrow(0, first, count),
        .slots = slots,
        .history_positions = g.history_positions.narrow(0, first, count),
        .rejected_tokens = g.rejected_tokens.narrow(0, first, count),
        .page_size = g.page_size,
    };
}

}  // namespace

state_rows state_layer_rows(const model::batch& g, std::size_t kv_layer, std::int64_t first, std::int64_t count) {
    return recurrent_rows(g, kv_layer, g.state_slots, first, count);
}

state_rows window_layer_rows(const model::batch& g, std::size_t kv_layer, std::int64_t first, std::int64_t count) {
    return recurrent_rows(g, kv_layer, 1, first, count);
}

std::int64_t count_step_tokens(const model::batch& g) {
    return g.single_query_request_count + g.per_query_state_request_count * g.state_slots;
}

at::Tensor gated_deltanet(
    const model::batch& b,
    const gated_deltanet_inputs& projected,
    const gated_deltanet_weights& weights,
    const kv& state_lines,
    const kv& conv_lines,
    std::size_t state_kv_layer,
    std::size_t conv_kv_layer,
    std::int64_t key_heads,
    double rms_norm_eps,
    gate_activation activation
) {
    const std::int64_t ntok = projected.qkv.size(0);
    const std::int64_t requests = b.history_positions.size(0);
    const std::int64_t step_requests = b.single_query_request_count + b.per_query_state_request_count;
    const std::int64_t step_tokens = count_step_tokens(b);
    at::Tensor convolved = at::empty({ntok, projected.qkv.size(1)}, projected.qkv.options());
    at::Tensor gated = at::empty({ntok, projected.z.size(1)}, projected.qkv.options());
    if (step_requests > 0) {
        const at::Tensor cu_seqlens_q = b.cu_seqlens_q.narrow(0, 0, step_requests + 1);
        causal_conv_step_cuda(
            projected.qkv.narrow(0, 0, step_tokens),
            weights.conv,
            conv_lines.k,
            window_layer_rows(b, conv_kv_layer, 0, step_requests),
            cu_seqlens_q,
            1,
            convolved.narrow(0, 0, step_tokens)
        );
        gated_delta_rule_step_cuda(
            convolved.narrow(0, 0, step_tokens),
            projected.a.narrow(0, 0, step_tokens),
            projected.b.narrow(0, 0, step_tokens),
            weights.dt_bias,
            weights.neg_a,
            state_lines.k,
            state_layer_rows(b, state_kv_layer, 0, step_requests),
            cu_seqlens_q,
            b.kept_state_counts,
            key_heads,
            projected.z.narrow(0, 0, step_tokens),
            weights.norm,
            rms_norm_eps,
            activation,
            b.arrival_counters,
            gated.narrow(0, 0, step_tokens)
        );
    }
    if (requests > step_requests) {
        const std::int64_t chunk_tokens = ntok - step_tokens;
        const std::int64_t chunks = requests - step_requests;
        causal_conv_chunked_cuda(
            projected.qkv.narrow(0, step_tokens, chunk_tokens),
            weights.conv,
            conv_lines.k,
            window_layer_rows(b, conv_kv_layer, step_requests, chunks),
            b.cu_seqlens_q_from_third_run,
            1,
            convolved.narrow(0, step_tokens, chunk_tokens)
        );
        gated_delta_rule_chunked_cuda(
            convolved.narrow(0, step_tokens, chunk_tokens),
            projected.a.narrow(0, step_tokens, chunk_tokens),
            projected.b.narrow(0, step_tokens, chunk_tokens),
            weights.dt_bias,
            weights.neg_a,
            state_lines.k,
            state_layer_rows(b, state_kv_layer, step_requests, chunks),
            b.cu_seqlens_q_from_third_run,
            key_heads,
            projected.z.narrow(0, step_tokens, chunk_tokens),
            weights.norm,
            rms_norm_eps,
            activation,
            b.arrival_counters,
            gated.narrow(0, step_tokens, chunk_tokens)
        );
    }
    return gated;
}

at::Tensor logit_inputs(const at::Tensor& x, const model::batch& g) {
    return x.index_select(0, g.logit_tokens);
}

void check_batch(const model::batch& b) {
    const std::int64_t requests = b.cu_seqlens_q.defined() ? b.cu_seqlens_q.size(0) - 1 : -1;
    assert(
        requests >= 0 && b.single_query_request_count >= 0 && b.per_query_state_request_count >= 0 &&
        b.single_query_request_count + b.per_query_state_request_count <= requests &&
        b.cu_seqlens_q_from_second_run.defined() && b.cu_seqlens_q_from_third_run.defined() && b.logit_tokens.defined()
    );
}

}  // namespace pulsarlm
