#include "pulsarlm/engine/model/paged_forward.hpp"

#include "pulsarlm/kernels/ops.hpp"

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
    const at::Tensor& k_pool = lines.k;
    const at::Tensor& v_pool = lines.v;
    const auto layer = static_cast<int64_t>(kv_layer);
    at::Tensor page_tables = g.page_tables.select(0, layer);
    at::Tensor page_positions = g.page_positions.select(0, layer);
    at::Tensor key_counts = g.key_counts.select(0, layer);
    if (k.defined()) {
        write_kv_cuda(
            k_pool,
            v_pool,
            k,
            v,
            page_tables,
            g.kind != model::batch_kind::decode ? std::optional<at::Tensor>(g.cu_seqlens_q) : std::nullopt,
            key_counts
        );
    }
    std::optional<at::Tensor> mass_sum, weight_sum, cu_view_pages, query_weights;
    at::Tensor reader_columns;
    if (!g.page_mass_sum.empty() && g.page_mass_sum[kv_layer].defined()) {
        const int64_t heads = q.size(1);
        reader_columns = g.page_mass_sum[kv_layer].narrow(1, static_cast<int64_t>(reader) * heads, heads);
        mass_sum = reader_columns.is_contiguous() ? reader_columns : at::zeros_like(reader_columns);
        const at::Tensor& page_weight_sum = g.page_weight_sum[kv_layer];
        weight_sum = reader == 0 ? page_weight_sum : at::zeros_like(page_weight_sum);
        cu_view_pages = g.cu_view_pages.select(0, layer);
        query_weights = g.query_weights;
    }
    at::Tensor o;
    if (g.kind == model::batch_kind::verify && mass_sum) {
        o = attn_prefill_verify_cuda(
            q,
            k_pool,
            v_pool,
            page_tables,
            g.cu_seqlens_q,
            key_counts,
            page_positions,
            rope,
            scale,
            sliding_window,
            k_norm_weight,
            *mass_sum,
            *weight_sum,
            *cu_view_pages,
            *query_weights
        );
    } else if (g.kind != model::batch_kind::decode) {
        o = attn_prefill_cuda(
            q,
            k_pool,
            v_pool,
            page_tables,
            g.cu_seqlens_q,
            key_counts,
            page_positions,
            rope,
            scale,
            sliding_window,
            k_norm_weight,
            mass_sum,
            weight_sum,
            cu_view_pages,
            query_weights
        );
    } else {
        o = attn_decode_cuda(
            q,
            k_pool,
            v_pool,
            page_tables,
            key_counts,
            page_positions,
            rope,
            scale,
            sliding_window,
            k_norm_weight,
            mass_sum,
            weight_sum,
            cu_view_pages,
            query_weights
        );
    }
    if (mass_sum && !mass_sum->is_same(reader_columns)) {
        reader_columns.copy_(*mass_sum);
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
    if (g.kind == model::batch_kind::decode) {
        at::Tensor lines = page_tables.gather(1, at::floor_divide(key_counts - 1, page_size).unsqueeze(1)).squeeze(1);
        line.last_hidden.index_copy_(0, lines, hidden);
        return;
    }
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

state_rows recurrent_rows(const model::batch& g, std::size_t kv_layer, std::int64_t slots) {
    return state_rows{
        .lines = g.page_tables.select(0, static_cast<std::int64_t>(kv_layer)),
        .slots = slots,
        .history_positions = g.history_positions,
        .rejected_tokens = g.rejected_tokens,
        .page_size = g.page_size,
    };
}

}  // namespace

state_rows state_layer_rows(const model::batch& g, std::size_t kv_layer) {
    return recurrent_rows(g, kv_layer, g.state_slots);
}

state_rows window_layer_rows(const model::batch& g, std::size_t kv_layer) {
    return recurrent_rows(g, kv_layer, 1);
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
    const state_rows window_rows = window_layer_rows(b, conv_kv_layer);
    const state_rows state_rows = state_layer_rows(b, state_kv_layer);
    if (b.kind == model::batch_kind::verify) {
        const std::int64_t ntok = projected.qkv.size(0);
        const std::int64_t sequences = window_rows.lines.size(0);
        assert(sequences > 0 && ntok > 0 && ntok % sequences == 0);
        const std::int64_t per_sequence = ntok / sequences;
        at::Tensor convolved = causal_conv_verify_cuda(
            projected.qkv,
            weights.conv,
            conv_lines.k,
            window_rows,
            per_sequence,
            1
        );
        return gated_delta_rule_verify_cuda(
            convolved,
            projected.a,
            projected.b,
            weights.dt_bias,
            weights.neg_a,
            state_lines.k,
            state_rows,
            per_sequence,
            key_heads,
            projected.z,
            weights.norm,
            rms_norm_eps,
            activation,
            b.arrival_counters
        );
    }
    std::optional<at::Tensor> cu_seqlens_q;
    if (b.kind != model::batch_kind::decode) {
        cu_seqlens_q = b.cu_seqlens_q;
    }
    at::Tensor convolved = causal_conv_cuda(projected.qkv, weights.conv, conv_lines.k, window_rows, cu_seqlens_q, 1);
    return gated_delta_rule_cuda(
        convolved,
        projected.a,
        projected.b,
        weights.dt_bias,
        weights.neg_a,
        state_lines.k,
        state_rows,
        cu_seqlens_q,
        key_heads,
        projected.z,
        weights.norm,
        rms_norm_eps,
        activation,
        b.arrival_counters
    );
}

at::Tensor logit_inputs(const at::Tensor& x, const model::batch& g) {
    return g.logit_rows.defined() ? x.index_select(0, g.logit_rows) : x;
}

void check_batch(const model::batch& b) {
    assert(b.cu_seqlens_q.defined() == (b.kind != model::batch_kind::decode));
}

}  // namespace pulsarlm
