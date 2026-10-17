#include "pulsarlm/model/paged_forward.hpp"

#include "pulsarlm/ops.hpp"  // rope_cuda, write_kv_cuda, attn_prefill/decode_cuda

#include <ATen/ATen.h>

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
    auto q = q_in.view({ntok, n_q_heads, head_dim}).transpose(0, 1).contiguous();
    q = rope_cuda(q, pos, rope);
    return q.transpose(0, 1).contiguous();  // transpose to paged layout [ntok, n_q_heads, head_dim]
}

at::Tensor paged_attention(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const kv& lines,
    std::size_t kv_layer,
    const GroupBatch& g,
    double scale,
    const rope_parameters& rope,
    std::optional<int64_t> sliding_window,
    const std::optional<at::Tensor>& k_norm_weight,
    const at::Tensor& lse_capture
) {
    const at::Tensor& k_pool = lines.k;
    const at::Tensor& v_pool = lines.v;
    const auto layer = static_cast<int64_t>(kv_layer);
    at::Tensor page_tables = g.page_tables.select(0, layer);
    at::Tensor key_counts = g.key_counts.select(0, layer);
    if (k.defined()) {
        write_kv_cuda(
            k_pool,
            v_pool,
            k,
            v,
            page_tables,
            g.is_prefill() ? std::optional<at::Tensor>(g.cu_seqlens_q) : std::nullopt,
            key_counts
        );
    }
    const auto lse = lse_capture.defined() ? std::optional<at::Tensor>(lse_capture) : std::nullopt;
    std::optional<at::Tensor> mass, cu_view_pages, mass_decay;
    at::Tensor page_mass = g.page_mass.empty() ? at::Tensor() : g.page_mass[kv_layer];
    if (page_mass.defined()) {
        mass = at::zeros({page_mass.size(0), k_pool.size(1), q.size(1)}, q.options().dtype(at::kFloat));
        cu_view_pages = g.cu_view_pages.select(0, layer);
        mass_decay = g.attention_mass_decay;
    }
    at::Tensor o = g.is_prefill() ? attn_prefill_cuda(
                                        q,
                                        k_pool,
                                        v_pool,
                                        page_tables,
                                        g.cu_seqlens_q,
                                        key_counts,
                                        g.rope_layout,
                                        rope,
                                        scale,
                                        sliding_window,
                                        k_norm_weight,
                                        mass,
                                        cu_view_pages,
                                        mass_decay,
                                        g.mass_length_gain,
                                        lse
                                    )
                                  : attn_decode_cuda(
                                        q,
                                        k_pool,
                                        v_pool,
                                        page_tables,
                                        key_counts,
                                        g.rope_layout,
                                        rope,
                                        scale,
                                        sliding_window,
                                        k_norm_weight,
                                        mass,
                                        cu_view_pages,
                                        mass_decay,
                                        g.mass_length_gain,
                                        lse
                                    );
    if (mass) {
        at::maximum_out(page_mass, page_mass, mass->mean(-1).amax(-1));
    }
    return o;
}

at::Tensor routed_experts(
    const at::Tensor& h,
    const at::Tensor& top_experts,
    const at::Tensor& top_weights,
    const at::Tensor& gate_up_proj,
    const at::Tensor& down_proj,
    const std::function<at::Tensor(const at::Tensor&)>& activation
) {
    const int64_t n_experts = gate_up_proj.size(0);
    const int64_t top_k = top_experts.size(-1);
    at::Tensor assigned_experts = top_experts.flatten();
    at::Tensor order = at::argsort(assigned_experts, /*stable=*/true);
    at::Tensor assigned_tokens = at::floor_divide(order, top_k);
    at::Tensor counts = at::zeros({n_experts}, h.options().dtype(at::kInt))
                            .index_add_(0, assigned_experts.index_select(0, order), at::ones_like(order, at::kInt));
    at::Tensor offsets = at::cumsum(counts, 0, at::kInt);

    at::Tensor rows = h.index_select(0, assigned_tokens);
    std::vector<at::Tensor> gate_up = at::_grouped_mm(rows, gate_up_proj.transpose(1, 2), offsets).chunk(2, -1);
    at::Tensor down = at::_grouped_mm(activation(gate_up[0]) * gate_up[1], down_proj.transpose(1, 2), offsets);
    at::Tensor weighted = down * top_weights.flatten().index_select(0, order).unsqueeze(1).to(down.scalar_type());
    return at::zeros_like(h).index_add_(0, assigned_tokens, weighted);
}

at::Tensor logit_inputs(const at::Tensor& x, const GroupBatch& g) {
    return g.logit_rows.defined() ? x.index_select(0, g.logit_rows) : x;
}

at::Tensor paged_group_forward(
    const at::Tensor& embed,
    const GroupBatch& g,
    int64_t n_layers,
    const std::function<at::Tensor(at::Tensor, int64_t)>& layer_fn,
    const std::function<at::Tensor(at::Tensor)>& head_fn
) {
    auto x = embed.index_select(0, g.tokens);  // [ntok, hidden]
    for (int64_t layer = 0; layer < n_layers; ++layer) {
        x = layer_fn(x, layer);
    }
    return head_fn(logit_inputs(x, g));
}

StepLogits paged_forward(const StepBatch& batch, const std::function<at::Tensor(const GroupBatch&)>& group_fn) {
    StepLogits out;
    if (batch.prefill.has_value()) {
        out.prefill = group_fn(*batch.prefill);
    }
    if (batch.decode.has_value()) {
        out.decode = group_fn(*batch.decode);
    }
    return out;
}

}  // namespace pulsarlm
