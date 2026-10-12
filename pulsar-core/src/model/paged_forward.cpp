#include "pulsar/model/paged_forward.hpp"

#include "pulsar/ops.hpp"  // rope_cuda, write_kv_cuda, attn_prefill/decode_cuda

#include <ATen/ATen.h>

namespace pulsar {

at::Tensor rope_q(const at::Tensor& q_in, const at::Tensor& pos, int64_t n_q_heads, int64_t head_dim, double theta) {
    const int64_t ntok = q_in.size(0);
    // rope_cuda requires [heads, seq, head_dim].
    auto q = q_in.view({ntok, n_q_heads, head_dim}).transpose(0, 1).contiguous();
    q = rope_cuda(q, pos, theta);
    return q.transpose(0, 1).contiguous();  // transpose to paged layout [ntok, n_q_heads, head_dim]
}

at::Tensor paged_attention(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const kv_lines& kv,
    int64_t layer,
    const GroupBatch& g,
    double scale,
    double rope_theta,
    const at::Tensor& lse_capture
) {
    auto k_pool = kv.k.select(0, layer);
    auto v_pool = kv.v.select(0, layer);
    write_kv_cuda(k_pool, v_pool, k, v, g.slot_mapping);
    const auto lse = lse_capture.defined() ? std::optional<at::Tensor>(lse_capture) : std::nullopt;
    if (g.is_prefill()) {
        return attn_prefill_cuda(
            q,
            k_pool,
            v_pool,
            g.page_tables,
            g.cu_seqlens_q,
            g.seqlens_k,
            g.rope_layout,
            rope_theta,
            scale,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            g.mass_length_gain,
            lse
        );
    }
    return attn_decode_cuda(
        q,
        k_pool,
        v_pool,
        g.page_tables,
        g.context_lens,
        g.rope_layout,
        rope_theta,
        scale,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        g.mass_length_gain,
        lse
    );
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
    return head_fn(x);
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

}  // namespace pulsar
