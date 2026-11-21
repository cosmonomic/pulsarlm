#pragma once

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <tuple>

namespace pulsarlm::ops::cuda {

// Each stream's RMSNorm of x as ops::cuda::stream_rmsnorm computes it, and that normed x times weight^T [out,
// n_streams * hidden], in x's dtype: (normed, projected). Batches of up to 4 tokens run as one kernel.
std::tuple<at::Tensor, at::Tensor> stream_rmsnorm_linear(
    const at::Tensor& x,
    const at::Tensor& norm_weight,
    int64_t n_streams,
    double eps,
    const at::Tensor& weight
);

// x [tokens, hc * hidden] plus, in each stream, out [tokens, hidden] times its injection weight [tokens, hc], as
// ops::cuda::hyper_connection_inject computes it, then ops::cuda::stream_rmsnorm_linear of that sum: (sum, normed,
// projected). Batches of up to 4 tokens run as one kernel.
std::tuple<at::Tensor, at::Tensor, at::Tensor> injected_stream_rmsnorm_linear(
    const at::Tensor& x,
    const at::Tensor& out,
    const at::Tensor& injection_weights,
    const at::Tensor& norm_weight,
    int64_t n_streams,
    double eps,
    const at::Tensor& weight
);

// A hyper-connection's block input and injection weights from its normed streams [tokens, hc * hidden], its low-rank
// mix logits [tokens, rank], the up projection [hc * hidden, rank] and the injection logits [tokens, hc], whose rows
// may lie apart: with gate logits silu(mix_logits / hc) times up^T, the mean over streams of sigmoid(gate) times the
// stream [tokens, hidden], and 2 * sigmoid(inject_logits / hc) [tokens, hc], each operation rounded to the dtype as the
// ATen ops round it. Batches of up to 4 tokens run as one kernel.
std::tuple<at::Tensor, at::Tensor> hyper_connection_mix(
    const at::Tensor& normed,
    const at::Tensor& mix_logits,
    const at::Tensor& up_weight,
    const at::Tensor& inject_logits
);

// x [tokens, hc * hidden] plus, in each stream, the block output out [tokens, hidden] times that stream's injection
// weight [tokens, hc], the product rounded to the dtype.
at::Tensor hyper_connection_inject(const at::Tensor& x, const at::Tensor& out, const at::Tensor& injection_weights);

}  // namespace pulsarlm::ops::cuda
