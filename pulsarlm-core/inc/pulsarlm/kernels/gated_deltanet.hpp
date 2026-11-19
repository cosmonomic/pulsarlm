#pragma once

#include "pulsarlm/kernels/gate_activation.hpp"

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <optional>

namespace pulsarlm {

// The Gated DeltaNet ops run a ragged batch: sequence s holds tokens [cu_seqlens_q[s], cu_seqlens_q[s + 1]) of x, or
// token s alone without cu_seqlens_q; a verify op's batch holds tokens_per_sequence tokens of each sequence in turn. The
// state after the batch's last token of a sequence goes to its last slot, the state after its newest page end to that
// page's boundary line; a verify op of n tokens per sequence also writes the state after each token t into slot slots - n
// + t. A pool's
// lines lie any number of elements apart, a state pool's lines 16-byte aligned.

// Where a ragged batch's sequences keep their recurrent state. Per sequence, lines [num_seqs, >= slots + 2] int32 holds
// slots lines holding states, then the boundary lines of its even and odd pages. history_positions (int64 [num_seqs])
// is the history position of each sequence's first token in the batch, and rejected_tokens (int32 [num_seqs]) the
// tokens its previous step forwarded and did not accept. A sequence continues from slot slots - 1 - min(rejected_tokens[s],
// slots - 1): a layer keeping a state per token has a slot per token a step may not accept and one more, a layer keeping
// a window of inputs one slot whose window it reads rejected_tokens[s] inputs earlier. A token at history position p
// ends a page when p + 1 is a multiple of page_size, and the state after it goes to the boundary line of page p /
// page_size, alternating between the two.
struct state_rows {
    at::Tensor lines;
    std::int64_t slots;
    at::Tensor history_positions;
    at::Tensor rejected_tokens;
    std::int64_t page_size;
};

// silu of the causal depthwise convolution of x [tokens, channels], whose rows may lie apart, by weight [channels, 1,
// kernel] with its taps dilation tokens apart, contiguous in x's dtype. A line of conv_pool [lines, channels, window]
// holds the last window inputs before a sequence's next token, window at least (kernel - 1) * dilation and the extra
// ones the inputs of up to slots - 1 tokens a step does not accept: the convolution reads those that end window -
// rejected_tokens[s] entries in. rows.slots is 1.
at::Tensor causal_conv_cuda(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_pool,
    const state_rows& rows,
    const std::optional<at::Tensor>& cu_seqlens_q,
    std::int64_t dilation
);

// causal_conv_cuda over tokens_per_sequence tokens of each sequence; the line keeps the inputs after the last.
at::Tensor causal_conv_verify_cuda(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_pool,
    const state_rows& rows,
    std::int64_t tokens_per_sequence,
    std::int64_t dilation
);

// The gated delta rule's gated outputs [tokens, value heads * value dim] in x's dtype. Each row of x [tokens, 2 *
// key_heads * key dim + value heads * value dim] holds the token's query heads, key heads and value heads; value head h
// reads key head h / (value heads / key_heads). Queries and keys are L2-normalized, with 1e-6 added to the sum of
// squares, and queries scaled by 1 / sqrt(key dim). x is contiguous. a and b are [tokens, value heads] in x's dtype,
// their rows equally far apart: the state decays by exp(neg_a * softplus(a + dt_bias)) and learns at sigmoid(b),
// rounded to x's dtype; dt_bias and neg_a are fp32 [value heads]. A line of state_pool, fp32 [lines, value heads, key
// dim, value dim], holds a sequence's state. Each value head's output o, rounded to x's dtype, leaves as norm_weight *
// RMSNorm(o) * activation(z), the normalized o and its product with norm_weight each rounded to x's dtype; z is
// [tokens, value heads * value dim] and norm_weight [value dim], both in x's dtype, each row of z contiguous.
at::Tensor gated_delta_rule_cuda(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& dt_bias,
    const at::Tensor& neg_a,
    const at::Tensor& state_pool,
    const state_rows& rows,
    const std::optional<at::Tensor>& cu_seqlens_q,
    int64_t key_heads,
    const at::Tensor& z,
    const at::Tensor& norm_weight,
    double eps,
    gate_activation activation,
    const at::Tensor& counters
);

// gated_delta_rule_cuda over tokens_per_sequence tokens of each sequence, at most rows.slots, writing the state after
// each token t into slot rows.slots - tokens_per_sequence + t.
at::Tensor gated_delta_rule_verify_cuda(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& dt_bias,
    const at::Tensor& neg_a,
    const at::Tensor& state_pool,
    const state_rows& rows,
    int64_t tokens_per_sequence,
    int64_t key_heads,
    const at::Tensor& z,
    const at::Tensor& norm_weight,
    double eps,
    gate_activation activation,
    const at::Tensor& counters
);

// The arrival counters (arrival_counters.cuh) gated_delta_rule_cuda and gated_delta_rule_verify_cuda need for num_seqs
// sequences.
int64_t gated_delta_rule_counters(int64_t num_seqs, int64_t value_heads);

}  // namespace pulsarlm
