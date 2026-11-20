#pragma once

#include <ATen/core/Tensor.h>

#include <cstdint>

namespace pulsarlm {

// Per running request, at the slot of its request id, the state its next step reads, on one GPU. Each tensor's first
// dimension is the slot, of max_running_requests slots, except recurrent_layers'.
struct request_table {
    // int32 [slots, n_kv_layers, view_page_capacity]: per view page its line and the RoPE position of its first key.
    at::Tensor page_tables;
    at::Tensor page_positions;
    // int64 [slots]: the history position and the RoPE position of the request's next token.
    at::Tensor history_positions;
    at::Tensor rope_positions;
    // int32 [slots, n_kv_layers]: the keys of each KV layer's view before the next token, request_state::next_index.
    // A recurrent layer's is the history position's offset in its page once the steps are written back.
    at::Tensor key_counts;
    // bool [n_kv_layers]: whether each KV layer is recurrent.
    at::Tensor recurrent_layers;
    std::int64_t page_size = 0;
    // int32 [slots]: the tokens the request's last step computed and did not accept (model::batch::rejected_tokens).
    at::Tensor rejected_tokens;
    // int64 [slots, 1 + draft_tokens]: the request's next token, then its drafts when it has them.
    at::Tensor tokens;
    // int64 [slots]: the history position at which the request's eviction cycle ends, and the one from which its
    // queries weigh in the page mass.
    at::Tensor cycle_end_positions;
    at::Tensor mass_start_positions;
    // int64 [slots]: the tokens of the request's eviction cycles, and of the window at the end of each whose queries
    // weigh in the page mass.
    at::Tensor cycle_tokens;
    at::Tensor mass_tokens;
    // fp64 [slots]: the request's attention_mass_decay.
    at::Tensor attention_mass_decays;
    // int64 [slots]: the tokens the request may still emit, below 0 once steps past its end were advanced, and the
    // largest int64 without max_tokens.
    at::Tensor remaining_tokens;
    // int64 [slots]: the seed, and the tokens sampled so far.
    at::Tensor seeds;
    at::Tensor draw_counts;
    // [slots]: the sampling parameters as sampling_inputs holds them.
    at::Tensor temperatures;
    at::Tensor top_k;
    at::Tensor top_p;
};

}  // namespace pulsarlm
