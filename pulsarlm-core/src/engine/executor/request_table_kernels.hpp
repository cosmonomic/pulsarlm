#pragma once

#include "pulsarlm/engine/model/interface.hpp"
#include "pulsarlm/engine/request_table.hpp"

#include <ATen/core/Tensor.h>

#include <cstddef>

namespace pulsarlm {

// Fills inputs, a group of the requests at request_slots (int64 on table's device), from their entries of table on the
// current stream: a decode or verify group's tokens, pos, history_positions, rejected_tokens, key_counts with the
// group's tokens of each request, page_tables and page_positions table_width columns wide, and query_weights when
// inputs.cu_view_pages is defined. A verify group's queries weigh 1 from mass_start_positions on, a decode or prefill
// group's from there as query_weights weighs them. inputs holds its kind, its cu_seqlens_q unless it decodes, and a
// prefill group's tokens.
void gather_group_inputs(
    const request_table& table,
    const at::Tensor& request_slots,
    std::size_t table_width,
    model::batch& inputs
);

// After a step, on the current stream: adds each request's computed_token_counts (int64 [entries]) to its history
// position, RoPE position and the key counts of its KV layers with pages, sets a recurrent layer's key count to the new
// history position's offset in its page, sets its rejected tokens to rejected_token_counts (int64 [entries]), its
// next token and drafts to its row of next_tokens (int64 [entries, table.tokens.size(1)]), and writes into ends_cycle
// (bool [entries]) whether its history position reached its cycle end. The requests at request_slots (int64) are
// distinct.
void advance_requests(
    request_table& table,
    const at::Tensor& request_slots,
    const at::Tensor& computed_token_counts,
    const at::Tensor& rejected_token_counts,
    const at::Tensor& next_tokens,
    at::Tensor& ends_cycle
);

}  // namespace pulsarlm
