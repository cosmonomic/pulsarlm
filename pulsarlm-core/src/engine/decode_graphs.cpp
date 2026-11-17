#include "pulsarlm/engine/decode_graphs.hpp"

#include <ATen/ATen.h>

#include <cstddef>
#include <optional>
#include <utility>

namespace pulsarlm {

namespace {

void copy_defined(const at::Tensor& to, const at::Tensor& from) {
    if (from.defined()) {
        to.copy_(from);
    }
}

at::Tensor empty_like_defined(const at::Tensor& t) {
    return t.defined() ? at::empty_like(t) : t;
}

}  // namespace

decode_graphs::decode_graphs(model& shard_model, std::span<const kv> lines)
: shard_model(shard_model),
  lines(lines.begin(), lines.end()) {}

std::unique_ptr<decode_graphs::captured_forward> decode_graphs::capture(const group_batch& g) {
    auto captured = std::make_unique<captured_forward>();
    group_batch& inputs = captured->inputs;
    inputs = group_batch{
        .tokens = at::empty_like(g.tokens),
        .pos = at::empty_like(g.pos),
        .page_tables = at::empty_like(g.page_tables),
        .page_end_tokens = at::empty_like(g.page_end_tokens),
        .key_counts = at::empty_like(g.key_counts),
        .page_positions = at::empty_like(g.page_positions),
        .cu_view_pages = empty_like_defined(g.cu_view_pages),
        .query_weights = empty_like_defined(g.query_weights),
    };
    for (std::size_t layer = 0; layer < g.page_mass_sum.size(); layer++) {
        inputs.page_mass_sum.push_back(empty_like_defined(g.page_mass_sum[layer]));
        inputs.page_weight_sum.push_back(empty_like_defined(g.page_weight_sum[layer]));
    }
    // Thread-local capture leaves the other GPUs' threads free to launch while this one captures.
    captured->graph.capture_begin({0, 0}, cudaStreamCaptureModeThreadLocal);
    for (std::size_t layer = 0; layer < inputs.page_mass_sum.size(); layer++) {
        if (inputs.page_mass_sum[layer].defined()) {
            inputs.page_mass_sum[layer].zero_();
            inputs.page_weight_sum[layer].zero_();
        }
    }
    captured->logits = *shard_model.forward(step_batch{.decode_group = inputs}, lines).decode_logits;
    captured->graph.capture_end();
    return captured;
}

at::Tensor decode_graphs::forward(const group_batch& g) {
    shape group_shape{g.page_tables.size(1), g.page_tables.size(2), g.cu_view_pages.defined()};
    std::unique_ptr<captured_forward>& captured = forwards[group_shape];
    if (!captured) {
        captured = capture(g);
    }
    group_batch& inputs = captured->inputs;
    inputs.tokens.copy_(g.tokens);
    inputs.pos.copy_(g.pos);
    inputs.page_tables.copy_(g.page_tables);
    inputs.page_end_tokens.copy_(g.page_end_tokens);
    inputs.key_counts.copy_(g.key_counts);
    inputs.page_positions.copy_(g.page_positions);
    copy_defined(inputs.cu_view_pages, g.cu_view_pages);
    copy_defined(inputs.query_weights, g.query_weights);
    captured->graph.replay();
    for (std::size_t layer = 0; layer < g.page_mass_sum.size(); layer++) {
        copy_defined(g.page_mass_sum[layer], inputs.page_mass_sum[layer]);
        copy_defined(g.page_weight_sum[layer], inputs.page_weight_sum[layer]);
    }
    return captured->logits;
}

}  // namespace pulsarlm
