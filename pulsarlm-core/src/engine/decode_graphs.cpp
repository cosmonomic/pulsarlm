#include "pulsarlm/engine/decode_graphs.hpp"

#include "pulsarlm/kernels/ops.hpp"

#include <ATen/ATen.h>

#include <algorithm>
#include <cstddef>
#include <utility>

namespace pulsarlm {

namespace {

void copy_defined(const at::Tensor& to, const at::Tensor& from) {
    if (from.defined()) {
        to.copy_(from);
    }
}

}  // namespace

decode_graphs::decode_graphs(
    model& shard_model,
    std::span<const kv> lines,
    std::size_t width,
    std::size_t page_size,
    std::size_t state_slots,
    std::vector<std::size_t> mass_heads,
    at::Tensor samples
)
: shard_model(shard_model),
  lines(lines.begin(), lines.end()),
  samples(std::move(samples)),
  width(static_cast<std::int64_t>(width)),
  page_size(static_cast<std::int64_t>(page_size)),
  state_slots(static_cast<std::int64_t>(state_slots)),
  mass_heads(std::move(mass_heads)) {
    auto floats = this->lines.front().k.options().dtype(at::kFloat);
    const std::int64_t rows = startup_sequences * this->width;
    for (std::size_t heads : this->mass_heads) {
        shared_mass_sum.push_back(
            heads > 0 ? at::empty({rows, static_cast<std::int64_t>(heads)}, floats) : at::Tensor()
        );
        shared_weight_sum.push_back(heads > 0 ? at::empty({rows}, floats) : at::Tensor());
    }
}

std::size_t decode_graphs::page_width() const noexcept {
    return static_cast<std::size_t>(width);
}

group_batch decode_graphs::make_inputs(std::int64_t sequences, bool measures_mass, bool verifies) const {
    auto device = lines.front().k.options();
    auto ints = device.dtype(at::kInt);
    auto longs = device.dtype(at::kLong);
    const auto layers = static_cast<std::int64_t>(lines.size());
    const std::int64_t rows = verifies ? state_slots : 1;
    const std::int64_t tokens = sequences * rows;
    group_batch inputs{
        .tokens = at::zeros({tokens}, longs),
        .pos = at::zeros({tokens}, longs),
        .page_tables = at::zeros({layers, sequences, width}, ints),
        .history_positions = at::zeros({sequences}, longs),
        .rejected_tokens = at::zeros({sequences}, ints),
        .page_size = page_size,
        .state_slots = state_slots,
        .key_counts = at::full({layers, sequences}, rows, ints),
        .page_positions = at::zeros({layers, sequences, width}, ints),
        .arrival_counters = at::zeros({shard_model.arrival_counter_count(sequences, tokens)}, ints),
    };
    if (verifies) {
        inputs.cu_seqlens_q = at::arange(sequences + 1, ints) * rows;
    }
    if (measures_mass) {
        // Each view is the one page its key lies on; a verify group's views are its rows'.
        const std::int64_t views = verifies ? tokens : sequences;
        inputs.cu_view_pages = at::arange(views + 1, ints).unsqueeze(0).repeat({layers, 1});
        inputs.query_weights = at::zeros({tokens}, device.dtype(at::kFloat));
        const std::int64_t mass_rows = views * width;
        auto floats = device.dtype(at::kFloat);
        for (std::size_t layer = 0; layer < mass_heads.size(); layer++) {
            if (mass_heads[layer] == 0) {
                inputs.page_mass_sum.emplace_back();
                inputs.page_weight_sum.emplace_back();
            } else if (sequences <= startup_sequences && !verifies) {
                inputs.page_mass_sum.push_back(shared_mass_sum[layer].narrow(0, 0, mass_rows));
                inputs.page_weight_sum.push_back(shared_weight_sum[layer].narrow(0, 0, mass_rows));
            } else {
                inputs.page_mass_sum.push_back(
                    at::empty({mass_rows, static_cast<std::int64_t>(mass_heads[layer])}, floats)
                );
                inputs.page_weight_sum.push_back(at::empty({mass_rows}, floats));
            }
        }
    }
    return inputs;
}

std::unique_ptr<decode_graphs::captured_forward>
decode_graphs::capture(std::int64_t sequences, bool measures_mass, bool verifies) {
    auto captured = std::make_unique<captured_forward>();
    captured->inputs = make_inputs(sequences, measures_mass, verifies);
    const group_batch& inputs = captured->inputs;
    // Thread-local capture leaves the other GPUs' threads free to launch while this one captures.
    captured->graph.capture_begin({0, 0}, cudaStreamCaptureModeThreadLocal);
    if (verifies) {
        step_logits logits = shard_model.forward(step_batch{.verify_group = inputs}, lines);
        captured->logits = *logits.verify_logits;
        captured->hidden = logits.verify_hidden.value_or(at::Tensor());
    } else {
        fill_awaited_tokens_cuda(inputs.tokens, samples);
        step_logits logits = shard_model.forward(step_batch{.decode_group = inputs}, lines);
        captured->logits = *logits.decode_logits;
        captured->hidden = logits.decode_hidden.value_or(at::Tensor());
    }
    captured->graph.capture_end();
    return captured;
}

// A capture cannot create what the forward creates at its first run, such as cuBLAS handles and RoPE tables, so an eager
// forward of each kind runs first. Its tables point every view at line 0, which it zeroes first so that the forward
// reads no uninitialized memory; no sequence holds a line yet.
void decode_graphs::capture_startup() {
    for (const kv& layer : lines) {
        for (const at::Tensor& part : {layer.k, layer.v, layer.index_keys, layer.last_hidden}) {
            if (part.defined()) {
                part.select(0, 0).zero_();
            }
        }
    }
    const bool any_mass = std::ranges::any_of(mass_heads, [](std::size_t heads) { return heads > 0; });
    std::vector<bool> kinds{false};
    if (state_slots > 1) {
        kinds.push_back(true);
    }
    for (bool verifies : kinds) {
        for (bool measures_mass : {false, true}) {
            if (measures_mass && !any_mass) {
                continue;
            }
            group_batch warmup = make_inputs(1, measures_mass, verifies);
            shard_model.forward(
                verifies ? step_batch{.verify_group = warmup} : step_batch{.decode_group = warmup},
                lines
            );
        }
    }
    if (state_slots > 1 && shard_model.drafter() != nullptr) {
        for (bool verify_rows : {true, false}) {
            auto [inputs, hidden] = make_draft_inputs(1, verify_rows);
            shard_model.drafter()->draft(inputs, hidden, lines);
        }
        for (std::int64_t sequences = 1; sequences <= startup_sequences; sequences++) {
            for (bool verify_rows : {true, false}) {
                std::unique_ptr<captured_draft>& captured = drafts[{sequences, verify_rows}];
                if (!captured) {
                    captured = capture_draft(sequences, verify_rows);
                }
            }
        }
    }
    for (bool verifies : kinds) {
        for (std::int64_t sequences = 1; sequences <= startup_sequences; sequences++) {
            for (bool measures_mass : {false, true}) {
                if (measures_mass && !any_mass) {
                    continue;
                }
                std::unique_ptr<captured_forward>& captured = forwards[{sequences, measures_mass, verifies}];
                if (!captured) {
                    captured = capture(sequences, measures_mass, verifies);
                }
            }
        }
    }
}

const decode_graphs::captured_forward& decode_graphs::replay(const group_batch& g, bool verifies) {
    const std::int64_t sequences = g.page_tables.size(1);
    const bool measures_mass = g.cu_view_pages.defined();
    std::unique_ptr<captured_forward>& captured = forwards[{sequences, measures_mass, verifies}];
    if (!captured) {
        captured = capture(sequences, measures_mass, verifies);
    }
    group_batch& inputs = captured->inputs;
    TORCH_CHECK(
        g.page_size == page_size && g.state_slots == state_slots && g.tokens.sizes() == inputs.tokens.sizes(),
        "decode_graphs: the group's page size, state slots or tokens are not the graph's"
    );
    inputs.tokens.copy_(g.tokens);
    inputs.pos.copy_(g.pos);
    inputs.page_tables.copy_(g.page_tables);
    inputs.history_positions.copy_(g.history_positions);
    inputs.rejected_tokens.copy_(g.rejected_tokens);
    inputs.key_counts.copy_(g.key_counts);
    inputs.page_positions.copy_(g.page_positions);
    copy_defined(inputs.cu_view_pages, g.cu_view_pages);
    copy_defined(inputs.query_weights, g.query_weights);
    // The graph adds the step's mass into the rows of its views, which start from zero.
    for (std::size_t layer = 0; layer < g.page_mass_sum.size(); layer++) {
        if (g.page_mass_sum[layer].defined()) {
            inputs.page_mass_sum[layer].narrow(0, 0, g.page_mass_sum[layer].size(0)).zero_();
            inputs.page_weight_sum[layer].narrow(0, 0, g.page_weight_sum[layer].size(0)).zero_();
        }
    }
    captured->graph.replay();
    for (std::size_t layer = 0; layer < g.page_mass_sum.size(); layer++) {
        if (g.page_mass_sum[layer].defined()) {
            g.page_mass_sum[layer].copy_(inputs.page_mass_sum[layer].narrow(0, 0, g.page_mass_sum[layer].size(0)));
            g.page_weight_sum[layer].copy_(
                inputs.page_weight_sum[layer].narrow(0, 0, g.page_weight_sum[layer].size(0))
            );
        }
    }
    return *captured;
}

std::pair<group_batch, at::Tensor> decode_graphs::make_draft_inputs(std::int64_t sequences, bool verify_rows) const {
    auto device = lines.front().k.options();
    auto ints = device.dtype(at::kInt);
    auto longs = device.dtype(at::kLong);
    const auto layers = static_cast<std::int64_t>(lines.size());
    const std::int64_t rows = verify_rows ? state_slots : 1;
    const std::int64_t tokens = sequences * rows;
    group_batch inputs{
        .tokens = at::zeros({tokens}, longs),
        .pos = at::zeros({tokens}, longs),
        .page_tables = at::zeros({layers, sequences, width}, ints),
        .page_size = page_size,
        .state_slots = state_slots,
        .key_counts = at::full({layers, sequences}, rows, ints),
        .page_positions = at::zeros({layers, sequences, width}, ints),
        .arrival_counters = at::zeros({shard_model.arrival_counter_count(sequences, tokens)}, ints),
    };
    if (verify_rows) {
        inputs.cu_seqlens_q = at::arange(sequences + 1, ints) * rows;
        inputs.logit_rows = at::arange(sequences, longs) * rows;
    }
    // The draft head's KV layer keeps hidden rows in the model's dtype.
    auto with_hidden = std::ranges::find_if(lines, [](const kv& line) { return line.last_hidden.defined(); });
    TORCH_CHECK(with_hidden != lines.end(), "decode_graphs: no KV layer keeps the draft head's hidden rows");
    at::Tensor hidden = at::zeros({tokens, shard_model.drafter()->hidden_dim()}, with_hidden->last_hidden.options());
    return {std::move(inputs), std::move(hidden)};
}

std::unique_ptr<decode_graphs::captured_draft> decode_graphs::capture_draft(std::int64_t sequences, bool verify_rows) {
    auto captured = std::make_unique<captured_draft>();
    std::tie(captured->inputs, captured->hidden_rows) = make_draft_inputs(sequences, verify_rows);
    captured->graph.capture_begin({0, 0}, cudaStreamCaptureModeThreadLocal);
    captured->output = shard_model.drafter()->draft(captured->inputs, captured->hidden_rows, lines);
    captured->graph.capture_end();
    return captured;
}

draft_output decode_graphs::draft(const group_batch& g, const at::Tensor& hidden, bool verify_rows) {
    const std::int64_t sequences = g.page_tables.size(1);
    std::unique_ptr<captured_draft>& captured = drafts[{sequences, verify_rows}];
    if (!captured) {
        captured = capture_draft(sequences, verify_rows);
    }
    group_batch& inputs = captured->inputs;
    TORCH_CHECK(
        g.tokens.sizes() == inputs.tokens.sizes() && hidden.sizes() == captured->hidden_rows.sizes(),
        "decode_graphs: the draft pass's rows are not the graph's"
    );
    inputs.tokens.copy_(g.tokens);
    inputs.pos.copy_(g.pos);
    inputs.page_tables.copy_(g.page_tables);
    inputs.key_counts.copy_(g.key_counts);
    inputs.page_positions.copy_(g.page_positions);
    copy_defined(inputs.logit_rows, g.logit_rows);
    captured->hidden_rows.copy_(hidden);
    captured->graph.replay();
    return captured->output;
}

std::pair<at::Tensor, at::Tensor> decode_graphs::decode(const group_batch& g) {
    const captured_forward& captured = replay(g, false);
    return {captured.logits, captured.hidden};
}

std::pair<at::Tensor, at::Tensor> decode_graphs::verify(const group_batch& g) {
    const captured_forward& captured = replay(g, true);
    return {captured.logits, captured.hidden};
}

}  // namespace pulsarlm
