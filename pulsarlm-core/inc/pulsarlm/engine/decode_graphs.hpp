#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAGraph.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "pulsarlm/engine/model/interface.hpp"

namespace pulsarlm {

// Replays one GPU's decode forward from CUDA graphs over page tables of one fixed width, a graph per number of decoding
// sequences, whether the step measures mass and whether the group is a verify group of state_slots tokens per sequence
// or a decode group. A verify group's forward reads nothing back to the host either. The model's decode forward reads nothing back to the host and launches
// the same kernels for every group of one such shape. Each graph first fills the group's awaited tokens from samples
// (fill_awaited_tokens_cuda).
class decode_graphs {
    struct captured_forward {
        at::cuda::CUDAGraph graph;
        group_batch inputs;
        at::Tensor logits;
        // Undefined for a model without a draft head.
        at::Tensor hidden;
    };
    // The sequences of a group, whether it measures mass, and whether it is a verify group.
    using shape = std::tuple<std::int64_t, bool, bool>;

    model& shard_model;
    std::vector<kv> lines;
    at::Tensor samples;
    std::int64_t width;
    std::int64_t page_size;
    std::int64_t state_slots;
    // Per KV layer, the query heads whose mass this GPU measures, 0 for a layer that measures none.
    std::vector<std::size_t> mass_heads;
    // Per KV layer, mass_sum then weight_sum rows for startup_sequences sequences' views, shared by the graphs of up to
    // that many sequences; graphs of more sequences hold their own.
    std::vector<at::Tensor> shared_mass_sum;
    std::vector<at::Tensor> shared_weight_sum;
    std::map<shape, std::unique_ptr<captured_forward>> forwards;

    // Inputs of a group of sequences on the lines' device, the mass rows of up to startup_sequences sequences in the
    // shared buffers, with the group's own arrival counters.
    group_batch make_inputs(std::int64_t sequences, bool measures_mass, bool verifies) const;
    std::unique_ptr<captured_forward> capture(std::int64_t sequences, bool measures_mass, bool verifies);
    // The captured forward of g's shape, captured first if it is not yet, with g's inputs copied into it and replayed.
    const captured_forward& replay(const group_batch& g, bool verifies);

  public:
    // The decoding sequences whose graphs capture_startup records.
    static constexpr std::int64_t startup_sequences = 8;

    // Graphs over page tables width pages wide, of pages page_size tokens long, with state_slots slots
    // (group_batch::state_slots). mass_heads holds, per entry of the model's kv_layers(), the query heads
    // whose mass this GPU measures. lines is on the current device, and so is samples, int64, which holds the previous
    // step's sampled tokens at each replay.
    decode_graphs(
        model& shard_model,
        std::span<const kv> lines,
        std::size_t width,
        std::size_t page_size,
        std::size_t state_slots,
        std::vector<std::size_t> mass_heads,
        at::Tensor samples
    );

    std::size_t page_width() const noexcept;
    // Captures the graphs of 1 to startup_sequences sequences, with mass when a layer measures it and without, decode
    // groups and with state_slots above 1 verify groups, on the current stream, which is not the default stream, after
    // an eager forward of each that zeroes line 0 of each layer and writes into it. Every GPU of the model runs it at
    // once, before any sequence holds a line.
    void capture_startup();
    // model::forward's decode logits and hidden rows of g, its awaited tokens filled from samples, on the current
    // stream, valid until the next call, with g's page mass written as model::forward writes it. g's page tables are
    // page_width() wide; a shape not captured yet is captured first.
    std::pair<at::Tensor, at::Tensor> decode(const group_batch& g);
    // decode's verify logits and hidden rows of g, a verify group of state_slots tokens per sequence.
    std::pair<at::Tensor, at::Tensor> verify(const group_batch& g);
};

}  // namespace pulsarlm
