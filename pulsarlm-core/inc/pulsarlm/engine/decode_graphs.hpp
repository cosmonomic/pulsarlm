#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAGraph.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "pulsarlm/engine/model/interface.hpp"

namespace pulsarlm {

// Replays one GPU's decode forward from CUDA graphs over page tables of one fixed width, a graph per number of decoding
// sequences and whether the step measures mass. The model's decode forward reads nothing back to the host and launches
// the same kernels for every group of one such shape. Each graph first fills the group's awaited tokens from samples
// (fill_awaited_tokens_cuda).
class decode_graphs {
    struct captured_forward {
        at::cuda::CUDAGraph graph;
        group_batch inputs;
        at::Tensor logits;
    };
    // The sequences of a group, and whether it measures mass.
    using shape = std::pair<std::int64_t, bool>;

    model& shard_model;
    std::vector<kv> lines;
    at::Tensor samples;
    std::int64_t width;
    // Per KV layer, the query heads whose mass this GPU measures, 0 for a layer that measures none.
    std::vector<std::size_t> mass_heads;
    // Per KV layer, mass_sum then weight_sum rows for startup_sequences sequences' views, shared by the graphs of up to
    // that many sequences; graphs of more sequences hold their own.
    std::vector<at::Tensor> shared_mass_sum;
    std::vector<at::Tensor> shared_weight_sum;
    std::map<shape, std::unique_ptr<captured_forward>> forwards;

    // Inputs of a decode group of sequences on the lines' device, the mass rows of up to startup_sequences sequences in
    // the shared buffers, with the group's own arrival counters.
    group_batch make_inputs(std::int64_t sequences, bool measures_mass) const;
    std::unique_ptr<captured_forward> capture(std::int64_t sequences, bool measures_mass);

  public:
    // The decoding sequences whose graphs capture_startup records.
    static constexpr std::int64_t startup_sequences = 8;

    // Graphs over page tables width pages wide. mass_heads holds, per entry of the model's kv_layers(), the query heads
    // whose mass this GPU measures. lines is on the current device, and so is samples, int64, which holds the previous
    // step's sampled tokens at each replay.
    decode_graphs(
        model& shard_model,
        std::span<const kv> lines,
        std::size_t width,
        std::vector<std::size_t> mass_heads,
        at::Tensor samples
    );

    std::size_t page_width() const noexcept;
    // Captures the graphs of 1 to startup_sequences sequences, with mass when a layer measures it and without, on the
    // current stream, which is not the default stream, after an eager forward that zeroes line 0 of each layer and
    // writes into it. Every GPU of the model runs it at once, before any sequence holds a line.
    void capture_startup();
    // model::forward's decode logits of g, its awaited tokens filled from samples, on the current stream, valid until
    // the next call, with g's page mass written as model::forward writes it. g's page tables are page_width() wide; a
    // shape not captured yet is captured first.
    at::Tensor forward(const group_batch& g);
};

}  // namespace pulsarlm
