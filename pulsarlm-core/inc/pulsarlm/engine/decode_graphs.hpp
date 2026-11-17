#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAGraph.h>

#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <tuple>
#include <vector>

#include "pulsarlm/engine/model/interface.hpp"

namespace pulsarlm {

// Replays one GPU's decode forward from CUDA graphs, one per shape of decode group, each captured at its shape's first
// forward. The model's decode forward reads nothing back to the host and launches the same kernels for every group of
// one shape.
class decode_graphs {
    struct captured_forward {
        at::cuda::CUDAGraph graph;
        group_batch inputs;
        at::Tensor logits;
    };
    // The sequences and the page-table width of a group, and whether it measures mass.
    using shape = std::tuple<std::int64_t, std::int64_t, bool>;

    model& shard_model;
    std::vector<kv> lines;
    std::map<shape, std::unique_ptr<captured_forward>> forwards;

    std::unique_ptr<captured_forward> capture(const group_batch& g);

  public:
    decode_graphs(model& shard_model, std::span<const kv> lines);

    // model::forward's decode logits of g on the current stream, valid until the next call, with g's page mass written
    // as model::forward writes it.
    at::Tensor forward(const group_batch& g);
};

}  // namespace pulsarlm
