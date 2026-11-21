#pragma once

#include <ATen/core/Tensor.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace pulsarlm::ops::cuda {

// int32 [n_layers, tables.size(), width]: row s of each layer holds the first page_counts[s] entries of that layer's
// row of tables[s], a contiguous int32 [n_layers, capacity] device tensor; the entries past them are left unwritten.
// The tables stay alive until the stream has run the gather.
at::Tensor gather_page_tables(
    std::span<const at::Tensor> tables,
    std::span<const std::size_t> page_counts,
    int64_t n_layers,
    int64_t width
);

}  // namespace pulsarlm::ops::cuda
