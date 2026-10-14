#include "pulsarlm/ops.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <span>

namespace pulsarlm {
namespace {

// One block per (layer, sequence); threads stride over the row's width.
__global__ void gather_page_tables_kernel(
    const int64_t* __restrict__ table_addresses,
    const int32_t* __restrict__ capacities,
    int32_t* __restrict__ out,
    int num_seqs,
    int width
) {
    const int layer = blockIdx.x / num_seqs;
    const int s = blockIdx.x % num_seqs;
    const int capacity = capacities[s];
    const auto* row = reinterpret_cast<const int32_t*>(table_addresses[s]) + static_cast<int64_t>(layer) * capacity;
    int32_t* out_row = out + static_cast<int64_t>(blockIdx.x) * width;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        out_row[j] = j < capacity ? row[j] : 0;
    }
}

}  // namespace

at::Tensor gather_page_tables_cuda(std::span<const at::Tensor> tables, int64_t n_layers, int64_t width) {
    TORCH_CHECK(!tables.empty(), "gather_page_tables: no tables");
    const auto device = tables.front().device();
    const auto num_seqs = static_cast<int64_t>(tables.size());
    at::Tensor addresses = at::empty({num_seqs}, at::TensorOptions().dtype(at::kLong).pinned_memory(true));
    at::Tensor capacities = at::empty({num_seqs}, at::TensorOptions().dtype(at::kInt).pinned_memory(true));
    auto* address = addresses.mutable_data_ptr<int64_t>();
    auto* capacity = capacities.mutable_data_ptr<int32_t>();
    for (int64_t s = 0; s < num_seqs; s++) {
        const at::Tensor& table = tables[s];
        TORCH_CHECK(
            table.device() == device && table.scalar_type() == at::kInt && table.dim() == 2 &&
                table.size(0) == n_layers && table.is_contiguous(),
            "gather_page_tables: each table must be a contiguous int32 [n_layers, capacity] tensor on one device"
        );
        address[s] = reinterpret_cast<int64_t>(table.const_data_ptr<int32_t>());
        capacity[s] = static_cast<int32_t>(table.size(1));
    }
    auto options = at::TensorOptions().device(device);
    at::Tensor out = at::empty({n_layers, num_seqs, width}, options.dtype(at::kInt));
    if (n_layers == 0 || width == 0) {
        return out;
    }
    at::Tensor device_addresses = addresses.to(options.dtype(at::kLong), /*non_blocking=*/true);
    at::Tensor device_capacities = capacities.to(options.dtype(at::kInt), /*non_blocking=*/true);
    gather_page_tables_kernel<<<static_cast<int>(n_layers * num_seqs), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        device_addresses.const_data_ptr<int64_t>(),
        device_capacities.const_data_ptr<int32_t>(),
        out.mutable_data_ptr<int32_t>(),
        static_cast<int>(num_seqs),
        static_cast<int>(width)
    );
    return out;
}

}  // namespace pulsarlm
