#include "pulsarlm/host_page_pool.hpp"

#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <optional>

namespace pulsarlm {

namespace {

std::size_t order_of(std::size_t bytes, std::size_t min_block_bytes) {
    return static_cast<std::size_t>(std::countr_zero(std::bit_ceil(std::max(bytes, min_block_bytes)) / min_block_bytes));
}

}  // namespace

host_page_pool::shared_state::~shared_state() {
    if (data != nullptr) {
        cudaFreeHost(data);
    }
}

std::optional<std::size_t> host_page_pool::shared_state::take(std::size_t order) {
    std::size_t found = order;
    while (found < free_blocks.size() && free_blocks[found].empty()) {
        found++;
    }
    if (found == free_blocks.size()) {
        return std::nullopt;
    }
    std::size_t offset = *free_blocks[found].begin();
    free_blocks[found].erase(free_blocks[found].begin());
    for (; found > order; found--) {
        free_blocks[found - 1].insert(offset + (min_block_bytes << (found - 1)));
    }
    return offset;
}

void host_page_pool::shared_state::give_back(std::size_t offset, std::size_t order) {
    for (; order + 1 < free_blocks.size(); order++) {
        std::size_t buddy = offset ^ (min_block_bytes << order);
        if (!free_blocks[order].erase(buddy)) {
            break;
        }
        offset = std::min(offset, buddy);
    }
    free_blocks[order].insert(offset);
}

host_page_pool::host_page_pool(std::size_t capacity_bytes, std::size_t largest_bytes)
: state(std::make_shared<shared_state>()) {
    const std::size_t top_order = order_of(largest_bytes, min_block_bytes);
    const std::size_t top_bytes = min_block_bytes << top_order;
    state->free_blocks.resize(top_order + 1);
    state->bytes = capacity_bytes / top_bytes * top_bytes;
    if (state->bytes == 0) {
        return;
    }
    void* data = nullptr;
    cudaError_t status = cudaHostAlloc(&data, state->bytes, cudaHostAllocDefault);
    if (status != cudaSuccess) {
        cudaGetLastError();
        throw c10::OutOfMemoryError(
            {__func__, __FILE__, static_cast<std::uint32_t>(__LINE__)},
            "host_page_pool: cannot pin " + std::to_string(state->bytes) + " bytes: " + cudaGetErrorString(status)
        );
    }
    state->data = static_cast<std::byte*>(data);
    for (std::size_t offset = 0; offset < state->bytes; offset += top_bytes) {
        state->free_blocks[top_order].insert(offset);
    }
    buffer_bytes = at::from_blob(
        data,
        {static_cast<std::int64_t>(state->bytes)},
        [shared = state](void*) {},
        at::TensorOptions().dtype(at::kByte)
    );
}

std::optional<at::Tensor> host_page_pool::allocate(std::int64_t elements, at::ScalarType dtype) {
    const std::size_t bytes = static_cast<std::size_t>(elements) * c10::elementSize(dtype);
    const std::size_t order = order_of(bytes, min_block_bytes);
    if (order >= state->free_blocks.size()) {
        return std::nullopt;
    }
    std::optional<std::size_t> offset;
    {
        std::lock_guard lock(state->mutex);
        offset = state->take(order);
    }
    if (!offset) {
        return std::nullopt;
    }
    return at::from_blob(
        state->data + *offset,
        {elements},
        [shared = state, freed = *offset, order](void*) {
            std::lock_guard lock(shared->mutex);
            shared->give_back(freed, order);
        },
        at::TensorOptions().dtype(dtype)
    );
}

std::optional<std::size_t> host_page_pool::offset_of(const void* data, std::size_t bytes) const noexcept {
    const auto* start = static_cast<const std::byte*>(data);
    if (state->data == nullptr || start < state->data || start + bytes > state->data + state->bytes) {
        return std::nullopt;
    }
    return static_cast<std::size_t>(start - state->data);
}

const at::Tensor& host_page_pool::buffer() const noexcept {
    return buffer_bytes;
}

}  // namespace pulsarlm
