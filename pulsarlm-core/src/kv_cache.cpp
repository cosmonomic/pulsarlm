#include "pulsarlm/kv_cache.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace pulsarlm {

static_assert(std::endian::native == std::endian::little);

namespace {

void append_u64(std::vector<std::byte>& seed, std::uint64_t value) {
    auto bytes = std::bit_cast<std::array<std::byte, sizeof(value)>>(value);
    seed.insert(seed.end(), bytes.begin(), bytes.end());
}

void append_string(std::vector<std::byte>& seed, std::string_view value) {
    append_u64(seed, value.size());
    auto bytes = std::as_bytes(std::span(value));
    seed.insert(seed.end(), bytes.begin(), bytes.end());
}

at::Tensor make_lines(const page_shape& shape, std::size_t n_lines, at::Device device) {
    TORCH_CHECK(n_lines < std::numeric_limits<std::uint32_t>::max(), "kv_cache: too many lines: ", n_lines);
    return at::empty(
        {static_cast<std::int64_t>(shape.n_layers),
         static_cast<std::int64_t>(n_lines),
         static_cast<std::int64_t>(shape.page_size),
         static_cast<std::int64_t>(shape.n_kv_heads),
         static_cast<std::int64_t>(shape.head_dim)},
        at::TensorOptions().dtype(shape.dtype).device(device)
    );
}

c10::cuda::CUDAStreamGuard guard_stream(const at::Tensor& lines, cudaStream_t stream) {
    return c10::cuda::CUDAStreamGuard(c10::cuda::getStreamFromExternal(stream, lines.device().index()));
}

at::Tensor layer_of(const at::Tensor& tensor, std::size_t layer) {
    return tensor.select(0, static_cast<std::int64_t>(layer));
}

at::Tensor line_layer(const at::Tensor& lines, std::size_t layer, std::uint32_t line_id) {
    return layer_of(lines, layer).select(0, static_cast<std::int64_t>(line_id));
}

// Copying layer by layer keeps both sides of each copy contiguous, so no copy allocates device memory.
void copy_into_line(const at::Tensor& lines, std::uint32_t line_id, const at::Tensor& source) {
    for (std::size_t layer = 0; layer < static_cast<std::size_t>(lines.size(0)); layer++) {
        line_layer(lines, layer, line_id).copy_(layer_of(source, layer), /*non_blocking=*/true);
    }
}

void check_page(const at::Tensor& lines, std::uint32_t line_id, const at::Tensor& source) {
    at::Tensor target = lines.select(1, static_cast<std::int64_t>(line_id));
    TORCH_CHECK(source.sizes() == target.sizes(), "kv_cache: page is ", source.sizes(), ", line is ", target.sizes());
    TORCH_CHECK(source.dtype() == target.dtype(), "kv_cache: page is ", source.dtype(), ", line is ", target.dtype());
}

at::Tensor copy_line_to_host(const at::Tensor& lines, std::uint32_t line_id) {
    at::Tensor source = lines.select(1, static_cast<std::int64_t>(line_id));
    at::Tensor host = at::empty(source.sizes(), source.options().device(at::kCPU).pinned_memory(true));
    for (std::size_t layer = 0; layer < static_cast<std::size_t>(lines.size(0)); layer++) {
        layer_of(host, layer).copy_(line_layer(lines, layer, line_id), /*non_blocking=*/true);
    }
    return host;
}

}  // namespace

std::size_t line_bytes(const page_shape& shape) {
    return 2 * shape.n_layers * shape.page_size * shape.n_kv_heads * shape.head_dim * c10::elementSize(shape.dtype);
}

digest make_root_digest(std::string_view model, const page_shape& shape) {
    std::vector<std::byte> seed;
    append_string(seed, digest_function());
    append_string(seed, model);
    append_string(seed, c10::toString(shape.dtype));
    append_u64(seed, shape.n_layers);
    append_u64(seed, shape.page_size);
    append_u64(seed, shape.n_kv_heads);
    append_u64(seed, shape.head_dim);
    return make_digest(seed);
}

line_pool::line_pool(const page_shape& shape, std::size_t n_lines, at::Device device)
: k(make_lines(shape, n_lines, device)),
  v(make_lines(shape, n_lines, device)) {}

void line_pool::copy_from_host(std::uint32_t line_id, const page& source, cudaStream_t stream) {
    check_page(k, line_id, source.k);
    check_page(v, line_id, source.v);
    auto guard = guard_stream(k, stream);
    copy_into_line(k, line_id, source.k);
    copy_into_line(v, line_id, source.v);
}

void line_pool::clone(std::uint32_t source, std::span<const std::uint32_t> targets, cudaStream_t stream) {
    auto guard = guard_stream(k, stream);
    at::Tensor k_source = k.select(1, static_cast<std::int64_t>(source));
    at::Tensor v_source = v.select(1, static_cast<std::int64_t>(source));
    for (std::uint32_t target : targets) {
        copy_into_line(k, target, k_source);
        copy_into_line(v, target, v_source);
    }
}

// A failed pinned allocation throws AcceleratorError with cudaErrorMemoryAllocation. Host tensors dropped while their
// copies are in flight stay reserved by the caching host allocator until the copies end.
std::optional<std::vector<page>>
line_pool::copy_to_host(std::span<const std::uint32_t> line_ids, cudaStream_t stream) const {
    auto guard = guard_stream(k, stream);
    std::vector<page> pages;
    pages.reserve(line_ids.size());
    try {
        for (std::uint32_t line_id : line_ids) {
            pages.push_back(page{.k = copy_line_to_host(k, line_id), .v = copy_line_to_host(v, line_id)});
        }
    } catch (const c10::AcceleratorError& error) {
        if (error.get_error_code() != cudaErrorMemoryAllocation) {
            throw;
        }
        return std::nullopt;
    }
    return pages;
}

}  // namespace pulsarlm
