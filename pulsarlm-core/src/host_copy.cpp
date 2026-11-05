#include "pulsarlm/host_copy.hpp"

#include "pulsarlm/ops.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace pulsarlm {

namespace {

// The device bytes of the whole pool a line views; segment offsets index into it.
at::Tensor lines_pool(const at::Tensor& line) {
    return at::empty({0}, line.options())
        .set_(line.storage(), 0, {static_cast<std::int64_t>(line.storage().nbytes())}, {1});
}

bool in_lines_pool(const at::Tensor& line, const at::Tensor& lines) {
    return line.storage().is_alias_of(lines.storage());
}

c10::cuda::CUDAStreamGuard guard_stream(const at::Tensor& lines, cudaStream_t stream) {
    return c10::cuda::CUDAStreamGuard(c10::cuda::getStreamFromExternal(stream, lines.device().index()));
}

void append_segment(std::vector<std::int64_t>& segments, std::size_t device, std::size_t host, std::size_t bytes) {
    segments.insert(
        segments.end(),
        {static_cast<std::int64_t>(device), static_cast<std::int64_t>(host), static_cast<std::int64_t>(bytes)}
    );
}

at::Tensor pinned_int64(const std::vector<std::int64_t>& values) {
    at::Tensor pinned = at::empty(
        {static_cast<std::int64_t>(values.size() / 3), 3},
        at::TensorOptions().dtype(at::kLong).pinned_memory(true)
    );
    std::ranges::copy(values, pinned.data_ptr<std::int64_t>());
    return pinned;
}

void check_page(const page& target, const page& source) {
    TORCH_CHECK(
        source.bytes.scalar_type() == at::kByte && source.bytes.dim() == 1 && source.bytes.is_contiguous() &&
            source.bytes.numel() == target.bytes.numel(),
        "a page must be contiguous uint8 [",
        target.bytes.numel(),
        "], got ",
        source.bytes.dtype(),
        " ",
        source.bytes.sizes()
    );
}

std::optional<std::vector<page>> allocate_host_pages(host_page_pool& pool, std::span<const page> sources) {
    std::vector<page> pages;
    pages.reserve(sources.size());
    try {
        for (const page& source : sources) {
            auto bytes = static_cast<std::size_t>(source.bytes.numel());
            std::optional<at::Tensor> block = pool.allocate(bytes);
            if (!block) {
                block = at::empty({static_cast<std::int64_t>(bytes)}, at::TensorOptions().dtype(at::kByte));
            }
            pages.push_back(page{.bytes = std::move(*block)});
        }
    } catch (const c10::OutOfMemoryError&) {
        return std::nullopt;
    }
    return pages;
}

}  // namespace

task<std::optional<std::vector<page>>>
copy_to_host(host_page_pool& pool, std::span<const page> sources, cudaStream_t stream) {
    std::optional<std::vector<page>> pages = allocate_host_pages(pool, sources);
    if (!pages || sources.empty()) {
        co_return pages;
    }
    at::Tensor lines = lines_pool(sources.front().bytes);
    std::vector<std::int64_t> segments;
    for (std::size_t i = 0; i < sources.size(); i++) {
        const at::Tensor& source = sources[i].bytes;
        at::Tensor& target = (*pages)[i].bytes;
        auto bytes = static_cast<std::size_t>(source.numel());
        std::optional<std::size_t> offset = pool.offset_of(target.data_ptr(), bytes);
        if (offset && in_lines_pool(source, lines)) {
            append_segment(segments, static_cast<std::size_t>(source.storage_offset()), *offset, bytes);
        } else {
            C10_CUDA_CHECK(
                cudaMemcpyAsync(target.data_ptr(), source.data_ptr(), bytes, cudaMemcpyDeviceToHost, stream)
            );
        }
    }
    at::Tensor pinned_segments;
    if (!segments.empty()) {
        pinned_segments = pinned_int64(segments);
        auto guard = guard_stream(lines, stream);
        copy_segments_to_host_cuda(lines, pool.buffer(), pinned_segments);
    }
    co_await synchronize(stream);
    co_return pages;
}

task<void>
copy_from_host(host_page_pool& pool, std::span<const page> targets, std::span<const page> pages, cudaStream_t stream) {
    if (targets.empty()) {
        co_return;
    }
    at::Tensor lines = lines_pool(targets.front().bytes);
    std::vector<std::int64_t> segments;
    std::vector<at::Tensor> staged;
    for (std::size_t i = 0; i < targets.size(); i++) {
        const at::Tensor& target = targets[i].bytes;
        check_page(targets[i], pages[i]);
        const at::Tensor& source = pages[i].bytes;
        auto bytes = static_cast<std::size_t>(source.numel());
        bool batched = in_lines_pool(target, lines);
        std::optional<std::size_t> offset = pool.offset_of(source.data_ptr(), bytes);
        if (!offset && batched) {
            if (std::optional<at::Tensor> block = pool.allocate(bytes)) {
                block->copy_(source);
                staged.push_back(*block);
                offset = pool.offset_of(block->data_ptr(), bytes);
            }
        }
        if (offset && batched) {
            append_segment(segments, static_cast<std::size_t>(target.storage_offset()), *offset, bytes);
        } else {
            C10_CUDA_CHECK(
                cudaMemcpyAsync(target.data_ptr(), source.data_ptr(), bytes, cudaMemcpyHostToDevice, stream)
            );
        }
    }
    at::Tensor pinned_segments;
    if (!segments.empty()) {
        pinned_segments = pinned_int64(segments);
        auto guard = guard_stream(lines, stream);
        copy_segments_from_host_cuda(lines, pool.buffer(), pinned_segments);
    }
    co_await synchronize(stream);
}

}  // namespace pulsarlm
