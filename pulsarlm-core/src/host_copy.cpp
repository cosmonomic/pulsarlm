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

c10::cuda::CUDAStreamGuard guard_stream(const at::Tensor& lines, cudaStream_t stream) {
    return c10::cuda::CUDAStreamGuard(c10::cuda::getStreamFromExternal(stream, lines.device().index()));
}

at::Tensor pinned_int64(const std::vector<std::int64_t>& values) {
    at::Tensor pinned = at::empty(
        {static_cast<std::int64_t>(values.size() / 3), 3},
        at::TensorOptions().dtype(at::kLong).pinned_memory(true)
    );
    std::ranges::copy(values, pinned.data_ptr<std::int64_t>());
    return pinned;
}

// One cache shard's batched copies: the pool its shards view, and (device, host, bytes) per segment.
struct shard_segments {
    at::Tensor lines;
    std::vector<std::int64_t> segments;

    // Whether bytes views lines, taking lines from the first shard seen.
    bool batches(const at::Tensor& bytes) {
        if (!lines.defined()) {
            lines = lines_pool(bytes);
        }
        return bytes.storage().is_alias_of(lines.storage());
    }

    void append(std::size_t device_offset, std::size_t host_offset, std::size_t bytes) {
        segments.insert(
            segments.end(),
            {static_cast<std::int64_t>(device_offset),
             static_cast<std::int64_t>(host_offset),
             static_cast<std::int64_t>(bytes)}
        );
    }
};

std::size_t host_bytes(const sharded_page& p) {
    std::size_t bytes = 0;
    for (const page_shard& shard : p.shards) {
        bytes = std::max(bytes, shard.host_offset + static_cast<std::size_t>(shard.bytes.numel()));
    }
    return bytes;
}

void check_page(const sharded_page& target, const page& source) {
    TORCH_CHECK(
        source.bytes.scalar_type() == at::kByte && source.bytes.dim() == 1 && source.bytes.is_contiguous() &&
            static_cast<std::size_t>(source.bytes.numel()) >= host_bytes(target),
        "a page must be contiguous uint8 of at least ",
        host_bytes(target),
        " bytes, got ",
        source.bytes.dtype(),
        " ",
        source.bytes.sizes()
    );
}

std::optional<std::vector<page>> allocate_host_pages(host_page_pool& pool, std::span<const sharded_page> sources) {
    std::vector<page> pages;
    pages.reserve(sources.size());
    try {
        for (const sharded_page& source : sources) {
            std::size_t bytes = host_bytes(source);
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

// Launches each cache shard's batched copies on its stream, keeping their pinned segments alive in launched.
template <typename copy_segments>
void launch_segments(
    std::span<shard_segments> by_shard,
    std::span<const cudaStream_t> streams,
    std::vector<at::Tensor>& launched,
    copy_segments copy
) {
    for (std::size_t shard = 0; shard < by_shard.size(); shard++) {
        if (by_shard[shard].segments.empty()) {
            continue;
        }
        launched.push_back(pinned_int64(by_shard[shard].segments));
        auto guard = guard_stream(by_shard[shard].lines, streams[shard]);
        copy(by_shard[shard].lines, launched.back());
    }
}

task<void> synchronize_all(std::span<const cudaStream_t> streams) {
    for (cudaStream_t stream : streams) {
        co_await synchronize(stream);
    }
}

}  // namespace

task<std::optional<std::vector<page>>>
copy_to_host(host_page_pool& pool, std::span<const sharded_page> sources, std::span<const cudaStream_t> streams) {
    std::optional<std::vector<page>> pages = allocate_host_pages(pool, sources);
    if (!pages || sources.empty()) {
        co_return pages;
    }
    std::vector<shard_segments> by_shard(streams.size());
    for (std::size_t i = 0; i < sources.size(); i++) {
        at::Tensor& target = (*pages)[i].bytes;
        std::optional<std::size_t> offset = pool.offset_of(target.data_ptr(), static_cast<std::size_t>(target.numel()));
        for (const page_shard& shard : sources[i].shards) {
            auto bytes = static_cast<std::size_t>(shard.bytes.numel());
            if (offset && by_shard[shard.cache_shard].batches(shard.bytes)) {
                by_shard[shard.cache_shard]
                    .append(static_cast<std::size_t>(shard.bytes.storage_offset()), *offset + shard.host_offset, bytes);
            } else {
                C10_CUDA_CHECK(cudaMemcpyAsync(
                    static_cast<std::byte*>(target.data_ptr()) + shard.host_offset,
                    shard.bytes.data_ptr(),
                    bytes,
                    cudaMemcpyDeviceToHost,
                    streams[shard.cache_shard]
                ));
            }
        }
    }
    std::vector<at::Tensor> launched;
    launch_segments(by_shard, streams, launched, [&](const at::Tensor& lines, const at::Tensor& segments) {
        copy_segments_to_host_cuda(lines, pool.buffer(), segments);
    });
    co_await synchronize_all(streams);
    co_return pages;
}

task<void> copy_from_host(
    host_page_pool& pool,
    std::span<const sharded_page> targets,
    std::span<const page> pages,
    std::span<const cudaStream_t> streams
) {
    if (targets.empty()) {
        co_return;
    }
    std::vector<shard_segments> by_shard(streams.size());
    std::vector<at::Tensor> staged;
    for (std::size_t i = 0; i < targets.size(); i++) {
        check_page(targets[i], pages[i]);
        const at::Tensor& source = pages[i].bytes;
        auto bytes = static_cast<std::size_t>(source.numel());
        std::optional<std::size_t> offset = pool.offset_of(source.data_ptr(), bytes);
        if (!offset) {
            if (std::optional<at::Tensor> block = pool.allocate(bytes)) {
                block->copy_(source);
                staged.push_back(*block);
                offset = pool.offset_of(block->data_ptr(), bytes);
            }
        }
        for (const page_shard& shard : targets[i].shards) {
            auto shard_bytes = static_cast<std::size_t>(shard.bytes.numel());
            if (offset && by_shard[shard.cache_shard].batches(shard.bytes)) {
                by_shard[shard.cache_shard].append(
                    static_cast<std::size_t>(shard.bytes.storage_offset()),
                    *offset + shard.host_offset,
                    shard_bytes
                );
            } else {
                C10_CUDA_CHECK(cudaMemcpyAsync(
                    shard.bytes.data_ptr(),
                    static_cast<const std::byte*>(source.data_ptr()) + shard.host_offset,
                    shard_bytes,
                    cudaMemcpyHostToDevice,
                    streams[shard.cache_shard]
                ));
            }
        }
    }
    std::vector<at::Tensor> launched;
    launch_segments(by_shard, streams, launched, [&](const at::Tensor& lines, const at::Tensor& segments) {
        copy_segments_from_host_cuda(lines, pool.buffer(), segments);
    });
    co_await synchronize_all(streams);
}

}  // namespace pulsarlm
