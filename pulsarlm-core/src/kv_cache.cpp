#include "pulsarlm/kv_cache.hpp"

#include "pulsarlm/ops.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <utility>
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

std::size_t ceil_div(std::size_t n, std::size_t d) {
    return (n + d - 1) / d;
}

std::vector<std::int64_t> page_sizes(const page_shape& shape) {
    if (!shape.state_sizes.empty()) {
        std::vector<std::int64_t> sizes{static_cast<std::int64_t>(shape.n_layers)};
        sizes.insert(sizes.end(), shape.state_sizes.begin(), shape.state_sizes.end());
        return sizes;
    }
    return {
        static_cast<std::int64_t>(shape.n_layers),
        static_cast<std::int64_t>(shape.page_size),
        static_cast<std::int64_t>(shape.n_kv_heads),
        static_cast<std::int64_t>(shape.head_dim),
    };
}

std::size_t tensor_count(const page_shape& shape) {
    return shape.values_only ? 1 : 2;
}

std::size_t tensor_bytes(const page_shape& shape) {
    return line_bytes(shape) / tensor_count(shape);
}

at::Tensor make_pool(std::span<const page_shape> shapes, std::size_t pool_bytes, at::Device device) {
    TORCH_CHECK(!shapes.empty(), "kv_cache: no KV layer");
    for (const page_shape& shape : shapes) {
        TORCH_CHECK(shape.n_layers == 1, "kv_cache: a KV layer's page shape has n_layers 1");
        TORCH_CHECK(shape.state_sizes.empty() || shape.values_only, "kv_cache: a recurrent page is values_only");
        TORCH_CHECK(
            pool_bytes / block_bytes(shape) < std::numeric_limits<std::uint32_t>::max(),
            "kv_cache: too many lines in a pool of ",
            pool_bytes,
            " bytes"
        );
    }
    return at::empty({static_cast<std::int64_t>(pool_bytes)}, at::TensorOptions().dtype(at::kByte).device(device));
}

std::vector<std::int64_t> line_strides(const page_shape& shape) {
    if (!shape.state_sizes.empty()) {
        std::vector<std::int64_t> sizes = page_sizes(shape);
        std::vector<std::int64_t> strides(sizes.size(), 1);
        for (std::size_t dim = sizes.size() - 1; dim > 0; dim--) {
            strides[dim - 1] = strides[dim] * sizes[dim];
        }
        return strides;
    }
    auto tensor_elements = static_cast<std::int64_t>(shape.page_size * shape.head_dim);
    auto head_elements = static_cast<std::int64_t>(tensor_count(shape)) * tensor_elements;
    return {
        head_elements * static_cast<std::int64_t>(shape.n_kv_heads),
        static_cast<std::int64_t>(shape.head_dim),
        head_elements,
        1,
    };
}

// count lines from element first of elements (1-D, in the shape's dtype), each [n_kv_heads, k then v, page_size,
// head_dim], as K and V views of [count, page_size, n_kv_heads, head_dim]; a recurrent shape's lines are contiguous
// [count, state_sizes...].
page line_views(const at::Tensor& elements, const page_shape& shape, std::size_t count, std::size_t first) {
    std::vector<std::int64_t> sizes = page_sizes(shape);
    sizes.front() = static_cast<std::int64_t>(count);
    std::vector<std::int64_t> strides = line_strides(shape);
    std::int64_t k_first = elements.storage_offset() + static_cast<std::int64_t>(first);
    at::Tensor k = elements.as_strided(sizes, strides, k_first);
    if (shape.values_only) {
        return page{.k = k, .v = k};
    }
    auto v_first = k_first + static_cast<std::int64_t>(shape.page_size * shape.head_dim);
    return page{.k = k, .v = elements.as_strided(sizes, strides, v_first)};
}

// Whether candidate is laid out as a line of shape, so its bytes copy as one block from k's first element.
bool in_line_layout(const page& candidate, const page_shape& shape) {
    std::vector<std::int64_t> strides = line_strides(shape);
    auto v_from_k = static_cast<const std::byte*>(candidate.v.data_ptr()) -
        static_cast<const std::byte*>(candidate.k.data_ptr());
    auto expected_v_from_k = shape.values_only ? 0 : static_cast<std::ptrdiff_t>(tensor_bytes(shape));
    return candidate.k.strides() == at::IntArrayRef(strides) && candidate.v.strides() == at::IntArrayRef(strides) &&
        v_from_k == expected_v_from_k;
}

// line_count lines of the pool, a block apart.
kv pool_views(const at::Tensor& pool, const page_shape& shape, std::size_t line_count) {
    page lines = line_views(pool.view(shape.dtype), shape, line_count, 0);
    auto block_elements = static_cast<std::int64_t>(block_bytes(shape) / c10::elementSize(shape.dtype));
    std::vector<std::int64_t> strides = line_strides(shape);
    strides.front() = block_elements;
    at::Tensor k = lines.k.as_strided(lines.k.sizes(), strides, lines.k.storage_offset());
    return kv{
        .k = k,
        .v = shape.values_only ? k : lines.v.as_strided(lines.v.sizes(), strides, lines.v.storage_offset())
    };
}

std::vector<std::size_t> layer_block_bytes(std::span<const page_shape> shapes) {
    std::vector<std::size_t> blocks;
    for (const page_shape& shape : shapes) {
        blocks.push_back(block_bytes(shape));
    }
    return blocks;
}

// Ascending; a layer's block class is the index of its block size.
std::vector<std::size_t> distinct_block_bytes(std::span<const page_shape> shapes) {
    std::vector<std::size_t> blocks = layer_block_bytes(shapes);
    std::ranges::sort(blocks);
    auto [first, last] = std::ranges::unique(blocks);
    blocks.erase(first, last);
    return blocks;
}

c10::cuda::CUDAStreamGuard guard_stream(const at::Tensor& lines, cudaStream_t stream) {
    return c10::cuda::CUDAStreamGuard(c10::cuda::getStreamFromExternal(stream, lines.device().index()));
}

at::Tensor line_of(const at::Tensor& lines, std::uint32_t line_id) {
    return lines.select(0, static_cast<std::int64_t>(line_id));
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

void check_page(const page_shape& shape, const at::Tensor& source) {
    std::vector<std::int64_t> sizes = page_sizes(shape);
    TORCH_CHECK(source.sizes() == at::IntArrayRef(sizes), "kv_cache: page is ", source.sizes(), ", line is ", sizes);
    TORCH_CHECK(source.scalar_type() == shape.dtype, "kv_cache: page is ", source.dtype(), ", line is ", shape.dtype);
}

std::size_t largest_line_bytes(std::span<const page_shape> shapes) {
    std::size_t largest = 0;
    for (const page_shape& shape : shapes) {
        largest = std::max(largest, line_bytes(shape));
    }
    return largest;
}

}  // namespace

std::size_t line_bytes(const page_shape& shape) {
    if (!shape.state_sizes.empty()) {
        std::size_t elements = shape.n_layers;
        for (std::int64_t size : shape.state_sizes) {
            elements *= static_cast<std::size_t>(size);
        }
        return elements * c10::elementSize(shape.dtype);
    }
    std::size_t tensors = shape.values_only ? 1 : 2;
    return tensors * shape.n_layers * shape.page_size * shape.n_kv_heads * shape.head_dim *
        c10::elementSize(shape.dtype);
}

std::size_t block_bytes(const page_shape& shape) {
    return std::bit_ceil(line_bytes(shape));
}

digest make_root_digest(std::string_view model, std::span<const page_shape> shapes) {
    std::vector<std::byte> seed;
    append_string(seed, digest_function());
    append_string(seed, model);
    append_u64(seed, shapes.size());
    for (const page_shape& shape : shapes) {
        append_string(seed, c10::toString(shape.dtype));
        append_u64(seed, shape.n_layers);
        append_u64(seed, shape.page_size);
        append_u64(seed, shape.n_kv_heads);
        append_u64(seed, shape.head_dim);
        append_u64(seed, shape.values_only);
        append_u64(seed, shape.state_sizes.size());
        for (std::int64_t size : shape.state_sizes) {
            append_u64(seed, static_cast<std::uint64_t>(size));
        }
    }
    return make_digest(seed);
}

digest store_key(const digest& page, std::size_t layer) {
    std::vector<std::byte> seed;
    append_u64(seed, page.hi);
    append_u64(seed, page.lo);
    append_u64(seed, layer);
    return make_digest(seed);
}

kv_cache::kv_cache(
    std::span<const page_shape> shapes,
    std::size_t pool_bytes,
    std::size_t host_page_bytes,
    at::Device device,
    page_store& store
)
: pool(make_pool(shapes, pool_bytes, device)),
  allocator(distinct_block_bytes(shapes), pool_bytes),
  store(&store),
  host_pages(host_page_bytes, largest_line_bytes(shapes)) {
    TORCH_CHECK(
        pool_bytes >= min_pool_bytes(layer_block_bytes(shapes)),
        "kv_cache: a pool of ",
        pool_bytes,
        " bytes holds no line of every layer"
    );
    std::vector<std::size_t> blocks = distinct_block_bytes(shapes);
    for (std::size_t c = 0; c < blocks.size(); c++) {
        classes.push_back(
            block_class{.block_bytes = blocks[c], .lines = std::vector<cache_line>(allocator.line_count(c))}
        );
    }
    layers.reserve(shapes.size());
    for (const page_shape& shape : shapes) {
        auto class_index = static_cast<std::size_t>(
            std::ranges::lower_bound(blocks, block_bytes(shape)) - blocks.begin()
        );
        layers.push_back(
            cache_layer{
                .shape = shape,
                .block_class = class_index,
                .device_kv = pool_views(pool, shape, classes[class_index].lines.size()),
            }
        );
    }
}

void kv_cache::assign_hash(std::size_t layer, std::uint32_t line_id, const digest& hash) {
    line(layer, line_id).hash = hash;
    layers[layer].digest_to_line_id.try_emplace(hash, line_id);
}

kv_cache::cache_line& kv_cache::line(std::size_t layer, std::uint32_t line_id) noexcept {
    return classes[layers[layer].block_class].lines[line_id];
}

const kv_cache::cache_line& kv_cache::line(std::size_t layer, std::uint32_t line_id) const noexcept {
    return classes[layers[layer].block_class].lines[line_id];
}

// A line keeps its hash unmapped when another line cached that hash before its fetch or sync completed.
bool kv_cache::is_mapped(std::size_t layer, std::uint32_t line_id) const noexcept {
    const std::optional<digest>& hash = line(layer, line_id).hash;
    if (!hash) {
        return false;
    }
    const std::unordered_map<digest, std::uint32_t, digest_hash>& mapped = layers[layer].digest_to_line_id;
    auto it = mapped.find(*hash);
    return it != mapped.end() && it->second == line_id;
}

void kv_cache::hold(line_allocator::transaction& taking, std::size_t layer, std::uint32_t line_id) {
    cache_line& held = line(layer, line_id);
    if (held.ref == 0) {
        taking.hold(layers[layer].block_class, line_id);
    }
    held.ref++;
}

std::optional<std::vector<std::vector<std::uint32_t>>>
kv_cache::take_lines(line_allocator::transaction& taking, std::span<const std::size_t> counts) {
    std::vector<std::size_t> by_block(layers.size());
    std::iota(by_block.begin(), by_block.end(), std::size_t{0});
    std::ranges::stable_sort(by_block, std::greater{}, [&](std::size_t layer) { return layers[layer].block_class; });
    std::vector<std::vector<std::uint32_t>> taken(layers.size());
    for (std::size_t layer : by_block) {
        for (std::size_t i = 0; i < counts[layer]; i++) {
            std::optional<std::uint32_t> line_id = taking.take(layers[layer].block_class);
            if (!line_id) {
                return std::nullopt;
            }
            taken[layer].push_back(*line_id);
        }
    }
    return taken;
}

// A reclaimed region holds no held line, so each line of it that maps its hash caches a page.
void kv_cache::commit(line_allocator::transaction& taking, std::span<const std::vector<std::uint32_t>> taken) {
    for (std::uint32_t region : taking.commit()) {
        for (std::size_t c = 0; c < classes.size(); c++) {
            auto [first, last] = allocator.lines_of_region(c, region);
            for (std::uint32_t line_id = first; line_id < last; line_id++) {
                cache_line& dropped = classes[c].lines[line_id];
                assert(dropped.ref == 0);
                if (dropped.hash && is_mapped(dropped.layer, line_id)) {
                    layers[dropped.layer].digest_to_line_id.erase(*dropped.hash);
                }
                dropped.hash.reset();
            }
        }
    }
    for (std::size_t layer = 0; layer < taken.size(); layer++) {
        for (std::uint32_t line_id : taken[layer]) {
            line(layer, line_id) = cache_line{.layer = layer, .ref = 1};
        }
    }
}

std::size_t kv_cache::line_offset(std::size_t layer, std::uint32_t line_id) const noexcept {
    return line_id * classes[layers[layer].block_class].block_bytes;
}

task<void>
kv_cache::copy_from_host(std::span<const layer_line> targets, std::span<const page> pages, cudaStream_t stream) {
    std::vector<std::int64_t> segments;
    std::vector<at::Tensor> copies;
    for (std::size_t i = 0; i < targets.size(); i++) {
        auto [layer, line_id] = targets[i];
        const page_shape& shape = layers[layer].shape;
        const page& source = pages[i];
        check_page(shape, source.v);
        check_page(shape, source.k);
        std::optional<std::size_t> offset;
        if (in_line_layout(source, shape)) {
            offset = host_pages.offset_of(source.k.data_ptr(), line_bytes(shape));
        }
        if (!offset) {
            auto elements = static_cast<std::int64_t>(line_bytes(shape) / c10::elementSize(shape.dtype));
            std::optional<at::Tensor> block = host_pages.allocate(elements, shape.dtype);
            at::Tensor copy = block ? *block : at::empty({elements}, at::TensorOptions().dtype(shape.dtype));
            page line = line_views(copy, shape, 1, 0);
            line.v.copy_(source.v);
            if (!shape.values_only) {
                line.k.copy_(source.k);
            }
            copies.push_back(copy);
            if (block) {
                offset = host_pages.offset_of(copy.data_ptr(), line_bytes(shape));
            } else {
                C10_CUDA_CHECK(cudaMemcpyAsync(
                    static_cast<std::byte*>(pool.data_ptr()) + line_offset(layer, line_id),
                    copy.data_ptr(),
                    line_bytes(shape),
                    cudaMemcpyHostToDevice,
                    stream
                ));
                continue;
            }
        }
        append_segment(segments, line_offset(layer, line_id), *offset, line_bytes(shape));
    }
    if (!segments.empty()) {
        auto guard = guard_stream(pool, stream);
        copy_segments_cuda(pool, host_pages.buffer(), pinned_int64(segments), /*to_host=*/false);
    }
    co_await synchronize(stream);
}

task<void> kv_cache::copy_to_host(std::span<const layer_line> sources, std::span<page> pages, cudaStream_t stream) {
    std::vector<std::int64_t> segments;
    for (std::size_t i = 0; i < sources.size(); i++) {
        auto [layer, line_id] = sources[i];
        const std::size_t bytes = line_bytes(layers[layer].shape);
        if (std::optional<std::size_t> offset = host_pages.offset_of(pages[i].k.data_ptr(), bytes)) {
            append_segment(segments, line_offset(layer, line_id), *offset, bytes);
        } else {
            C10_CUDA_CHECK(cudaMemcpyAsync(
                pages[i].k.data_ptr(),
                static_cast<const std::byte*>(pool.data_ptr()) + line_offset(layer, line_id),
                bytes,
                cudaMemcpyDeviceToHost,
                stream
            ));
        }
    }
    if (!segments.empty()) {
        auto guard = guard_stream(pool, stream);
        copy_segments_cuda(pool, host_pages.buffer(), pinned_int64(segments), /*to_host=*/true);
    }
    co_await synchronize(stream);
}

std::optional<std::vector<page>> kv_cache::allocate_host_pages(std::span<const layer_line> sources) {
    std::vector<page> pages;
    pages.reserve(sources.size());
    try {
        for (const layer_line& source : sources) {
            const page_shape& shape = layers[source.layer].shape;
            auto elements = static_cast<std::int64_t>(line_bytes(shape) / c10::elementSize(shape.dtype));
            std::optional<at::Tensor> block = host_pages.allocate(elements, shape.dtype);
            at::Tensor buffer = block ? *block : at::empty({elements}, at::TensorOptions().dtype(shape.dtype));
            pages.push_back(line_views(buffer, shape, 1, 0));
        }
    } catch (const c10::OutOfMemoryError&) {
        return std::nullopt;
    }
    return pages;
}

void kv_cache::release_all(std::span<const std::vector<std::uint32_t>> line_ids) noexcept {
    for (std::size_t layer = 0; layer < line_ids.size(); layer++) {
        release(layer, line_ids[layer]);
    }
}

// Every layer's resident pages are held before any line is taken, so taking one never drops a page this call needs.
task<std::expected<std::vector<std::vector<std::uint32_t>>, page_error>>
kv_cache::get(std::span<const std::span<const digest>> hashes, cudaStream_t stream) {
    assert(hashes.size() == layers.size());
    std::vector<std::vector<std::uint32_t>> line_ids(layers.size());
    std::vector<std::vector<std::size_t>> joined(layers.size());
    std::vector<std::vector<std::size_t>> to_fetch(layers.size());
    std::vector<layer_line> held;
    line_allocator::transaction taking(allocator);
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        cache_layer& cached = layers[layer];
        std::span<const digest> layer_hashes = hashes[layer];
        std::vector<std::uint32_t>& layer_lines = line_ids[layer];
        layer_lines.resize(layer_hashes.size());
        for (std::size_t i = 0; i < layer_hashes.size(); i++) {
            if (auto it = cached.digest_to_line_id.find(layer_hashes[i]); it != cached.digest_to_line_id.end()) {
                layer_lines[i] = it->second;
            } else if (auto it = cached.fetching.find(layer_hashes[i]); it != cached.fetching.end()) {
                layer_lines[i] = it->second.line_id;
                joined[layer].push_back(i);
            } else {
                to_fetch[layer].push_back(i);
                continue;
            }
            hold(taking, layer, layer_lines[i]);
            held.push_back(layer_line{.layer = layer, .line_id = layer_lines[i]});
        }
    }
    std::vector<std::size_t> fetch_counts;
    for (const std::vector<std::size_t>& fetched : to_fetch) {
        fetch_counts.push_back(fetched.size());
    }
    std::optional<std::vector<std::vector<std::uint32_t>>> taken = take_lines(taking, fetch_counts);
    if (!taken) {
        for (const layer_line& undone : held) {
            line(undone.layer, undone.line_id).ref--;
        }
        co_return std::unexpected(page_error::oom);
    }
    commit(taking, *taken);

    std::vector<digest> fetch_hashes;
    std::vector<digest> fetch_keys;
    std::vector<layer_line> fetch_lines;
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        for (std::size_t j = 0; j < to_fetch[layer].size(); j++) {
            std::size_t i = to_fetch[layer][j];
            const digest& hash = hashes[layer][i];
            line_ids[layer][i] = (*taken)[layer][j];
            layers[layer].fetching.try_emplace(hash, line_ids[layer][i]);
            fetch_hashes.push_back(hash);
            fetch_keys.push_back(store_key(hash, layer));
            fetch_lines.push_back(layer_line{.layer = layer, .line_id = line_ids[layer][i]});
        }
    }

    if (!fetch_keys.empty()) {
        std::vector<std::optional<page>> pages = co_await store->fetch(fetch_keys);
        assert(pages.size() == fetch_keys.size());
        std::vector<layer_line> arrived_lines;
        std::vector<page> arrived;
        for (std::size_t j = 0; j < pages.size(); j++) {
            if (pages[j]) {
                arrived_lines.push_back(fetch_lines[j]);
                arrived.push_back(std::move(*pages[j]));
            }
        }
        co_await copy_from_host(arrived_lines, arrived, stream);
        for (std::size_t j = 0; j < pages.size(); j++) {
            auto [layer, line_id] = fetch_lines[j];
            if (pages[j]) {
                assign_hash(layer, line_id, fetch_hashes[j]);
            }
            layers[layer].fetching.extract(fetch_hashes[j]).mapped().done.set();
        }
    }

    // An entry for the same hash in another line is a later fetch, after the one this call joined ended.
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        for (std::size_t i : joined[layer]) {
            auto it = layers[layer].fetching.find(hashes[layer][i]);
            if (it != layers[layer].fetching.end() && it->second.line_id == line_ids[layer][i]) {
                co_await it->second.done;
            }
        }
    }

    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        for (std::size_t i = 0; i < hashes[layer].size(); i++) {
            if (line(layer, line_ids[layer][i]).hash != hashes[layer][i]) {
                release_all(line_ids);
                co_return std::unexpected(page_error::miss);
            }
        }
    }
    co_return line_ids;
}

std::expected<std::vector<std::vector<std::uint32_t>>, page_error>
kv_cache::allocate(std::span<const std::size_t> counts) {
    assert(counts.size() == layers.size());
    line_allocator::transaction taking(allocator);
    std::optional<std::vector<std::vector<std::uint32_t>>> taken = take_lines(taking, counts);
    if (!taken) {
        return std::unexpected(page_error::oom);
    }
    commit(taking, *taken);
    return std::move(*taken);
}

std::expected<std::vector<std::uint32_t>, page_error>
kv_cache::clone(std::size_t layer, std::uint32_t line_id, std::size_t count, cudaStream_t stream) {
    assert(line(layer, line_id).ref > 0);
    std::vector<std::size_t> counts(layers.size(), 0);
    counts[layer] = count;
    line_allocator::transaction taking(allocator);
    std::optional<std::vector<std::vector<std::uint32_t>>> taken = take_lines(taking, counts);
    if (!taken) {
        return std::unexpected(page_error::oom);
    }
    commit(taking, *taken);
    std::vector<std::uint32_t> copies = std::move((*taken)[layer]);
    const kv& tensors = layers[layer].device_kv;
    auto guard = guard_stream(pool, stream);
    for (std::uint32_t copy : copies) {
        line_of(tensors.v, copy).copy_(line_of(tensors.v, line_id), /*non_blocking=*/true);
        if (!layers[layer].shape.values_only) {
            line_of(tensors.k, copy).copy_(line_of(tensors.k, line_id), /*non_blocking=*/true);
        }
    }
    return copies;
}

task<std::expected<void, page_error>> kv_cache::sync(
    const std::vector<std::vector<digest>>& hashes,
    const std::vector<std::vector<std::uint64_t>>& owners,
    const std::vector<std::vector<std::uint32_t>>& line_ids,
    cudaStream_t stream
) {
    assert(hashes.size() == layers.size() && owners.size() == layers.size() && line_ids.size() == layers.size());
    std::vector<layer_line> sources;
    std::vector<digest> keys;
    std::vector<page_owner> page_owners;
    std::vector<std::pair<layer_line, digest>> named;
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        assert(line_ids[layer].size() == hashes[layer].size() && owners[layer].size() == hashes[layer].size());
        std::unordered_map<digest, std::size_t, digest_hash> put_index;
        for (std::size_t i = 0; i < hashes[layer].size(); i++) {
            std::uint32_t line_id = line_ids[layer][i];
            assert(line(layer, line_id).ref > 0 && !line(layer, line_id).hash);
            auto [found, inserted] = put_index.try_emplace(hashes[layer][i], sources.size());
            if (inserted) {
                sources.push_back(layer_line{.layer = layer, .line_id = line_id});
                keys.push_back(store_key(hashes[layer][i], layer));
                named.emplace_back(layer_line{.layer = layer, .line_id = line_id}, hashes[layer][i]);
            }
            page_owners.push_back(page_owner{.request = owners[layer][i], .page = found->second});
        }
    }
    if (sources.empty()) {
        co_return {};
    }

    std::optional<std::vector<page>> copies = allocate_host_pages(sources);
    if (!copies) {
        co_return std::unexpected(page_error::oom);
    }
    for (const auto& [target, hash] : named) {
        assign_hash(target.layer, target.line_id, hash);
    }
    co_await copy_to_host(sources, *copies, stream);
    store->put(std::move(keys), std::move(*copies), std::move(page_owners));
    co_return {};
}

void kv_cache::release(std::size_t layer, std::span<const std::uint32_t> line_ids) noexcept {
    for (std::uint32_t line_id : line_ids) {
        cache_line& released = line(layer, line_id);
        assert(released.ref > 0);
        if (--released.ref > 0) {
            continue;
        }
        bool keeps_page = is_mapped(layer, line_id);
        if (!keeps_page) {
            released.hash.reset();
        }
        allocator.release(layers[layer].block_class, line_id, keeps_page);
    }
}

std::size_t kv_cache::free_count() {
    std::size_t block_total = 0;
    for (const cache_layer& layer : layers) {
        block_total += classes[layer.block_class].block_bytes;
    }
    std::size_t low = 0;
    std::size_t high = allocator.line_count(0) * classes.front().block_bytes / block_total;
    while (low < high) {
        std::size_t count = high - (high - low) / 2;
        line_allocator::transaction trial(allocator);
        if (take_lines(trial, std::vector<std::size_t>(layers.size(), count))) {
            low = count;
        } else {
            high = count - 1;
        }
    }
    return low;
}

std::size_t kv_cache::layer_count() const noexcept {
    return layers.size();
}

const kv& kv_cache::tensors(std::size_t layer) const noexcept {
    return layers[layer].device_kv;
}

}  // namespace pulsarlm
