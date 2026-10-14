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
#include <limits>
#include <memory>
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
        TORCH_CHECK(shape.dtype == shapes.front().dtype, "kv_cache: the KV layers share a dtype");
        TORCH_CHECK(line_bytes(shape) <= pool_bytes, "kv_cache: a pool of ", pool_bytes, " bytes holds no line");
        TORCH_CHECK(
            pool_bytes / line_bytes(shape) < std::numeric_limits<std::uint32_t>::max(),
            "kv_cache: too many lines in a pool of ",
            pool_bytes,
            " bytes"
        );
    }
    return at::empty({static_cast<std::int64_t>(pool_bytes)}, at::TensorOptions().dtype(at::kByte).device(device));
}

std::vector<std::int64_t> line_strides(const page_shape& shape) {
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
// head_dim], as K and V views of [count, page_size, n_kv_heads, head_dim].
page line_views(const at::Tensor& elements, const page_shape& shape, std::size_t count, std::size_t first) {
    std::vector<std::int64_t> sizes = page_sizes(shape);
    sizes.front() = static_cast<std::int64_t>(count);
    std::vector<std::int64_t> strides = line_strides(shape);
    std::int64_t k_first = elements.storage_offset() + static_cast<std::int64_t>(first);
    at::Tensor k = elements.as_strided(sizes, strides, k_first);
    auto v_first = k_first + static_cast<std::int64_t>(shape.page_size * shape.head_dim);
    return page{.k = k, .v = shape.values_only ? k : elements.as_strided(sizes, strides, v_first)};
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

kv pool_views(const at::Tensor& pool, const page_shape& shape) {
    page lines = line_views(
        pool.view(shape.dtype),
        shape,
        static_cast<std::size_t>(pool.numel()) / line_bytes(shape),
        0
    );
    return kv{.k = lines.k, .v = lines.v};
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
    std::size_t tensors = shape.values_only ? 1 : 2;
    return tensors * shape.n_layers * shape.page_size * shape.n_kv_heads * shape.head_dim *
        c10::elementSize(shape.dtype);
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

void kv_cache::size_class::link_free_front(std::uint32_t line_id) noexcept {
    lines[line_id].previous_free = no_line;
    lines[line_id].next_free = free_front;
    if (free_front == no_line) {
        free_back = line_id;
    } else {
        lines[free_front].previous_free = line_id;
    }
    free_front = line_id;
    free_count++;
}

void kv_cache::size_class::link_free_back(std::uint32_t line_id) noexcept {
    lines[line_id].previous_free = free_back;
    lines[line_id].next_free = no_line;
    if (free_back == no_line) {
        free_front = line_id;
    } else {
        lines[free_back].next_free = line_id;
    }
    free_back = line_id;
    free_count++;
}

void kv_cache::size_class::unlink_free(std::uint32_t line_id) noexcept {
    cache_line& unlinked = lines[line_id];
    if (unlinked.previous_free == no_line) {
        free_front = unlinked.next_free;
    } else {
        lines[unlinked.previous_free].next_free = unlinked.next_free;
    }
    if (unlinked.next_free == no_line) {
        free_back = unlinked.previous_free;
    } else {
        lines[unlinked.next_free].previous_free = unlinked.previous_free;
    }
    free_count--;
}

kv_cache::kv_cache(
    std::span<const page_shape> shapes,
    std::size_t pool_bytes,
    std::size_t host_page_bytes,
    at::Device device,
    page_store& store
)
: pool(make_pool(shapes, pool_bytes, device)),
  store(&store),
  host_pages(host_page_bytes, largest_line_bytes(shapes)) {
    layers.reserve(shapes.size());
    for (const page_shape& shape : shapes) {
        auto same_size = [&](const size_class& c) { return c.line_bytes == line_bytes(shape); };
        auto found = std::ranges::find_if(classes, same_size);
        std::size_t class_index = static_cast<std::size_t>(found - classes.begin());
        if (found == classes.end()) {
            classes.push_back(size_class{.line_bytes = line_bytes(shape)});
        }
        layers.push_back(
            cache_layer{
                .shape = shape,
                .size_class = class_index,
                .device_kv = pool_views(pool, shape),
            }
        );
    }
}

std::size_t kv_cache::pool_bytes() const noexcept {
    return static_cast<std::size_t>(pool.numel());
}

kv_cache::cache_line& kv_cache::line(std::size_t layer, std::uint32_t line_id) noexcept {
    return classes[layers[layer].size_class].lines[line_id];
}

const kv_cache::cache_line& kv_cache::line(std::size_t layer, std::uint32_t line_id) const noexcept {
    return classes[layers[layer].size_class].lines[line_id];
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

void kv_cache::hold(std::size_t layer, std::uint32_t line_id) noexcept {
    cache_line& held = line(layer, line_id);
    if (held.ref == 0) {
        classes[layers[layer].size_class].unlink_free(line_id);
    }
    held.ref++;
}

// Reusing a mapped line drops its page from the layer that cached it, which may be another layer of the class.
std::uint32_t kv_cache::take_free_line(std::size_t layer) noexcept {
    size_class& taken_from = classes[layers[layer].size_class];
    std::uint32_t line_id = taken_from.free_front;
    if (line_id != no_line) {
        taken_from.unlink_free(line_id);
        const cache_line& reused = taken_from.lines[line_id];
        if (is_mapped(reused.layer, line_id)) {
            layers[reused.layer].digest_to_line_id.erase(*reused.hash);
        }
    } else {
        std::size_t first = ceil_div(bump, taken_from.line_bytes) * taken_from.line_bytes;
        assert(first + taken_from.line_bytes <= pool_bytes());
        line_id = static_cast<std::uint32_t>(first / taken_from.line_bytes);
        if (taken_from.lines.size() <= line_id) {
            taken_from.lines.resize(line_id + 1);
        }
        bump = first + taken_from.line_bytes;
    }
    cache_line& taken = taken_from.lines[line_id];
    taken.hash.reset();
    taken.layer = layer;
    taken.ref = 1;
    return line_id;
}

std::vector<std::uint32_t> kv_cache::take_free_lines(std::size_t layer, std::size_t count) noexcept {
    std::vector<std::uint32_t> line_ids(count);
    for (std::uint32_t& line_id : line_ids) {
        line_id = take_free_line(layer);
    }
    return line_ids;
}

void kv_cache::assign_hash(std::size_t layer, std::uint32_t line_id, const digest& hash) {
    line(layer, line_id).hash = hash;
    layers[layer].digest_to_line_id.try_emplace(hash, line_id);
}

bool kv_cache::covers(std::span<const std::size_t> counts, std::vector<std::size_t>&& free_lines) const noexcept {
    std::size_t carved_end = bump;
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        std::size_t class_index = layers[layer].size_class;
        std::size_t from_free = std::min(free_lines[class_index], counts[layer]);
        free_lines[class_index] -= from_free;
        std::size_t carved = counts[layer] - from_free;
        if (carved == 0) {
            continue;
        }
        std::size_t size = classes[class_index].line_bytes;
        std::size_t first = ceil_div(carved_end, size) * size;
        if (first > pool_bytes() || carved > (pool_bytes() - first) / size) {
            return false;
        }
        carved_end = first + carved * size;
    }
    return true;
}

std::vector<std::size_t> kv_cache::free_counts() const {
    std::vector<std::size_t> counts;
    counts.reserve(classes.size());
    for (const size_class& c : classes) {
        counts.push_back(c.free_count);
    }
    return counts;
}

std::size_t kv_cache::line_offset(std::size_t layer, std::uint32_t line_id) const noexcept {
    return line_id * classes[layers[layer].size_class].line_bytes;
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

// Every layer's resident pages are held before any free line is taken, so taking one never drops a page this call
// needs, also when the layers share a size class.
task<std::expected<std::vector<std::vector<std::uint32_t>>, page_error>>
kv_cache::get(std::span<const std::span<const digest>> hashes, cudaStream_t stream) {
    assert(hashes.size() == layers.size());
    std::vector<std::size_t> lines_needed(layers.size());
    std::vector<std::size_t> free_lines = free_counts();
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        const cache_layer& cached = layers[layer];
        for (const digest& hash : hashes[layer]) {
            if (auto it = cached.digest_to_line_id.find(hash); it != cached.digest_to_line_id.end()) {
                free_lines[cached.size_class] -= line(layer, it->second).ref == 0;
            } else if (!cached.fetching.contains(hash)) {
                lines_needed[layer]++;
            }
        }
    }
    if (!covers(lines_needed, std::move(free_lines))) {
        co_return std::unexpected(page_error::oom);
    }

    std::vector<std::vector<std::uint32_t>> line_ids(layers.size());
    std::vector<std::vector<std::size_t>> joined(layers.size());
    std::vector<std::vector<std::size_t>> to_fetch(layers.size());
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        cache_layer& cached = layers[layer];
        std::span<const digest> layer_hashes = hashes[layer];
        std::vector<std::uint32_t>& layer_lines = line_ids[layer];
        layer_lines.resize(layer_hashes.size());
        for (std::size_t i = 0; i < layer_hashes.size(); i++) {
            if (auto it = cached.digest_to_line_id.find(layer_hashes[i]); it != cached.digest_to_line_id.end()) {
                layer_lines[i] = it->second;
                hold(layer, layer_lines[i]);
            } else if (auto it = cached.fetching.find(layer_hashes[i]); it != cached.fetching.end()) {
                layer_lines[i] = it->second.line_id;
                hold(layer, layer_lines[i]);
                joined[layer].push_back(i);
            } else {
                to_fetch[layer].push_back(i);
            }
        }
    }

    std::vector<digest> fetch_hashes;
    std::vector<digest> fetch_keys;
    std::vector<layer_line> fetch_lines;
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        for (std::size_t i : to_fetch[layer]) {
            const digest& hash = hashes[layer][i];
            line_ids[layer][i] = take_free_line(layer);
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

std::expected<std::vector<std::vector<std::uint32_t>>, page_error> kv_cache::allocate(std::size_t count) {
    std::vector<std::size_t> counts(layers.size(), count);
    if (!covers(counts, free_counts())) {
        return std::unexpected(page_error::oom);
    }
    std::vector<std::vector<std::uint32_t>> line_ids;
    line_ids.reserve(layers.size());
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        line_ids.push_back(take_free_lines(layer, count));
    }
    return line_ids;
}

std::expected<std::vector<std::uint32_t>, page_error>
kv_cache::clone(std::size_t layer, std::uint32_t line_id, std::size_t count, cudaStream_t stream) {
    assert(line(layer, line_id).ref > 0);
    std::vector<std::size_t> counts(layers.size(), 0);
    counts[layer] = count;
    if (!covers(counts, free_counts())) {
        return std::unexpected(page_error::oom);
    }
    std::vector<std::uint32_t> copies = take_free_lines(layer, count);
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
    const std::vector<digest>& hashes,
    const std::vector<std::uint64_t>& owners,
    const std::vector<std::vector<std::uint32_t>>& line_ids,
    cudaStream_t stream
) {
    assert(line_ids.size() == layers.size() && owners.size() == hashes.size());
    std::vector<std::size_t> distinct;
    std::vector<std::size_t> distinct_index(hashes.size());
    std::unordered_map<digest, std::size_t, digest_hash> seen;
    for (std::size_t i = 0; i < hashes.size(); i++) {
        auto [found, inserted] = seen.try_emplace(hashes[i], distinct.size());
        if (inserted) {
            distinct.push_back(i);
        }
        distinct_index[i] = found->second;
    }
    std::vector<layer_line> sources;
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        assert(line_ids[layer].size() == hashes.size());
        for (std::uint32_t line_id : line_ids[layer]) {
            assert(line(layer, line_id).ref > 0 && !line(layer, line_id).hash);
        }
        for (std::size_t i : distinct) {
            sources.push_back(layer_line{.layer = layer, .line_id = line_ids[layer][i]});
        }
    }
    if (sources.empty()) {
        co_return {};
    }

    std::optional<std::vector<page>> copies = allocate_host_pages(sources);
    if (!copies) {
        co_return std::unexpected(page_error::oom);
    }
    std::vector<digest> keys;
    keys.reserve(sources.size());
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        for (std::size_t i : distinct) {
            assign_hash(layer, line_ids[layer][i], hashes[i]);
            keys.push_back(store_key(hashes[i], layer));
        }
    }
    std::vector<page_owner> page_owners;
    page_owners.reserve(layers.size() * hashes.size());
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        for (std::size_t i = 0; i < hashes.size(); i++) {
            page_owners.push_back(page_owner{.request = owners[i], .page = layer * distinct.size() + distinct_index[i]});
        }
    }
    co_await copy_to_host(sources, *copies, stream);
    store->put(std::move(keys), std::move(*copies), std::move(page_owners));
    co_return {};
}

void kv_cache::release(std::size_t layer, std::span<const std::uint32_t> line_ids) noexcept {
    size_class& owner = classes[layers[layer].size_class];
    for (std::uint32_t line_id : line_ids) {
        cache_line& released = line(layer, line_id);
        assert(released.ref > 0);
        if (--released.ref > 0) {
            continue;
        }
        if (is_mapped(layer, line_id)) {
            owner.link_free_back(line_id);
        } else {
            released.hash.reset();
            owner.link_free_front(line_id);
        }
    }
}

std::size_t kv_cache::free_count() const {
    std::size_t low = 0;
    std::size_t high = std::numeric_limits<std::size_t>::max();
    for (const size_class& c : classes) {
        high = std::min(high, c.free_count + (pool_bytes() - bump) / c.line_bytes);
    }
    while (low < high) {
        std::size_t count = high - (high - low) / 2;
        if (covers(std::vector<std::size_t>(layers.size(), count), free_counts())) {
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
