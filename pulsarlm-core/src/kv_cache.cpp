#include "pulsarlm/kv_cache.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>

#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_set>
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

at::Tensor make_lines(const page_shape& shape, std::size_t n_lines, at::Device device) {
    TORCH_CHECK(n_lines < std::numeric_limits<std::uint32_t>::max(), "kv_cache: too many lines: ", n_lines);
    std::vector<std::int64_t> sizes = page_sizes(shape);
    sizes.insert(sizes.begin() + 1, static_cast<std::int64_t>(n_lines));
    return at::empty(sizes, at::TensorOptions().dtype(shape.dtype).device(device));
}

kv make_line_tensors(const page_shape& shape, std::size_t n_lines, at::Device device) {
    at::Tensor v = make_lines(shape, n_lines, device);
    return kv{.k = shape.values_only ? v : make_lines(shape, n_lines, device), .v = v};
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

void copy_out_of_line(const at::Tensor& lines, std::uint32_t line_id, const at::Tensor& target) {
    for (std::size_t layer = 0; layer < static_cast<std::size_t>(lines.size(0)); layer++) {
        layer_of(target, layer).copy_(line_layer(lines, layer, line_id), /*non_blocking=*/true);
    }
}

at::Tensor make_staging(std::span<const page_shape> shapes, std::size_t staging_bytes) {
    for (const page_shape& shape : shapes) {
        TORCH_CHECK(
            line_bytes(shape) <= staging_bytes,
            "kv_cache: a staging buffer of ",
            staging_bytes,
            " bytes holds no page of ",
            line_bytes(shape),
            " bytes"
        );
    }
    return at::empty(
        {static_cast<std::int64_t>(staging_bytes)},
        at::TensorOptions().dtype(at::kByte).pinned_memory(true)
    );
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

task<void> kv_cache::hold_staging() {
    while (staging_held) {
        if (!staging_released) {
            staging_released = std::make_unique<event>();
        }
        co_await *staging_released;
    }
    staging_held = true;
}

// The awaiters a set event scheduled no longer refer to it, so it is destroyed right away.
void kv_cache::release_staging() noexcept {
    staging_held = false;
    if (staging_released) {
        staging_released->set();
        staging_released.reset();
    }
}

// A slot's tensors start at a multiple of their element size, as a typed view of the bytes requires.
std::optional<page> kv_cache::staging_slot(std::size_t layer, std::size_t& used) const {
    const page_shape& shape = layers[layer].shape;
    std::size_t element_bytes = c10::elementSize(shape.dtype);
    std::size_t tensor_bytes = line_bytes(shape) / (shape.values_only ? 1 : 2);
    std::size_t first = ceil_div(used, element_bytes) * element_bytes;
    if (first + line_bytes(shape) > static_cast<std::size_t>(staging.numel())) {
        return std::nullopt;
    }
    std::vector<std::int64_t> sizes = page_sizes(shape);
    auto tensor_at = [&](std::size_t offset) {
        return staging.narrow(0, static_cast<std::int64_t>(offset), static_cast<std::int64_t>(tensor_bytes))
            .view(shape.dtype)
            .view(sizes);
    };
    at::Tensor v = tensor_at(first);
    used = first + line_bytes(shape);
    return page{.k = shape.values_only ? v : tensor_at(first + tensor_bytes), .v = v};
}

// Slots are reused only after the stream finished the copies out of them. The staging buffer holds a page of any
// layer, so every round copies at least one.
task<void>
kv_cache::copy_from_host(std::span<const layer_line> targets, std::span<const page> pages, cudaStream_t stream) {
    co_await hold_staging();
    std::size_t first = 0;
    while (first < targets.size()) {
        std::size_t end = first;
        {
            auto guard = guard_stream(layers.front().device_kv.v, stream);
            std::size_t used = 0;
            for (; end < targets.size(); end++) {
                auto [layer, line_id] = targets[end];
                std::optional<page> slot = staging_slot(layer, used);
                if (!slot) {
                    break;
                }
                const cache_layer& target = layers[layer];
                const page& source = pages[end];
                check_page(target.device_kv.v, line_id, source.v);
                slot->v.copy_(source.v);
                copy_into_line(target.device_kv.v, line_id, slot->v);
                if (!target.shape.values_only) {
                    check_page(target.device_kv.k, line_id, source.k);
                    slot->k.copy_(source.k);
                    copy_into_line(target.device_kv.k, line_id, slot->k);
                }
            }
        }
        co_await synchronize(stream);
        first = end;
    }
    release_staging();
}

task<void> kv_cache::copy_to_host(std::span<const layer_line> sources, std::span<page> pages, cudaStream_t stream) {
    co_await hold_staging();
    std::size_t first = 0;
    while (first < sources.size()) {
        std::vector<page> slots;
        {
            auto guard = guard_stream(layers.front().device_kv.v, stream);
            std::size_t used = 0;
            while (first + slots.size() < sources.size()) {
                auto [layer, line_id] = sources[first + slots.size()];
                std::optional<page> slot = staging_slot(layer, used);
                if (!slot) {
                    break;
                }
                const cache_layer& source = layers[layer];
                copy_out_of_line(source.device_kv.v, line_id, slot->v);
                if (!source.shape.values_only) {
                    copy_out_of_line(source.device_kv.k, line_id, slot->k);
                }
                slots.push_back(std::move(*slot));
            }
        }
        co_await synchronize(stream);
        for (std::size_t j = 0; j < slots.size(); j++) {
            page& target = pages[first + j];
            target.v.copy_(slots[j].v);
            if (!layers[sources[first + j].layer].shape.values_only) {
                target.k.copy_(slots[j].k);
            }
        }
        first += slots.size();
    }
    release_staging();
}

// A failed host allocation throws OutOfMemoryError.
std::optional<std::vector<page>> kv_cache::allocate_host_pages(std::span<const layer_line> sources) const {
    std::vector<page> pages;
    pages.reserve(sources.size());
    try {
        for (const layer_line& source : sources) {
            const page_shape& shape = layers[source.layer].shape;
            at::TensorOptions options = at::TensorOptions().dtype(shape.dtype);
            at::Tensor v = at::empty(page_sizes(shape), options);
            pages.push_back(page{.k = shape.values_only ? v : at::empty(page_sizes(shape), options), .v = v});
        }
    } catch (const c10::OutOfMemoryError&) {
        return std::nullopt;
    }
    return pages;
}

void kv_cache::release_all(std::span<const std::vector<std::uint32_t>> line_ids) noexcept {
    for (std::size_t layer = 0; layer < line_ids.size(); layer++) {
        layers[layer].release(line_ids[layer]);
    }
}

kv_cache::cache_layer::cache_layer(const page_shape& shape, std::size_t n_lines, at::Device device)
: shape(shape),
  device_kv(make_line_tensors(shape, n_lines, device)),
  lines(n_lines) {
    for (std::uint32_t line_id = 0; line_id < n_lines; line_id++) {
        link_free_back(line_id);
    }
}

void kv_cache::cache_layer::link_free_front(std::uint32_t line_id) noexcept {
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

void kv_cache::cache_layer::link_free_back(std::uint32_t line_id) noexcept {
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

void kv_cache::cache_layer::unlink_free(std::uint32_t line_id) noexcept {
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

// A line keeps its hash unmapped when another line cached that hash before its fetch or sync completed.
bool kv_cache::cache_layer::is_mapped(std::uint32_t line_id) const noexcept {
    const std::optional<digest>& hash = lines[line_id].hash;
    if (!hash) {
        return false;
    }
    auto it = digest_to_line_id.find(*hash);
    return it != digest_to_line_id.end() && it->second == line_id;
}

void kv_cache::cache_layer::hold(std::uint32_t line_id) noexcept {
    if (lines[line_id].ref == 0) {
        unlink_free(line_id);
    }
    lines[line_id].ref++;
}

std::uint32_t kv_cache::cache_layer::take_free_line() noexcept {
    assert(free_front != no_line);
    std::uint32_t line_id = free_front;
    unlink_free(line_id);
    if (is_mapped(line_id)) {
        digest_to_line_id.erase(*lines[line_id].hash);
    }
    lines[line_id].hash.reset();
    lines[line_id].ref = 1;
    return line_id;
}

std::vector<std::uint32_t> kv_cache::cache_layer::take_free_lines(std::size_t count) noexcept {
    assert(count <= free_count);
    std::vector<std::uint32_t> line_ids(count);
    for (std::uint32_t& line_id : line_ids) {
        line_id = take_free_line();
    }
    return line_ids;
}

void kv_cache::cache_layer::assign_hash(std::uint32_t line_id, const digest& hash) {
    lines[line_id].hash = hash;
    digest_to_line_id.try_emplace(hash, line_id);
}

void kv_cache::cache_layer::release(std::span<const std::uint32_t> line_ids) noexcept {
    for (std::uint32_t line_id : line_ids) {
        cache_line& released = lines[line_id];
        assert(released.ref > 0);
        if (--released.ref > 0) {
            continue;
        }
        if (is_mapped(line_id)) {
            link_free_back(line_id);
        } else {
            released.hash.reset();
            link_free_front(line_id);
        }
    }
}

kv_cache::kv_cache(
    std::span<const page_shape> shapes,
    std::span<const std::size_t> line_counts,
    std::size_t staging_bytes,
    at::Device device,
    page_store& store
)
: store(&store),
  staging(make_staging(shapes, staging_bytes)) {
    TORCH_CHECK(!shapes.empty() && shapes.size() == line_counts.size(), "kv_cache: one line count per KV layer");
    layers.reserve(shapes.size());
    for (std::size_t layer = 0; layer < shapes.size(); layer++) {
        layers.emplace_back(shapes[layer], line_counts[layer], device);
    }
}

task<std::expected<std::vector<std::vector<std::uint32_t>>, page_error>>
kv_cache::get(std::span<const std::span<const digest>> hashes, cudaStream_t stream) {
    assert(hashes.size() == layers.size());
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        const cache_layer& cached = layers[layer];
        std::size_t lines_needed = 0;
        std::size_t free_resident_count = 0;
        for (const digest& hash : hashes[layer]) {
            if (auto it = cached.digest_to_line_id.find(hash); it != cached.digest_to_line_id.end()) {
                free_resident_count += cached.lines[it->second].ref == 0;
            } else if (!cached.fetching.contains(hash)) {
                lines_needed++;
            }
        }
        if (lines_needed > cached.free_count - free_resident_count) {
            co_return std::unexpected(page_error::oom);
        }
    }

    // A layer's resident pages are held before any of its free lines is taken, so taking one never drops a page this
    // call needs.
    std::vector<std::vector<std::uint32_t>> line_ids(layers.size());
    std::vector<std::vector<std::size_t>> joined(layers.size());
    std::vector<digest> fetch_hashes;
    std::vector<digest> fetch_keys;
    std::vector<layer_line> fetch_lines;
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        cache_layer& cached = layers[layer];
        std::span<const digest> layer_hashes = hashes[layer];
        std::vector<std::uint32_t>& layer_lines = line_ids[layer];
        layer_lines.resize(layer_hashes.size());
        std::vector<std::size_t> to_fetch;
        for (std::size_t i = 0; i < layer_hashes.size(); i++) {
            if (auto it = cached.digest_to_line_id.find(layer_hashes[i]); it != cached.digest_to_line_id.end()) {
                layer_lines[i] = it->second;
                cached.hold(layer_lines[i]);
            } else if (auto it = cached.fetching.find(layer_hashes[i]); it != cached.fetching.end()) {
                layer_lines[i] = it->second.line_id;
                cached.hold(layer_lines[i]);
                joined[layer].push_back(i);
            } else {
                to_fetch.push_back(i);
            }
        }
        for (std::size_t i : to_fetch) {
            layer_lines[i] = cached.take_free_line();
            cached.fetching.try_emplace(layer_hashes[i], layer_lines[i]);
            fetch_hashes.push_back(layer_hashes[i]);
            fetch_keys.push_back(store_key(layer_hashes[i], layer));
            fetch_lines.push_back(layer_line{.layer = layer, .line_id = layer_lines[i]});
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
            cache_layer& cached = layers[fetch_lines[j].layer];
            if (pages[j]) {
                cached.assign_hash(fetch_lines[j].line_id, fetch_hashes[j]);
            }
            cached.fetching.extract(fetch_hashes[j]).mapped().done.set();
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
            if (layers[layer].lines[line_ids[layer][i]].hash != hashes[layer][i]) {
                release_all(line_ids);
                co_return std::unexpected(page_error::miss);
            }
        }
    }
    co_return line_ids;
}

std::expected<std::vector<std::vector<std::uint32_t>>, page_error> kv_cache::allocate(std::size_t count) {
    for (const cache_layer& cached : layers) {
        if (count > cached.free_count) {
            return std::unexpected(page_error::oom);
        }
    }
    std::vector<std::vector<std::uint32_t>> line_ids;
    line_ids.reserve(layers.size());
    for (cache_layer& cached : layers) {
        line_ids.push_back(cached.take_free_lines(count));
    }
    return line_ids;
}

std::expected<std::vector<std::uint32_t>, page_error>
kv_cache::clone(std::size_t layer, std::uint32_t line_id, std::size_t count, cudaStream_t stream) {
    cache_layer& cached = layers[layer];
    assert(cached.lines[line_id].ref > 0);
    if (count > cached.free_count) {
        return std::unexpected(page_error::oom);
    }
    std::vector<std::uint32_t> copies = cached.take_free_lines(count);
    auto guard = guard_stream(cached.device_kv.v, stream);
    at::Tensor k_source = cached.device_kv.k.select(1, static_cast<std::int64_t>(line_id));
    at::Tensor v_source = cached.device_kv.v.select(1, static_cast<std::int64_t>(line_id));
    for (std::uint32_t copy : copies) {
        copy_into_line(cached.device_kv.v, copy, v_source);
        if (!cached.shape.values_only) {
            copy_into_line(cached.device_kv.k, copy, k_source);
        }
    }
    return copies;
}

task<std::expected<void, page_error>> kv_cache::sync(
    const std::vector<digest>& hashes,
    const std::vector<std::vector<std::uint32_t>>& line_ids,
    cudaStream_t stream
) {
    assert(line_ids.size() == layers.size());
    std::vector<std::size_t> distinct;
    std::unordered_set<digest, digest_hash> seen;
    for (std::size_t i = 0; i < hashes.size(); i++) {
        if (seen.insert(hashes[i]).second) {
            distinct.push_back(i);
        }
    }
    std::vector<layer_line> sources;
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        assert(line_ids[layer].size() == hashes.size());
        for (std::uint32_t line_id : line_ids[layer]) {
            assert(layers[layer].lines[line_id].ref > 0 && !layers[layer].lines[line_id].hash);
        }
        for (std::size_t i : distinct) {
            sources.push_back(layer_line{.layer = layer, .line_id = line_ids[layer][i]});
        }
    }
    if (sources.empty()) {
        co_return {};
    }

    std::optional<std::vector<page>> host_pages = allocate_host_pages(sources);
    if (!host_pages) {
        co_return std::unexpected(page_error::oom);
    }
    std::vector<digest> keys;
    keys.reserve(sources.size());
    for (std::size_t layer = 0; layer < layers.size(); layer++) {
        for (std::size_t i : distinct) {
            layers[layer].assign_hash(line_ids[layer][i], hashes[i]);
            keys.push_back(store_key(hashes[i], layer));
        }
    }
    co_await copy_to_host(sources, *host_pages, stream);
    store->put(std::move(keys), std::move(*host_pages));
    co_return {};
}

void kv_cache::release(std::size_t layer, std::span<const std::uint32_t> line_ids) noexcept {
    layers[layer].release(line_ids);
}

std::size_t kv_cache::layer_count() const noexcept {
    return layers.size();
}

std::size_t kv_cache::line_count(std::size_t layer) const noexcept {
    return layers[layer].lines.size();
}

std::size_t kv_cache::free_line_count(std::size_t layer) const noexcept {
    return layers[layer].free_count;
}

const kv& kv_cache::tensors(std::size_t layer) const noexcept {
    return layers[layer].device_kv;
}

}  // namespace pulsarlm
