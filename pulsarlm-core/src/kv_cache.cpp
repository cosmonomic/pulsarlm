#include "pulsarlm/kv_cache.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>

#include <algorithm>
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

std::size_t page_bytes(const page_shape& shape) {
    return line_bytes(shape) / (shape.values_only ? 1 : 2);
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
    return at::empty(
        {static_cast<std::int64_t>(pool_bytes)},
        at::TensorOptions().dtype(at::kByte).device(device)
    );
}

// A K and V layer's V view starts page_bytes into the pool, so both views hold one line fewer than fit in the pool.
kv pool_views(const at::Tensor& pool, const page_shape& shape) {
    std::size_t bytes = page_bytes(shape);
    std::size_t n_lines = static_cast<std::size_t>(pool.numel()) / bytes - (shape.values_only ? 0 : 1);
    std::vector<std::int64_t> sizes = page_sizes(shape);
    sizes.front() = static_cast<std::int64_t>(n_lines);
    auto view_from = [&](std::size_t first_byte) {
        return pool.narrow(0, static_cast<std::int64_t>(first_byte), static_cast<std::int64_t>(n_lines * bytes))
            .view(shape.dtype)
            .view(sizes);
    };
    at::Tensor k = view_from(0);
    return kv{.k = k, .v = shape.values_only ? k : view_from(bytes)};
}

c10::cuda::CUDAStreamGuard guard_stream(const at::Tensor& lines, cudaStream_t stream) {
    return c10::cuda::CUDAStreamGuard(c10::cuda::getStreamFromExternal(stream, lines.device().index()));
}

at::Tensor line_of(const at::Tensor& lines, std::uint32_t line_id) {
    return lines.select(0, static_cast<std::int64_t>(line_id));
}

void copy_into_line(const at::Tensor& lines, std::uint32_t line_id, const at::Tensor& source) {
    line_of(lines, line_id).copy_(source.select(0, 0), /*non_blocking=*/true);
}

void copy_out_of_line(const at::Tensor& lines, std::uint32_t line_id, const at::Tensor& target) {
    target.select(0, 0).copy_(line_of(lines, line_id), /*non_blocking=*/true);
}

void check_page(const page_shape& shape, const at::Tensor& source) {
    std::vector<std::int64_t> sizes = page_sizes(shape);
    TORCH_CHECK(source.sizes() == at::IntArrayRef(sizes), "kv_cache: page is ", source.sizes(), ", line is ", sizes);
    TORCH_CHECK(source.scalar_type() == shape.dtype, "kv_cache: page is ", source.dtype(), ", line is ", shape.dtype);
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

void kv_cache::size_class::link_free_front(std::uint32_t slot) noexcept {
    lines[slot].previous_free = no_slot;
    lines[slot].next_free = free_front;
    if (free_front == no_slot) {
        free_back = slot;
    } else {
        lines[free_front].previous_free = slot;
    }
    free_front = slot;
    free_count++;
}

void kv_cache::size_class::link_free_back(std::uint32_t slot) noexcept {
    lines[slot].previous_free = free_back;
    lines[slot].next_free = no_slot;
    if (free_back == no_slot) {
        free_front = slot;
    } else {
        lines[free_back].next_free = slot;
    }
    free_back = slot;
    free_count++;
}

void kv_cache::size_class::unlink_free(std::uint32_t slot) noexcept {
    cache_line& unlinked = lines[slot];
    if (unlinked.previous_free == no_slot) {
        free_front = unlinked.next_free;
    } else {
        lines[unlinked.previous_free].next_free = unlinked.next_free;
    }
    if (unlinked.next_free == no_slot) {
        free_back = unlinked.previous_free;
    } else {
        lines[unlinked.next_free].previous_free = unlinked.previous_free;
    }
    free_count--;
}

kv_cache::kv_cache(
    std::span<const page_shape> shapes,
    std::size_t pool_bytes,
    std::size_t staging_bytes,
    at::Device device,
    page_store& store
)
: pool(make_pool(shapes, pool_bytes, device)),
  store(&store),
  staging(make_staging(shapes, staging_bytes)) {
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
                .page_bytes = page_bytes(shape),
                .device_kv = pool_views(pool, shape),
            }
        );
    }
}

std::size_t kv_cache::pool_bytes() const noexcept {
    return static_cast<std::size_t>(pool.numel());
}

std::uint32_t kv_cache::slot_of(std::size_t layer, std::uint32_t line_id) const noexcept {
    const cache_layer& owner = layers[layer];
    return static_cast<std::uint32_t>(line_id * owner.page_bytes / classes[owner.size_class].line_bytes);
}

std::uint32_t kv_cache::line_id_of(std::size_t layer, std::uint32_t slot) const noexcept {
    const cache_layer& owner = layers[layer];
    return static_cast<std::uint32_t>(slot * classes[owner.size_class].line_bytes / owner.page_bytes);
}

kv_cache::cache_line& kv_cache::line(std::size_t layer, std::uint32_t line_id) noexcept {
    return classes[layers[layer].size_class].lines[slot_of(layer, line_id)];
}

const kv_cache::cache_line& kv_cache::line(std::size_t layer, std::uint32_t line_id) const noexcept {
    return classes[layers[layer].size_class].lines[slot_of(layer, line_id)];
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
        classes[layers[layer].size_class].unlink_free(slot_of(layer, line_id));
    }
    held.ref++;
}

// Reusing a mapped line drops its page from the layer that cached it, which may be another layer of the class.
std::uint32_t kv_cache::take_free_line(std::size_t layer) noexcept {
    size_class& taken_from = classes[layers[layer].size_class];
    std::uint32_t slot = taken_from.free_front;
    if (slot != no_slot) {
        taken_from.unlink_free(slot);
        const cache_line& reused = taken_from.lines[slot];
        std::uint32_t reused_id = line_id_of(reused.layer, slot);
        if (is_mapped(reused.layer, reused_id)) {
            layers[reused.layer].digest_to_line_id.erase(*reused.hash);
        }
    } else {
        std::size_t first = ceil_div(bump, taken_from.line_bytes) * taken_from.line_bytes;
        assert(first + taken_from.line_bytes <= pool_bytes());
        slot = static_cast<std::uint32_t>(first / taken_from.line_bytes);
        if (taken_from.lines.size() <= slot) {
            taken_from.lines.resize(slot + 1);
        }
        taken_from.line_count++;
        bump = first + taken_from.line_bytes;
    }
    cache_line& taken = taken_from.lines[slot];
    taken.hash.reset();
    taken.layer = layer;
    taken.ref = 1;
    return line_id_of(layer, slot);
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
    std::size_t tensor_bytes = page_bytes(shape);
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
            auto guard = guard_stream(pool, stream);
            std::size_t used = 0;
            for (; end < targets.size(); end++) {
                auto [layer, line_id] = targets[end];
                std::optional<page> slot = staging_slot(layer, used);
                if (!slot) {
                    break;
                }
                const cache_layer& target = layers[layer];
                const page& source = pages[end];
                check_page(target.shape, source.v);
                slot->v.copy_(source.v);
                copy_into_line(target.device_kv.v, line_id, slot->v);
                if (!target.shape.values_only) {
                    check_page(target.shape, source.k);
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
            auto guard = guard_stream(pool, stream);
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
    if (!fits_free(counts)) {
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
    if (!fits_free(counts)) {
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
            assert(line(layer, line_id).ref > 0 && !line(layer, line_id).hash);
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
            assign_hash(layer, line_ids[layer][i], hashes[i]);
            keys.push_back(store_key(hashes[i], layer));
        }
    }
    co_await copy_to_host(sources, *host_pages, stream);
    store->put(std::move(keys), std::move(*host_pages));
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
            owner.link_free_back(slot_of(layer, line_id));
        } else {
            released.hash.reset();
            owner.link_free_front(slot_of(layer, line_id));
        }
    }
}

bool kv_cache::fits(std::span<const std::size_t> counts) const {
    std::vector<std::size_t> lines;
    lines.reserve(classes.size());
    for (const size_class& c : classes) {
        lines.push_back(c.line_count);
    }
    return covers(counts, std::move(lines));
}

bool kv_cache::fits_free(std::span<const std::size_t> counts) const {
    return covers(counts, free_counts());
}

std::size_t kv_cache::layer_count() const noexcept {
    return layers.size();
}

const kv& kv_cache::tensors(std::size_t layer) const noexcept {
    return layers[layer].device_kv;
}

}  // namespace pulsarlm
