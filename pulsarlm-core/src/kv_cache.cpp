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

at::Tensor make_staging(const page_shape& shape, std::size_t staging_pages) {
    TORCH_CHECK(staging_pages > 0, "kv_cache: the staging buffer must hold a page");
    return at::empty(
        {static_cast<std::int64_t>(staging_pages),
         static_cast<std::int64_t>(shape.n_layers),
         static_cast<std::int64_t>(shape.page_size),
         static_cast<std::int64_t>(shape.n_kv_heads),
         static_cast<std::int64_t>(shape.head_dim)},
        at::TensorOptions().dtype(shape.dtype).pinned_memory(true)
    );
}

void copy_out_of_line(const at::Tensor& lines, std::uint32_t line_id, const at::Tensor& target) {
    for (std::size_t layer = 0; layer < static_cast<std::size_t>(lines.size(0)); layer++) {
        layer_of(target, layer).copy_(line_layer(lines, layer, line_id), /*non_blocking=*/true);
    }
}

at::Tensor slot(const at::Tensor& staging, std::size_t index) {
    return staging.select(0, static_cast<std::int64_t>(index));
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

// Slots are reused only after the stream finished the copies out of them.
task<void>
kv_cache::copy_from_host(std::span<const std::uint32_t> line_ids, std::span<const page> pages, cudaStream_t stream) {
    co_await hold_staging();
    auto slot_count = static_cast<std::size_t>(staging.k.size(0));
    for (std::size_t first = 0; first < line_ids.size(); first += slot_count) {
        std::size_t count = std::min(slot_count, line_ids.size() - first);
        {
            auto guard = guard_stream(device_kv.k, stream);
            for (std::size_t j = 0; j < count; j++) {
                std::uint32_t line_id = line_ids[first + j];
                const page& source = pages[first + j];
                check_page(device_kv.k, line_id, source.k);
                check_page(device_kv.v, line_id, source.v);
                slot(staging.k, j).copy_(source.k);
                slot(staging.v, j).copy_(source.v);
                copy_into_line(device_kv.k, line_id, slot(staging.k, j));
                copy_into_line(device_kv.v, line_id, slot(staging.v, j));
            }
        }
        co_await synchronize(stream);
    }
    release_staging();
}

task<void> kv_cache::copy_to_host(std::span<const std::uint32_t> line_ids, std::span<page> pages, cudaStream_t stream) {
    co_await hold_staging();
    auto slot_count = static_cast<std::size_t>(staging.k.size(0));
    for (std::size_t first = 0; first < line_ids.size(); first += slot_count) {
        std::size_t count = std::min(slot_count, line_ids.size() - first);
        {
            auto guard = guard_stream(device_kv.k, stream);
            for (std::size_t j = 0; j < count; j++) {
                copy_out_of_line(device_kv.k, line_ids[first + j], slot(staging.k, j));
                copy_out_of_line(device_kv.v, line_ids[first + j], slot(staging.v, j));
            }
        }
        co_await synchronize(stream);
        for (std::size_t j = 0; j < count; j++) {
            pages[first + j].k.copy_(slot(staging.k, j));
            pages[first + j].v.copy_(slot(staging.v, j));
        }
    }
    release_staging();
}

void kv_cache::copy_line(std::uint32_t source, std::span<const std::uint32_t> targets, cudaStream_t stream) {
    auto guard = guard_stream(device_kv.k, stream);
    at::Tensor k_source = device_kv.k.select(1, static_cast<std::int64_t>(source));
    at::Tensor v_source = device_kv.v.select(1, static_cast<std::int64_t>(source));
    for (std::uint32_t target : targets) {
        copy_into_line(device_kv.k, target, k_source);
        copy_into_line(device_kv.v, target, v_source);
    }
}

// A failed host allocation throws OutOfMemoryError.
std::optional<std::vector<page>> kv_cache::allocate_host_pages(std::size_t count) const {
    at::Tensor line = device_kv.k.select(1, 0);
    at::TensorOptions options = line.options().device(at::kCPU);
    std::vector<page> pages;
    pages.reserve(count);
    try {
        for (std::size_t i = 0; i < count; i++) {
            pages.push_back(page{.k = at::empty(line.sizes(), options), .v = at::empty(line.sizes(), options)});
        }
    } catch (const c10::OutOfMemoryError&) {
        return std::nullopt;
    }
    return pages;
}

kv_cache::kv_cache(
    const page_shape& shape,
    std::size_t n_lines,
    std::size_t staging_pages,
    at::Device device,
    std::unique_ptr<page_store>&& store
)
: device_kv{.k = make_lines(shape, n_lines, device), .v = make_lines(shape, n_lines, device)},
  lines(n_lines),
  store(std::move(store)),
  staging{.k = make_staging(shape, staging_pages), .v = make_staging(shape, staging_pages)} {
    for (std::uint32_t line_id = 0; line_id < n_lines; line_id++) {
        link_free_back(line_id);
    }
}

void kv_cache::link_free_front(std::uint32_t line_id) noexcept {
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

void kv_cache::link_free_back(std::uint32_t line_id) noexcept {
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

void kv_cache::unlink_free(std::uint32_t line_id) noexcept {
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
bool kv_cache::is_mapped(std::uint32_t line_id) const noexcept {
    const std::optional<digest>& hash = lines[line_id].hash;
    if (!hash) {
        return false;
    }
    auto it = digest_to_line_id.find(*hash);
    return it != digest_to_line_id.end() && it->second == line_id;
}

void kv_cache::hold(std::uint32_t line_id) noexcept {
    if (lines[line_id].ref == 0) {
        unlink_free(line_id);
    }
    lines[line_id].ref++;
}

std::uint32_t kv_cache::take_free_line() noexcept {
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

void kv_cache::assign_hash(std::uint32_t line_id, const digest& hash) {
    lines[line_id].hash = hash;
    digest_to_line_id.try_emplace(hash, line_id);
}

task<std::expected<std::vector<std::uint32_t>, page_error>>
kv_cache::get(const std::vector<digest>& hashes, cudaStream_t stream) {
    std::size_t lines_needed = 0;
    std::size_t free_resident_count = 0;
    for (const digest& hash : hashes) {
        if (auto it = digest_to_line_id.find(hash); it != digest_to_line_id.end()) {
            free_resident_count += lines[it->second].ref == 0;
        } else if (!fetching.contains(hash)) {
            lines_needed++;
        }
    }
    if (lines_needed > free_count - free_resident_count) {
        co_return std::unexpected(page_error::oom);
    }

    // Resident pages are held before any free line is taken, so taking one never drops a page this call needs.
    std::vector<std::uint32_t> line_ids(hashes.size());
    std::vector<std::size_t> joined;
    std::vector<std::size_t> to_fetch;
    for (std::size_t i = 0; i < hashes.size(); i++) {
        if (auto it = digest_to_line_id.find(hashes[i]); it != digest_to_line_id.end()) {
            line_ids[i] = it->second;
            hold(line_ids[i]);
        } else if (auto it = fetching.find(hashes[i]); it != fetching.end()) {
            line_ids[i] = it->second.line_id;
            hold(line_ids[i]);
            joined.push_back(i);
        } else {
            to_fetch.push_back(i);
        }
    }

    std::vector<digest> fetch_hashes;
    std::vector<std::uint32_t> fetch_lines;
    for (std::size_t i : to_fetch) {
        line_ids[i] = take_free_line();
        fetching.try_emplace(hashes[i], line_ids[i]);
        fetch_hashes.push_back(hashes[i]);
        fetch_lines.push_back(line_ids[i]);
    }

    if (!fetch_hashes.empty()) {
        std::vector<std::optional<page>> pages = co_await store->fetch(fetch_hashes);
        assert(pages.size() == fetch_hashes.size());
        std::vector<std::uint32_t> arrived_lines;
        std::vector<page> arrived;
        for (std::size_t j = 0; j < pages.size(); j++) {
            if (pages[j]) {
                arrived_lines.push_back(fetch_lines[j]);
                arrived.push_back(std::move(*pages[j]));
            }
        }
        co_await copy_from_host(arrived_lines, arrived, stream);
        for (std::size_t j = 0; j < pages.size(); j++) {
            if (pages[j]) {
                assign_hash(fetch_lines[j], fetch_hashes[j]);
            }
            fetching.extract(fetch_hashes[j]).mapped().done.set();
        }
    }

    // An entry for the same hash in another line is a later fetch, after the one this call joined ended.
    for (std::size_t i : joined) {
        auto it = fetching.find(hashes[i]);
        if (it != fetching.end() && it->second.line_id == line_ids[i]) {
            co_await it->second.done;
        }
    }

    for (std::size_t i = 0; i < hashes.size(); i++) {
        if (lines[line_ids[i]].hash != hashes[i]) {
            release(line_ids);
            co_return std::unexpected(page_error::miss);
        }
    }
    co_return line_ids;
}

std::expected<std::vector<std::uint32_t>, page_error> kv_cache::allocate(std::size_t count) {
    if (count > free_count) {
        return std::unexpected(page_error::oom);
    }
    std::vector<std::uint32_t> line_ids(count);
    for (std::uint32_t& line_id : line_ids) {
        line_id = take_free_line();
    }
    return line_ids;
}

std::expected<std::vector<std::uint32_t>, page_error>
kv_cache::clone(std::uint32_t line_id, std::size_t count, cudaStream_t stream) {
    assert(lines[line_id].ref > 0);
    std::expected<std::vector<std::uint32_t>, page_error> line_ids = allocate(count);
    if (line_ids) {
        copy_line(line_id, *line_ids, stream);
    }
    return line_ids;
}

task<std::expected<void, page_error>>
kv_cache::sync(const std::vector<digest>& hashes, const std::vector<std::uint32_t>& line_ids, cudaStream_t stream) {
    assert(hashes.size() == line_ids.size());
    std::vector<digest> distinct_hashes;
    std::vector<std::uint32_t> distinct_lines;
    std::unordered_set<digest, digest_hash> seen;
    for (std::size_t i = 0; i < hashes.size(); i++) {
        assert(lines[line_ids[i]].ref > 0 && !lines[line_ids[i]].hash);
        if (seen.insert(hashes[i]).second) {
            distinct_hashes.push_back(hashes[i]);
            distinct_lines.push_back(line_ids[i]);
        }
    }
    if (distinct_hashes.empty()) {
        co_return {};
    }

    std::optional<std::vector<page>> host_pages = allocate_host_pages(distinct_lines.size());
    if (!host_pages) {
        co_return std::unexpected(page_error::oom);
    }
    for (std::size_t j = 0; j < distinct_hashes.size(); j++) {
        assign_hash(distinct_lines[j], distinct_hashes[j]);
    }
    co_await copy_to_host(distinct_lines, *host_pages, stream);
    store->put(std::move(distinct_hashes), std::move(*host_pages));
    co_return {};
}

void kv_cache::release(std::span<const std::uint32_t> line_ids) noexcept {
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

std::size_t kv_cache::free_line_count() const noexcept {
    return free_count;
}

const kv& kv_cache::tensors() const noexcept {
    return device_kv;
}

}  // namespace pulsarlm
