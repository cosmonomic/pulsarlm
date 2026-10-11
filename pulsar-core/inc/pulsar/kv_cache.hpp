#pragma once

#include <ATen/core/Tensor.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pulsar/digest.hpp"
#include "pulsar/event_loop.hpp"

namespace pulsar {

struct page {
    at::Tensor k;  // [n_layers, page_size, n_kv_heads, head_dim]
    at::Tensor v;
};

struct page_shape {
    std::size_t n_layers;
    std::size_t page_size;
    std::size_t n_kv_heads;
    std::size_t head_dim;
    at::ScalarType dtype;
};

// fetch returns the pages in the order of hashes, nullopt for those it does not have. put takes ownership of host
// pages and their hashes, matched by index, and returns without waiting for the write.
template <typename t>
concept page_store = requires(t s, std::span<const digest> hashes, std::vector<digest> keys, std::vector<page> pages) {
    { s.fetch(hashes) } -> std::same_as<task<std::vector<std::optional<page>>>>;
    { s.put(std::move(keys), std::move(pages)) } -> std::same_as<void>;
};

enum class page_error { miss, oom };

template <page_store t_store> class kv_cache;

class line_pool {
    at::Tensor k;  // [n_layers, n_lines, page_size, n_kv_heads, head_dim]
    at::Tensor v;

    line_pool(const page_shape& shape, std::size_t n_lines, at::Device device);

    void upload(std::uint32_t line_id, const page& source, cudaStream_t stream);
    void clone(std::uint32_t source, std::span<const std::uint32_t> targets, cudaStream_t stream);
    // Pinned host copies, complete once stream reaches this point. nullopt when pinned host memory runs out.
    std::optional<std::vector<page>> download(std::span<const std::uint32_t> line_ids, cudaStream_t stream) const;

    at::Tensor k_pool(std::size_t layer) const;
    at::Tensor v_pool(std::size_t layer) const;

    template <page_store> friend class kv_cache;
};

template <page_store t_store> class kv_cache {
  private:
    struct line;
    struct fetch;

    static constexpr std::uint32_t no_line = std::numeric_limits<std::uint32_t>::max();

    line_pool pool;
    // Indexed by line.
    std::vector<line> lines;
    // Free lines in reuse order: unmapped lines first, then mapped lines least recently released first.
    std::uint32_t free_front = no_line;
    std::uint32_t free_back = no_line;
    std::size_t free_count = 0;
    std::unordered_map<digest, std::uint32_t, digest_hash> digest_to_line_id;
    // Pages a get is fetching into a line, not yet in digest_to_line_id.
    std::unordered_map<digest, fetch, digest_hash> fetching;
    t_store store;

    void link_free_front(std::uint32_t line_id) noexcept;
    void link_free_back(std::uint32_t line_id) noexcept;
    void unlink_free(std::uint32_t line_id) noexcept;
    bool is_mapped(std::uint32_t line_id) const noexcept;
    void hold(std::uint32_t line_id) noexcept;
    std::uint32_t take_free_line() noexcept;
    void assign_hash(std::uint32_t line_id, const digest& hash);

  public:
    kv_cache(const page_shape& shape, std::size_t n_lines, at::Device device, t_store store);

    // Copies run on stream. Callers release lines only after the GPU work using them completes. get, allocate and
    // clone never wait for lines. A line is free when it is not held; free lines are reused least recently released
    // first and their pages dropped. Without enough free lines the call fails with oom. A failed call holds nothing.

    // Gives every page a line in the order of hashes, fetching the pages that are not resident from the store, and
    // holds the lines until release. A page another get is fetching shares that line and fetch. Fails with miss when
    // the store does not have a page.
    task<std::expected<std::vector<std::uint32_t>, page_error>> get(std::vector<digest> hashes, cudaStream_t stream);
    // Gives count empty lines, held until release. get does not find their pages until they are put, and releasing a
    // line before it is put drops its page.
    std::expected<std::vector<std::uint32_t>, page_error> allocate(std::size_t count);
    // Gives count unmapped lines holding a copy of the page at the held line_id, held until release.
    std::expected<std::vector<std::uint32_t>, page_error>
    clone(std::uint32_t line_id, std::size_t count, cudaStream_t stream);
    // Caches the full page at each held line under the hash at the same index, and puts a host copy of each into the
    // store. A hash already cached at another line leaves its line unmapped. Completes once the copies are handed to
    // the store. Fails with oom, caching nothing, when the host copies cannot be allocated.
    task<std::expected<void, page_error>>
    put(std::vector<digest> hashes, std::vector<std::uint32_t> line_ids, cudaStream_t stream);
    void release(std::span<const std::uint32_t> line_ids) noexcept;

    std::size_t free_line_count() const noexcept;
    at::Tensor k_pool(std::size_t layer) const;
    at::Tensor v_pool(std::size_t layer) const;
};

template <page_store t_store> struct kv_cache<t_store>::line {
    std::optional<digest> hash;
    std::uint32_t ref = 0;
    std::uint32_t previous_free = no_line;
    std::uint32_t next_free = no_line;
};

// done is set when the fetch completes. The page arrived when its line's hash then equals its digest.
template <page_store t_store> struct kv_cache<t_store>::fetch {
    std::uint32_t line_id;
    event done;
};

template <page_store t_store>
kv_cache<t_store>::kv_cache(const page_shape& shape, std::size_t n_lines, at::Device device, t_store store)
: pool(shape, n_lines, device),
  lines(n_lines),
  store(std::move(store)) {
    for (std::uint32_t line_id = 0; line_id < n_lines; line_id++) {
        link_free_back(line_id);
    }
}

template <page_store t_store> void kv_cache<t_store>::link_free_front(std::uint32_t line_id) noexcept {
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

template <page_store t_store> void kv_cache<t_store>::link_free_back(std::uint32_t line_id) noexcept {
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

template <page_store t_store> void kv_cache<t_store>::unlink_free(std::uint32_t line_id) noexcept {
    line& unlinked = lines[line_id];
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

// A line keeps its hash unmapped when another line cached that hash before its fetch or put completed.
template <page_store t_store> bool kv_cache<t_store>::is_mapped(std::uint32_t line_id) const noexcept {
    const std::optional<digest>& hash = lines[line_id].hash;
    if (!hash) {
        return false;
    }
    auto it = digest_to_line_id.find(*hash);
    return it != digest_to_line_id.end() && it->second == line_id;
}

template <page_store t_store> void kv_cache<t_store>::hold(std::uint32_t line_id) noexcept {
    if (lines[line_id].ref == 0) {
        unlink_free(line_id);
    }
    lines[line_id].ref++;
}

template <page_store t_store> std::uint32_t kv_cache<t_store>::take_free_line() noexcept {
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

template <page_store t_store> void kv_cache<t_store>::assign_hash(std::uint32_t line_id, const digest& hash) {
    lines[line_id].hash = hash;
    digest_to_line_id.try_emplace(hash, line_id);
}

template <page_store t_store>
task<std::expected<std::vector<std::uint32_t>, page_error>>
kv_cache<t_store>::get(std::vector<digest> hashes, cudaStream_t stream) {
    std::size_t lines_needed = 0;
    std::size_t free_resident_count = 0;
    std::unordered_set<digest, digest_hash> seen;
    for (const digest& hash : hashes) {
        if (!seen.insert(hash).second) {
            continue;
        }
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
        if (auto it = fetching.find(hashes[i]); it != fetching.end()) {
            line_ids[i] = it->second.line_id;
            hold(line_ids[i]);
            joined.push_back(i);
            continue;
        }
        line_ids[i] = take_free_line();
        fetching.try_emplace(hashes[i], line_ids[i]);
        fetch_hashes.push_back(hashes[i]);
        fetch_lines.push_back(line_ids[i]);
    }

    if (!fetch_hashes.empty()) {
        std::vector<std::optional<page>> pages = co_await store.fetch(fetch_hashes);
        assert(pages.size() == fetch_hashes.size());
        bool uploaded = false;
        for (std::size_t j = 0; j < pages.size(); j++) {
            if (pages[j]) {
                pool.upload(fetch_lines[j], *pages[j], stream);
                uploaded = true;
            }
        }
        if (uploaded) {
            co_await synchronize(stream);
        }
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

template <page_store t_store>
std::expected<std::vector<std::uint32_t>, page_error> kv_cache<t_store>::allocate(std::size_t count) {
    if (count > free_count) {
        return std::unexpected(page_error::oom);
    }
    std::vector<std::uint32_t> line_ids(count);
    for (std::uint32_t& line_id : line_ids) {
        line_id = take_free_line();
    }
    return line_ids;
}

template <page_store t_store>
std::expected<std::vector<std::uint32_t>, page_error>
kv_cache<t_store>::clone(std::uint32_t line_id, std::size_t count, cudaStream_t stream) {
    assert(lines[line_id].ref > 0);
    std::expected<std::vector<std::uint32_t>, page_error> line_ids = allocate(count);
    if (line_ids) {
        pool.clone(line_id, *line_ids, stream);
    }
    return line_ids;
}

template <page_store t_store>
task<std::expected<void, page_error>>
kv_cache<t_store>::put(std::vector<digest> hashes, std::vector<std::uint32_t> line_ids, cudaStream_t stream) {
    assert(hashes.size() == line_ids.size());
    std::vector<digest> new_hashes;
    std::vector<std::uint32_t> new_lines;
    std::unordered_set<digest, digest_hash> seen;
    for (std::size_t i = 0; i < hashes.size(); i++) {
        assert(lines[line_ids[i]].ref > 0 && !lines[line_ids[i]].hash);
        if (!digest_to_line_id.contains(hashes[i]) && seen.insert(hashes[i]).second) {
            new_hashes.push_back(hashes[i]);
            new_lines.push_back(line_ids[i]);
        }
    }
    if (new_hashes.empty()) {
        co_return {};
    }

    std::optional<std::vector<page>> host_pages = pool.download(new_lines, stream);
    if (!host_pages) {
        co_return std::unexpected(page_error::oom);
    }
    for (std::size_t j = 0; j < new_hashes.size(); j++) {
        assign_hash(new_lines[j], new_hashes[j]);
    }
    co_await synchronize(stream);
    store.put(std::move(new_hashes), std::move(*host_pages));
    co_return {};
}

template <page_store t_store> void kv_cache<t_store>::release(std::span<const std::uint32_t> line_ids) noexcept {
    for (std::uint32_t line_id : line_ids) {
        line& released = lines[line_id];
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

template <page_store t_store> std::size_t kv_cache<t_store>::free_line_count() const noexcept {
    return free_count;
}

template <page_store t_store> at::Tensor kv_cache<t_store>::k_pool(std::size_t layer) const {
    return pool.k_pool(layer);
}

template <page_store t_store> at::Tensor kv_cache<t_store>::v_pool(std::size_t layer) const {
    return pool.v_pool(layer);
}

}  // namespace pulsar
