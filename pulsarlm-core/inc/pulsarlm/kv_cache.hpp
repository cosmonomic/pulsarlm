#pragma once

#include <ATen/core/Tensor.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "pulsarlm/digest.hpp"
#include "pulsarlm/event_loop.hpp"
#include "pulsarlm/model/interface.hpp"

namespace pulsarlm {

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

// The device bytes of one line: its K and V in every layer.
std::size_t line_bytes(const page_shape& shape);

// The digest that the page chains of an engine start from. model names the model and its weights version; with shape
// and digest_function it covers everything that changes a page's KV or how tokens split into pages. The seed is
// digest_function, model and the dtype name, each as an 8-byte little-endian length then its bytes, then n_layers,
// page_size, n_kv_heads and head_dim, each as 8 little-endian bytes.
digest make_root_digest(std::string_view model, const page_shape& shape);

class page_store {
  public:
    virtual ~page_store() = default;

    // The pages in the order of hashes, nullopt for those it does not have.
    virtual task<std::vector<std::optional<page>>> fetch(std::span<const digest> hashes) = 0;
    // Takes host pages and their hashes, matched by index, and returns without waiting for the write.
    virtual void put(std::vector<digest>&& hashes, std::vector<page>&& pages) = 0;
};

enum class page_error { miss, oom };

class kv_cache {
  private:
    static constexpr std::uint32_t no_line = std::numeric_limits<std::uint32_t>::max();

    struct cache_line {
        std::optional<digest> hash;
        std::uint32_t ref = 0;
        std::uint32_t previous_free = no_line;
        std::uint32_t next_free = no_line;
    };

    // done is set when the fetch completes. The page arrived when its line's hash then equals its digest.
    struct fetch {
        std::uint32_t line_id;
        event done;
    };

    kv device_kv;
    // Indexed by line.
    std::vector<cache_line> lines;
    // Free lines in reuse order: unmapped lines first, then mapped lines least recently released first.
    std::uint32_t free_front = no_line;
    std::uint32_t free_back = no_line;
    std::size_t free_count = 0;
    std::unordered_map<digest, std::uint32_t, digest_hash> digest_to_line_id;
    // Pages a get is fetching into a line, not yet in digest_to_line_id.
    std::unordered_map<digest, fetch, digest_hash> fetching;
    std::unique_ptr<page_store> store;
    // Pinned host memory every transfer goes through, [staging_pages, n_layers, page_size, n_kv_heads, head_dim] each:
    // each slot has a page's layout. One transfer holds it at a time.
    page staging;
    bool staging_held = false;
    // Set when the staging buffer is released.
    std::unique_ptr<event> staging_released;

    void link_free_front(std::uint32_t line_id) noexcept;
    void link_free_back(std::uint32_t line_id) noexcept;
    void unlink_free(std::uint32_t line_id) noexcept;
    bool is_mapped(std::uint32_t line_id) const noexcept;
    void hold(std::uint32_t line_id) noexcept;
    std::uint32_t take_free_line() noexcept;
    void assign_hash(std::uint32_t line_id, const digest& hash);
    task<void> hold_staging();
    void release_staging() noexcept;
    // Device copies run on the stream, layer by layer, so no copy allocates device memory. The transfers go through
    // the staging buffer as many pages at a time as it holds, and complete once the stream has finished them.
    task<void> copy_from_host(std::span<const std::uint32_t> line_ids, std::span<const page> pages, cudaStream_t stream);
    task<void> copy_to_host(std::span<const std::uint32_t> line_ids, std::span<page> pages, cudaStream_t stream);
    void copy_line(std::uint32_t source, std::span<const std::uint32_t> targets, cudaStream_t stream);
    // Pageable host pages shaped like a line. nullopt when host memory runs out.
    std::optional<std::vector<page>> allocate_host_pages(std::size_t count) const;

  public:
    // staging_pages is at least 1.
    kv_cache(
        const page_shape& shape,
        std::size_t n_lines,
        std::size_t staging_pages,
        at::Device device,
        std::unique_ptr<page_store>&& store
    );

    // Copies run on stream. Callers release lines only after the GPU work using them completes. get, allocate and
    // clone never wait for lines. A line is free when it is not held; free lines are reused least recently released
    // first and their pages dropped. Without enough free lines the call fails with oom. A failed call holds nothing.

    // hashes are distinct. Gives every page a line in their order, fetching the pages that are not resident from the
    // store, and holds the lines until release. A page another get is fetching shares that line and fetch. Fails with
    // miss when the store does not have a page.
    task<std::expected<std::vector<std::uint32_t>, page_error>>
    get(const std::vector<digest>& hashes, cudaStream_t stream);
    // Gives count empty lines, held until release. get does not find their pages until they are synced, and releasing
    // a line before it is synced drops its page.
    std::expected<std::vector<std::uint32_t>, page_error> allocate(std::size_t count);
    // Gives count unmapped lines holding a copy of the page at the held line_id, held until release.
    std::expected<std::vector<std::uint32_t>, page_error>
    clone(std::uint32_t line_id, std::size_t count, cudaStream_t stream);
    // Names the full page at each held, unmapped line by the hash at the same index and hands a host copy of each
    // distinct page to the store, also when another line caches its hash: the store may have dropped that page. A hash
    // already cached at another line leaves its line unmapped. Completes once the copies are handed to the store. Fails
    // with oom, caching nothing, when host memory for the copies cannot be allocated.
    task<std::expected<void, page_error>>
    sync(const std::vector<digest>& hashes, const std::vector<std::uint32_t>& line_ids, cudaStream_t stream);
    void release(std::span<const std::uint32_t> line_ids) noexcept;

    std::size_t free_line_count() const noexcept;
    const kv& tensors() const noexcept;
};

}  // namespace pulsarlm
