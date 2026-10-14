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

// A page of a values_only shape holds one tensor, as both k and v. The cache's pages and lines lay a page out head
// first, [n_kv_heads, k then v, page_size, head_dim], and k and v view it.
struct page {
    at::Tensor k;  // [n_layers, page_size, n_kv_heads, head_dim]
    at::Tensor v;
};

// With values_only, one tensor per layer serves as both K and V.
struct page_shape {
    std::size_t n_layers;
    std::size_t page_size;
    std::size_t n_kv_heads;
    std::size_t head_dim;
    at::ScalarType dtype;
    bool values_only = false;
};

// The device bytes of one line: its K and V in every layer, or its one tensor for values_only.
std::size_t line_bytes(const page_shape& shape);

// The digest that the page chains of an engine start from. model names the model and its weights version; with the
// page shape of every KV layer (in KV layer order) and digest_function it covers everything that changes a
// page's KV or how tokens split into pages. The seed is digest_function and model, each as an 8-byte little-endian
// length then its bytes, then the shape count as 8 little-endian bytes, then per shape the dtype name as a length and
// its bytes, followed by n_layers, page_size, n_kv_heads, head_dim and values_only, each as 8 little-endian bytes.
digest make_root_digest(std::string_view model, std::span<const page_shape> shapes);

// The key of KV layer layer's page in the store: the digest of the page's digest bytes followed by layer as 8
// little-endian bytes.
digest store_key(const digest& page, std::size_t layer);

class page_store {
  public:
    virtual ~page_store() = default;

    // The pages in the order of hashes, nullopt for those it does not have.
    virtual task<std::vector<std::optional<page>>> fetch(std::span<const digest> hashes) = 0;
    // Takes host pages and their hashes, matched by index, and returns without waiting for the write.
    virtual void put(std::vector<digest>&& hashes, std::vector<page>&& pages) = 0;
};

enum class page_error { miss, oom };

// The lines of every KV layer, in one device pool of bytes. A line holds one page of one layer. The layers whose lines
// have the same size share a size class, which reuses only its own freed lines and carves new ones from a bump pointer
// shared by every class. A layer's page is stored under store_key(page, layer).
class kv_cache {
  private:
    static constexpr std::uint32_t no_line = std::numeric_limits<std::uint32_t>::max();

    struct cache_line {
        std::optional<digest> hash;
        std::size_t layer = 0;
        std::uint32_t ref = 0;
        std::uint32_t previous_free = no_line;
        std::uint32_t next_free = no_line;
    };

    // lines is indexed by line id: line i covers pool bytes [i line_bytes, (i + 1) line_bytes). Ids whose
    // bytes another class carved hold no line of this class.
    struct size_class {
        std::size_t line_bytes;
        std::vector<cache_line> lines;
        // Free lines in reuse order: unmapped lines first, then mapped lines least recently released first.
        std::uint32_t free_front = no_line;
        std::uint32_t free_back = no_line;
        std::size_t free_count = 0;

        void link_free_front(std::uint32_t line_id) noexcept;
        void link_free_back(std::uint32_t line_id) noexcept;
        void unlink_free(std::uint32_t line_id) noexcept;
    };

    // done is set when the fetch completes. The page arrived when its line's hash then equals its digest.
    struct fetch {
        std::uint32_t line_id;
        event done;
    };

    struct cache_layer {
        page_shape shape;
        std::size_t size_class;
        kv device_kv;
        std::unordered_map<digest, std::uint32_t, digest_hash> digest_to_line_id;
        // Pages a get is fetching into a line, not yet in digest_to_line_id.
        std::unordered_map<digest, fetch, digest_hash> fetching;
    };

    struct layer_line {
        std::size_t layer;
        std::uint32_t line_id;
    };

    at::Tensor pool;
    // The pool bytes below it are carved into lines.
    std::size_t bump = 0;
    std::vector<size_class> classes;
    std::vector<cache_layer> layers;
    page_store* store;
    // Pinned host bytes every transfer goes through, one page after another, each in its layer's page layout. One
    // transfer holds it at a time.
    at::Tensor staging;
    bool staging_held = false;
    // Set when the staging buffer is released.
    std::unique_ptr<event> staging_released;

    std::size_t pool_bytes() const noexcept;
    cache_line& line(std::size_t layer, std::uint32_t line_id) noexcept;
    const cache_line& line(std::size_t layer, std::uint32_t line_id) const noexcept;
    bool is_mapped(std::size_t layer, std::uint32_t line_id) const noexcept;
    void hold(std::size_t layer, std::uint32_t line_id) noexcept;
    std::uint32_t take_free_line(std::size_t layer) noexcept;
    std::vector<std::uint32_t> take_free_lines(std::size_t layer, std::size_t count) noexcept;
    void assign_hash(std::size_t layer, std::uint32_t line_id, const digest& hash);
    // Whether taking counts[layer] lines in each layer, in layer order, succeeds when size class c has free_lines[c]
    // free lines: a class takes its free lines first, then lines carved at the bump pointer.
    bool covers(std::span<const std::size_t> counts, std::vector<std::size_t>&& free_lines) const noexcept;
    std::vector<std::size_t> free_counts() const;
    task<void> hold_staging();
    void release_staging() noexcept;
    // The byte offset of room in the staging buffer for a page of layer, laid out as a line, past its first used bytes,
    // which then cover it too. nullopt when the page does not fit.
    std::optional<std::size_t> staging_offset(std::size_t layer, std::size_t& used) const;
    std::size_t line_offset(std::size_t layer, std::uint32_t line_id) const noexcept;
    // The transfers go through the staging buffer as many pages at a time as it holds, and complete once the stream has
    // finished them.
    task<void> copy_from_host(std::span<const layer_line> targets, std::span<const page> pages, cudaStream_t stream);
    task<void> copy_to_host(std::span<const layer_line> sources, std::span<page> pages, cudaStream_t stream);
    // Pageable host pages laid out as lines, one for each line of sources. nullopt when host memory runs out.
    std::optional<std::vector<page>> allocate_host_pages(std::span<const layer_line> sources) const;
    void release_all(std::span<const std::vector<std::uint32_t>> line_ids) noexcept;

  public:
    // shapes holds one page shape per KV layer, each with n_layers 1 and one dtype for all. The pool takes pool_bytes,
    // and staging_bytes holds a page of any layer. store outlives the cache.
    kv_cache(
        std::span<const page_shape> shapes,
        std::size_t pool_bytes,
        std::size_t staging_bytes,
        at::Device device,
        page_store& store
    );

    // Copies run on stream. Callers release lines only after the GPU work using them completes. get, allocate and
    // clone never wait for lines; without enough of them the call fails with oom, holding nothing in any layer, which is
    // also true of every other failure. A line is free when it is not held; free lines are reused least recently
    // released first and their pages dropped. Line ids come per layer, indexed by layer.

    // Each layer's hashes are distinct. Gives every page of hashes[layer] a line of that layer in their order, fetching
    // the pages that are not resident from the store in one fetch, and holds the lines until release. A page another
    // get is fetching shares that line and fetch. Fails with miss when the store does not have a page.
    task<std::expected<std::vector<std::vector<std::uint32_t>>, page_error>>
    get(std::span<const std::span<const digest>> hashes, cudaStream_t stream);
    // Gives count empty lines in every layer, held until release. get does not find their pages until they are synced,
    // and releasing a line before it is synced drops its page.
    std::expected<std::vector<std::vector<std::uint32_t>>, page_error> allocate(std::size_t count);
    // Gives count unmapped lines of layer holding a copy of the page at its held line_id, held until release.
    std::expected<std::vector<std::uint32_t>, page_error>
    clone(std::size_t layer, std::uint32_t line_id, std::size_t count, cudaStream_t stream);
    // Names the full page at line_ids[layer][i], held and unmapped, by hashes[i] in every layer and hands a host copy
    // of each distinct page to the store in one put, also when another line caches its hash: the store may have
    // dropped that page. A hash already cached at another line leaves its line unmapped. Completes once the copies are
    // handed to the store. Fails with oom, caching nothing, when host memory for the copies cannot be allocated.
    task<std::expected<void, page_error>> sync(
        const std::vector<digest>& hashes,
        const std::vector<std::vector<std::uint32_t>>& line_ids,
        cudaStream_t stream
    );
    void release(std::size_t layer, std::span<const std::uint32_t> line_ids) noexcept;

    // The largest count for which allocate(count) succeeds now.
    std::size_t free_count() const;
    std::size_t layer_count() const noexcept;
    // [n_lines, page_size, n_kv_heads, head_dim] views of the pool, indexed by line id, lines line_bytes apart.
    const kv& tensors(std::size_t layer) const noexcept;
};

}  // namespace pulsarlm
