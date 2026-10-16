#pragma once

#include <ATen/core/Tensor.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "pulsarlm/digest.hpp"
#include "pulsarlm/event_loop.hpp"
#include "pulsarlm/host_page_pool.hpp"
#include "pulsarlm/line_allocator.hpp"
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
// The bytes of the pool block a line takes: line_bytes rounded up to a power of two.
std::size_t block_bytes(const page_shape& shape);

// The digest that the page chains of an engine start from. model names the model and its weights version; with the
// page shape of every KV layer (in KV layer order) and digest_function it covers everything that changes a
// page's KV or how tokens split into pages. The seed is digest_function and model, each as an 8-byte little-endian
// length then its bytes, then the shape count as 8 little-endian bytes, then per shape the dtype name as a length and
// its bytes, followed by n_layers, page_size, n_kv_heads, head_dim and values_only, each as 8 little-endian bytes.
digest make_root_digest(std::string_view model, std::span<const page_shape> shapes);

// The key of KV layer layer's page in the store: the digest of the page's digest bytes followed by layer as 8
// little-endian bytes.
digest store_key(const digest& page, std::size_t layer);

// A request whose step produced pages[page] of a put.
struct page_owner {
    std::uint64_t request;
    std::size_t page;
};

class page_store {
  public:
    virtual ~page_store() = default;

    // The pages in the order of hashes, nullopt for those it does not have.
    virtual task<std::vector<std::optional<page>>> fetch(std::span<const digest> hashes) = 0;
    // Takes host pages and their hashes, matched by index, and returns without waiting for the write. A page has an
    // owner for every request that produced it.
    virtual void put(std::vector<digest>&& hashes, std::vector<page>&& pages, std::vector<page_owner>&& owners) = 0;
};

enum class page_error { miss, oom };

// The lines of every KV layer, in one device pool of bytes. A line holds one page of one layer, in a block of the pool
// a line_allocator gives it; the layers whose lines take blocks of one size share their line ids. A line is free when
// it is neither held nor caching a page; taking a line may drop the cached pages of a region, all of them, oldest
// first, as line_allocator describes. A layer's page is stored under store_key(page, layer).
class kv_cache {
  private:
    struct cache_line {
        std::optional<digest> hash;
        std::size_t layer = 0;
        std::uint32_t ref = 0;
    };

    // The lines of the layers whose blocks are block_bytes, indexed by line id.
    struct block_class {
        std::size_t block_bytes;
        std::vector<cache_line> lines;
    };

    // done is set when the fetch completes. The page arrived when its line's hash then equals its digest.
    struct fetch {
        std::uint32_t line_id;
        event done;
    };

    struct cache_layer {
        page_shape shape;
        std::size_t block_class;
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
    std::vector<block_class> classes;
    std::vector<cache_layer> layers;
    line_allocator allocator;
    page_store* store;
    host_page_pool host_pages;

    cache_line& line(std::size_t layer, std::uint32_t line_id) noexcept;
    const cache_line& line(std::size_t layer, std::uint32_t line_id) const noexcept;
    bool is_mapped(std::size_t layer, std::uint32_t line_id) const noexcept;
    // Holds a line another holder holds or that caches a page.
    void hold(line_allocator::transaction& taking, std::size_t layer, std::uint32_t line_id);
    // counts[layer] lines of each layer, larger blocks first so that they pack; nullopt when they do not fit.
    std::optional<std::vector<std::vector<std::uint32_t>>>
    take_lines(line_allocator::transaction& taking, std::span<const std::size_t> counts);
    // Commits taking: drops the pages of the regions it reclaimed, then names taken, empty and held, by their layers.
    void commit(line_allocator::transaction& taking, std::span<const std::vector<std::uint32_t>> taken);
    void assign_hash(std::size_t layer, std::uint32_t line_id, const digest& hash);
    std::size_t line_offset(std::size_t layer, std::uint32_t line_id) const noexcept;
    // The transfers complete once the stream has finished them. Pages in the host pool copy in one launch; a page
    // outside it is copied into a pool block first, and without one goes through a blocking cudaMemcpyAsync.
    task<void> copy_from_host(std::span<const layer_line> targets, std::span<const page> pages, cudaStream_t stream);
    task<void> copy_to_host(std::span<const layer_line> sources, std::span<page> pages, cudaStream_t stream);
    // Host pages laid out as lines, one for each line of sources, in the host pool or, past its room, pageable.
    // nullopt when host memory runs out.
    std::optional<std::vector<page>> allocate_host_pages(std::span<const layer_line> sources);
    void release_all(std::span<const std::vector<std::uint32_t>> line_ids) noexcept;

  public:
    // shapes holds one page shape per KV layer, each with n_layers 1 and one dtype for all. The pool takes pool_bytes.
    // The host copies sync hands the store come from a host_page_pool of host_page_bytes. store outlives the cache.
    kv_cache(
        std::span<const page_shape> shapes,
        std::size_t pool_bytes,
        std::size_t host_page_bytes,
        at::Device device,
        page_store& store
    );

    // Copies run on stream. Callers release lines only after the GPU work using them completes. get, allocate and
    // clone never wait for lines; without enough of them the call fails with oom, holding nothing in any layer and
    // dropping no page, which is also true of every other failure. Line ids come per layer, indexed by layer.

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
    // of each distinct page, owned by every owners[i] of its hash, to the store in one put, also when another line
    // caches its hash: the store may have
    // dropped that page. A hash already cached at another line leaves its line unmapped. Completes once the copies are
    // handed to the store. Fails with oom, caching nothing, when host memory for the copies cannot be allocated.
    task<std::expected<void, page_error>> sync(
        const std::vector<digest>& hashes,
        const std::vector<std::uint64_t>& owners,
        const std::vector<std::vector<std::uint32_t>>& line_ids,
        cudaStream_t stream
    );
    void release(std::size_t layer, std::span<const std::uint32_t> line_ids) noexcept;

    // The largest count for which allocate(count) succeeds now.
    std::size_t free_count();
    std::size_t layer_count() const noexcept;
    // [n_lines, page_size, n_kv_heads, head_dim] views of the pool, indexed by line id, lines a block apart.
    const kv& tensors(std::size_t layer) const noexcept;
};

}  // namespace pulsarlm
