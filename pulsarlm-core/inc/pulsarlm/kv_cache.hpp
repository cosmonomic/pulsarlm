#pragma once

#include <ATen/core/Tensor.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pulsarlm/digest.hpp"
#include "pulsarlm/event_loop.hpp"
#include "pulsarlm/line_allocator.hpp"
#include "pulsarlm/model/interface.hpp"

namespace pulsarlm {

// With values_only, one tensor per layer serves as both K and V. A recurrent layer's page (state_sizes not empty) is one
// tensor [n_layers, state_sizes...] with values_only set; page_size, n_kv_heads and head_dim do not apply to it. A line
// lays a page out head first, [n_kv_heads, k then v, page_size, head_dim], then an indexed attention layer's index keys
// [page_size, index_head_dim].
struct page_shape {
    std::size_t n_layers;
    std::size_t page_size;
    std::size_t n_kv_heads;
    std::size_t head_dim;
    at::ScalarType dtype;
    bool values_only = false;
    std::size_t index_head_dim = 0;
    std::vector<std::int64_t> state_sizes;
};

// The device bytes of one line: its K and V in every layer, or its one tensor for values_only, and its index keys.
std::size_t line_bytes(const page_shape& shape);

// One line of a layer of the whole model as uint8 [line_bytes(shape)], laid out as the line, in host memory.
struct page {
    at::Tensor bytes;
};

// The part of a page one GPU's line holds: a device view of those bytes of the line of kv_cache_shard cache_shard, and
// their offset in the page.
struct page_shard {
    at::Tensor bytes;
    std::size_t host_offset;
    std::size_t cache_shard;
};

// One page across the GPUs holding it; on one GPU holding whole lines, a single shard covering the line.
struct sharded_page {
    std::vector<page_shard> shards;
};

// The K and V of a page of shape, [n_layers, page_size, n_kv_heads, head_dim] views of its bytes, or its one tensor as
// both for values_only, and its index keys [n_layers, page_size, index_head_dim].
kv page_tensors(const page& p, const page_shape& shape);

// The digest that the page chains of an engine start from. model names the model and its weights version; with the
// page shape of every KV layer (in KV layer order) and digest_function it covers everything that changes a
// page's KV or how tokens split into pages. The seed is digest_function and model, each as an 8-byte little-endian
// length then its bytes, then the shape count as 8 little-endian bytes, then per shape the dtype name as a length and
// its bytes, followed by n_layers, page_size, n_kv_heads, head_dim, values_only and index_head_dim, each as 8
// little-endian bytes, then the count of state sizes and each size, as 8 little-endian bytes each.
digest make_root_digest(std::string_view model, std::span<const page_shape> shapes);

// The key of KV layer layer's page in the store: the digest of the page's digest bytes followed by layer as 8
// little-endian bytes.
digest store_key(const digest& page, std::size_t layer);

// A request whose step produced pages[page] of a put.
struct page_owner {
    std::uint64_t request;
    std::size_t page;
};

enum class page_error { miss, oom };

class page_store {
  public:
    virtual ~page_store() = default;

    // A shard's copies run on streams[shard.cache_shard].

    // Copies the page of hashes[i] into every shard of lines[i], and completes once the copies are done with true for
    // each page it had.
    virtual task<std::vector<bool>>
    fetch(std::span<const digest> hashes, std::vector<sharded_page>&& lines, std::span<const cudaStream_t> streams) = 0;
    // Takes the pages of hashes, matched by index, and copies the ones it keeps, completing once the copies are done.
    // The shards of a page cover it once. A page has an owner for every request that produced it. Fails with oom when
    // it has no host memory for a page it keeps.
    virtual task<std::expected<void, page_error>>
    put(std::vector<digest>&& hashes,
        std::vector<sharded_page>&& pages,
        std::vector<page_owner>&& owners,
        std::span<const cudaStream_t> streams) = 0;
};

// One GPU's pool: the page shape of its line of each KV layer, and per KV layer the ranges along the full line's first
// dimension (KV heads, or a state's first size) its line holds, in order; index keys are always whole.
struct kv_cache_shard {
    at::Device device;
    std::vector<page_shape> shapes;
    std::vector<std::vector<std::pair<std::size_t, std::size_t>>> rows;
};

// The lines of every KV layer, in one device pool of bytes per shard. A line holds one page of one layer, in a block of
// every shard's pool a line_allocator gives it, each shard holding its part of the page; the layers whose lines take
// blocks of one size share their line ids. A line is free when it is neither held nor caching a page; taking a line may
// drop the cached pages of a region, all of them, oldest first, as line_allocator describes. A layer's page is stored
// under store_key(page, layer).
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
        event done_event;
    };

    // Bytes [line_offset, line_offset + bytes) of a shard's line hold bytes [host_offset, host_offset + bytes) of the
    // page. A put copies only the pieces that cover the page once.
    struct line_piece {
        std::size_t line_offset;
        std::size_t host_offset;
        std::size_t bytes;
        bool put_source;
    };

    struct shard_layer {
        page_shape shape;
        std::vector<line_piece> pieces;
        kv device_kv;
    };

    struct cache_layer {
        page_shape full_shape;
        std::size_t block_class;
        std::vector<shard_layer> shards;
        std::unordered_map<digest, std::uint32_t, digest_hash> digest_to_line_id;
        // Pages a get is fetching into a line, not yet in digest_to_line_id.
        std::unordered_map<digest, fetch, digest_hash> fetches;
    };

    struct layer_line {
        std::size_t layer;
        std::uint32_t line_id;
    };

    std::vector<at::Tensor> pools;
    std::vector<block_class> classes;
    std::vector<cache_layer> layers;
    line_allocator allocator;
    page_store* store;
    // The block bytes of the lines in use, read and written through std::atomic_ref so used_bytes is safe from any
    // thread.
    alignas(std::atomic_ref<std::size_t>::required_alignment) mutable std::size_t used_line_bytes = 0;

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
    void count_used(std::size_t layer) noexcept;
    void count_unused(std::size_t layer) noexcept;
    void assign_hash(std::size_t layer, std::uint32_t line_id, const digest& hash);
    std::size_t line_offset(std::size_t layer, std::uint32_t line_id) const noexcept;
    at::Tensor shard_line(std::size_t layer, std::size_t shard, std::uint32_t line_id) const;
    // Every piece of every shard's line, for a fetch; the pieces that cover the page once, for a put.
    sharded_page fetch_target(const layer_line& line_of_layer) const;
    sharded_page put_source(const layer_line& line_of_layer) const;
    void release_all(std::span<const std::vector<std::uint32_t>> line_ids) noexcept;

  public:
    // full_shapes holds one page shape per KV layer of the whole model, each with n_layers 1 and one dtype for all;
    // each shard holds its part of every layer, every shard's line of a layer taking the same bytes. Each shard's pool
    // takes pool_bytes_per_device; throws std::invalid_argument when that holds no line of every layer, or when a
    // shard's rows do not lay out its lines or do not cover the full lines between them. store outlives the cache.
    kv_cache(
        std::span<const page_shape> full_shapes,
        std::span<const kv_cache_shard> shards,
        std::size_t pool_bytes_per_device,
        page_store& store
    );
    // One shard on device holding whole lines of shapes.
    kv_cache(std::span<const page_shape> shapes, std::size_t pool_bytes, at::Device device, page_store& store);

    // A shard's copies run on streams[shard], one stream per shard. Callers release lines only after the GPU work
    // using them completes. get, allocate and clone never wait for lines; without enough of them the call fails with
    // oom, holding nothing in any layer and dropping no page, which is also true of every other failure. Line ids come
    // per layer, indexed by layer.

    // Each layer's hashes are distinct. Gives every page of hashes[layer] a line of that layer in their order, fetching
    // the pages that are not resident from the store in one fetch, and holds the lines until release. A page another
    // get is fetching shares that line and fetch. Fails with miss when the store does not have a page.
    task<std::expected<std::vector<std::vector<std::uint32_t>>, page_error>>
    get(std::span<const std::span<const digest>> hashes, std::span<const cudaStream_t> streams);
    // Gives counts[layer] empty lines of each layer, held until release. get does not find their pages until they are
    // synced, and releasing a line before it is synced drops its page.
    std::expected<std::vector<std::vector<std::uint32_t>>, page_error> allocate(std::span<const std::size_t> counts);
    // Gives count unmapped lines of layer holding a copy of the page at its held line_id, held until release.
    std::expected<std::vector<std::uint32_t>, page_error>
    clone(std::size_t layer, std::uint32_t line_id, std::size_t count, std::span<const cudaStream_t> streams);
    // Names the page at line_ids[layer][i], held and unmapped, by hashes[layer][i] and hands each distinct page of a
    // layer, owned by every owners[layer][i] of its hash, to the store in one put, also when another line caches its
    // hash: the store may have dropped that page. A hash already cached at another line leaves its line unmapped.
    // Completes once the put does, and fails as it does.
    task<std::expected<void, page_error>> sync(
        const std::vector<std::vector<digest>>& hashes,
        const std::vector<std::vector<std::uint64_t>>& owners,
        const std::vector<std::vector<std::uint32_t>>& line_ids,
        std::span<const cudaStream_t> streams
    );
    void release(std::size_t layer, std::span<const std::uint32_t> line_ids) noexcept;

    // The largest count for which allocate gives count lines of every layer now.
    std::size_t free_count();
    // The bytes of each shard's pool that lines in use take: lines get, allocate or clone gave and release has not
    // taken back, not lines only caching a page. Safe to call from any thread.
    std::size_t used_bytes() const noexcept;
    std::size_t layer_count() const noexcept;
    std::size_t shard_count() const noexcept;
    // [n_lines, page_size, n_kv_heads, head_dim] views of shard's pool, indexed by line id, lines a block apart.
    const kv& tensors(std::size_t layer, std::size_t shard) const noexcept;
    // tensors(layer, 0).
    const kv& tensors(std::size_t layer) const noexcept;
};

}  // namespace pulsarlm
