#include "pulsarlm/engine/kv_cache.hpp"
#include "pulsarlm/digest.hpp"
#include "pulsarlm/engine/event_loop.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#undef CHECK
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using pulsarlm::digest;
using pulsarlm::digest_hash;
using pulsarlm::event_loop;
using pulsarlm::external_task;
using pulsarlm::page;
using pulsarlm::page_error;
using pulsarlm::sharded_page;
using pulsarlm::task;

using line_ids = std::vector<std::uint32_t>;
using lines_result = std::expected<line_ids, page_error>;
using layer_lines_result = std::expected<std::vector<line_ids>, page_error>;

constexpr pulsarlm::page_shape
    shape{.n_layers = 1, .page_size = 4, .n_kv_heads = 2, .head_dim = 8, .dtype = at::kFloat};

const lines_result oom = std::unexpected(page_error::oom);
const lines_result miss = std::unexpected(page_error::miss);

struct store_state {
    std::unordered_map<digest, page, digest_hash> pages;
    std::vector<std::vector<digest>> fetches;
    std::size_t puts = 0;
    std::vector<std::pair<std::uint64_t, digest>> owners;
    bool gated = false;
    std::vector<external_task<>::resolver> gates;
};

std::size_t page_bytes_of(const sharded_page& p) {
    std::size_t bytes = 0;
    for (const pulsarlm::page_shard& shard : p.shards) {
        bytes = std::max(bytes, shard.host_offset + static_cast<std::size_t>(shard.bytes.numel()));
    }
    return bytes;
}

// Pageable copies of sources, each shard into its bytes of the page on streams[shard.cache_shard].
std::vector<page> copy_pages_to_host(std::span<const sharded_page> sources, std::span<const cudaStream_t> streams) {
    std::vector<page> pages;
    pages.reserve(sources.size());
    for (const sharded_page& source : sources) {
        at::Tensor bytes = at::empty({static_cast<std::int64_t>(page_bytes_of(source))}, at::kByte);
        for (const pulsarlm::page_shard& shard : source.shards) {
            c10::cuda::CUDAStreamGuard guard(
                c10::cuda::getStreamFromExternal(streams[shard.cache_shard], shard.bytes.device().index())
            );
            bytes.narrow(0, static_cast<std::int64_t>(shard.host_offset), shard.bytes.numel()).copy_(shard.bytes);
        }
        pages.push_back(page{.bytes = std::move(bytes)});
    }
    return pages;
}

// Copies each shard's bytes of pages[i] into the shards of targets[i] on streams[shard.cache_shard].
void copy_pages_from_host(
    std::span<const sharded_page> targets,
    std::span<const page> pages,
    std::span<const cudaStream_t> streams
) {
    for (std::size_t i = 0; i < targets.size(); i++) {
        for (const pulsarlm::page_shard& shard : targets[i].shards) {
            c10::cuda::CUDAStreamGuard guard(
                c10::cuda::getStreamFromExternal(streams[shard.cache_shard], shard.bytes.device().index())
            );
            shard.bytes.copy_(pages[i].bytes.narrow(0, static_cast<std::int64_t>(shard.host_offset), shard.bytes.numel()));
        }
    }
}

// Keeps every page it is written, copied to pageable host memory. kv_cache reads and writes no state.
struct memory_store final : pulsarlm::page_store {
    std::shared_ptr<store_state> state = std::make_shared<store_state>();

    task<std::vector<bool>> read_pages(
        std::span<const digest> hashes,
        std::vector<sharded_page>&& lines,
        std::span<const cudaStream_t> streams
    ) override {
        std::shared_ptr<store_state> shared = state;
        std::vector<digest> requested(hashes.begin(), hashes.end());
        std::vector<sharded_page> targets = std::move(lines);
        shared->fetches.push_back(requested);
        std::vector<bool> found(requested.size());
        std::vector<sharded_page> found_targets;
        std::vector<page> found_pages;
        for (std::size_t i = 0; i < requested.size(); i++) {
            if (auto it = shared->pages.find(requested[i]); it != shared->pages.end()) {
                found[i] = true;
                found_targets.push_back(targets[i]);
                found_pages.push_back(it->second);
            }
        }
        if (shared->gated) {
            auto [gate, opener] = external_task<>::make();
            shared->gates.push_back(opener);
            co_await std::move(gate);
        }
        copy_pages_from_host(found_targets, found_pages, streams);
        co_return found;
    }

    task<std::expected<void, page_error>> write_pages(
        std::vector<digest>&& keys,
        std::vector<sharded_page>&& pages,
        std::vector<std::pair<pulsarlm::state::lease_id_type, page_index_type>>&& owners,
        std::span<const cudaStream_t> streams
    ) override {
        std::shared_ptr<store_state> shared = state;
        std::vector<digest> put_keys = std::move(keys);
        std::vector<sharded_page> sources = std::move(pages);
        shared->puts++;
        for (const auto& [lease_id, index] : owners) {
            shared->owners.emplace_back(lease_id, put_keys[index]);
        }
        std::vector<page> copies = copy_pages_to_host(sources, streams);
        for (std::size_t i = 0; i < put_keys.size(); i++) {
            shared->pages.insert_or_assign(put_keys[i], std::move(copies[i]));
        }
        co_return {};
    }

    task<std::pair<std::shared_ptr<const pulsarlm::state>, pulsarlm::state::lease_id_type>>
    read_state(std::span<const std::int64_t>, std::span<const digest>) override {
        co_return std::pair(std::make_shared<const pulsarlm::state>(), pulsarlm::state::lease_id_type{0});
    }

    task<void> write_state(pulsarlm::state::lease_id_type, std::shared_ptr<const pulsarlm::state>&&) override {
        co_return;
    }

    void release(pulsarlm::state::lease_id_type) noexcept override {}
};

std::vector<cudaStream_t> compute_streams() {
    return {c10::cuda::getCurrentCUDAStream().stream()};
}

std::size_t lines_bytes(const std::vector<pulsarlm::page_shape>& shapes) {
    std::size_t bytes = 0;
    for (const pulsarlm::page_shape& layer_shape : shapes) {
        bytes += pulsarlm::line_bytes(layer_shape);
    }
    return bytes;
}

struct cache_fixture {
    memory_store backing;
    std::shared_ptr<store_state> store;
    pulsarlm::kv_cache kv;

    // One layer of shape unless shapes gives one per layer. The pool has room for n_lines lines of every layer.
    explicit cache_fixture(
        std::size_t n_lines,
        memory_store backing_store = {},
        const std::vector<pulsarlm::page_shape>& shapes = {shape}
    )
    : backing(std::move(backing_store)),
      store(backing.state),
      kv(shapes, n_lines * lines_bytes(shapes), at::Device(at::kCUDA, 0), backing) {}
};

digest hash_of(std::uint64_t n) {
    return digest{.hi = n, .lo = n * 31 + 7};
}

digest key_of(std::uint64_t n, std::size_t layer = 0) {
    return pulsarlm::store_key(hash_of(n), layer);
}

// K filled with value and V with -value, in pageable memory.
page host_page(float value, const pulsarlm::page_shape& page_shape = shape) {
    page filled{.bytes = at::empty({static_cast<std::int64_t>(pulsarlm::line_bytes(page_shape))}, at::kByte)};
    pulsarlm::kv tensors = pulsarlm::page_tensors(filled, page_shape);
    tensors.k.fill_(value);
    tensors.v.fill_(-value);
    return filled;
}

pulsarlm::kv
stored_tensors(const store_state& store, const digest& key, const pulsarlm::page_shape& page_shape = shape) {
    return pulsarlm::page_tensors(store.pages.at(key), page_shape);
}

void fill_line(pulsarlm::kv_cache& kv, std::uint32_t line_id, float value, std::size_t layer = 0) {
    kv.tensors(layer).k.select(0, line_id).fill_(value);
    kv.tensors(layer).v.select(0, line_id).fill_(-value);
}

bool line_filled_with(pulsarlm::kv_cache& kv, std::uint32_t line_id, float value, std::size_t layer = 0) {
    return kv.tensors(layer).k.select(0, line_id).eq(value).all().item<bool>() &&
        kv.tensors(layer).v.select(0, line_id).eq(-value).all().item<bool>();
}

lines_result only_layer(layer_lines_result result) {
    if (!result) {
        return std::unexpected(result.error());
    }
    REQUIRE(result->size() == 1);
    return std::move(result->front());
}

layer_lines_result allocate_each(pulsarlm::kv_cache& kv, std::size_t count) {
    return kv.allocate(std::vector<std::size_t>(kv.layer_count(), count));
}

lines_result allocate(pulsarlm::kv_cache& kv, std::size_t count) {
    return only_layer(allocate_each(kv, count));
}

void release(pulsarlm::kv_cache& kv, const line_ids& held) {
    kv.release(0, held);
}

task<lines_result> get_one_layer(pulsarlm::kv_cache& kv, std::vector<digest> hashes) {
    std::vector<std::span<const digest>> layer_hashes{hashes};
    layer_lines_result got = co_await kv.get(layer_hashes, compute_streams());
    co_return only_layer(std::move(got));
}

// Names lines[layer][i] by hashes[i] in every layer, owned by owners[i].
task<std::expected<void, page_error>> sync_each(
    pulsarlm::kv_cache& kv,
    std::vector<digest> hashes,
    std::vector<std::uint64_t> owners,
    std::vector<line_ids> lines
) {
    std::vector<std::vector<digest>> layer_hashes(kv.layer_count(), hashes);
    std::vector<std::vector<std::uint64_t>> layer_owners(kv.layer_count(), owners);
    std::expected<void, page_error> result = co_await kv.sync(layer_hashes, layer_owners, lines, compute_streams());
    co_return result;
}

task<std::expected<void, page_error>>
sync_one_layer(pulsarlm::kv_cache& kv, std::vector<digest> hashes, line_ids synced) {
    std::vector<std::uint64_t> owners(hashes.size(), 0);
    co_return co_await sync_each(kv, std::move(hashes), std::move(owners), {std::move(synced)});
}

template <typename t> task<void> store_result(task<t> work, std::optional<t>* result) {
    result->emplace(co_await std::move(work));
    event_loop::current()->stop();
}

template <typename t> t run_to_completion(task<t> work) {
    event_loop loop;
    std::optional<t> result;
    loop.spawn(store_result(std::move(work), &result));
    REQUIRE(loop.run() == nullptr);
    REQUIRE(result.has_value());
    return std::move(*result);
}

task<void> get_into(pulsarlm::kv_cache* kv, std::vector<digest> hashes, lines_result* result, std::size_t* remaining) {
    *result = co_await get_one_layer(*kv, std::move(hashes));
    if (--*remaining == 0) {
        event_loop::current()->stop();
    }
}

task<void> open_gate(std::shared_ptr<store_state> store) {
    REQUIRE(store->gates.size() == 1);
    store->gates.front().resolve();
    co_return;
}

task<void> sync_filled_line(pulsarlm::kv_cache* kv, digest hash, float value, std::uint32_t* synced_line) {
    line_ids allocated = allocate(*kv, 1).value();
    fill_line(*kv, allocated.front(), value);
    std::expected<void, page_error> sync_result = co_await sync_one_layer(*kv, {hash}, allocated);
    REQUIRE(sync_result.has_value());
    *synced_line = allocated.front();
}

task<void> sync_into(
    pulsarlm::kv_cache* kv,
    std::vector<digest> hashes,
    line_ids synced,
    std::expected<void, page_error>* result,
    std::size_t* remaining
) {
    *result = co_await sync_one_layer(*kv, std::move(hashes), std::move(synced));
    if (--*remaining == 0) {
        event_loop::current()->stop();
    }
}

lines_result run_get(pulsarlm::kv_cache& kv, std::vector<digest> hashes) {
    return run_to_completion(get_one_layer(kv, std::move(hashes)));
}

void sync_page(pulsarlm::kv_cache& kv, digest hash, float value) {
    line_ids allocated = allocate(kv, 1).value();
    fill_line(kv, allocated.front(), value);
    REQUIRE(run_to_completion(sync_one_layer(kv, {hash}, allocated)).has_value());
    release(kv, allocated);
}

// A size class takes its free lines before carving new ones; carving count lines first makes them all free.
void carve(pulsarlm::kv_cache& kv, std::size_t count) {
    line_ids carved = allocate(kv, count).value();
    release(kv, carved);
}

void require_all_free(pulsarlm::kv_cache& kv, std::size_t n_lines) {
    REQUIRE(kv.free_count() == n_lines);
    lines_result all = allocate(kv, n_lines);
    REQUIRE(all.has_value());
    release(kv, *all);
}

}  // namespace

TEST_CASE("allocate fails with oom holding nothing when lines run out") {
    cache_fixture fixture(4);
    REQUIRE(allocate(fixture.kv, 5) == oom);
    line_ids held = allocate(fixture.kv, 4).value();
    REQUIRE(fixture.kv.free_count() == 0);
    REQUIRE(allocate(fixture.kv, 1) == oom);
    release(fixture.kv, held);
    require_all_free(fixture.kv, 4);
}

TEST_CASE("used_bytes counts each line in use once and no line only caching a page") {
    cache_fixture fixture(4);
    std::size_t block = std::bit_ceil(pulsarlm::line_bytes(shape));
    REQUIRE(fixture.kv.used_bytes() == 0);
    sync_page(fixture.kv, hash_of(1), 1.0f);
    REQUIRE(fixture.kv.used_bytes() == 0);

    line_ids allocated = allocate(fixture.kv, 2).value();
    REQUIRE(fixture.kv.used_bytes() == 2 * block);
    lines_result got = run_get(fixture.kv, {hash_of(1)});
    lines_result got_again = run_get(fixture.kv, {hash_of(1)});
    REQUIRE(got.has_value());
    REQUIRE(got_again.has_value());
    REQUIRE(fixture.kv.used_bytes() == 3 * block);

    release(fixture.kv, *got);
    REQUIRE(fixture.kv.used_bytes() == 3 * block);
    release(fixture.kv, *got_again);
    release(fixture.kv, allocated);
    REQUIRE(fixture.kv.used_bytes() == 0);
}

TEST_CASE("a get failing with oom leaves used_bytes as it was") {
    cache_fixture fixture(1);
    sync_page(fixture.kv, hash_of(1), 1.0f);
    fixture.store->pages.emplace(key_of(2), host_page(2.0f));

    REQUIRE(run_get(fixture.kv, {hash_of(1), hash_of(2)}) == oom);
    REQUIRE(fixture.kv.used_bytes() == 0);
}

TEST_CASE("sync stores host copies and get finds the page without fetching") {
    cache_fixture fixture(4);
    sync_page(fixture.kv, hash_of(1), 3.0f);
    REQUIRE(fixture.store->puts == 1);
    REQUIRE(stored_tensors(*fixture.store, key_of(1)).k.eq(3.0f).all().item<bool>());
    REQUIRE(stored_tensors(*fixture.store, key_of(1)).v.eq(-3.0f).all().item<bool>());

    lines_result got = run_get(fixture.kv, {hash_of(1)});
    REQUIRE(got.has_value());
    REQUIRE(line_filled_with(fixture.kv, got->front(), 3.0f));
    REQUIRE(fixture.store->fetches.empty());
    release(fixture.kv, *got);
}

TEST_CASE("an evicted page is fetched back from the store") {
    cache_fixture fixture(2);
    sync_page(fixture.kv, hash_of(1), 1.0f);
    sync_page(fixture.kv, hash_of(2), 2.0f);
    line_ids evicting = allocate(fixture.kv, 2).value();
    fill_line(fixture.kv, evicting[0], 0.0f);
    fill_line(fixture.kv, evicting[1], 0.0f);
    release(fixture.kv, evicting);

    lines_result got = run_get(fixture.kv, {hash_of(2), hash_of(1)});
    REQUIRE(got.has_value());
    REQUIRE(fixture.store->fetches.size() == 1);
    REQUIRE(line_filled_with(fixture.kv, (*got)[0], 2.0f));
    REQUIRE(line_filled_with(fixture.kv, (*got)[1], 1.0f));
    release(fixture.kv, *got);
}

TEST_CASE("free lines are reused least recently released first") {
    cache_fixture fixture(2);
    carve(fixture.kv, 2);
    sync_page(fixture.kv, hash_of(1), 1.0f);
    sync_page(fixture.kv, hash_of(2), 2.0f);
    line_ids evicting = allocate(fixture.kv, 1).value();
    release(fixture.kv, evicting);

    lines_result got = run_get(fixture.kv, {hash_of(2)});
    REQUIRE(got.has_value());
    REQUIRE(fixture.store->fetches.empty());
    release(fixture.kv, *got);
}

TEST_CASE("a page the store does not have fails with miss holding nothing") {
    cache_fixture fixture(2);
    REQUIRE(run_get(fixture.kv, {hash_of(9)}) == miss);
    require_all_free(fixture.kv, 2);
}

TEST_CASE("the oom check does not count the free line of a resident page the get holds") {
    cache_fixture fixture(1);
    sync_page(fixture.kv, hash_of(1), 1.0f);
    fixture.store->pages.emplace(key_of(2), host_page(2.0f));

    REQUIRE(run_get(fixture.kv, {hash_of(1), hash_of(2)}) == oom);
    REQUIRE(fixture.store->fetches.empty());
    lines_result got = run_get(fixture.kv, {hash_of(1)});
    REQUIRE(got.has_value());
    release(fixture.kv, *got);
}

TEST_CASE("concurrent gets of a page share its line and fetch") {
    memory_store backing;
    backing.state->gated = true;
    cache_fixture fixture(2, std::move(backing));
    fixture.store->pages.emplace(key_of(5), host_page(5.0f));

    lines_result first;
    lines_result second;
    std::size_t remaining = 2;
    {
        event_loop loop;
        loop.spawn(get_into(&fixture.kv, {hash_of(5)}, &first, &remaining));
        loop.spawn(get_into(&fixture.kv, {hash_of(5)}, &second, &remaining));
        loop.spawn(open_gate(fixture.store));
        REQUIRE(loop.run() == nullptr);
    }
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());
    REQUIRE(*first == *second);
    REQUIRE(fixture.store->fetches.size() == 1);
    REQUIRE(line_filled_with(fixture.kv, first->front(), 5.0f));

    release(fixture.kv, *first);
    REQUIRE(fixture.kv.free_count() == 1);
    release(fixture.kv, *second);
    require_all_free(fixture.kv, 2);
}

TEST_CASE("a get sharing a fetch that misses fails with miss and frees the line") {
    memory_store backing;
    backing.state->gated = true;
    cache_fixture fixture(2, std::move(backing));

    lines_result first;
    lines_result second;
    std::size_t remaining = 2;
    {
        event_loop loop;
        loop.spawn(get_into(&fixture.kv, {hash_of(5)}, &first, &remaining));
        loop.spawn(get_into(&fixture.kv, {hash_of(5)}, &second, &remaining));
        loop.spawn(open_gate(fixture.store));
        REQUIRE(loop.run() == nullptr);
    }
    REQUIRE(first == miss);
    REQUIRE(second == miss);
    require_all_free(fixture.kv, 2);
}

TEST_CASE("a fetch finishing after sync cached its hash leaves the fetched line unmapped") {
    memory_store backing;
    backing.state->gated = true;
    cache_fixture fixture(2, std::move(backing));
    fixture.store->pages.emplace(key_of(5), host_page(5.0f));

    lines_result fetched;
    std::size_t remaining = 1;
    std::uint32_t synced_line = 0;
    {
        event_loop loop;
        loop.spawn(get_into(&fixture.kv, {hash_of(5)}, &fetched, &remaining));
        loop.spawn(sync_filled_line(&fixture.kv, hash_of(5), 6.0f, &synced_line));
        loop.spawn(open_gate(fixture.store));
        REQUIRE(loop.run() == nullptr);
    }
    REQUIRE(fetched.has_value());
    REQUIRE(fetched->front() != synced_line);
    REQUIRE(line_filled_with(fixture.kv, fetched->front(), 5.0f));
    release(fixture.kv, *fetched);
    release(fixture.kv, line_ids{synced_line});

    lines_result found = run_get(fixture.kv, {hash_of(5)});
    REQUIRE(found.has_value());
    REQUIRE(found->front() == synced_line);
    REQUIRE(line_filled_with(fixture.kv, synced_line, 6.0f));
    release(fixture.kv, *found);
    REQUIRE(allocate(fixture.kv, 1) == lines_result(line_ids{fetched->front()}));
}

TEST_CASE("clone copies a held page into unmapped held lines") {
    cache_fixture fixture(3);
    line_ids tail = allocate(fixture.kv, 1).value();
    fill_line(fixture.kv, tail.front(), 4.0f);
    line_ids copies = fixture.kv.clone(0, tail.front(), 2, compute_streams()).value();
    REQUIRE(copies.size() == 2);
    REQUIRE(line_filled_with(fixture.kv, copies[0], 4.0f));
    REQUIRE(line_filled_with(fixture.kv, copies[1], 4.0f));
    REQUIRE(fixture.kv.clone(0, tail.front(), 1, compute_streams()) == oom);
    release(fixture.kv, tail);
    release(fixture.kv, copies);
    require_all_free(fixture.kv, 3);
}

TEST_CASE("syncing a hash already cached leaves the line unmapped and still stores the page") {
    cache_fixture fixture(3);
    carve(fixture.kv, 3);
    sync_page(fixture.kv, hash_of(1), 1.0f);
    fixture.store->pages.clear();
    line_ids duplicate = allocate(fixture.kv, 1).value();
    fill_line(fixture.kv, duplicate.front(), 1.0f);
    REQUIRE(run_to_completion(sync_one_layer(fixture.kv, {hash_of(1)}, duplicate)).has_value());
    REQUIRE(fixture.store->puts == 2);
    REQUIRE(fixture.store->pages.contains(key_of(1)));
    release(fixture.kv, duplicate);

    line_ids reused = allocate(fixture.kv, 1).value();
    REQUIRE(reused == duplicate);
    release(fixture.kv, reused);
    lines_result got = run_get(fixture.kv, {hash_of(1)});
    REQUIRE(got.has_value());
    REQUIRE(got->front() != duplicate.front());
    release(fixture.kv, *got);
}

TEST_CASE("sync copies each line into its page in the store") {
    cache_fixture fixture(4);
    line_ids synced = allocate(fixture.kv, 3).value();
    for (std::size_t j = 0; j < synced.size(); j++) {
        fill_line(fixture.kv, synced[j], static_cast<float>(j + 1));
    }
    REQUIRE(run_to_completion(sync_one_layer(fixture.kv, {hash_of(1), hash_of(2), hash_of(3)}, synced)).has_value());

    for (std::uint64_t n = 1; n <= 3; n++) {
        pulsarlm::kv stored = stored_tensors(*fixture.store, key_of(n));
        REQUIRE(stored.k.eq(static_cast<float>(n)).all().item<bool>());
        REQUIRE(stored.v.eq(-static_cast<float>(n)).all().item<bool>());
    }
    release(fixture.kv, synced);
}

TEST_CASE("get copies the store's pages into their lines") {
    cache_fixture fixture(4);
    for (std::uint64_t n = 1; n <= 3; n++) {
        fixture.store->pages.emplace(key_of(n), host_page(static_cast<float>(n)));
    }
    lines_result got = run_get(fixture.kv, {hash_of(1), hash_of(2), hash_of(3)});
    REQUIRE(got.has_value());
    for (std::size_t j = 0; j < got->size(); j++) {
        REQUIRE(line_filled_with(fixture.kv, (*got)[j], static_cast<float>(j + 1)));
    }
    release(fixture.kv, *got);
}

TEST_CASE("a get and a sync run together") {
    cache_fixture fixture(6);
    for (std::uint64_t n = 1; n <= 3; n++) {
        fixture.store->pages.emplace(key_of(n), host_page(static_cast<float>(n)));
    }
    line_ids synced = allocate(fixture.kv, 2).value();
    fill_line(fixture.kv, synced[0], 4.0f);
    fill_line(fixture.kv, synced[1], 5.0f);

    lines_result got;
    std::expected<void, page_error> sync_result;
    std::size_t remaining = 2;
    {
        event_loop loop;
        loop.spawn(get_into(&fixture.kv, {hash_of(1), hash_of(2), hash_of(3)}, &got, &remaining));
        loop.spawn(sync_into(&fixture.kv, {hash_of(4), hash_of(5)}, synced, &sync_result, &remaining));
        REQUIRE(loop.run() == nullptr);
    }

    REQUIRE(got.has_value());
    for (std::size_t j = 0; j < got->size(); j++) {
        REQUIRE(line_filled_with(fixture.kv, (*got)[j], static_cast<float>(j + 1)));
    }
    REQUIRE(sync_result.has_value());
    REQUIRE(stored_tensors(*fixture.store, key_of(4)).k.eq(4.0f).all().item<bool>());
    REQUIRE(stored_tensors(*fixture.store, key_of(5)).k.eq(5.0f).all().item<bool>());
    release(fixture.kv, *got);
    release(fixture.kv, synced);
}

TEST_CASE("a values_only cache keeps and stores one tensor as both k and v") {
    pulsarlm::page_shape values_only = shape;
    values_only.values_only = true;
    REQUIRE(pulsarlm::line_bytes(values_only) * 2 == pulsarlm::line_bytes(shape));
    cache_fixture fixture(2, {}, {values_only});
    REQUIRE(fixture.kv.tensors(0).k.is_same(fixture.kv.tensors(0).v));

    line_ids allocated = allocate(fixture.kv, 1).value();
    fixture.kv.tensors(0).v.select(0, allocated.front()).fill_(5.0f);
    REQUIRE(run_to_completion(sync_one_layer(fixture.kv, {hash_of(1)}, allocated)).has_value());
    release(fixture.kv, allocated);
    pulsarlm::kv stored = stored_tensors(*fixture.store, key_of(1), values_only);
    REQUIRE(stored.k.is_same(stored.v));
    REQUIRE(stored.v.eq(5.0f).all().item<bool>());

    line_ids evicting = allocate(fixture.kv, 2).value();
    fixture.kv.tensors(0).v.select(0, evicting[0]).fill_(0.0f);
    fixture.kv.tensors(0).v.select(0, evicting[1]).fill_(0.0f);
    release(fixture.kv, evicting);
    lines_result got = run_get(fixture.kv, {hash_of(1)});
    REQUIRE(got.has_value());
    REQUIRE(fixture.store->fetches.size() == 1);
    REQUIRE(fixture.kv.tensors(0).v.select(0, got->front()).eq(5.0f).all().item<bool>());
    release(fixture.kv, *got);
}

TEST_CASE("each layer's page is stored under its own key") {
    REQUIRE(pulsarlm::store_key(hash_of(1), 0) != pulsarlm::store_key(hash_of(1), 1));
}

TEST_CASE("sync stores a page of every layer in one put") {
    cache_fixture fixture(2, {}, {shape, shape});
    std::vector<line_ids> allocated = allocate_each(fixture.kv, 1).value();
    fill_line(fixture.kv, allocated[0].front(), 1.0f, 0);
    fill_line(fixture.kv, allocated[1].front(), 2.0f, 1);
    std::vector<digest> hashes{hash_of(1)};
    REQUIRE(run_to_completion(sync_each(fixture.kv, hashes, std::vector<std::uint64_t>(hashes.size(), 0), allocated))
                .has_value());

    REQUIRE(fixture.store->puts == 1);
    REQUIRE(stored_tensors(*fixture.store, key_of(1, 0)).k.eq(1.0f).all().item<bool>());
    REQUIRE(stored_tensors(*fixture.store, key_of(1, 1)).k.eq(2.0f).all().item<bool>());
    fixture.kv.release(0, allocated[0]);
    fixture.kv.release(1, allocated[1]);
}

TEST_CASE("sync owns a page by every request whose hash it is, in every layer") {
    cache_fixture fixture(4, {}, {shape, shape});
    std::vector<line_ids> allocated = allocate_each(fixture.kv, 3).value();
    std::vector<digest> hashes{hash_of(1), hash_of(2), hash_of(1)};
    std::vector<std::uint64_t> owners{7, 7, 9};
    REQUIRE(run_to_completion(sync_each(fixture.kv, hashes, owners, allocated)).has_value());

    REQUIRE(fixture.store->puts == 1);
    REQUIRE(fixture.store->pages.size() == 4);
    std::vector<std::pair<std::uint64_t, digest>> expected{
        {7, key_of(1, 0)},
        {7, key_of(2, 0)},
        {9, key_of(1, 0)},
        {7, key_of(1, 1)},
        {7, key_of(2, 1)},
        {9, key_of(1, 1)}
    };
    REQUIRE((fixture.store->owners == expected));
    fixture.kv.release(0, allocated[0]);
    fixture.kv.release(1, allocated[1]);
}

TEST_CASE("get fetches the pages of every layer in one fetch") {
    cache_fixture fixture(2, {}, {shape, shape});
    fixture.store->pages.emplace(key_of(1, 0), host_page(1.0f));
    fixture.store->pages.emplace(key_of(2, 0), host_page(2.0f));
    fixture.store->pages.emplace(key_of(2, 1), host_page(3.0f));
    std::vector<digest> first_layer{hash_of(1), hash_of(2)};
    std::vector<digest> second_layer{hash_of(2)};
    std::vector<std::span<const digest>> hashes{first_layer, second_layer};

    layer_lines_result got = run_to_completion(fixture.kv.get(hashes, compute_streams()));
    REQUIRE(got.has_value());
    REQUIRE((fixture.store->fetches == std::vector<std::vector<digest>>{{key_of(1, 0), key_of(2, 0), key_of(2, 1)}}));
    REQUIRE(line_filled_with(fixture.kv, (*got)[0][0], 1.0f, 0));
    REQUIRE(line_filled_with(fixture.kv, (*got)[0][1], 2.0f, 0));
    REQUIRE(line_filled_with(fixture.kv, (*got)[1][0], 3.0f, 1));
    fixture.kv.release(0, (*got)[0]);
    fixture.kv.release(1, (*got)[1]);
}

TEST_CASE("a page missing in one layer fails the get with miss holding nothing in any layer") {
    cache_fixture fixture(2, {}, {shape, shape});
    fixture.store->pages.emplace(key_of(1, 0), host_page(1.0f));
    std::vector<digest> page_hashes{hash_of(1)};
    std::vector<std::span<const digest>> hashes{page_hashes, page_hashes};

    layer_lines_result got = run_to_completion(fixture.kv.get(hashes, compute_streams()));
    REQUIRE((got == layer_lines_result(std::unexpected(page_error::miss))));
    REQUIRE(fixture.kv.free_count() == 2);
}

TEST_CASE("a line holds each head's K then V together") {
    cache_fixture fixture(2);
    const pulsarlm::kv& lines = fixture.kv.tensors(0);
    auto head_block = static_cast<std::int64_t>(2 * shape.page_size * shape.head_dim);
    REQUIRE(lines.k.stride(2) == head_block);
    REQUIRE(lines.k.stride(1) == static_cast<std::int64_t>(shape.head_dim));
    REQUIRE(lines.k.stride(0) == head_block * static_cast<std::int64_t>(shape.n_kv_heads));
    REQUIRE(lines.v.data_ptr<float>() - lines.k.data_ptr<float>() == head_block / 2);
}

TEST_CASE("allocate fails with oom holding nothing when the layers together run out") {
    cache_fixture fixture(2, {}, {shape, shape});
    REQUIRE((allocate_each(fixture.kv, 3) == layer_lines_result(std::unexpected(page_error::oom))));
    REQUIRE(fixture.kv.free_count() == 2);
    std::vector<line_ids> held = allocate_each(fixture.kv, 2).value();
    REQUIRE(fixture.kv.free_count() == 0);
    fixture.kv.release(0, held[0]);
    fixture.kv.release(1, held[1]);
}

TEST_CASE("a layer's line reclaims the region whose cached page is oldest, whichever layer cached it") {
    cache_fixture fixture(1, {}, {shape, shape});
    fixture.store->pages.emplace(key_of(1, 0), host_page(1.0f));
    fixture.store->pages.emplace(key_of(2, 1), host_page(2.0f));
    fixture.store->pages.emplace(key_of(3, 1), host_page(3.0f));
    std::vector<digest> none;
    auto get_released = [&](std::size_t layer, std::uint64_t n) {
        std::vector<digest> wanted{hash_of(n)};
        std::vector<std::span<const digest>> hashes{none, none};
        hashes[layer] = wanted;
        layer_lines_result got = run_to_completion(fixture.kv.get(hashes, compute_streams()));
        REQUIRE(got.has_value());
        REQUIRE(line_filled_with(fixture.kv, (*got)[layer].front(), static_cast<float>(n), layer));
        fixture.kv.release(layer, (*got)[layer]);
    };

    get_released(0, 1);
    get_released(1, 2);
    REQUIRE(fixture.store->fetches.size() == 2);
    get_released(1, 3);
    get_released(1, 2);
    REQUIRE(fixture.store->fetches.size() == 3);
    get_released(0, 1);
    REQUIRE(fixture.store->fetches.size() == 4);
}

TEST_CASE("a large line reclaims a region of cached small pages and drops them all") {
    pulsarlm::page_shape wide = shape;
    wide.n_kv_heads = 2 * shape.n_kv_heads;
    std::vector<pulsarlm::page_shape> shapes{shape, wide};
    memory_store backing = memory_store{};
    pulsarlm::kv_cache kv(shapes, 2 * pulsarlm::line_bytes(wide), at::Device(at::kCUDA, 0), backing);
    std::vector<digest> narrow_hashes;
    for (std::uint64_t n = 1; n <= 4; n++) {
        backing.state->pages.emplace(key_of(n, 0), host_page(static_cast<float>(n)));
        narrow_hashes.push_back(hash_of(n));
    }
    backing.state->pages.emplace(key_of(5, 1), host_page(5.0f, wide));
    std::vector<digest> none;
    layer_lines_result got = run_to_completion(
        kv.get(std::vector<std::span<const digest>>{narrow_hashes, none}, compute_streams())
    );
    REQUIRE(got.has_value());
    kv.release(0, (*got)[0]);

    std::vector<digest> wide_hashes{hash_of(5)};
    got = run_to_completion(kv.get(std::vector<std::span<const digest>>{none, wide_hashes}, compute_streams()));
    REQUIRE(got.has_value());
    kv.release(1, (*got)[1]);
    REQUIRE(backing.state->fetches.size() == 2);

    std::vector<digest> newer{hash_of(3), hash_of(4)};
    got = run_to_completion(kv.get(std::vector<std::span<const digest>>{newer, none}, compute_streams()));
    REQUIRE(got.has_value());
    kv.release(0, (*got)[0]);
    REQUIRE(backing.state->fetches.size() == 2);
    std::vector<digest> older{hash_of(1)};
    got = run_to_completion(kv.get(std::vector<std::span<const digest>>{older, none}, compute_streams()));
    REQUIRE(got.has_value());
    kv.release(0, (*got)[0]);
    REQUIRE(backing.state->fetches.size() == 3);
}

TEST_CASE("lines of several sizes pack into regions of the largest size") {
    pulsarlm::page_shape wide = shape;
    wide.n_kv_heads = 2 * shape.n_kv_heads;
    std::vector<pulsarlm::page_shape> shapes{shape, wide};
    std::size_t narrow_bytes = pulsarlm::line_bytes(shape);
    memory_store backing = memory_store{};

    REQUIRE_THROWS(pulsarlm::kv_cache(shapes, 3 * narrow_bytes, at::Device(at::kCUDA, 0), backing));

    pulsarlm::kv_cache aligned(shapes, 4 * narrow_bytes, at::Device(at::kCUDA, 0), backing);
    REQUIRE(aligned.free_count() == 1);
    std::vector<line_ids> held = allocate_each(aligned, 1).value();
    fill_line(aligned, held[0].front(), 1.0f, 0);
    fill_line(aligned, held[1].front(), 2.0f, 1);
    REQUIRE(line_filled_with(aligned, held[0].front(), 1.0f, 0));
    REQUIRE(line_filled_with(aligned, held[1].front(), 2.0f, 1));
    REQUIRE(aligned.free_count() == 0);
    line_ids narrow = aligned.clone(0, held[0].front(), 1, compute_streams()).value();
    REQUIRE(line_filled_with(aligned, narrow.front(), 1.0f, 0));
    REQUIRE(
        (aligned.clone(0, held[0].front(), 1, compute_streams()) == lines_result(std::unexpected(page_error::oom)))
    );
    aligned.release(0, narrow);
    aligned.release(0, held[0]);
    aligned.release(1, held[1]);
}

TEST_CASE("an indexed attention line keeps its index keys after K and V, stored and fetched with them") {
    pulsarlm::page_shape indexed = shape;
    indexed.index_head_dim = 6;
    REQUIRE(pulsarlm::line_bytes(indexed) == pulsarlm::line_bytes(shape) + shape.page_size * 6 * 4);
    REQUIRE(std::bit_ceil(pulsarlm::line_bytes(indexed)) == 1024);
    cache_fixture fixture(4, {}, {indexed});
    const pulsarlm::kv& lines = fixture.kv.tensors(0);
    REQUIRE((lines.index_keys.sizes().slice(1) == at::IntArrayRef{static_cast<std::int64_t>(shape.page_size), 6}));
    REQUIRE(lines.index_keys.stride(0) == lines.k.stride(0));
    REQUIRE(
        lines.index_keys.data_ptr<float>() - lines.k.data_ptr<float>() ==
        static_cast<std::int64_t>(2 * shape.page_size * shape.n_kv_heads * shape.head_dim)
    );

    line_ids allocated = allocate(fixture.kv, 1).value();
    fill_line(fixture.kv, allocated.front(), 1.0f);
    at::Tensor index_keys = at::arange(static_cast<std::int64_t>(shape.page_size) * 6, at::kFloat)
                                .view({static_cast<std::int64_t>(shape.page_size), 6});
    lines.index_keys.select(0, allocated.front()).copy_(index_keys);
    REQUIRE(run_to_completion(sync_one_layer(fixture.kv, {hash_of(1)}, allocated)).has_value());
    release(fixture.kv, allocated);
    pulsarlm::kv stored = stored_tensors(*fixture.store, key_of(1), indexed);
    REQUIRE(stored.k.eq(1.0f).all().item<bool>());
    REQUIRE(stored.v.eq(-1.0f).all().item<bool>());
    REQUIRE(stored.index_keys.select(0, 0).equal(index_keys));

    line_ids evicting = allocate(fixture.kv, 2).value();
    lines.index_keys.zero_();
    release(fixture.kv, evicting);
    lines_result got = run_get(fixture.kv, {hash_of(1)});
    REQUIRE(got.has_value());
    REQUIRE(line_filled_with(fixture.kv, got->front(), 1.0f));
    REQUIRE(lines.index_keys.select(0, got->front()).equal(index_keys.cuda()));
    release(fixture.kv, *got);
}

TEST_CASE("a recurrent layer's line is one tensor of its state sizes and dtype, stored and fetched as one") {
    pulsarlm::page_shape state{
        .n_layers = 1,
        .page_size = shape.page_size,
        .n_kv_heads = 0,
        .head_dim = 0,
        .dtype = at::kDouble,
        .values_only = true,
        .state_sizes = {3, 5},
    };
    REQUIRE(pulsarlm::line_bytes(state) == 3 * 5 * 8);
    cache_fixture fixture(3, {}, {shape, state});
    const pulsarlm::kv& lines = fixture.kv.tensors(1);
    REQUIRE(lines.k.is_same(lines.v));
    REQUIRE(lines.k.scalar_type() == at::kDouble);
    REQUIRE((lines.k.sizes().slice(1) == at::IntArrayRef{3, 5}));
    REQUIRE(lines.k.stride(0) == static_cast<std::int64_t>(std::bit_ceil(pulsarlm::line_bytes(state)) / 8));

    std::vector<line_ids> allocated = allocate_each(fixture.kv, 1).value();
    lines.k.select(0, allocated[1].front()).copy_(at::arange(15, at::kDouble).view({3, 5}));
    fill_line(fixture.kv, allocated[0].front(), 1.0f, 0);
    REQUIRE(run_to_completion(sync_each(fixture.kv, {hash_of(1)}, {0}, allocated)).has_value());
    fixture.kv.release(0, allocated[0]);
    fixture.kv.release(1, allocated[1]);
    REQUIRE(stored_tensors(*fixture.store, key_of(1, 1), state).k.equal(at::arange(15, at::kDouble).view({1, 3, 5})));

    std::vector<line_ids> evicting = allocate_each(fixture.kv, 2).value();
    lines.k.zero_();
    fixture.kv.release(0, evicting[0]);
    fixture.kv.release(1, evicting[1]);
    std::vector<digest> none;
    std::vector<digest> wanted{hash_of(1)};
    layer_lines_result got = run_to_completion(
        fixture.kv.get(std::vector<std::span<const digest>>{none, wanted}, compute_streams())
    );
    REQUIRE(got.has_value());
    REQUIRE(lines.k.select(0, (*got)[1].front()).equal(at::arange(15, at::kDouble).view({3, 5}).cuda()));
    fixture.kv.release(1, (*got)[1]);
}

TEST_CASE("the store carries pages of layers with different shapes") {
    pulsarlm::page_shape values_only = shape;
    values_only.values_only = true;
    cache_fixture fixture(4, {}, {values_only, shape});
    std::vector<line_ids> allocated = allocate_each(fixture.kv, 2).value();
    for (std::size_t j = 0; j < 2; j++) {
        fixture.kv.tensors(0).v.select(0, allocated[0][j]).fill_(static_cast<float>(j + 1));
        fill_line(fixture.kv, allocated[1][j], static_cast<float>(j + 3), 1);
    }
    std::vector<digest> hashes{hash_of(1), hash_of(2)};
    REQUIRE(run_to_completion(sync_each(fixture.kv, hashes, std::vector<std::uint64_t>(hashes.size(), 0), allocated))
                .has_value());
    for (std::size_t layer = 0; layer < 2; layer++) {
        fixture.kv.release(layer, allocated[layer]);
    }
    REQUIRE(stored_tensors(*fixture.store, key_of(1, 0), values_only).v.eq(1.0f).all().item<bool>());
    REQUIRE(stored_tensors(*fixture.store, key_of(2, 0), values_only).v.eq(2.0f).all().item<bool>());
    REQUIRE(stored_tensors(*fixture.store, key_of(1, 1)).k.eq(3.0f).all().item<bool>());
    REQUIRE(stored_tensors(*fixture.store, key_of(2, 1)).v.eq(-4.0f).all().item<bool>());

    std::vector<line_ids> evicting = allocate_each(fixture.kv, 4).value();
    for (std::size_t layer = 0; layer < 2; layer++) {
        fixture.kv.tensors(layer).k.zero_();
        fixture.kv.tensors(layer).v.zero_();
        fixture.kv.release(layer, evicting[layer]);
    }
    std::vector<std::span<const digest>> layer_hashes{hashes, hashes};
    layer_lines_result got = run_to_completion(fixture.kv.get(layer_hashes, compute_streams()));
    REQUIRE(got.has_value());
    REQUIRE(fixture.kv.tensors(0).v.select(0, (*got)[0][1]).eq(2.0f).all().item<bool>());
    REQUIRE(line_filled_with(fixture.kv, (*got)[1][0], 3.0f, 1));
    REQUIRE(line_filled_with(fixture.kv, (*got)[1][1], 4.0f, 1));
    for (std::size_t layer = 0; layer < 2; layer++) {
        fixture.kv.release(layer, (*got)[layer]);
    }
}

namespace {

using row_ranges = std::vector<std::pair<std::size_t, std::size_t>>;

const pulsarlm::page_shape indexed_shape{
    .n_layers = 1,
    .page_size = 4,
    .n_kv_heads = 2,
    .head_dim = 2,
    .dtype = at::kFloat,
    .index_head_dim = 2,
};
const pulsarlm::page_shape state_shape{
    .n_layers = 1,
    .page_size = 4,
    .n_kv_heads = 0,
    .head_dim = 0,
    .dtype = at::kFloat,
    .values_only = true,
    .state_sizes = {4, 3},
};

std::size_t first_dim(const pulsarlm::page_shape& page_shape) {
    return page_shape.state_sizes.empty() ? page_shape.n_kv_heads
                                          : static_cast<std::size_t>(page_shape.state_sizes.front());
}

pulsarlm::page_shape with_rows(pulsarlm::page_shape page_shape, const row_ranges& rows) {
    std::size_t count = 0;
    for (auto [first, last] : rows) {
        count += last - first;
    }
    if (page_shape.state_sizes.empty()) {
        page_shape.n_kv_heads = count;
    } else {
        page_shape.state_sizes.front() = static_cast<std::int64_t>(count);
    }
    return page_shape;
}

// A full page of distinct floats from base, in pageable memory.
page full_page(const pulsarlm::page_shape& page_shape, float base) {
    auto floats = static_cast<std::int64_t>(pulsarlm::line_bytes(page_shape) / sizeof(float));
    return page{.bytes = (at::arange(floats, at::kFloat) + base).view(at::kByte)};
}

// The bytes of full's rows, then its index keys: one shard's line of it.
page shard_page(const page& full, const pulsarlm::page_shape& page_shape, const row_ranges& rows) {
    std::size_t index_bytes = page_shape.page_size * page_shape.index_head_dim * sizeof(float);
    std::size_t row_bytes = (pulsarlm::line_bytes(page_shape) - index_bytes) / first_dim(page_shape);
    std::vector<at::Tensor> parts;
    for (auto [first, last] : rows) {
        parts.push_back(full.bytes.narrow(
            0,
            static_cast<std::int64_t>(first * row_bytes),
            static_cast<std::int64_t>((last - first) * row_bytes)
        ));
    }
    if (index_bytes > 0) {
        parts.push_back(full.bytes.narrow(
            0,
            static_cast<std::int64_t>(first_dim(page_shape) * row_bytes),
            static_cast<std::int64_t>(index_bytes)
        ));
    }
    return page{.bytes = at::cat(parts).contiguous()};
}

void write_line(
    const pulsarlm::kv& lines,
    std::uint32_t line_id,
    const page& p,
    const pulsarlm::page_shape& page_shape
) {
    pulsarlm::kv source = pulsarlm::page_tensors(p, page_shape);
    lines.k.select(0, line_id).copy_(source.k.select(0, 0));
    if (!page_shape.values_only) {
        lines.v.select(0, line_id).copy_(source.v.select(0, 0));
    }
    if (page_shape.index_head_dim > 0) {
        lines.index_keys.select(0, line_id).copy_(source.index_keys.select(0, 0));
    }
}

bool line_equals(
    const pulsarlm::kv& lines,
    std::uint32_t line_id,
    const page& p,
    const pulsarlm::page_shape& page_shape
) {
    pulsarlm::kv expected = pulsarlm::page_tensors(p, page_shape);
    bool equal = at::equal(lines.k.select(0, line_id).cpu(), expected.k.select(0, 0)) &&
        at::equal(lines.v.select(0, line_id).cpu(), expected.v.select(0, 0));
    if (page_shape.index_head_dim > 0) {
        equal = equal && at::equal(lines.index_keys.select(0, line_id).cpu(), expected.index_keys.select(0, 0));
    }
    return equal;
}

struct sharded_fixture {
    std::vector<pulsarlm::page_shape> full_shapes;
    // Per shard, per layer.
    std::vector<std::vector<row_ranges>> rows;
    memory_store backing;

    std::vector<pulsarlm::kv_cache_shard> cache_shards() const {
        std::vector<pulsarlm::kv_cache_shard> shards;
        for (const std::vector<row_ranges>& shard_rows : rows) {
            pulsarlm::kv_cache_shard shard{.device = at::Device(at::kCUDA, 0), .rows = shard_rows};
            for (std::size_t layer = 0; layer < full_shapes.size(); layer++) {
                shard.shapes.push_back(with_rows(full_shapes[layer], shard_rows[layer]));
            }
            shards.push_back(std::move(shard));
        }
        return shards;
    }

    pulsarlm::kv_cache make_cache() {
        return pulsarlm::kv_cache(full_shapes, cache_shards(), 4 * lines_bytes(full_shapes), backing);
    }

    std::vector<cudaStream_t> streams() const {
        return std::vector<cudaStream_t>(rows.size(), c10::cuda::getCurrentCUDAStream().stream());
    }
};

task<std::expected<void, page_error>>
sync_sharded(pulsarlm::kv_cache& kv, std::vector<line_ids> lines, std::vector<cudaStream_t> streams) {
    std::vector<std::vector<digest>> hashes(kv.layer_count(), {hash_of(1)});
    std::vector<std::vector<std::uint64_t>> owners(kv.layer_count(), {0});
    co_return co_await kv.sync(hashes, owners, lines, streams);
}

task<layer_lines_result> get_sharded(pulsarlm::kv_cache& kv, std::vector<cudaStream_t> streams) {
    std::vector<digest> wanted{hash_of(1)};
    std::vector<std::span<const digest>> hashes(kv.layer_count(), wanted);
    co_return co_await kv.get(hashes, streams);
}

// Writes each shard's rows of a full page per layer, puts the lines, and fetches them into a second cache.
void check_round_trip(sharded_fixture& fixture) {
    fixture.backing = memory_store{};
    std::vector<pulsarlm::kv_cache_shard> shards = fixture.cache_shards();
    std::vector<page> full_pages;
    for (std::size_t layer = 0; layer < fixture.full_shapes.size(); layer++) {
        full_pages.push_back(full_page(fixture.full_shapes[layer], 100.0f * static_cast<float>(layer + 1)));
    }

    pulsarlm::kv_cache writer = fixture.make_cache();
    layer_lines_result lines = allocate_each(writer, 1);
    REQUIRE(lines);
    for (std::size_t shard = 0; shard < shards.size(); shard++) {
        for (std::size_t layer = 0; layer < fixture.full_shapes.size(); layer++) {
            write_line(
                writer.tensors(layer, shard),
                (*lines)[layer].front(),
                shard_page(full_pages[layer], fixture.full_shapes[layer], shards[shard].rows[layer]),
                shards[shard].shapes[layer]
            );
        }
    }
    REQUIRE(run_to_completion(sync_sharded(writer, *lines, fixture.streams())));
    for (std::size_t layer = 0; layer < fixture.full_shapes.size(); layer++) {
        REQUIRE(at::equal(fixture.backing.state->pages.at(key_of(1, layer)).bytes, full_pages[layer].bytes));
    }

    pulsarlm::kv_cache reader = fixture.make_cache();
    layer_lines_result got = run_to_completion(get_sharded(reader, fixture.streams()));
    REQUIRE(got);
    for (std::size_t shard = 0; shard < shards.size(); shard++) {
        for (std::size_t layer = 0; layer < fixture.full_shapes.size(); layer++) {
            REQUIRE(line_equals(
                reader.tensors(layer, shard),
                (*got)[layer].front(),
                shard_page(full_pages[layer], fixture.full_shapes[layer], shards[shard].rows[layer]),
                shards[shard].shapes[layer]
            ));
        }
    }
}

}  // namespace

TEST_CASE("a sharded cache puts each byte of a page once and fetches every shard's rows of it", "[runtime][cuda]") {
    sharded_fixture fixture{
        .full_shapes = {indexed_shape, state_shape},
        .rows = {{{{0, 1}}, {{0, 1}, {2, 3}}}, {{{1, 2}}, {{1, 2}, {3, 4}}}},
    };
    check_round_trip(fixture);
}

TEST_CASE("shards holding the same KV head put it once and both fetch it", "[runtime][cuda]") {
    pulsarlm::page_shape one_head = indexed_shape;
    one_head.n_kv_heads = 1;
    sharded_fixture fixture{
        .full_shapes = {one_head, state_shape},
        .rows = {{{{0, 1}}, {{0, 2}}}, {{{0, 1}}, {{2, 4}}}},
    };
    check_round_trip(fixture);
}

TEST_CASE("a sharded cache rejects rows that overlap in part or leave rows out", "[runtime][cuda]") {
    sharded_fixture partial{
        .full_shapes = {state_shape},
        .rows = {{{{0, 2}}}, {{{1, 3}}}},
        .backing = memory_store{},
    };
    REQUIRE_THROWS_AS(partial.make_cache(), std::invalid_argument);
    sharded_fixture uncovered{
        .full_shapes = {state_shape},
        .rows = {{{{0, 2}}}, {{{0, 2}}}},
        .backing = memory_store{},
    };
    REQUIRE_THROWS_AS(uncovered.make_cache(), std::invalid_argument);
}
