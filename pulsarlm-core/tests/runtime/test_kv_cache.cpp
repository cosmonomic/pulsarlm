#include "pulsarlm/kv_cache.hpp"
#include "pulsarlm/digest.hpp"
#include "pulsarlm/event_loop.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>

#undef CHECK
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
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
using pulsarlm::task;

using line_ids = std::vector<std::uint32_t>;
using lines_result = std::expected<line_ids, page_error>;
using layer_lines_result = std::expected<std::vector<line_ids>, page_error>;

constexpr pulsarlm::page_shape
    shape{.n_layers = 2, .page_size = 4, .n_kv_heads = 2, .head_dim = 8, .dtype = at::kFloat};

const lines_result oom = std::unexpected(page_error::oom);
const lines_result miss = std::unexpected(page_error::miss);

struct store_state {
    std::unordered_map<digest, page, digest_hash> pages;
    std::vector<std::vector<digest>> fetches;
    std::size_t puts = 0;
    bool gated = false;
    std::vector<external_task<>::resolver> gates;
};

struct memory_store final : pulsarlm::page_store {
    std::shared_ptr<store_state> state = std::make_shared<store_state>();

    task<std::vector<std::optional<page>>> fetch(std::span<const digest> hashes) override {
        std::shared_ptr<store_state> shared = state;
        std::vector<digest> requested(hashes.begin(), hashes.end());
        shared->fetches.push_back(requested);
        if (shared->gated) {
            auto [gate, opener] = external_task<>::make();
            shared->gates.push_back(opener);
            co_await std::move(gate);
        }
        std::vector<std::optional<page>> found;
        for (const digest& hash : requested) {
            auto it = shared->pages.find(hash);
            found.push_back(it == shared->pages.end() ? std::nullopt : std::optional<page>(it->second));
        }
        co_return found;
    }

    void put(std::vector<digest>&& keys, std::vector<page>&& pages) override {
        state->puts++;
        for (std::size_t i = 0; i < keys.size(); i++) {
            state->pages.insert_or_assign(keys[i], std::move(pages[i]));
        }
    }
};

cudaStream_t compute_stream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

struct cache_fixture {
    memory_store backing;
    std::shared_ptr<store_state> store;
    pulsarlm::kv_cache kv;

    // One layer of shape unless shapes gives one per layer, each with n_lines lines. The staging buffer holds
    // staging_pages pages of shape.
    explicit cache_fixture(
        std::size_t n_lines,
        memory_store backing_store = {},
        std::size_t staging_pages = 2,
        const std::vector<pulsarlm::page_shape>& shapes = {shape}
    )
    : backing(std::move(backing_store)),
      store(backing.state),
      kv(shapes,
         std::vector<std::size_t>(shapes.size(), n_lines),
         staging_pages * pulsarlm::line_bytes(shape),
         at::Device(at::kCUDA, 0),
         backing) {}
};

digest hash_of(std::uint64_t n) {
    return digest{.hi = n, .lo = n * 31 + 7};
}

digest key_of(std::uint64_t n, std::size_t layer = 0) {
    return pulsarlm::store_key(hash_of(n), layer);
}

at::Tensor page_tensor(float value) {
    return at::full(
        {static_cast<std::int64_t>(shape.n_layers),
         static_cast<std::int64_t>(shape.page_size),
         static_cast<std::int64_t>(shape.n_kv_heads),
         static_cast<std::int64_t>(shape.head_dim)},
        value
    );
}

page host_page(float value) {
    return page{.k = page_tensor(value), .v = page_tensor(-value)};
}

void fill_line(pulsarlm::kv_cache& kv, std::uint32_t line_id, float value, std::size_t layer = 0) {
    kv.tensors(layer).k.select(1, line_id).fill_(value);
    kv.tensors(layer).v.select(1, line_id).fill_(-value);
}

bool line_filled_with(pulsarlm::kv_cache& kv, std::uint32_t line_id, float value, std::size_t layer = 0) {
    return kv.tensors(layer).k.select(1, line_id).eq(value).all().item<bool>() &&
        kv.tensors(layer).v.select(1, line_id).eq(-value).all().item<bool>();
}

lines_result only_layer(layer_lines_result result) {
    if (!result) {
        return std::unexpected(result.error());
    }
    REQUIRE(result->size() == 1);
    return std::move(result->front());
}

lines_result allocate(pulsarlm::kv_cache& kv, std::size_t count) {
    return only_layer(kv.allocate(count));
}

void release(pulsarlm::kv_cache& kv, const line_ids& held) {
    kv.release(0, held);
}

task<lines_result> get_one_layer(pulsarlm::kv_cache& kv, std::vector<digest> hashes) {
    std::vector<std::span<const digest>> layer_hashes{hashes};
    layer_lines_result got = co_await kv.get(layer_hashes, compute_stream());
    co_return only_layer(std::move(got));
}

task<std::expected<void, page_error>>
sync_one_layer(pulsarlm::kv_cache& kv, std::vector<digest> hashes, line_ids synced) {
    std::vector<line_ids> layer_lines{std::move(synced)};
    std::expected<void, page_error> result = co_await kv.sync(hashes, layer_lines, compute_stream());
    co_return result;
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

void require_all_free(pulsarlm::kv_cache& kv, std::size_t n_lines) {
    REQUIRE(kv.free_line_count(0) == n_lines);
    lines_result all = allocate(kv, n_lines);
    REQUIRE(all.has_value());
    release(kv, *all);
}

}  // namespace

TEST_CASE("allocate fails with oom holding nothing when lines run out") {
    cache_fixture fixture(4);
    REQUIRE(allocate(fixture.kv, 5) == oom);
    line_ids held = allocate(fixture.kv, 4).value();
    REQUIRE(fixture.kv.free_line_count(0) == 0);
    REQUIRE(allocate(fixture.kv, 1) == oom);
    release(fixture.kv, held);
    require_all_free(fixture.kv, 4);
}

TEST_CASE("sync stores host copies and get finds the page without fetching") {
    cache_fixture fixture(4);
    sync_page(fixture.kv, hash_of(1), 3.0f);
    REQUIRE(fixture.store->puts == 1);
    REQUIRE(fixture.store->pages.at(key_of(1)).k.eq(3.0f).all().item<bool>());
    REQUIRE(fixture.store->pages.at(key_of(1)).v.eq(-3.0f).all().item<bool>());

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
    REQUIRE(fixture.kv.free_line_count(0) == 1);
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
    line_ids copies = fixture.kv.clone(0, tail.front(), 2, compute_stream()).value();
    REQUIRE(copies.size() == 2);
    REQUIRE(line_filled_with(fixture.kv, copies[0], 4.0f));
    REQUIRE(line_filled_with(fixture.kv, copies[1], 4.0f));
    REQUIRE(fixture.kv.clone(0, tail.front(), 1, compute_stream()) == oom);
    release(fixture.kv, tail);
    release(fixture.kv, copies);
    require_all_free(fixture.kv, 3);
}

TEST_CASE("syncing a hash already cached leaves the line unmapped and still stores the page") {
    cache_fixture fixture(3);
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

TEST_CASE("sync hands the store pageable copies, a staging buffer's worth at a time") {
    cache_fixture fixture(4, {}, 1);
    line_ids synced = allocate(fixture.kv, 3).value();
    for (std::size_t j = 0; j < synced.size(); j++) {
        fill_line(fixture.kv, synced[j], static_cast<float>(j + 1));
    }
    REQUIRE(run_to_completion(sync_one_layer(fixture.kv, {hash_of(1), hash_of(2), hash_of(3)}, synced)).has_value());

    for (std::uint64_t n = 1; n <= 3; n++) {
        const page& stored = fixture.store->pages.at(key_of(n));
        REQUIRE(stored.k.eq(static_cast<float>(n)).all().item<bool>());
        REQUIRE(stored.v.eq(-static_cast<float>(n)).all().item<bool>());
        REQUIRE(!stored.k.is_pinned());
    }
    release(fixture.kv, synced);
}

TEST_CASE("get copies the fetched pages a staging buffer's worth at a time") {
    cache_fixture fixture(4, {}, 1);
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

TEST_CASE("a get and a sync take turns on the staging buffer") {
    cache_fixture fixture(6, {}, 1);
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
    REQUIRE(fixture.store->pages.at(key_of(4)).k.eq(4.0f).all().item<bool>());
    REQUIRE(fixture.store->pages.at(key_of(5)).k.eq(5.0f).all().item<bool>());
    release(fixture.kv, *got);
    release(fixture.kv, synced);
}

TEST_CASE("a values_only cache keeps and stores one tensor as both k and v") {
    pulsarlm::page_shape values_only = shape;
    values_only.values_only = true;
    REQUIRE(pulsarlm::line_bytes(values_only) * 2 == pulsarlm::line_bytes(shape));
    cache_fixture fixture(2, {}, 2, {values_only});
    REQUIRE(fixture.kv.tensors(0).k.is_same(fixture.kv.tensors(0).v));

    line_ids allocated = allocate(fixture.kv, 1).value();
    fixture.kv.tensors(0).v.select(1, allocated.front()).fill_(5.0f);
    REQUIRE(run_to_completion(sync_one_layer(fixture.kv, {hash_of(1)}, allocated)).has_value());
    release(fixture.kv, allocated);
    const page& stored = fixture.store->pages.at(key_of(1));
    REQUIRE(stored.k.is_same(stored.v));
    REQUIRE(stored.v.eq(5.0f).all().item<bool>());

    line_ids evicting = allocate(fixture.kv, 2).value();
    fixture.kv.tensors(0).v.select(1, evicting[0]).fill_(0.0f);
    fixture.kv.tensors(0).v.select(1, evicting[1]).fill_(0.0f);
    release(fixture.kv, evicting);
    lines_result got = run_get(fixture.kv, {hash_of(1)});
    REQUIRE(got.has_value());
    REQUIRE(fixture.store->fetches.size() == 1);
    REQUIRE(fixture.kv.tensors(0).v.select(1, got->front()).eq(5.0f).all().item<bool>());
    release(fixture.kv, *got);
}

TEST_CASE("each layer's page is stored under its own key") {
    REQUIRE(pulsarlm::store_key(hash_of(1), 0) != pulsarlm::store_key(hash_of(1), 1));
}

TEST_CASE("sync stores a page of every layer in one put") {
    cache_fixture fixture(2, {}, 2, {shape, shape});
    std::vector<line_ids> allocated = fixture.kv.allocate(1).value();
    fill_line(fixture.kv, allocated[0].front(), 1.0f, 0);
    fill_line(fixture.kv, allocated[1].front(), 2.0f, 1);
    std::vector<digest> hashes{hash_of(1)};
    REQUIRE(run_to_completion(fixture.kv.sync(hashes, allocated, compute_stream())).has_value());

    REQUIRE(fixture.store->puts == 1);
    REQUIRE(fixture.store->pages.at(key_of(1, 0)).k.eq(1.0f).all().item<bool>());
    REQUIRE(fixture.store->pages.at(key_of(1, 1)).k.eq(2.0f).all().item<bool>());
    fixture.kv.release(0, allocated[0]);
    fixture.kv.release(1, allocated[1]);
}

TEST_CASE("get fetches the pages of every layer in one fetch") {
    cache_fixture fixture(2, {}, 2, {shape, shape});
    fixture.store->pages.emplace(key_of(1, 0), host_page(1.0f));
    fixture.store->pages.emplace(key_of(2, 0), host_page(2.0f));
    fixture.store->pages.emplace(key_of(2, 1), host_page(3.0f));
    std::vector<digest> first_layer{hash_of(1), hash_of(2)};
    std::vector<digest> second_layer{hash_of(2)};
    std::vector<std::span<const digest>> hashes{first_layer, second_layer};

    layer_lines_result got = run_to_completion(fixture.kv.get(hashes, compute_stream()));
    REQUIRE(got.has_value());
    REQUIRE((fixture.store->fetches == std::vector<std::vector<digest>>{{key_of(1, 0), key_of(2, 0), key_of(2, 1)}}));
    REQUIRE(line_filled_with(fixture.kv, (*got)[0][0], 1.0f, 0));
    REQUIRE(line_filled_with(fixture.kv, (*got)[0][1], 2.0f, 0));
    REQUIRE(line_filled_with(fixture.kv, (*got)[1][0], 3.0f, 1));
    fixture.kv.release(0, (*got)[0]);
    fixture.kv.release(1, (*got)[1]);
}

TEST_CASE("a page missing in one layer fails the get with miss holding nothing in any layer") {
    cache_fixture fixture(2, {}, 2, {shape, shape});
    fixture.store->pages.emplace(key_of(1, 0), host_page(1.0f));
    std::vector<digest> page_hashes{hash_of(1)};
    std::vector<std::span<const digest>> hashes{page_hashes, page_hashes};

    layer_lines_result got = run_to_completion(fixture.kv.get(hashes, compute_stream()));
    REQUIRE((got == layer_lines_result(std::unexpected(page_error::miss))));
    REQUIRE(fixture.kv.free_line_count(0) == 2);
    REQUIRE(fixture.kv.free_line_count(1) == 2);
}

TEST_CASE("allocate fails with oom holding nothing when one layer runs out") {
    memory_store backing;
    pulsarlm::page_shape shapes[] = {shape, shape};
    std::size_t line_counts[] = {3, 1};
    pulsarlm::kv_cache kv(shapes, line_counts, pulsarlm::line_bytes(shape), at::Device(at::kCUDA, 0), backing);

    REQUIRE((kv.allocate(2) == layer_lines_result(std::unexpected(page_error::oom))));
    REQUIRE(kv.free_line_count(0) == 3);
    REQUIRE(kv.free_line_count(1) == 1);
}

TEST_CASE("the staging buffer carries pages of layers with different shapes") {
    pulsarlm::page_shape values_only = shape;
    values_only.values_only = true;
    cache_fixture fixture(4, {}, 1, {values_only, shape});
    std::vector<line_ids> allocated = fixture.kv.allocate(2).value();
    for (std::size_t j = 0; j < 2; j++) {
        fixture.kv.tensors(0).v.select(1, allocated[0][j]).fill_(static_cast<float>(j + 1));
        fill_line(fixture.kv, allocated[1][j], static_cast<float>(j + 3), 1);
    }
    std::vector<digest> hashes{hash_of(1), hash_of(2)};
    REQUIRE(run_to_completion(fixture.kv.sync(hashes, allocated, compute_stream())).has_value());
    for (std::size_t layer = 0; layer < 2; layer++) {
        fixture.kv.release(layer, allocated[layer]);
    }
    REQUIRE(fixture.store->pages.at(key_of(1, 0)).v.eq(1.0f).all().item<bool>());
    REQUIRE(fixture.store->pages.at(key_of(2, 0)).v.eq(2.0f).all().item<bool>());
    REQUIRE(fixture.store->pages.at(key_of(1, 1)).k.eq(3.0f).all().item<bool>());
    REQUIRE(fixture.store->pages.at(key_of(2, 1)).v.eq(-4.0f).all().item<bool>());

    std::vector<line_ids> evicting = fixture.kv.allocate(4).value();
    for (std::size_t layer = 0; layer < 2; layer++) {
        fixture.kv.tensors(layer).k.zero_();
        fixture.kv.tensors(layer).v.zero_();
        fixture.kv.release(layer, evicting[layer]);
    }
    std::vector<std::span<const digest>> layer_hashes{hashes, hashes};
    layer_lines_result got = run_to_completion(fixture.kv.get(layer_hashes, compute_stream()));
    REQUIRE(got.has_value());
    REQUIRE(fixture.kv.tensors(0).v.select(1, (*got)[0][1]).eq(2.0f).all().item<bool>());
    REQUIRE(line_filled_with(fixture.kv, (*got)[1][0], 3.0f, 1));
    REQUIRE(line_filled_with(fixture.kv, (*got)[1][1], 4.0f, 1));
    for (std::size_t layer = 0; layer < 2; layer++) {
        fixture.kv.release(layer, (*got)[layer]);
    }
}
