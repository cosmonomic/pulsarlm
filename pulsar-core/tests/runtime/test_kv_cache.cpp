#include "pulsar/kv_cache.hpp"
#include "pulsar/digest.hpp"
#include "pulsar/event_loop.hpp"

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

using pulsar::digest;
using pulsar::digest_hash;
using pulsar::event_loop;
using pulsar::external_task;
using pulsar::page;
using pulsar::page_error;
using pulsar::task;

using line_ids = std::vector<std::uint32_t>;
using lines_result = std::expected<line_ids, page_error>;

constexpr pulsar::page_shape shape{.n_layers = 2, .page_size = 4, .n_kv_heads = 2, .head_dim = 8, .dtype = at::kFloat};

const lines_result oom = std::unexpected(page_error::oom);
const lines_result miss = std::unexpected(page_error::miss);

struct store_state {
    std::unordered_map<digest, page, digest_hash> pages;
    std::vector<std::vector<digest>> fetches;
    std::size_t puts = 0;
    bool gated = false;
    std::vector<external_task<>::resolver> gates;
};

struct memory_store {
    std::shared_ptr<store_state> state = std::make_shared<store_state>();

    task<std::vector<std::optional<page>>> fetch(std::span<const digest> hashes) {
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

    void put(std::vector<digest> keys, std::vector<page> pages) {
        state->puts++;
        for (std::size_t i = 0; i < keys.size(); i++) {
            state->pages.insert_or_assign(keys[i], std::move(pages[i]));
        }
    }
};

static_assert(pulsar::page_store<memory_store>);

using memory_kv_cache = pulsar::kv_cache<memory_store>;

cudaStream_t compute_stream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

struct cache_fixture {
    std::shared_ptr<store_state> store;
    memory_kv_cache kv;

    explicit cache_fixture(std::size_t n_lines, memory_store backing = {})
    : store(backing.state),
      kv(shape, n_lines, at::Device(at::kCUDA, 0), std::move(backing)) {}
};

digest hash_of(std::uint64_t n) {
    return digest{.hi = n, .lo = n * 31 + 7};
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

void fill_line(memory_kv_cache& kv, std::uint32_t line_id, float value) {
    kv.k_lines().select(1, line_id).fill_(value);
    kv.v_lines().select(1, line_id).fill_(-value);
}

bool line_filled_with(memory_kv_cache& kv, std::uint32_t line_id, float value) {
    return kv.k_lines().select(1, line_id).eq(value).all().item<bool>() &&
           kv.v_lines().select(1, line_id).eq(-value).all().item<bool>();
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

task<void> get_into(memory_kv_cache* kv, std::vector<digest> hashes, lines_result* result, std::size_t* remaining) {
    *result = co_await kv->get(std::move(hashes), compute_stream());
    if (--*remaining == 0) {
        event_loop::current()->stop();
    }
}

task<void> open_gate(std::shared_ptr<store_state> store) {
    REQUIRE(store->gates.size() == 1);
    store->gates.front().resolve();
    co_return;
}

task<void> put_filled_line(memory_kv_cache* kv, digest hash, float value, std::uint32_t* put_line) {
    line_ids allocated = kv->allocate(1).value();
    fill_line(*kv, allocated.front(), value);
    std::expected<void, page_error> put_result = co_await kv->put({hash}, allocated, compute_stream());
    REQUIRE(put_result.has_value());
    *put_line = allocated.front();
}

lines_result run_get(memory_kv_cache& kv, std::vector<digest> hashes) {
    return run_to_completion(kv.get(std::move(hashes), compute_stream()));
}

void put_page(memory_kv_cache& kv, digest hash, float value) {
    line_ids allocated = kv.allocate(1).value();
    fill_line(kv, allocated.front(), value);
    REQUIRE(run_to_completion(kv.put({hash}, allocated, compute_stream())).has_value());
    kv.release(allocated);
}

void require_all_free(memory_kv_cache& kv, std::size_t n_lines) {
    REQUIRE(kv.free_line_count() == n_lines);
    lines_result all = kv.allocate(n_lines);
    REQUIRE(all.has_value());
    kv.release(*all);
}

}  // namespace

TEST_CASE("allocate fails with oom holding nothing when lines run out") {
    cache_fixture fixture(4);
    REQUIRE(fixture.kv.allocate(5) == oom);
    line_ids held = fixture.kv.allocate(4).value();
    REQUIRE(fixture.kv.free_line_count() == 0);
    REQUIRE(fixture.kv.allocate(1) == oom);
    fixture.kv.release(held);
    require_all_free(fixture.kv, 4);
}

TEST_CASE("put stores host copies and get finds the page without fetching") {
    cache_fixture fixture(4);
    put_page(fixture.kv, hash_of(1), 3.0f);
    REQUIRE(fixture.store->puts == 1);
    REQUIRE(fixture.store->pages.at(hash_of(1)).k.eq(3.0f).all().item<bool>());
    REQUIRE(fixture.store->pages.at(hash_of(1)).v.eq(-3.0f).all().item<bool>());

    lines_result got = run_get(fixture.kv, {hash_of(1)});
    REQUIRE(got.has_value());
    REQUIRE(line_filled_with(fixture.kv, got->front(), 3.0f));
    REQUIRE(fixture.store->fetches.empty());
    fixture.kv.release(*got);
}

TEST_CASE("an evicted page is fetched back from the store") {
    cache_fixture fixture(2);
    put_page(fixture.kv, hash_of(1), 1.0f);
    put_page(fixture.kv, hash_of(2), 2.0f);
    line_ids evicting = fixture.kv.allocate(2).value();
    fill_line(fixture.kv, evicting[0], 0.0f);
    fill_line(fixture.kv, evicting[1], 0.0f);
    fixture.kv.release(evicting);

    lines_result got = run_get(fixture.kv, {hash_of(2), hash_of(1)});
    REQUIRE(got.has_value());
    REQUIRE(fixture.store->fetches.size() == 1);
    REQUIRE(line_filled_with(fixture.kv, (*got)[0], 2.0f));
    REQUIRE(line_filled_with(fixture.kv, (*got)[1], 1.0f));
    fixture.kv.release(*got);
}

TEST_CASE("free lines are reused least recently released first") {
    cache_fixture fixture(2);
    put_page(fixture.kv, hash_of(1), 1.0f);
    put_page(fixture.kv, hash_of(2), 2.0f);
    line_ids evicting = fixture.kv.allocate(1).value();
    fixture.kv.release(evicting);

    lines_result got = run_get(fixture.kv, {hash_of(2)});
    REQUIRE(got.has_value());
    REQUIRE(fixture.store->fetches.empty());
    fixture.kv.release(*got);
}

TEST_CASE("a page the store does not have fails with miss holding nothing") {
    cache_fixture fixture(2);
    REQUIRE(run_get(fixture.kv, {hash_of(9)}) == miss);
    require_all_free(fixture.kv, 2);
}

TEST_CASE("the oom check does not count the free line of a resident page the get holds") {
    cache_fixture fixture(1);
    put_page(fixture.kv, hash_of(1), 1.0f);
    fixture.store->pages.emplace(hash_of(2), host_page(2.0f));

    REQUIRE(run_get(fixture.kv, {hash_of(1), hash_of(2)}) == oom);
    REQUIRE(fixture.store->fetches.empty());
    lines_result got = run_get(fixture.kv, {hash_of(1)});
    REQUIRE(got.has_value());
    fixture.kv.release(*got);
}

TEST_CASE("a page repeated in one get shares one line and one fetch") {
    cache_fixture fixture(1);
    fixture.store->pages.emplace(hash_of(2), host_page(2.0f));
    lines_result got = run_get(fixture.kv, {hash_of(2), hash_of(2)});
    REQUIRE(got.has_value());
    REQUIRE((*got)[0] == (*got)[1]);
    REQUIRE(fixture.store->fetches.size() == 1);
    REQUIRE(line_filled_with(fixture.kv, (*got)[0], 2.0f));
    fixture.kv.release(*got);
    require_all_free(fixture.kv, 1);
}

TEST_CASE("concurrent gets of a page share its line and fetch") {
    memory_store backing;
    backing.state->gated = true;
    cache_fixture fixture(2, std::move(backing));
    fixture.store->pages.emplace(hash_of(5), host_page(5.0f));

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

    fixture.kv.release(*first);
    REQUIRE(fixture.kv.free_line_count() == 1);
    fixture.kv.release(*second);
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

TEST_CASE("a fetch finishing after put cached its hash leaves the fetched line unmapped") {
    memory_store backing;
    backing.state->gated = true;
    cache_fixture fixture(2, std::move(backing));
    fixture.store->pages.emplace(hash_of(5), host_page(5.0f));

    lines_result fetched;
    std::size_t remaining = 1;
    std::uint32_t put_line = 0;
    {
        event_loop loop;
        loop.spawn(get_into(&fixture.kv, {hash_of(5)}, &fetched, &remaining));
        loop.spawn(put_filled_line(&fixture.kv, hash_of(5), 6.0f, &put_line));
        loop.spawn(open_gate(fixture.store));
        REQUIRE(loop.run() == nullptr);
    }
    REQUIRE(fetched.has_value());
    REQUIRE(fetched->front() != put_line);
    REQUIRE(line_filled_with(fixture.kv, fetched->front(), 5.0f));
    fixture.kv.release(*fetched);
    fixture.kv.release(line_ids{put_line});

    lines_result found = run_get(fixture.kv, {hash_of(5)});
    REQUIRE(found.has_value());
    REQUIRE(found->front() == put_line);
    REQUIRE(line_filled_with(fixture.kv, put_line, 6.0f));
    fixture.kv.release(*found);
    REQUIRE(fixture.kv.allocate(1) == lines_result(line_ids{fetched->front()}));
}

TEST_CASE("clone copies a held page into unmapped held lines") {
    cache_fixture fixture(3);
    line_ids tail = fixture.kv.allocate(1).value();
    fill_line(fixture.kv, tail.front(), 4.0f);
    line_ids copies = fixture.kv.clone(tail.front(), 2, compute_stream()).value();
    REQUIRE(copies.size() == 2);
    REQUIRE(line_filled_with(fixture.kv, copies[0], 4.0f));
    REQUIRE(line_filled_with(fixture.kv, copies[1], 4.0f));
    REQUIRE(fixture.kv.clone(tail.front(), 1, compute_stream()) == oom);
    fixture.kv.release(tail);
    fixture.kv.release(copies);
    require_all_free(fixture.kv, 3);
}

TEST_CASE("putting a hash already cached leaves the line unmapped and stores nothing") {
    cache_fixture fixture(3);
    put_page(fixture.kv, hash_of(1), 1.0f);
    line_ids duplicate = fixture.kv.allocate(1).value();
    fill_line(fixture.kv, duplicate.front(), 1.0f);
    REQUIRE(run_to_completion(fixture.kv.put({hash_of(1)}, duplicate, compute_stream())).has_value());
    REQUIRE(fixture.store->puts == 1);
    fixture.kv.release(duplicate);

    line_ids reused = fixture.kv.allocate(1).value();
    REQUIRE(reused == duplicate);
    fixture.kv.release(reused);
    lines_result got = run_get(fixture.kv, {hash_of(1)});
    REQUIRE(got.has_value());
    REQUIRE(got->front() != duplicate.front());
    fixture.kv.release(*got);
}
