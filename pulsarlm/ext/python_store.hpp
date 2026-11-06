#pragma once

#include <pybind11/pybind11.h>

#include <cstddef>
#include <expected>
#include <optional>
#include <span>
#include <vector>

#include "pulsarlm/digest.hpp"
#include "pulsarlm/event_loop.hpp"
#include "pulsarlm/host_page_pool.hpp"
#include "pulsarlm/kv_cache.hpp"

namespace pulsarlm {

// A page_store backed by a Python object with three methods, called on the loop thread with the GIL held:
//   fetch(hashes: list[bytes]) -> concurrent.futures.Future[list[Tensor | None]]
//   keep(hashes: list[bytes], owners: list[tuple[int, int]]) -> list[int]
//   put(hashes: list[bytes], pages: list[Tensor], owners: list[tuple[int, int]]) -> None
// A page is its host uint8 tensor; an owner is a (request id, index into hashes) pair. keep gives the indices of the
// pages the object takes; put then hands it their host copies, with the owners of those pages. fetch and put return
// without waiting for the store. A fetch whose call raises, or whose future fails, is cancelled, is dropped unfinished
// or yields a malformed result, misses every page it asked for. An exception raised by keep, which then keeps nothing,
// or by put is reported through sys.unraisablehook. Host copies take blocks of a pinned host_page_pool, and pageable
// memory past its room.
class python_store final : public page_store {
    pybind11::object hooks;
    host_page_pool host_pages;

    void
    start_fetch(std::span<const digest> hashes, external_task<std::vector<std::optional<page>>>::resolver&& resolver);
    std::vector<std::size_t> keep(std::span<const digest> hashes, std::span<const page_owner> owners);

  public:
    // host_page_bytes sizes the pool; largest_page_bytes is the largest page it takes.
    python_store(pybind11::object&& hooks, std::size_t host_page_bytes, std::size_t largest_page_bytes);
    python_store(const python_store&) = delete;
    python_store& operator=(const python_store&) = delete;
    ~python_store() override;

    task<std::vector<bool>> fetch(
        std::span<const digest> hashes,
        std::vector<sharded_page>&& lines,
        std::span<const cudaStream_t> streams
    ) override;
    task<std::expected<void, page_error>>
    put(std::vector<digest>&& hashes,
        std::vector<sharded_page>&& pages,
        std::vector<page_owner>&& owners,
        std::span<const cudaStream_t> streams) override;
};

}  // namespace pulsarlm
