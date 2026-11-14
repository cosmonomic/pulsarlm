#pragma once

#include <pybind11/pybind11.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "host_page_pool.hpp"
#include "pulsarlm/digest.hpp"
#include "pulsarlm/engine.hpp"
#include "pulsarlm/task.hpp"

namespace pulsarlm {

// A page_store backed by a Python object with these methods, called on the loop thread with the GIL held:
//   async read_pages(hashes: list[bytes]) -> list[Tensor | None]
//   async read_state(tokens: list[int], page_digests: list[bytes]) -> tuple[State, int]
//   claim(hashes: list[bytes], owners: list[tuple[int, int]]) -> list[int]
//   write_pages(hashes: list[bytes], pages: list[Tensor], owners: list[tuple[int, int]]) -> None
//   write_state(lease_id: int, state: State, page_keys: list[bytes]) -> None
//   release(lease_id: int) -> None
// The coroutines run on loop, an asyncio event loop. A page is its host uint8 tensor; an owner is a (lease id, index
// into hashes) pair. claim gives the indices of the pages the object takes; write_pages then hands it their host copies,
// with the owners of those pages. A read_pages that raises, is cancelled or yields a malformed result misses every page
// it asked for. A read_state that raises, is cancelled or yields a malformed result panics the loop. An exception raised
// by claim, which then claims nothing, or by write_pages, write_state or release is reported through
// sys.unraisablehook. Host copies take blocks of a pinned host_page_pool, and pageable memory past its room.
class python_store final : public page_store {
    pybind11::object hooks;
    pybind11::object loop;
    pybind11::object run_coroutine_threadsafe;
    std::size_t host_page_bytes;
    std::optional<host_page_pool> host_pages;

    // Calls the hook coroutine_function and runs its coroutine on the loop; returns its concurrent.futures.Future.
    template <typename... argument_types>
    pybind11::object start_coroutine(const char* coroutine_function, argument_types&&... arguments);
    std::vector<std::size_t>
    claim(std::span<const digest> hashes, std::span<const std::pair<state::lease_id_type, page_index_type>> owners);

  public:
    // host_page_bytes sizes the pool, which size_host_pages makes before the engine reads or writes a page.
    python_store(pybind11::object&& hooks, pybind11::object&& loop, std::size_t host_page_bytes);
    python_store(const python_store&) = delete;
    python_store& operator=(const python_store&) = delete;
    ~python_store() override;

    // Makes the pool, whose largest block holds largest_page_bytes.
    void size_host_pages(std::size_t largest_page_bytes);

    task<std::vector<bool>> read_pages(
        std::span<const digest> hashes,
        std::vector<sharded_page>&& lines,
        std::span<const cudaStream_t> streams
    ) override;
    task<std::expected<void, page_error>> write_pages(
        std::vector<digest>&& hashes,
        std::vector<sharded_page>&& pages,
        std::vector<std::pair<state::lease_id_type, page_index_type>>&& owners,
        std::span<const cudaStream_t> streams
    ) override;
    task<std::pair<std::shared_ptr<const state>, state::lease_id_type>>
    read_state(std::span<const std::int64_t> tokens, std::span<const digest> page_digests) override;
    task<void> write_state(state::lease_id_type lease_id, std::shared_ptr<const state>&& s) override;
    void release(state::lease_id_type lease_id) noexcept override;
};

template <typename... argument_types>
pybind11::object python_store::start_coroutine(const char* coroutine_function, argument_types&&... arguments) {
    return run_coroutine_threadsafe(hooks.attr(coroutine_function)(std::forward<argument_types>(arguments)...), loop);
}

}  // namespace pulsarlm
