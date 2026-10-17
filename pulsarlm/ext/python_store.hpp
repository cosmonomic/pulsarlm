#pragma once

#include <pybind11/pybind11.h>

#include <optional>
#include <span>
#include <vector>

#include "pulsarlm/digest.hpp"
#include "pulsarlm/event_loop.hpp"
#include "pulsarlm/kv_cache.hpp"

namespace pulsarlm {

// A page_store backed by a Python object with two methods, called on the loop thread with the GIL held:
//   fetch(hashes: list[bytes]) -> concurrent.futures.Future[list[Tensor | None]]
//   put(hashes: list[bytes], pages: list[Tensor], owners: list[tuple[int, int]]) -> None
// A page is its host uint8 tensor; an owner is a (request id, index into pages) pair. Both return without waiting for
// the store. A fetch whose call raises, or whose future fails, is cancelled, is dropped unfinished or yields a malformed
// result, misses every page it asked for. An exception raised by put is reported through sys.unraisablehook.
class python_store final : public page_store {
    pybind11::object hooks;

    void
    start_fetch(std::span<const digest> hashes, external_task<std::vector<std::optional<page>>>::resolver&& resolver);

  public:
    explicit python_store(pybind11::object&& hooks) noexcept;
    python_store(const python_store&) = delete;
    python_store& operator=(const python_store&) = delete;
    ~python_store() override;

    task<std::vector<std::optional<page>>> fetch(std::span<const digest> hashes) override;
    void put(std::vector<digest>&& hashes, std::vector<page>&& pages, std::vector<page_owner>&& owners) override;
};

}  // namespace pulsarlm
