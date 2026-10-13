#pragma once

#include <pybind11/pybind11.h>

#include <optional>
#include <span>
#include <vector>

#include "pulsar/digest.hpp"
#include "pulsar/event_loop.hpp"
#include "pulsar/kv_cache.hpp"

namespace pulsar {

// A page_store backed by a Python object with two methods, called on the loop thread with the GIL held:
//   fetch(hashes: list[bytes]) -> concurrent.futures.Future[list[tuple[Tensor, Tensor] | None]]
//   put(hashes: list[bytes], pages: list[tuple[Tensor, Tensor]]) -> None
// A page is its (k, v) host tensors. Both return without waiting for the store. A fetch whose call raises, or whose
// future fails, is cancelled, is dropped unfinished or yields a malformed result, misses every page it asked for. An
// exception raised by put is reported through sys.unraisablehook.
class python_store {
    pybind11::object hooks;

    void
    start_fetch(std::span<const digest> hashes, external_task<std::vector<std::optional<page>>>::resolver&& resolver);

  public:
    explicit python_store(pybind11::object&& hooks) noexcept;
    python_store(python_store&&) noexcept = default;
    python_store& operator=(python_store&&) = delete;
    ~python_store();

    task<std::vector<std::optional<page>>> fetch(std::span<const digest> hashes);
    void put(std::vector<digest>&& hashes, std::vector<page>&& pages);
};

}  // namespace pulsar
