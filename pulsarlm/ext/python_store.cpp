#include "python_store.hpp"

#include <torch/csrc/utils/pybind.h>

#include <pybind11/stl.h>

#include <cstddef>
#include <exception>
#include <format>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "digest_bytes.hpp"
#include "host_copy.hpp"
#include "pulsarlm/engine.hpp"

namespace pulsarlm {

namespace py = pybind11;

namespace {

using fetched_pages = std::vector<std::optional<page>>;

fetched_pages pages_from_python(const py::handle& result, std::size_t count) {
    auto tensors = result.cast<std::vector<std::optional<at::Tensor>>>();
    if (tensors.size() != count) {
        throw std::length_error(std::format("the store returned {} pages for {} hashes", tensors.size(), count));
    }
    fetched_pages pages(count);
    for (std::size_t i = 0; i < count; i++) {
        if (tensors[i]) {
            pages[i] = page{.bytes = std::move(*tensors[i])};
        }
    }
    return pages;
}

py::list pages_to_python(std::vector<page>&& pages) {
    py::list list(pages.size());
    for (std::size_t i = 0; i < pages.size(); i++) {
        list[i] = py::cast(std::move(pages[i].bytes));
    }
    return list;
}

// Resolves its fetch exactly once; destroyed unresolved, it resolves with every page missing.
class fetch_resolver {
    external_task<fetched_pages>::resolver resolver;
    std::size_t count;
    bool resolved = false;

  public:
    fetch_resolver(external_task<fetched_pages>::resolver&& resolver, std::size_t count) noexcept
    : resolver(std::move(resolver)),
      count(count) {}
    fetch_resolver(const fetch_resolver&) = delete;
    fetch_resolver& operator=(const fetch_resolver&) = delete;

    ~fetch_resolver() {
        if (!resolved) {
            resolver.resolve(fetched_pages(count));
        }
    }

    // Called with the GIL held.
    void resolve_from(const py::handle& future) {
        fetched_pages pages(count);
        try {
            pages = pages_from_python(future.attr("result")(), count);
        } catch (const std::exception&) {
        }
        resolved = true;
        resolver.resolve(std::move(pages));
    }
};

py::list owners_to_python(std::span<const page_owner> owners) {
    py::list list(owners.size());
    for (std::size_t i = 0; i < owners.size(); i++) {
        list[i] = py::make_tuple(owners[i].request, owners[i].page);
    }
    return list;
}

}  // namespace

python_store::python_store(py::object&& hooks, std::size_t host_page_bytes, std::size_t largest_page_bytes)
: hooks(std::move(hooks)),
  host_pages(host_page_bytes, largest_page_bytes) {}

python_store::~python_store() {
    if (hooks) {
        py::gil_scoped_acquire gil;
        hooks = py::object();
    }
}

task<std::vector<bool>> python_store::fetch(
    std::span<const digest> hashes,
    std::vector<sharded_page>&& lines,
    std::span<const cudaStream_t> streams
) {
    auto [fetched, resolver] = external_task<fetched_pages>::make();
    start_fetch(hashes, std::move(resolver));
    fetched_pages pages = co_await std::move(fetched);
    std::vector<bool> found(pages.size());
    std::vector<sharded_page> targets;
    std::vector<page> arrived;
    for (std::size_t i = 0; i < pages.size(); i++) {
        if (pages[i]) {
            found[i] = true;
            targets.push_back(std::move(lines[i]));
            arrived.push_back(std::move(*pages[i]));
        }
    }
    co_await copy_from_host(host_pages, targets, arrived, streams);
    co_return found;
}

// A callback added to a finished future runs inside add_done_callback.
void python_store::start_fetch(std::span<const digest> hashes, external_task<fetched_pages>::resolver&& resolver) {
    auto fetch = std::make_shared<fetch_resolver>(std::move(resolver), hashes.size());
    py::gil_scoped_acquire gil;
    try {
        py::object future = hooks.attr("fetch")(to_bytes_list(hashes));
        future.attr("add_done_callback")(py::cpp_function([fetch](const py::handle& done) {
            fetch->resolve_from(done);
        }));
    } catch (const std::exception&) {
    }
}

std::vector<std::size_t> python_store::keep(std::span<const digest> hashes, std::span<const page_owner> owners) {
    py::gil_scoped_acquire gil;
    try {
        return hooks.attr("keep")(to_bytes_list(hashes), owners_to_python(owners)).cast<std::vector<std::size_t>>();
    } catch (py::error_already_set& e) {
        e.discard_as_unraisable("pulsarlm page store keep");
    } catch (const py::cast_error&) {
    }
    return {};
}

task<std::expected<void, page_error>> python_store::put(
    std::vector<digest>&& hashes,
    std::vector<sharded_page>&& pages,
    std::vector<page_owner>&& owners,
    std::span<const cudaStream_t> streams
) {
    std::vector<std::size_t> kept = keep(hashes, owners);
    std::erase_if(kept, [&](std::size_t i) { return i >= hashes.size(); });
    if (kept.empty()) {
        co_return {};
    }
    std::vector<sharded_page> sources;
    std::vector<digest> kept_hashes;
    std::unordered_map<std::size_t, std::size_t> kept_index;
    for (std::size_t i : kept) {
        if (kept_index.try_emplace(i, sources.size()).second) {
            sources.push_back(std::move(pages[i]));
            kept_hashes.push_back(hashes[i]);
        }
    }
    std::optional<std::vector<page>> copies = co_await copy_to_host(host_pages, sources, streams);
    if (!copies) {
        co_return std::unexpected(page_error::oom);
    }
    std::vector<page_owner> kept_owners;
    for (const page_owner& owner : owners) {
        if (auto it = kept_index.find(owner.page); it != kept_index.end()) {
            kept_owners.push_back(page_owner{.request = owner.request, .page = it->second});
        }
    }
    py::gil_scoped_acquire gil;
    try {
        hooks.attr("put")(
            to_bytes_list(kept_hashes),
            pages_to_python(std::move(*copies)),
            owners_to_python(kept_owners)
        );
    } catch (py::error_already_set& e) {
        e.discard_as_unraisable("pulsarlm page store put");
    }
    co_return {};
}

}  // namespace pulsarlm
