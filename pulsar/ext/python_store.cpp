#include "python_store.hpp"

#include <torch/csrc/utils/pybind.h>

#include <pybind11/stl.h>

#include <cstddef>
#include <exception>
#include <format>
#include <memory>
#include <stdexcept>
#include <utility>

#include "digest_bytes.hpp"

namespace pulsar {

namespace py = pybind11;

namespace {

using fetched_pages = std::vector<std::optional<page>>;

fetched_pages pages_from_python(const py::handle& result, std::size_t count) {
    auto tensors = result.cast<std::vector<std::optional<std::pair<at::Tensor, at::Tensor>>>>();
    if (tensors.size() != count) {
        throw std::length_error(std::format("the store returned {} pages for {} hashes", tensors.size(), count));
    }
    fetched_pages pages(count);
    for (std::size_t i = 0; i < count; i++) {
        if (tensors[i]) {
            pages[i] = page{std::move(tensors[i]->first), std::move(tensors[i]->second)};
        }
    }
    return pages;
}

py::list pages_to_python(std::vector<page>&& pages) {
    py::list list(pages.size());
    for (std::size_t i = 0; i < pages.size(); i++) {
        list[i] = py::make_tuple(std::move(pages[i].k), std::move(pages[i].v));
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

}  // namespace

python_store::python_store(py::object&& hooks) noexcept
: hooks(std::move(hooks)) {}

python_store::~python_store() {
    if (hooks) {
        py::gil_scoped_acquire gil;
        hooks = py::object();
    }
}

task<fetched_pages> python_store::fetch(std::span<const digest> hashes) {
    auto [fetched, resolver] = external_task<fetched_pages>::make();
    start_fetch(hashes, std::move(resolver));
    co_return co_await std::move(fetched);
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

void python_store::put(std::vector<digest>&& hashes, std::vector<page>&& pages) {
    py::gil_scoped_acquire gil;
    try {
        hooks.attr("put")(to_bytes_list(hashes), pages_to_python(std::move(pages)));
    } catch (py::error_already_set& e) {
        e.discard_as_unraisable("pulsar page store put");
    }
}

}  // namespace pulsar
