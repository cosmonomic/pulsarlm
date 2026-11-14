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

// Resolves its read exactly once; destroyed unresolved, it resolves with every page missing.
class read_pages_resolver {
    external_task<fetched_pages>::resolver resolver;
    std::size_t count;
    bool resolved = false;

  public:
    read_pages_resolver(external_task<fetched_pages>::resolver&& resolver, std::size_t count) noexcept
    : resolver(std::move(resolver)),
      count(count) {}
    read_pages_resolver(const read_pages_resolver&) = delete;
    read_pages_resolver& operator=(const read_pages_resolver&) = delete;

    ~read_pages_resolver() {
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

using state_with_lease = std::pair<std::shared_ptr<const state>, state::lease_id_type>;

// The result of a read_state hook on its way to the loop: the state_and_lease state, or a miss. Destroyed with the lease untaken,
// as when the task awaiting it is gone, it releases the lease through release_hook.
class read_state_result {
    py::object release_hook;
    std::optional<state_with_lease> state_and_lease;

  public:
    read_state_result() noexcept = default;
    read_state_result(py::object&& release_hook, state_with_lease&& state_and_lease) noexcept
    : release_hook(std::move(release_hook)),
      state_and_lease(std::move(state_and_lease)) {}
    read_state_result(const read_state_result&) = delete;
    read_state_result& operator=(const read_state_result&) = delete;
    read_state_result(read_state_result&& other) noexcept
    : release_hook(std::move(other.release_hook)),
      state_and_lease(std::exchange(other.state_and_lease, std::nullopt)) {}
    read_state_result& operator=(read_state_result&&) = delete;

    ~read_state_result() {
        if (!release_hook) {
            return;
        }
        py::gil_scoped_acquire gil;
        if (state_and_lease) {
            try {
                release_hook(state_and_lease->second);
            } catch (py::error_already_set& e) {
                e.discard_as_unraisable("pulsarlm page store release");
            } catch (...) {
            }
        }
        release_hook = py::object();
    }

    std::expected<state_with_lease, page_error> take() {
        if (!state_and_lease) {
            return std::unexpected(page_error::miss);
        }
        return *std::exchange(state_and_lease, std::nullopt);
    }
};

// Resolves its read exactly once; destroyed unresolved, it resolves with a miss. Created and destroyed with the GIL held.
class read_state_resolver {
    external_task<read_state_result>::resolver resolver;
    py::object release_hook;
    bool resolved = false;

  public:
    read_state_resolver(external_task<read_state_result>::resolver&& resolver, py::object&& release_hook) noexcept
    : resolver(std::move(resolver)),
      release_hook(std::move(release_hook)) {}
    read_state_resolver(const read_state_resolver&) = delete;
    read_state_resolver& operator=(const read_state_resolver&) = delete;

    ~read_state_resolver() {
        if (!resolved) {
            resolver.resolve(read_state_result());
        }
    }

    void resolve(read_state_result&& result) {
        resolved = true;
        resolver.resolve(std::move(result));
    }

    // Called with the GIL held. A failed or malformed result is a miss.
    void resolve_from(const py::handle& future) {
        std::shared_ptr<state> stored_state;
        state::lease_id_type lease_id = 0;
        try {
            auto hook_result = future.attr("result")().cast<py::tuple>();
            if (hook_result.size() == 2) {
                lease_id = hook_result[1].cast<state::lease_id_type>();
                stored_state = hook_result[0].cast<std::shared_ptr<state>>();
            }
        } catch (const std::exception&) {
            stored_state = nullptr;
        }
        if (!stored_state) {
            resolve(read_state_result());
            return;
        }
        resolve(read_state_result(py::object(release_hook), state_with_lease(std::move(stored_state), lease_id)));
    }
};

py::list owners_to_python(std::span<const std::pair<state::lease_id_type, page_store::page_index_type>> owners) {
    py::list list(owners.size());
    for (std::size_t i = 0; i < owners.size(); i++) {
        list[i] = py::make_tuple(owners[i].first, owners[i].second);
    }
    return list;
}

std::vector<digest> page_keys(const state& s) {
    std::vector<digest> keys;
    for (std::size_t layer = 0; layer < s.layers.size(); layer++) {
        for (const active_page& page : s.layers[layer].pages) {
            keys.push_back(store_key(page.hash, layer));
        }
    }
    return keys;
}

}  // namespace

python_store::python_store(py::object&& hooks, py::object&& loop, std::size_t host_page_bytes)
: hooks(std::move(hooks)),
  loop(std::move(loop)),
  run_coroutine_threadsafe(py::module_::import("asyncio").attr("run_coroutine_threadsafe")),
  host_page_bytes(host_page_bytes) {}

void python_store::size_host_pages(std::size_t largest_page_bytes) {
    host_pages.emplace(host_page_bytes, largest_page_bytes);
}

python_store::~python_store() {
    if (hooks) {
        py::gil_scoped_acquire gil;
        hooks = py::object();
        loop = py::object();
        run_coroutine_threadsafe = py::object();
    }
}

// A callback added to a finished future runs inside add_done_callback.
task<std::vector<bool>> python_store::read_pages(
    std::span<const digest> hashes,
    std::vector<sharded_page>&& lines,
    std::span<const cudaStream_t> streams
) {
    auto [pages_task, resolver] = external_task<fetched_pages>::make();
    {
        py::gil_scoped_acquire gil;
        auto reader = std::make_shared<read_pages_resolver>(std::move(resolver), hashes.size());
        try {
            start_coroutine("read_pages", to_bytes_list(hashes))
                .attr("add_done_callback")(py::cpp_function([reader](const py::handle& done) {
                    reader->resolve_from(done);
                }));
        } catch (const std::exception&) {
        }
    }
    fetched_pages pages = co_await std::move(pages_task);
    std::vector<bool> page_found(pages.size());
    std::vector<sharded_page> targets;
    std::vector<page> arrived;
    for (std::size_t i = 0; i < pages.size(); i++) {
        if (pages[i]) {
            page_found[i] = true;
            targets.push_back(std::move(lines[i]));
            arrived.push_back(std::move(*pages[i]));
        }
    }
    co_await copy_from_host(*host_pages, targets, arrived, streams);
    co_return page_found;
}

task<std::expected<std::pair<std::shared_ptr<const state>, state::lease_id_type>, page_error>>
python_store::read_state(std::span<const std::int64_t> tokens, std::span<const digest> page_digests) {
    auto [state_task, resolver] = external_task<read_state_result>::make();
    {
        py::gil_scoped_acquire gil;
        auto reader = std::make_shared<read_state_resolver>(std::move(resolver), hooks.attr("release"));
        try {
            py::list token_list(tokens.size());
            for (std::size_t i = 0; i < tokens.size(); i++) {
                token_list[i] = py::int_(tokens[i]);
            }
            start_coroutine("read_state", std::move(token_list), to_bytes_list(page_digests))
                .attr("add_done_callback")(py::cpp_function([reader](const py::handle& done) {
                    reader->resolve_from(done);
                }));
        } catch (const std::exception&) {
            reader->resolve(read_state_result());
        }
    }
    read_state_result result = co_await std::move(state_task);
    co_return result.take();
}

std::vector<std::size_t> python_store::claim(
    std::span<const digest> hashes,
    std::span<const std::pair<state::lease_id_type, page_index_type>> owners
) {
    py::gil_scoped_acquire gil;
    try {
        return hooks.attr("claim")(to_bytes_list(hashes), owners_to_python(owners)).cast<std::vector<std::size_t>>();
    } catch (py::error_already_set& e) {
        e.discard_as_unraisable("pulsarlm page store claim");
    } catch (const py::cast_error&) {
    }
    return {};
}

task<std::expected<void, page_error>> python_store::write_pages(
    std::vector<digest>&& hashes,
    std::vector<sharded_page>&& pages,
    std::vector<std::pair<state::lease_id_type, page_index_type>>&& owners,
    std::span<const cudaStream_t> streams
) {
    std::vector<std::size_t> missing_indices = claim(hashes, owners);
    std::erase_if(missing_indices, [&](std::size_t i) { return i >= hashes.size(); });
    if (missing_indices.empty()) {
        co_return {};
    }
    std::vector<sharded_page> sources;
    std::vector<digest> copied_hashes;
    std::unordered_map<std::size_t, std::size_t> copied_index;
    for (std::size_t i : missing_indices) {
        if (copied_index.try_emplace(i, sources.size()).second) {
            sources.push_back(std::move(pages[i]));
            copied_hashes.push_back(hashes[i]);
        }
    }
    std::optional<std::vector<page>> copies = co_await copy_to_host(*host_pages, sources, streams);
    if (!copies) {
        co_return std::unexpected(page_error::oom);
    }
    std::vector<std::pair<state::lease_id_type, page_index_type>> copied_owners;
    for (const auto& [lease_id, index] : owners) {
        if (auto it = copied_index.find(index); it != copied_index.end()) {
            copied_owners.emplace_back(lease_id, it->second);
        }
    }
    py::gil_scoped_acquire gil;
    try {
        hooks.attr("write_pages")(
            to_bytes_list(copied_hashes),
            pages_to_python(std::move(*copies)),
            owners_to_python(copied_owners)
        );
    } catch (py::error_already_set& e) {
        e.discard_as_unraisable("pulsarlm page store write_pages");
    }
    co_return {};
}

task<void> python_store::write_state(state::lease_id_type lease_id, std::shared_ptr<const state>&& s) {
    std::vector<digest> keys = page_keys(*s);
    py::gil_scoped_acquire gil;
    try {
        hooks.attr("write_state")(lease_id, std::const_pointer_cast<state>(std::move(s)), to_bytes_list(keys));
    } catch (py::error_already_set& e) {
        e.discard_as_unraisable("pulsarlm page store write_state");
    }
    co_return;
}

void python_store::release(state::lease_id_type lease_id) noexcept {
    py::gil_scoped_acquire gil;
    try {
        hooks.attr("release")(lease_id);
    } catch (py::error_already_set& e) {
        e.discard_as_unraisable("pulsarlm page store release");
    } catch (...) {
    }
}

}  // namespace pulsarlm
