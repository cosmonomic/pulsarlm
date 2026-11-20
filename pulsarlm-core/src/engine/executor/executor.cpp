#include "batch.hpp"

#include "pulsarlm/engine/executor.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace pulsarlm {

std::size_t executor::slots_of(const kv_layer& layer) const noexcept {
    if (!layer.state) {
        return 0;
    }
    return layer.state->input_window ? 1 : draft_state_slots;
}

std::vector<cudaStream_t> executor::streams_of(cudaStream_t shard::* stream) const {
    std::vector<cudaStream_t> streams;
    for (const shard& s : shards) {
        streams.push_back(s.*stream);
    }
    return streams;
}

executor::executor(
    std::vector<model*>&& models,
    kv_cache cache,
    page_store& store,
    std::vector<cudaStream_t>&& streams,
    std::vector<cudaStream_t>&& copy_streams,
    std::vector<event_loop*>&& worker_loops,
    digest root,
    std::vector<std::int64_t>&& stop_ids,
    std::size_t view_page_capacity,
    std::optional<std::size_t> graph_pages,
    std::size_t draft_tokens,
    std::size_t sampled_row_limit,
    std::size_t max_running_requests
)
: cache(std::move(cache)),
  store(store),
  layout{.layers = {}, .page_size = 0},
  root(root),
  stop_ids(std::move(stop_ids)),
  view_page_capacity(view_page_capacity),
  graph_pages(graph_pages),
  max_running_requests(max_running_requests),
  draft_state_slots(draft_tokens + 1),
  sampled_row_capacity(sampled_row_limit) {
    assert(sampled_row_limit >= 1);
    assert(sampled_row_limit >= draft_tokens + 1);
    assert(max_running_requests >= 1 && view_page_capacity >= 1);
    assert(!graph_pages || *graph_pages == view_page_capacity);
    assert(
        !models.empty() && models.size() == this->cache.shard_count() && streams.size() == models.size() &&
        copy_streams.size() == models.size() && worker_loops.size() + 1 == models.size()
    );
    for (std::size_t s = 0; s < models.size(); s++) {
        shards.push_back(
            shard{
                .model = models[s],
                .lines = {},
                .mass_heads = {},
                .stop_tokens = {},
                .compute_stream = streams[s],
                .copy_stream = copy_streams[s],
                .worker_loop = s == 0 ? nullptr : worker_loops[s - 1],
                .graphs = nullptr,
                .samples = {},
                .arrival_counters = {},
                .table = {},
            }
        );
    }
    for (std::size_t shard = 0; shard < shards.size(); shard++) {
        std::vector<kv_layer> layers = shards[shard].model->kv_layers();
        assert(layers.size() == this->cache.layer_count());
        for (std::size_t layer = 0; layer < layers.size(); layer++) {
            shards[shard].lines.push_back(this->cache.tensors(layer, shard));
            bool layer_recurrent = layers[layer].state.has_value();
            bool evictable = !layers[layer].sliding_window && !layer_recurrent && !layers[layer].mirrors_held_pages;
            require(
                !evictable || layers[layer].n_q_heads > 0,
                "executor: a full-attention KV layer must state its query heads"
            );
            shards[shard].mass_heads.push_back(evictable ? layers[layer].n_readers * layers[layer].n_q_heads : 0);
            if (shard == 0) {
                layout.layers.push_back(
                    view_layout::layer{
                        .is_recurrent = layer_recurrent,
                        .mirrors_held_pages = layers[layer].mirrors_held_pages,
                        .sliding_window = layers[layer].sliding_window,
                        .state_slots = slots_of(layers[layer]),
                        .mass_heads = 0,
                    }
                );
                if (layout.page_size == 0 && !layer_recurrent) {
                    layout.page_size = static_cast<std::size_t>(shards[0].lines.back().k.size(1));
                }
            }
            view_layout::layer& layer_layout = layout.layers[layer];
            require(
                layer_layout.sliding_window == layers[layer].sliding_window &&
                    layer_layout.is_recurrent == layer_recurrent &&
                    layer_layout.mirrors_held_pages == layers[layer].mirrors_held_pages &&
                    layer_layout.state_slots == slots_of(layers[layer]),
                "executor: the model shards must agree on the kind of each KV layer"
            );
            layer_layout.mass_heads += shards[shard].mass_heads.back();
            if (shard == 0 && draft_tokens > 0 && layers[layer].last_hidden_dim > 0) {
                draft_layer = layer;
            }
        }
    }
    require(layout.page_size > 0, "executor: the model must have a KV layer with pages");
    for (shard& s : shards) {
        s.samples = at::zeros(
            {static_cast<std::int64_t>(sampled_row_capacity)},
            s.lines.front().k.options().dtype(at::kLong)
        );
        s.stop_tokens = at::tensor(this->stop_ids, at::TensorOptions().dtype(at::kLong)).to(s.lines.front().k.device());
        const at::TensorOptions options = s.lines.front().k.options();
        const auto slots = static_cast<std::int64_t>(max_running_requests);
        s.table.page_tables = at::zeros(
            {slots, static_cast<std::int64_t>(layout.layers.size()), static_cast<std::int64_t>(view_page_capacity)},
            options.dtype(at::kInt)
        );
        s.table.page_positions = at::zeros_like(s.table.page_tables);
        s.table.seeds = at::zeros({slots}, options.dtype(at::kLong));
        s.table.draw_counts = at::zeros({slots}, options.dtype(at::kLong));
        s.table.temperatures = at::zeros({slots}, options.dtype(at::kFloat));
        s.table.top_k = at::zeros({slots}, options.dtype(at::kLong));
        s.table.top_p = at::zeros({slots}, options.dtype(at::kFloat));
    }
    if (this->graph_pages) {
        for (shard& s : shards) {
            c10::cuda::CUDAGuard device_guard(s.lines.front().k.device());
            s.graphs = std::make_unique<step_graphs>(
                *s.model,
                s.lines,
                *this->graph_pages,
                layout.page_size,
                draft_state_slots,
                s.mass_heads,
                s.samples,
                s.stop_tokens
            );
        }
        capture_step_graphs();
    }
}

executor::~executor() {
    end_requests(request_error::loop_exited);
}

executor::request_ref::request_ref(executor& owner, request_id_type id) noexcept
: owner(&owner),
  id(id) {}

request_state& executor::request_ref::operator*() const {
    return owner->state_of(id);
}

request_state* executor::request_ref::operator->() const {
    return &**this;
}

executor::request_id_type executor::request_ref::key() const noexcept {
    return id;
}

request_state& executor::state_of(request_id_type id) {
    return requests[id].state;
}

const request_state& executor::state_of(request_id_type id) const {
    return requests[id].state;
}

std::size_t executor::slot_of(request_id_type id) noexcept {
    return static_cast<std::uint32_t>(id);
}

std::size_t executor::max_running_request_count() const noexcept {
    return max_running_requests;
}

std::size_t executor::request_count() const noexcept {
    return requests.size();
}

void executor::mark_stale_tables(request_id_type id, std::size_t first_column) {
    auto [stale, inserted] = stale_tables.try_emplace(id, first_column);
    if (!inserted) {
        stale->second = std::min(stale->second, first_column);
    }
}

std::size_t executor::table_width_after(const request_state& req, std::span<const std::size_t> line_counts) const {
    std::size_t width = 0;
    for (std::size_t layer = 0; layer < layout.layers.size(); layer++) {
        width = std::max(width, req.views()[layer].lines.size() + line_counts[layer]);
    }
    return width;
}

std::vector<std::size_t> executor::start_line_counts(const request_state& req) const {
    std::vector<std::size_t> counts;
    for (std::size_t layer = 0; layer < layout.layers.size(); layer++) {
        counts.push_back(
            layout.layers[layer].is_recurrent ? layout.layers[layer].state_slots + 2
                                              : req.views()[layer].pages.size() + req.cycle_lines()
        );
    }
    return counts;
}

// The requests erased since they were marked are passed over; a request in a reused slot was marked from column 0 at
// its start.
void executor::move_stale_tables(batch& b) {
    const std::size_t n_layers = layout.layers.size();
    std::vector<std::int64_t> cells;
    std::vector<std::int32_t> lines;
    std::vector<std::int32_t> positions;
    for (const auto& [id, first_column] : stale_tables) {
        const running_request* running = requests.find(id);
        if (running == nullptr) {
            continue;
        }
        const request_state& req = running->state;
        const std::size_t width = req.table_width();
        for (std::size_t layer = 0; layer < n_layers; layer++) {
            const layer_view& view = req.views()[layer];
            const std::size_t row_start = (slot_of(id) * n_layers + layer) * view_page_capacity;
            for (std::size_t column = first_column; column < width; column++) {
                const bool used = column < view.lines.size();
                cells.push_back(static_cast<std::int64_t>(row_start + column));
                lines.push_back(used ? static_cast<std::int32_t>(view.lines[column]) : 0);
                positions.push_back(
                    used && !layout.layers[layer].is_recurrent
                        ? static_cast<std::int32_t>(req.page_position(layer, column))
                        : 0
                );
            }
        }
    }
    stale_tables.clear();
    if (cells.empty()) {
        return;
    }
    const auto count = static_cast<std::int64_t>(cells.size());
    b.table_cells = pinned({count}, at::kLong);
    b.table_lines = pinned({count}, at::kInt);
    b.table_positions = pinned({count}, at::kInt);
    std::ranges::copy(cells, b.table_cells.mutable_data_ptr<std::int64_t>());
    std::ranges::copy(lines, b.table_lines.mutable_data_ptr<std::int32_t>());
    std::ranges::copy(positions, b.table_positions.mutable_data_ptr<std::int32_t>());
}

// The requests erased since they were marked are passed over.
void executor::move_stale_sampling(batch& b) {
    std::vector<std::int64_t> slots;
    std::vector<std::int64_t> seeds;
    std::vector<std::int64_t> draw_counts;
    std::vector<float> temperatures;
    std::vector<std::int64_t> top_k;
    std::vector<float> top_p;
    for (const request_id_type id : stale_sampling) {
        const running_request* running = requests.find(id);
        if (running == nullptr) {
            continue;
        }
        const sampling_draw draw = running->state.next_draw();
        const sampling_params& params = running->state.sampling_parameters();
        slots.push_back(static_cast<std::int64_t>(slot_of(id)));
        seeds.push_back(static_cast<std::int64_t>(draw.seed));
        draw_counts.push_back(static_cast<std::int64_t>(draw.count));
        temperatures.push_back(input_temperature(params));
        top_k.push_back(input_top_k(params));
        top_p.push_back(input_top_p(params));
    }
    stale_sampling.clear();
    if (slots.empty()) {
        return;
    }
    const auto count = static_cast<std::int64_t>(slots.size());
    b.sampling_slots = pinned({count}, at::kLong);
    b.seeds = pinned({count}, at::kLong);
    b.draw_counts = pinned({count}, at::kLong);
    b.temperatures = pinned({count}, at::kFloat);
    b.top_k = pinned({count}, at::kLong);
    b.top_p = pinned({count}, at::kFloat);
    std::ranges::copy(slots, b.sampling_slots.mutable_data_ptr<std::int64_t>());
    std::ranges::copy(seeds, b.seeds.mutable_data_ptr<std::int64_t>());
    std::ranges::copy(draw_counts, b.draw_counts.mutable_data_ptr<std::int64_t>());
    std::ranges::copy(temperatures, b.temperatures.mutable_data_ptr<float>());
    std::ranges::copy(top_k, b.top_k.mutable_data_ptr<std::int64_t>());
    std::ranges::copy(top_p, b.top_p.mutable_data_ptr<float>());
}

bool executor::is_steppable(request_id_type id) const {
    const request_state& req = state_of(id);
    return !req.has_launched_last_step() && !req.has_exit_cause();
}

std::size_t executor::pending_token_count(request_id_type id) const {
    return state_of(id).pending_token_list().size();
}

std::size_t executor::cycle_tokens_left(request_id_type id) const {
    return state_of(id).cycle_tokens_left();
}

bool executor::has_exited(request_id_type id) const {
    return state_of(id).has_exited();
}

bool executor::is_in_launched_batch(request_id_type id) const {
    return state_of(id).unfinished_step_count() > 0;
}

bool executor::is_cancelled(request_id_type id) const {
    return requests[id].callbacks->cancelled();
}

void executor::cancel(request_id_type id) {
    request_state& req = state_of(id);
    if (req.has_exited()) {
        return;
    }
    req.cancel();
    if (req.unfinished_step_count() == 0) {
        req.exit();
    }
}

std::size_t executor::kv_cache_bytes() const noexcept {
    return cache.used_bytes();
}

std::uint64_t executor::drafted_token_count() const noexcept {
    return drafted_tokens.load(std::memory_order_relaxed);
}

std::uint64_t executor::accepted_draft_count() const noexcept {
    return accepted_drafts.load(std::memory_order_relaxed);
}

bool executor::completes_page(const request_state& req, std::size_t token_count) const noexcept {
    std::size_t start = req.tail_tokens().size();
    return (start + token_count) / layout.page_size * layout.page_size > start;
}

// A drafting request's verify step ends neither its cycle nor max_tokens (executor::advance).
bool executor::entries_allow_overlap(std::span<const scheduled_request> scheduled) const {
    if (draft_layer) {
        return std::ranges::all_of(scheduled, [&](const scheduled_request& e) {
            return !state_of(e.request_id).draft_tokens().empty();
        });
    }
    const bool sliding = std::ranges::any_of(layout.layers, [](const view_layout::layer& layer) {
        return layer.sliding_window.has_value();
    });
    return std::ranges::none_of(scheduled, [&](const scheduled_request& e) {
        const auto& [id, token_count] = e;
        const request_state& req = state_of(id);
        return req.ends_cycle(token_count) || (sliding && completes_page(req, token_count));
    });
}

bool executor::allows_overlap(std::span<const scheduled_request> scheduled) const {
    return (launched_batches.empty() || launched_batches.back()->allows_overlap) && entries_allow_overlap(scheduled);
}

std::size_t executor::launched_batch_count() const noexcept {
    return launched_batches.size();
}

std::size_t executor::sampled_row_limit() const noexcept {
    return sampled_row_capacity;
}

std::size_t executor::sampled_row_count(request_id_type id, std::size_t token_count) const {
    const request_state& req = state_of(id);
    if (token_count != req.pending_token_list().size() || !req.may_sample()) {
        return 0;
    }
    return req.draft_tokens().empty() ? 1 : token_count + req.draft_tokens().size();
}

bool executor::is_launchable(std::span<const scheduled_request> scheduled) const {
    if (scheduled.empty() || launched_batches.size() >= 2) {
        return false;
    }
    std::unordered_set<request_id_type> entered;
    for (const auto& [id, token_count] : scheduled) {
        if (!entered.insert(id).second || token_count == 0 || token_count > pending_token_count(id) ||
            !is_steppable(id)) {
            return false;
        }
    }
    return launched_batches.empty() || allows_overlap(scheduled);
}

// A verify entry's tokens are accepted, and its samples taken, at its write-back. Its pending token and drafts are
// awaited_token until then, and a request whose previous step is launched lags it by up to that step's drafts.
void executor::advance(batch& b) {
    const std::size_t draft_count = draft_state_slots - 1;
    for (batch::group* g : b.groups()) {
        for (batch::entry& e : g->entries) {
            request_state& req = *e.req;
            if (draft_layer && req.end_of_draft_rows() == req.history_position() && req.history_position() > 0) {
                const layer_view& view = req.views()[*draft_layer];
                e.prior_row = b.prior_lines.size();
                b.prior_lines.push_back(view.lines[(req.next_index(*draft_layer) - 1) / layout.page_size]);
            }
        }
    }
    std::size_t first_hidden_row = 1 + b.prior_lines.size();
    for (batch::group* g : b.groups()) {
        const bool verifies = g == &b.verify_group;
        for (batch::entry& e : g->entries) {
            request_state& req = *e.req;
            const std::size_t lag = req.unfinished_step_count() > 0 ? draft_count : 0;
            e.is_measured = batch::measures_mass(e);
            e.start_position = req.history_position();
            bool drafts = false;
            if (draft_layer) {
                const std::optional<std::size_t>& token_limit = req.token_limit();
                const bool ends_request = e.is_sampling ? token_limit && req.prepared_sample_count() + 1 >= *token_limit
                                                        : req.pending_token_list().size() == e.token_count;
                const bool next_row = verifies || (!e.ends_cycle && !ends_request);
                // The next step's verify group then ends neither the cycle nor max_tokens, whatever this one accepts.
                drafts = next_row && (verifies || e.is_sampling) &&
                    req.cycle_tokens_left() >= batch::forwarded(e) + lag + draft_count + 2;
                b.plan_draft_rows(e, verifies, first_hidden_row, next_row, drafts, *draft_layer, draft_count);
                first_hidden_row += batch::forwarded(e);
                if (verifies) {
                    const std::size_t remaining = token_limit ? *token_limit - req.prepared_sample_count()
                                                              : b.verify_rows;
                    e.max_accepted_token_count = std::clamp<std::size_t>(remaining, 1, b.verify_rows);
                    e.mass_start_position = req.mass_start_position();
                }
            }
            if (verifies) {
                e.end_position = req.history_position();
                e.sampled_position = req.history_position() + e.token_count;
                req.launch_verify_step(e.token_count, drafts ? draft_count : 0);
            } else {
                e.end_position = req.history_position() + e.token_count;
                e.sampled_position = req.history_position() + req.pending_token_list().size();
                req.launch_step(
                    e.token_count,
                    e.is_sampling ? std::optional(e.first_sample_row) : std::nullopt,
                    e.draft ? e.draft->row_count : 0,
                    drafts ? draft_count : 0
                );
            }
            e.ends_request = !verifies && req.has_launched_last_step();
        }
    }
}

// Shard 0 queues on the executor's thread while the others queue on their own, so every shard's kernels launch at
// once. Each shard's completion is taken once every shard has queued, before any later step queues.
task<void> executor::launch(std::span<const scheduled_request> scheduled) {
    const std::chrono::steady_clock::time_point launch_time = std::chrono::steady_clock::now();
    assert(is_launchable(scheduled));
    std::unique_ptr<batch> b(new batch(*this, scheduled));
    b->launch_time = launch_time;
    assert(static_cast<std::size_t>(b->sample_slots.size(0)) <= sampled_row_capacity);
    b->allows_overlap = entries_allow_overlap(scheduled);
    move_stale_tables(*b);
    move_stale_sampling(*b);
    link_previous(*b);
    advance(*b);
    for (batch::group* g : b->groups()) {
        g->mass_by_shard.assign(shards.size(), at::Tensor());
        g->hidden_by_shard.assign(shards.size(), at::Tensor());
    }
    b->shard_prior_hidden.assign(shards.size(), at::Tensor());
    b->verify_tokens.assign(shards.size(), at::Tensor());
    b->accepted_token_counts_by_shard.assign(shards.size(), at::Tensor());
    b->shard_next_tokens.assign(shards.size(), at::Tensor());
    std::vector<external_task<>> other_shards;
    for (std::size_t shard = 1; shard < shards.size(); shard++) {
        other_shards.push_back(run_on(*shards[shard].worker_loop, enqueue_on_shard(*b, shard)));
    }
    co_await enqueue_on_shard(*b, 0);
    for (external_task<>& other : other_shards) {
        co_await std::move(other);
    }
    for (std::size_t shard = 0; shard < shards.size(); shard++) {
        at::Device device = shards[shard].lines.front().k.device();
        c10::cuda::CUDAGuard device_guard(device);
        b->ends.emplace_back().record(c10::cuda::getStreamFromExternal(shards[shard].compute_stream, device.index()));
        b->completions.push_back(completion_of(shards[shard].compute_stream));
    }
    launched_batches.push_back(std::move(b));
}

void executor::link_previous(batch& b) const {
    if (launched_batches.empty() || b.verify_group.entries.empty()) {
        return;
    }
    const batch& previous = *launched_batches.back();
    b.previous_batch = &previous;
    const std::vector<batch::entry> previous_drafting_entries = previous.drafting_entries();
    for (batch::entry& e : b.verify_group.entries) {
        if (e.req->unfinished_step_count() == 0) {
            continue;
        }
        for (std::size_t pv = 0; pv < previous.verify_group.entries.size(); pv++) {
            if (previous.verify_group.entries[pv].req == e.req) {
                e.previous_verify_row = pv;
            }
        }
        for (std::size_t d = 0; d < previous_drafting_entries.size(); d++) {
            if (previous_drafting_entries[d].req == e.req) {
                e.previous_drafting_row = d;
            }
        }
        assert(e.previous_verify_row && e.previous_drafting_row);
    }
}

task<void> executor::enqueue_on_shard(batch& b, std::size_t shard) {
    c10::cuda::CUDAGuard device_guard(shards[shard].lines.front().k.device());
    b.enqueue(
        *shards[shard].model,
        shards[shard].lines,
        shards[shard].mass_heads,
        draft_layer,
        shards[shard].stop_tokens,
        shards[shard].compute_stream,
        shard,
        shards[shard].graphs.get(),
        shards[shard].samples,
        shards[shard].arrival_counters,
        shards[shard].table
    );
    co_return;
}

// Each shard on its own thread, as the eager forwards' collectives need every shard at once.
void executor::capture_step_graphs() {
    std::vector<std::exception_ptr> failures(shards.size());
    std::vector<std::thread> threads;
    for (std::size_t shard = 0; shard < shards.size(); shard++) {
        threads.emplace_back([this, shard, &failures] {
            try {
                at::Device device = shards[shard].lines.front().k.device();
                c10::cuda::CUDAGuard device_guard(device);
                c10::cuda::CUDAStreamGuard stream_guard(
                    c10::cuda::getStreamFromExternal(shards[shard].compute_stream, device.index())
                );
                shards[shard].graphs->capture_startup();
                C10_CUDA_CHECK(cudaStreamSynchronize(shards[shard].compute_stream));
            } catch (...) {
                failures[shard] = std::current_exception();
            }
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
    for (const std::exception_ptr& failure : failures) {
        if (failure) {
            std::rethrow_exception(failure);
        }
    }
}

}  // namespace pulsarlm
