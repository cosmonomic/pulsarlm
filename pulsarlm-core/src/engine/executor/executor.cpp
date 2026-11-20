#include "batch.hpp"

#include "pulsarlm/engine/executor.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace pulsarlm {

namespace {

// A step's requests decode one token or verify state_slots tokens each, except one or more prefill chunks of at most
// prefill_tokens_per_step tokens together, each in place of a request's decode.
std::vector<std::int64_t>
make_token_buckets(std::size_t max_running_requests, std::size_t state_slots, std::size_t prefill_tokens_per_step) {
    const auto most = static_cast<std::int64_t>(
        (max_running_requests - 1) * state_slots + std::max(state_slots, prefill_tokens_per_step)
    );
    std::vector<std::int64_t> buckets;
    for (std::int64_t count : {1, 2, 3}) {
        if (count < most) {
            buckets.push_back(count);
        }
    }
    for (std::int64_t power = 1; 4 * power < most; power *= 2) {
        buckets.push_back(4 * power);
        if (6 * power < most) {
            buckets.push_back(6 * power);
        }
    }
    buckets.push_back(most);
    return buckets;
}

}  // namespace

std::size_t executor::count_slots(const kv_layer& layer) const noexcept {
    if (!layer.state) {
        return 0;
    }
    return layer.state->input_window ? 1 : draft_state_slots;
}

std::vector<cudaStream_t> executor::collect_streams(cudaStream_t shard::* stream) const {
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
    std::size_t max_running_requests,
    std::size_t prefill_tokens_per_step
)
: cache(std::move(cache)),
  store(store),
  layout{.layers = {}, .page_size = 0},
  root(root),
  stop_ids(std::move(stop_ids)),
  view_page_capacity(view_page_capacity),
  max_running_requests(max_running_requests),
  draft_state_slots(draft_tokens + 1),
  sampled_row_capacity(sampled_row_limit),
  token_buckets(make_token_buckets(max_running_requests, draft_tokens + 1, prefill_tokens_per_step)) {
    assert(prefill_tokens_per_step >= 1);
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
                        .state_slots = count_slots(layers[layer]),
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
                    layer_layout.state_slots == count_slots(layers[layer]),
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
        s.stop_tokens = at::tensor(this->stop_ids, at::TensorOptions().dtype(at::kLong)).to(s.lines.front().k.device());
        const at::TensorOptions options = s.lines.front().k.options();
        const auto slots = static_cast<std::int64_t>(max_running_requests);
        s.table.page_tables = at::zeros(
            {slots, static_cast<std::int64_t>(layout.layers.size()), static_cast<std::int64_t>(view_page_capacity)},
            options.dtype(at::kInt)
        );
        s.table.page_positions = at::zeros_like(s.table.page_tables);
        s.table.history_positions = at::zeros({slots}, options.dtype(at::kLong));
        s.table.rope_positions = at::zeros({slots}, options.dtype(at::kLong));
        s.table.key_counts = at::zeros(
            {slots, static_cast<std::int64_t>(layout.layers.size())},
            options.dtype(at::kInt)
        );
        s.table.rejected_tokens = at::zeros({slots}, options.dtype(at::kInt));
        s.table.tokens = at::zeros({slots, static_cast<std::int64_t>(draft_state_slots)}, options.dtype(at::kLong));
        s.table.cycle_end_positions = at::zeros({slots}, options.dtype(at::kLong));
        s.table.mass_start_positions = at::zeros({slots}, options.dtype(at::kLong));
        s.table.attention_mass_decays = at::zeros({slots}, options.dtype(at::kDouble));
        std::vector<std::uint8_t> recurrent_layers;
        for (const view_layout::layer& layer : layout.layers) {
            recurrent_layers.push_back(layer.is_recurrent ? 1 : 0);
        }
        s.table.recurrent_layers = at::tensor(recurrent_layers, at::TensorOptions().dtype(at::kByte))
                                       .to(options.device())
                                       .to(at::kBool);
        s.table.page_size = static_cast<std::int64_t>(layout.page_size);
        s.table.seeds = at::zeros({slots}, options.dtype(at::kLong));
        s.table.draw_counts = at::zeros({slots}, options.dtype(at::kLong));
        s.table.temperatures = at::zeros({slots}, options.dtype(at::kFloat));
        s.table.top_k = at::zeros({slots}, options.dtype(at::kLong));
        s.table.top_p = at::zeros({slots}, options.dtype(at::kFloat));
    }
    if (graph_pages) {
        for (shard& s : shards) {
            c10::cuda::CUDAGuard device_guard(s.lines.front().k.device());
            s.graphs = std::make_unique<step_graphs>(
                *s.model,
                s.lines,
                *graph_pages,
                layout.page_size,
                draft_state_slots,
                s.mass_heads,
                s.stop_tokens,
                s.table,
                token_buckets
            );
        }
        capture_step_graphs();
    }
}

executor::~executor() {
    end_requests(request_error::loop_exited);
    for (shard& s : shards) {
        s.graphs.reset();
    }
    // The memory pools of destroyed graphs stay reserved by the caching allocator until its cache is emptied.
    c10::cuda::CUDACachingAllocator::emptyCache();
}

request_state& executor::get_state(request_id_type id) {
    return requests[id].state;
}

const request_state& executor::get_state(request_id_type id) const {
    return requests[id].state;
}

std::size_t executor::get_slot(request_id_type id) noexcept {
    return static_cast<std::uint32_t>(id);
}

std::size_t executor::find_token_bucket(std::size_t token_count) const {
    auto bucket = std::ranges::lower_bound(token_buckets, static_cast<std::int64_t>(token_count));
    assert(bucket != token_buckets.end());
    return static_cast<std::size_t>(*bucket);
}

const request_table& executor::get_table(std::size_t shard) const {
    return shards[shard].table;
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
            layout.layers[layer].is_recurrent ? layout.layers[layer].state_slots + 2 : req.views()[layer].pages.size()
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
            const std::size_t row_start = (get_slot(id) * n_layers + layer) * view_page_capacity;
            for (std::size_t column = first_column; column < width; column++) {
                const bool used = column < view.lines.size();
                cells.push_back(static_cast<std::int64_t>(row_start + column));
                if (!used) {
                    lines.push_back(0);
                } else if (view.lines[column] == layer_view::no_line) {
                    lines.push_back(-1);
                } else {
                    lines.push_back(static_cast<std::int32_t>(view.lines[column]));
                }
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
void executor::move_stale_requests(batch& b) {
    const std::size_t n_layers = layout.layers.size();
    const std::size_t token_width = draft_state_slots;
    std::vector<std::int64_t> slots;
    std::vector<std::int64_t> history_positions;
    std::vector<std::int64_t> rope_positions;
    std::vector<std::int32_t> key_counts;
    std::vector<std::int32_t> rejected_tokens;
    std::vector<std::int64_t> tokens;
    std::vector<std::int64_t> cycle_end_positions;
    std::vector<std::int64_t> mass_start_positions;
    std::vector<double> attention_mass_decays;
    std::vector<std::int64_t> seeds;
    std::vector<std::int64_t> draw_counts;
    std::vector<float> temperatures;
    std::vector<std::int64_t> top_k;
    std::vector<float> top_p;
    for (const request_id_type id : stale_requests) {
        const running_request* running = requests.find(id);
        if (running == nullptr) {
            continue;
        }
        const request_state& req = running->state;
        assert(req.unfinished_step_count() == 0);
        slots.push_back(static_cast<std::int64_t>(get_slot(id)));
        history_positions.push_back(static_cast<std::int64_t>(req.history_position()));
        rope_positions.push_back(static_cast<std::int64_t>(req.next_position()));
        for (std::size_t layer = 0; layer < n_layers; layer++) {
            key_counts.push_back(static_cast<std::int32_t>(req.next_index(layer)));
        }
        rejected_tokens.push_back(static_cast<std::int32_t>(req.rejected_token_count()));
        const std::span<const std::int64_t> drafts = req.draft_tokens();
        assert(drafts.size() == req.prepared_draft_count() && drafts.size() < token_width);
        tokens.push_back(req.pending_token().value_or(0));
        tokens.insert(tokens.end(), drafts.begin(), drafts.end());
        tokens.resize(slots.size() * token_width, 0);
        cycle_end_positions.push_back(static_cast<std::int64_t>(req.cycle_end_position()));
        mass_start_positions.push_back(static_cast<std::int64_t>(req.mass_start_position()));
        attention_mass_decays.push_back(req.eviction_policy_in_pages().attention_mass_decay);
        const sampling_draw draw = req.next_draw();
        const sampling_params& params = req.sampling_parameters();
        seeds.push_back(static_cast<std::int64_t>(draw.seed));
        draw_counts.push_back(static_cast<std::int64_t>(draw.count));
        temperatures.push_back(input_temperature(params));
        top_k.push_back(input_top_k(params));
        top_p.push_back(input_top_p(params));
    }
    stale_requests.clear();
    if (slots.empty()) {
        return;
    }
    const auto count = static_cast<std::int64_t>(slots.size());
    b.stale_slots = pinned_values(slots);
    b.stale_entries = request_table{
        .history_positions = pinned_values(history_positions),
        .rope_positions = pinned_values(rope_positions),
        .key_counts = pinned_values(key_counts).view({count, static_cast<std::int64_t>(n_layers)}),
        .rejected_tokens = pinned_values(rejected_tokens),
        .tokens = pinned_values(tokens).view({count, static_cast<std::int64_t>(token_width)}),
        .cycle_end_positions = pinned_values(cycle_end_positions),
        .mass_start_positions = pinned_values(mass_start_positions),
        .attention_mass_decays = pinned_values(attention_mass_decays),
        .seeds = pinned_values(seeds),
        .draw_counts = pinned_values(draw_counts),
        .temperatures = pinned_values(temperatures),
        .top_k = pinned_values(top_k),
        .top_p = pinned_values(top_p),
    };
}

bool executor::is_steppable(request_id_type id) const {
    const request_state& req = get_state(id);
    return !req.has_prepared_last_step() && !req.has_exit_cause();
}

std::size_t executor::pending_token_count(request_id_type id) const {
    return get_state(id).unprepared_token_count();
}

std::size_t executor::cycle_tokens_left(request_id_type id) const {
    return get_state(id).cycle_tokens_left();
}

bool executor::has_exited(request_id_type id) const {
    return get_state(id).has_exited();
}

bool executor::is_in_launched_batch(request_id_type id) const {
    return get_state(id).unfinished_step_count() > 0;
}

bool executor::is_cancelled(request_id_type id) const {
    return requests[id].callbacks->cancelled();
}

void executor::cancel(request_id_type id) {
    request_state& req = get_state(id);
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
    std::size_t start = req.prepared_position_bound() % layout.page_size;
    return start + token_count >= layout.page_size;
}

// A drafting request's verify step ends neither its cycle nor max_tokens (executor::advance).
bool executor::entries_allow_overlap(std::span<const scheduled_request> scheduled) const {
    if (draft_layer) {
        return std::ranges::all_of(scheduled, [&](const scheduled_request& e) {
            return get_state(e.request_id).prepared_draft_count() > 0;
        });
    }
    const bool sliding = std::ranges::any_of(layout.layers, [](const view_layout::layer& layer) {
        return layer.sliding_window.has_value();
    });
    return std::ranges::none_of(scheduled, [&](const scheduled_request& e) {
        const auto& [id, token_count] = e;
        const request_state& req = get_state(id);
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
    const request_state& req = get_state(id);
    if (token_count != req.unprepared_token_count() || !req.may_sample()) {
        return 0;
    }
    return req.prepared_draft_count() == 0 ? 1 : token_count + req.prepared_draft_count();
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

// The tokens of an entry of the second run are accepted, and its samples taken, at its write-back. A prepared step's
// draft rows are relative to the request's history position at its start, which the GPU's request table holds.
void executor::advance(batch& b) {
    const std::size_t draft_count = draft_state_slots - 1;
    std::size_t prior_hidden_count = 0;
    if (draft_layer) {
        b.draft_count = draft_count;
        for (const batch_entry& e : b.entries) {
            const request_state& req = get_state(e.request_id);
            if (req.reads_last_hidden() && req.prepared_position_bound() > 0) {
                prior_hidden_count++;
            }
        }
    }
    std::size_t first_hidden_row = 1 + prior_hidden_count;
    for (std::size_t index = 0; index < b.entries.size(); index++) {
        batch_entry& e = b.entries[index];
        request_state& req = get_state(e.request_id);
        const bool verifies = b.verifies(index);
        const std::size_t token_count = verifies ? 1 : e.query_token_count;
        e.is_measured = batch::measures_mass(req, e.query_token_count);
        e.start_position = req.prepared_position_bound();
        bool drafts = false;
        bool next_row = false;
        if (draft_layer) {
            std::optional<std::size_t> prior_hidden_index;
            if (req.reads_last_hidden() && req.prepared_position_bound() > 0) {
                prior_hidden_index = b.prior_hidden_slots.size();
                b.prior_hidden_slots.push_back(static_cast<std::int64_t>(get_slot(e.request_id)));
            }
            std::tie(next_row, drafts) = plan_drafting(req, token_count);
            b.plan_draft_rows(e, req, verifies, first_hidden_row, prior_hidden_index, next_row, drafts);
            first_hidden_row += e.query_token_count;
            const std::optional<std::size_t>& token_limit = req.token_limit();
            if (verifies) {
                const std::size_t remaining = token_limit ? *token_limit - req.prepared_sample_count()
                                                          : e.query_token_count;
                e.max_accepted_token_count = std::clamp<std::size_t>(remaining, 1, e.query_token_count);
            }
        }
        if (verifies) {
            req.prepare_verify_step(drafts ? draft_count : 0);
        } else {
            req.prepare_step(token_count, e.first_sample_index.has_value(), drafts ? draft_count : 0, next_row);
        }
        e.ends_request = !verifies && req.has_prepared_last_step();
    }
}

// The step verifying drafts then ends neither the cycle nor max_tokens, whatever the step drafting them accepts.
std::pair<bool, bool> executor::plan_drafting(const request_state& req, std::size_t token_count) const {
    const std::size_t draft_count = draft_state_slots - 1;
    const bool verifies = req.unprepared_token_count() == 1 && req.prepared_draft_count() > 0;
    const std::size_t query_token_count = token_count + req.prepared_draft_count();
    const bool samples = token_count == req.unprepared_token_count() && req.may_sample();
    const std::optional<std::size_t>& token_limit = req.token_limit();
    const bool ends_request = samples ? token_limit && req.prepared_sample_count() + 1 >= *token_limit
                                      : req.unprepared_token_count() == token_count;
    const bool next_key = verifies || (!req.ends_cycle(query_token_count) && !ends_request);
    const bool drafts = next_key && (verifies || samples) &&
        req.cycle_tokens_left() >= query_token_count + draft_count + 2;
    return {next_key, drafts};
}

// The draft head's KV layer takes the key of the token after the step's tokens, and the keys of the drafts after it
// but the last.
bool executor::add_step_lines(const scheduled_request& scheduled) {
    request_state& req = get_state(scheduled.request_id);
    const std::size_t query_token_count = scheduled.token_count + req.prepared_draft_count();
    std::vector<std::size_t> key_counts(layout.layers.size(), query_token_count);
    if (draft_layer) {
        const auto [next_key, drafts] = plan_drafting(req, scheduled.token_count);
        key_counts[*draft_layer] += drafts ? draft_state_slots - 1 : next_key ? 1 : 0;
    }
    const std::vector<std::size_t> missing_counts = req.missing_lines(key_counts);
    if (std::ranges::all_of(missing_counts, [](std::size_t count) { return count == 0; })) {
        return true;
    }
    if (table_width_after(req, missing_counts) > view_page_capacity) {
        req.exceed_context();
        return false;
    }
    std::expected<std::vector<std::vector<std::uint32_t>>, page_error> allocation = cache.allocate(missing_counts);
    if (!allocation) {
        req.run_out_of_lines();
        return false;
    }
    std::optional<std::size_t> first_stale_column;
    for (std::size_t layer = 0; layer < layout.layers.size(); layer++) {
        const std::vector<std::uint32_t>& lines = req.views()[layer].lines;
        if (missing_counts[layer] > 0) {
            const auto first_added = static_cast<std::size_t>(
                std::ranges::find(lines, layer_view::no_line) - lines.begin()
            );
            first_stale_column = std::min(first_stale_column.value_or(first_added), first_added);
        }
    }
    req.add_step_lines(key_counts, *allocation);
    assert(std::ranges::all_of(req.missing_lines(key_counts), [](std::size_t count) { return count == 0; }));
    mark_stale_tables(scheduled.request_id, *first_stale_column);
    return true;
}

// Shard 0 queues on the executor's thread while the others queue on their own, so every shard's kernels launch at
// once. Each shard's completion is taken once every shard has queued, before any later step queues.
task<void> executor::launch(std::span<const scheduled_request> scheduled) {
    const std::chrono::steady_clock::time_point launch_time = std::chrono::steady_clock::now();
    assert(is_launchable(scheduled));
    std::vector<scheduled_request> lined;
    for (const scheduled_request& s : scheduled) {
        if (add_step_lines(s)) {
            lined.push_back(s);
        } else if (get_state(s.request_id).unfinished_step_count() == 0) {
            get_state(s.request_id).exit();
        }
    }
    if (lined.empty()) {
        co_return;
    }
    std::unique_ptr<batch> b(new batch(*this, lined));
    b->launch_time = launch_time;
    assert(static_cast<std::size_t>(b->sample_slots.size(0)) <= sampled_row_capacity);
    b->allows_overlap = entries_allow_overlap(lined);
    move_stale_tables(*b);
    move_stale_requests(*b);
    advance(*b);
    b->padded_token_count = b->is_uniform(draft_layer.has_value()) ? b->count_tokens()
                                                                   : find_token_bucket(b->count_tokens());
    b->mass_by_shard.assign(shards.size(), at::Tensor());
    b->hidden_by_shard.assign(shards.size(), at::Tensor());
    b->shard_prior_hidden.assign(shards.size(), at::Tensor());
    b->verify_tokens.assign(shards.size(), at::Tensor());
    b->accepted_token_counts_by_shard.assign(shards.size(), at::Tensor());
    b->shard_next_tokens.assign(shards.size(), at::Tensor());
    b->device_input_bytes.assign(shards.size(), at::Tensor());
    std::vector<external_task<void>> other_shards;
    for (std::size_t shard = 1; shard < shards.size(); shard++) {
        other_shards.push_back(run_on(*shards[shard].worker_loop, enqueue_on_shard(*b, shard)));
    }
    co_await enqueue_on_shard(*b, 0);
    for (external_task<void>& other : other_shards) {
        co_await std::move(other);
    }
    for (std::size_t shard = 0; shard < shards.size(); shard++) {
        at::Device device = shards[shard].lines.front().k.device();
        c10::cuda::CUDAGuard device_guard(device);
        b->ends.emplace_back().record(c10::cuda::getStreamFromExternal(shards[shard].compute_stream, device.index()));
        b->completions.push_back(await_stream(shards[shard].compute_stream));
    }
    launched_batches.push_back(std::move(b));
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
