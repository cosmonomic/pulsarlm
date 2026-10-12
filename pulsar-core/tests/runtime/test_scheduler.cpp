#include "pulsar/digest.hpp"
#include "pulsar/event_loop.hpp"
#include "pulsar/kv_cache.hpp"
#include "pulsar/model/interface.hpp"
#include "pulsar/result_stream.hpp"
#include "pulsar/scheduler.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>

#undef CHECK
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using pulsar::digest;
using pulsar::digest_hash;
using pulsar::event_loop;
using pulsar::page;
using pulsar::request;
using pulsar::request_error;
using pulsar::result_stream;
using pulsar::result_writer;
using pulsar::task;

using token_list = std::vector<std::int64_t>;

constexpr pulsar::page_shape shape{.n_layers = 2, .page_size = 4, .n_kv_heads = 1, .head_dim = 2, .dtype = at::kFloat};
constexpr std::int64_t vocab = 1000;
// sum_model predicts tokens below vocab only.
constexpr std::int64_t unreachable_eos = vocab;

struct store_contents {
    std::unordered_map<digest, page, digest_hash> pages;
    std::vector<std::vector<digest>> fetches;
};

struct memory_store {
    std::shared_ptr<store_contents> contents = std::make_shared<store_contents>();

    task<std::vector<std::optional<page>>> fetch(std::span<const digest> hashes) {
        std::shared_ptr<store_contents> shared = contents;
        std::vector<digest> requested(hashes.begin(), hashes.end());
        shared->fetches.push_back(requested);
        std::vector<std::optional<page>> found;
        for (const digest& hash : requested) {
            auto it = shared->pages.find(hash);
            found.push_back(it == shared->pages.end() ? std::nullopt : std::optional<page>(it->second));
        }
        co_return found;
    }

    void put(std::vector<digest> keys, std::vector<page> pages) {
        for (std::size_t i = 0; i < keys.size(); i++) {
            contents->pages.insert_or_assign(keys[i], std::move(pages[i]));
        }
    }
};

static_assert(pulsar::page_store<memory_store>);

using test_scheduler = pulsar::scheduler<memory_store>;

// Writes each token's id as its K and its view index as its V. The token after a query is the sum of the K of every
// key the query attends, mod vocab.
class sum_model final : public pulsar::Model {
  public:
    std::size_t forward_count = 0;
    std::vector<token_list> prefill_chunks;
    std::vector<std::int64_t> decode_sizes;
    std::function<void()> on_forward;

    pulsar::StepLogits forward(const pulsar::StepBatch& batch, const pulsar::kv_lines& kv) override {
        forward_count++;
        if (on_forward) {
            on_forward();
        }
        pulsar::StepLogits logits;
        if (batch.prefill) {
            at::Tensor starts = batch.prefill->cu_seqlens_q.cpu().to(at::kLong);
            prefill_chunks.push_back(token_list(starts.size(0) - 1));
            for (std::size_t s = 0; s < prefill_chunks.back().size(); s++) {
                prefill_chunks.back()[s] = starts[s + 1].item<std::int64_t>() - starts[s].item<std::int64_t>();
            }
            logits.prefill = predict(*batch.prefill, kv);
        }
        if (batch.decode) {
            decode_sizes.push_back(batch.decode->tokens.size(0));
            logits.decode = predict(*batch.decode, kv);
        }
        return logits;
    }

  private:
    static at::Tensor predict(const pulsar::GroupBatch& g, const pulsar::kv_lines& kv) {
        std::int64_t n_tokens = g.tokens.size(0);
        std::vector<std::int64_t> sizes{kv.k.size(0), n_tokens, kv.k.size(3), kv.k.size(4)};
        at::Tensor slots = g.slot_mapping.to(at::kLong);
        kv.k.flatten(1, 2).index_copy_(1, slots, g.tokens.to(at::kFloat).view({1, n_tokens, 1, 1}).expand(sizes));
        kv.v.flatten(1, 2).index_copy_(1, slots, g.pos.to(at::kFloat).view({1, n_tokens, 1, 1}).expand(sizes));

        at::Tensor keys = kv.k.select(0, 0).select(2, 0).select(2, 0).cpu();
        at::Tensor positions = g.pos.cpu();
        at::Tensor page_tables = g.page_tables.cpu();
        at::Tensor rope_layouts = g.rope_layout.cpu();
        at::Tensor key_counts = (g.is_prefill() ? g.seqlens_k : g.context_lens).cpu();
        at::Tensor query_starts = g.is_prefill() ? g.cu_seqlens_q.cpu() : at::arange(page_tables.size(0) + 1, at::kInt);
        at::Tensor logits = at::zeros({n_tokens, vocab});

        auto key = keys.accessor<float, 2>();
        auto position = positions.accessor<std::int64_t, 1>();
        auto page_table = page_tables.accessor<std::int32_t, 2>();
        auto rope_layout = rope_layouts.accessor<std::int32_t, 2>();
        auto key_count = key_counts.accessor<std::int32_t, 1>();
        auto query_start = query_starts.accessor<std::int32_t, 1>();
        auto logit = logits.accessor<float, 2>();
        for (std::size_t s = 0; s < static_cast<std::size_t>(page_tables.size(0)); s++) {
            REQUIRE((rope_layout[s][0] == 0 && rope_layout[s][1] == 0 && rope_layout[s][2] == -1));
            auto end_row = static_cast<std::size_t>(query_start[s + 1]);
            for (auto row = static_cast<std::size_t>(query_start[s]); row < end_row; row++) {
                std::size_t view_index = static_cast<std::size_t>(key_count[s]) - (end_row - row);
                REQUIRE(position[row] == static_cast<std::int64_t>(view_index));
                std::int64_t sum = 0;
                for (std::size_t j = 0; j <= view_index; j++) {
                    sum += static_cast<std::int64_t>(key[page_table[s][j / shape.page_size]][j % shape.page_size]);
                }
                logit[row][sum % vocab] = 1.0f;
            }
        }
        return logits.to(g.tokens.device());
    }
};

cudaStream_t compute_stream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

digest root() {
    return pulsar::make_root_digest("sum-model", shape);
}

struct fixture {
    std::shared_ptr<store_contents> store;
    sum_model model;
    test_scheduler sched;

    explicit fixture(
        std::size_t n_lines,
        std::size_t prefill_tokens_per_step = 64,
        std::int64_t eos_id = unreachable_eos,
        memory_store backing = {}
    )
    : store(backing.contents),
      sched(
          model,
          pulsar::kv_cache<memory_store>(shape, n_lines, at::Device(at::kCUDA, 0), std::move(backing)),
          compute_stream(),
          root(),
          eos_id,
          prefill_tokens_per_step
      ) {}
};

token_list concat(token_list front, const token_list& back) {
    front.insert(front.end(), back.begin(), back.end());
    return front;
}

token_list without_last(token_list tokens) {
    tokens.pop_back();
    return tokens;
}

// The first count tokens sum_model samples greedily after context.
token_list continuation(token_list context, std::size_t count) {
    token_list generated;
    for (std::size_t i = 0; i < count; i++) {
        std::int64_t token = std::accumulate(context.begin(), context.end(), std::int64_t{0}) % vocab;
        generated.push_back(token);
        context.push_back(token);
    }
    return generated;
}

pulsar::state ended_state(const result_stream::taken& result) {
    REQUIRE(result.end.has_value());
    REQUIRE(result.end->has_value());
    return **result.end;
}

request_error ended_error(const result_stream::taken& result) {
    REQUIRE(result.end.has_value());
    REQUIRE(!result.end->has_value());
    return result.end->error();
}

// p holds tokens as K and their view indices, from first_index on, as V.
bool page_holds(const page& p, const token_list& tokens, std::size_t first_index) {
    auto n_tokens = static_cast<std::int64_t>(tokens.size());
    auto first = static_cast<std::int64_t>(first_index);
    at::Tensor k = at::tensor(tokens).to(at::kFloat).view({1, n_tokens, 1, 1}).expand_as(p.k);
    at::Tensor v = at::arange(first, first + n_tokens).to(at::kFloat).view({1, n_tokens, 1, 1}).expand_as(p.v);
    return at::equal(p.k, k) && at::equal(p.v, v);
}

struct submission {
    request r;
    std::shared_ptr<result_stream> stream;
    result_writer writer;
};

submission make_submission(request r) {
    std::pair<std::shared_ptr<result_stream>, result_writer> made = result_stream::make();
    return submission{.r = std::move(r), .stream = made.first, .writer = std::move(made.second)};
}

task<void> serve(test_scheduler& sched, request r, result_writer writer, std::size_t& remaining) {
    co_await sched.submit(std::move(r), std::move(writer));
    if (--remaining == 0) {
        event_loop::current()->stop();
    }
}

std::vector<result_stream::taken> serve_submissions(test_scheduler& sched, std::vector<submission> submissions) {
    std::size_t remaining = submissions.size();
    event_loop loop;
    loop.spawn(sched.run());
    for (submission& s : submissions) {
        loop.spawn(serve(sched, std::move(s.r), std::move(s.writer), remaining));
    }
    REQUIRE(loop.run() == nullptr);
    std::vector<result_stream::taken> results;
    for (submission& s : submissions) {
        results.push_back(s.stream->take());
    }
    return results;
}

std::vector<result_stream::taken> serve_all(test_scheduler& sched, std::vector<request> requests) {
    std::vector<submission> submissions;
    for (request& r : requests) {
        submissions.push_back(make_submission(std::move(r)));
    }
    return serve_submissions(sched, std::move(submissions));
}

task<void> serve_then(
    test_scheduler& sched,
    request first,
    std::function<request(const result_stream::taken&)> next,
    std::vector<result_stream::taken>& results
) {
    std::pair<std::shared_ptr<result_stream>, result_writer> first_stream = result_stream::make();
    co_await sched.submit(std::move(first), std::move(first_stream.second));
    results.push_back(first_stream.first->take());
    std::pair<std::shared_ptr<result_stream>, result_writer> second_stream = result_stream::make();
    co_await sched.submit(next(results.front()), std::move(second_stream.second));
    results.push_back(second_stream.first->take());
    event_loop::current()->stop();
}

// Serves first, then the request next makes from first's result.
std::vector<result_stream::taken>
serve_in_turn(test_scheduler& sched, request first, std::function<request(const result_stream::taken&)> next) {
    std::vector<result_stream::taken> results;
    event_loop loop;
    loop.spawn(sched.run());
    loop.spawn(serve_then(sched, std::move(first), std::move(next), results));
    REQUIRE(loop.run() == nullptr);
    return results;
}

TEST_CASE("a request from an empty state prefills, decodes and ends with its state") {
    fixture f(16);
    token_list prompt{1, 2, 3, 4, 5, 6};
    std::vector<result_stream::taken> results = serve_all(f.sched, {request{.tokens = prompt, .max_tokens = 5}});

    token_list generated = continuation(prompt, 5);
    REQUIRE(results[0].tokens == generated);
    pulsar::state end = ended_state(results[0]);
    token_list forwarded = concat(prompt, without_last(generated));
    std::size_t full_tokens = forwarded.size() - forwarded.size() % shape.page_size;
    REQUIRE((end.page_hashes == pulsar::make_page_digests(root(), forwarded, shape.page_size)));
    REQUIRE(end.tail == token_list(forwarded.begin() + full_tokens, forwarded.end()));
    REQUIRE(end.page_hashes.size() == 2);
    for (std::size_t i = 0; i < end.page_hashes.size(); i++) {
        std::size_t first_index = i * shape.page_size;
        token_list page_tokens(forwarded.begin() + first_index, forwarded.begin() + first_index + shape.page_size);
        REQUIRE(page_holds(f.store->pages.at(end.page_hashes[i]), page_tokens, first_index));
    }
}

TEST_CASE("a request continues from the end state of another in the same cache") {
    fixture f(16);
    token_list prompt{1, 2, 3, 4, 5, 6};
    token_list user{7, 8};
    std::vector<result_stream::taken> results =
        serve_in_turn(f.sched, request{.tokens = prompt, .max_tokens = 5}, [&](const result_stream::taken& first) {
            return request{
                .initial_state = ended_state(first),
                .tokens = concat({first.tokens.back()}, user),
                .max_tokens = 4
            };
        });

    token_list context = concat(concat(prompt, results[0].tokens), user);
    REQUIRE(results[1].tokens == continuation(context, 4));
    REQUIRE((
        ended_state(results[1]).page_hashes ==
        pulsar::make_page_digests(root(), concat(context, without_last(results[1].tokens)), shape.page_size)
    ));
    REQUIRE(f.store->fetches.empty());
}

TEST_CASE("a request continues from pages another cache fetches from the store") {
    fixture first_engine(16);
    token_list prompt{1, 2, 3, 4, 5, 6};
    token_list user{7, 8};
    result_stream::taken first = serve_all(first_engine.sched, {request{.tokens = prompt, .max_tokens = 5}})[0];
    pulsar::state first_end = ended_state(first);

    fixture second_engine(16, 64, unreachable_eos, memory_store{.contents = first_engine.store});
    result_stream::taken second = serve_all(
        second_engine.sched,
        {request{.initial_state = first_end, .tokens = concat({first.tokens.back()}, user), .max_tokens = 4}}
    )[0];

    REQUIRE(second.tokens == continuation(concat(concat(prompt, first.tokens), user), 4));
    REQUIRE((first_engine.store->fetches == std::vector<std::vector<digest>>{first_end.page_hashes}));
}

TEST_CASE("a state page missing from the store ends the request with miss") {
    fixture f(16);
    pulsar::state unknown{.page_hashes = {digest{.hi = 1, .lo = 2}}};
    std::vector<result_stream::taken> results =
        serve_all(f.sched, {request{.initial_state = unknown, .tokens = {3}, .max_tokens = 1}});

    REQUIRE(results[0].tokens.empty());
    REQUIRE(ended_error(results[0]) == request_error::miss);
    REQUIRE(f.model.forward_count == 0);
}

TEST_CASE("the eos token ends the request after it is emitted") {
    token_list prompt{1, 2, 3};
    token_list free_run = continuation(prompt, 4);
    REQUIRE(free_run == token_list{6, 12, 24, 48});
    fixture f(16, 64, 24);
    std::vector<result_stream::taken> results = serve_all(f.sched, {request{.tokens = prompt, .max_tokens = 10}});

    REQUIRE(results[0].tokens == token_list{6, 12, 24});
    pulsar::state end = ended_state(results[0]);
    REQUIRE((end.page_hashes == pulsar::make_page_digests(root(), token_list{1, 2, 3, 6}, shape.page_size)));
    REQUIRE(end.tail == token_list{12});
}

TEST_CASE("max_tokens 0 forwards the tokens and ends with their state") {
    fixture f(16);
    token_list prompt{1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<result_stream::taken> results = serve_all(f.sched, {request{.tokens = prompt, .max_tokens = 0}});

    REQUIRE(results[0].tokens.empty());
    pulsar::state end = ended_state(results[0]);
    REQUIRE((end.page_hashes == pulsar::make_page_digests(root(), prompt, shape.page_size)));
    REQUIRE(end.tail == token_list{9});
    REQUIRE(f.store->pages.size() == 2);
}

TEST_CASE("a request cancelled before it is admitted ends with cancelled") {
    fixture f(16);
    submission s = make_submission(request{.tokens = {1, 2, 3}, .max_tokens = 10});
    s.stream->cancel();
    std::vector<submission> submissions;
    submissions.push_back(std::move(s));
    std::vector<result_stream::taken> results = serve_submissions(f.sched, std::move(submissions));

    REQUIRE(results[0].tokens.empty());
    REQUIRE(ended_error(results[0]) == request_error::cancelled);
    REQUIRE(f.model.forward_count == 0);
}

TEST_CASE("a request cancelled while decoding ends with cancelled after its step") {
    fixture f(16);
    token_list prompt{1, 2, 3};
    submission s = make_submission(request{.tokens = prompt, .max_tokens = 10});
    f.model.on_forward = [&f, stream = s.stream] {
        if (f.model.forward_count == 3) {
            stream->cancel();
        }
    };
    std::vector<submission> submissions;
    submissions.push_back(std::move(s));
    std::vector<result_stream::taken> results = serve_submissions(f.sched, std::move(submissions));

    REQUIRE(results[0].tokens == continuation(prompt, 3));
    REQUIRE(ended_error(results[0]) == request_error::cancelled);
}

TEST_CASE("a request waits for the lines another request releases") {
    fixture f(3);
    token_list first{1, 2, 3, 4, 5};
    token_list second{6, 7, 8, 9, 10};
    std::vector<result_stream::taken> results = serve_all(
        f.sched,
        {request{.tokens = first, .max_tokens = 3}, request{.tokens = second, .max_tokens = 3}}
    );

    REQUIRE(results[0].tokens == continuation(first, 3));
    REQUIRE(results[1].tokens == continuation(second, 3));
    REQUIRE(f.model.forward_count == 6);
}

TEST_CASE("a request needing more lines than the cache has ends with oom") {
    fixture f(2);
    std::vector<result_stream::taken> results =
        serve_all(f.sched, {request{.tokens = token_list(12, 1), .max_tokens = 1}});

    REQUIRE(results[0].tokens.empty());
    REQUIRE(ended_error(results[0]) == request_error::oom);
    REQUIRE(f.model.forward_count == 0);
}

TEST_CASE("a request that outgrows the cache alone ends with oom") {
    fixture f(2);
    token_list prompt{1, 2, 3, 4};
    std::vector<result_stream::taken> results = serve_all(f.sched, {request{.tokens = prompt, .max_tokens = 10}});

    REQUIRE(results[0].tokens == continuation(prompt, 5));
    REQUIRE(ended_error(results[0]) == request_error::oom);
}

TEST_CASE("a request that outgrows the free lines waits for them and completes") {
    fixture f(5);
    token_list first{1, 2, 3, 4};
    token_list second{5, 6, 7, 8};
    std::vector<result_stream::taken> results = serve_all(
        f.sched,
        {request{.tokens = first, .max_tokens = 10}, request{.tokens = second, .max_tokens = 10}}
    );

    REQUIRE(results[0].tokens == continuation(first, 10));
    REQUIRE(results[1].tokens == continuation(second, 10));
    REQUIRE((
        ended_state(results[1]).page_hashes ==
        pulsar::make_page_digests(root(), concat(second, without_last(results[1].tokens)), shape.page_size)
    ));
    // Five steps together; the second then waits for a third line while the first runs its last five steps.
    REQUIRE(f.model.forward_count == 15);
}

TEST_CASE("prefill forwards at most the step's prefill tokens") {
    fixture f(16, 4);
    token_list prompt{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<result_stream::taken> results = serve_all(f.sched, {request{.tokens = prompt, .max_tokens = 2}});

    REQUIRE(results[0].tokens == continuation(prompt, 2));
    REQUIRE(f.model.prefill_chunks == std::vector<token_list>{{4}, {4}, {2}});
    REQUIRE(f.model.decode_sizes == std::vector<std::int64_t>{1});
}

TEST_CASE("running requests are forwarded in one step") {
    fixture f(16);
    token_list first{1, 2, 3};
    token_list second{4, 5, 6};
    std::vector<result_stream::taken> results = serve_all(
        f.sched,
        {request{.tokens = first, .max_tokens = 3}, request{.tokens = second, .max_tokens = 3}}
    );

    REQUIRE(results[0].tokens == continuation(first, 3));
    REQUIRE(results[1].tokens == continuation(second, 3));
    REQUIRE(f.model.prefill_chunks == std::vector<token_list>{{3, 3}});
    REQUIRE(f.model.decode_sizes == std::vector<std::int64_t>{2, 2});
}

}  // namespace
