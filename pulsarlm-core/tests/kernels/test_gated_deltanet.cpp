#include "pulsarlm/kernels/gated_deltanet.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAFunctions.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

namespace {

constexpr int64_t line_count = 40;
constexpr int64_t pool_page_size = 16;
// The DeltaNet slots of the cases, and the conv windows' inputs past the kernel's history.
constexpr int64_t slot_count = 3;
constexpr int64_t extra_inputs = slot_count - 1;

at::Tensor int32_cuda(const std::vector<int32_t>& v) {
    return at::tensor(v, at::dtype(at::kInt)).to(at::kCUDA);
}

// A pool of random lines with 4 elements between each line's end and the next line.
at::Tensor spaced_pool(const std::vector<int64_t>& line_sizes, at::ScalarType dtype) {
    int64_t line_elements = 1;
    for (int64_t size : line_sizes) {
        line_elements *= size;
    }
    std::vector<int64_t> sizes{line_count};
    sizes.insert(sizes.end(), line_sizes.begin(), line_sizes.end());
    return at::randn({line_count, line_elements + 4}, at::device(at::kCUDA).dtype(dtype))
        .narrow(1, 0, line_elements)
        .view(sizes);
}

enum class group_kind { prefill, decode, verify };

struct ragged_case {
    std::vector<int32_t> counts;
    std::vector<int64_t> history_positions;
    std::vector<int32_t> rejected;
    group_kind kind;

    int64_t total() const {
        int64_t total = 0;
        for (int32_t count : counts) {
            total += count;
        }
        return total;
    }

    int64_t num_seqs() const {
        return static_cast<int64_t>(counts.size());
    }

    std::optional<at::Tensor> cu_seqlens_q() const {
        if (kind != group_kind::prefill) {
            return std::nullopt;
        }
        std::vector<int32_t> starts{0};
        for (int32_t count : counts) {
            starts.push_back(starts.back() + count);
        }
        return int32_cuda(starts);
    }

    // The tokens of sequence s up to the newest page end among them, 0 when none ends a page.
    int64_t page_end(size_t s) const {
        const int64_t first = history_positions[s];
        const int64_t end = (first + counts[s]) / pool_page_size * pool_page_size;
        return end > first ? end - first : 0;
    }

    // Sequence s's line j: its slots, then the boundary lines of its even and odd pages, scattered over the pool.
    static int32_t line(size_t s, int64_t slots, int64_t j) {
        return static_cast<int32_t>((static_cast<int64_t>(s) * (slots + 2) + j) * 7 % line_count);
    }

    // The boundary line of the newest page sequence s's tokens end.
    int32_t boundary_line(size_t s, int64_t slots) const {
        const int64_t page = (history_positions[s] + page_end(s)) / pool_page_size - 1;
        return line(s, slots, slots + page % 2);
    }

    int32_t other_boundary_line(size_t s, int64_t slots) const {
        const int64_t page = (history_positions[s] + page_end(s)) / pool_page_size - 1;
        return line(s, slots, slots + (page + 1) % 2);
    }

    // Each sequence's lines, then a column the ops do not read.
    pulsarlm::state_rows rows(int64_t slots) const {
        std::vector<int32_t> ids;
        for (size_t s = 0; s < counts.size(); s++) {
            for (int64_t j = 0; j < slots + 2; j++) {
                ids.push_back(line(s, slots, j));
            }
            ids.push_back(-1);
        }
        return pulsarlm::state_rows{
            .lines = int32_cuda(ids).view({num_seqs(), slots + 3}),
            .slots = slots,
            .history_positions = at::tensor(history_positions, at::dtype(at::kLong)).to(at::kCUDA),
            .rejected_tokens = int32_cuda(rejected),
            .page_size = pool_page_size,
        };
    }
};

ragged_case prefill_case() {
    return {
        .counts = {1, 2, 5, 7, 140},
        .history_positions = {15, 30, 0, 13, 0},
        .rejected = {0, 1, 2, 0, 1},
        .kind = group_kind::prefill,
    };
}

ragged_case decode_case() {
    return {
        .counts = {1, 1, 1, 1, 1},
        .history_positions = {3, 15, 4, 31, 47},
        .rejected = {0, 1, 2, 0, 1},
        .kind = group_kind::decode,
    };
}

// A verify group of fewer tokens per sequence than slot_count, as when a step drafts fewer tokens than the layout allows.
ragged_case short_verify_case() {
    return {
        .counts = {2, 2, 2, 2, 2},
        .history_positions = {14, 30, 0, 31, 46},
        .rejected = {0, 1, 2, 0, 1},
        .kind = group_kind::verify,
    };
}

// slot_count tokens per sequence.
ragged_case verify_case() {
    return {
        .counts = {3, 3, 3, 3, 3},
        .history_positions = {14, 29, 0, 30, 45},
        .rejected = {0, 1, 2, 0, 1},
        .kind = group_kind::verify,
    };
}

struct conv_expected {
    at::Tensor out;
    at::Tensor window_after;
    std::optional<at::Tensor> boundary;
};

// old is the line's window; its first window - rejected entries are the inputs before the group's tokens, the latest
// last, and entries before the first input read zero.
conv_expected reference_conv(
    const at::Tensor& x_seq,
    const at::Tensor& weight,
    const at::Tensor& old,
    int64_t rejected,
    int64_t page_end,
    int64_t dilation
) {
    const int64_t window = old.size(-1);
    const int64_t history = (weight.size(2) - 1) * dilation;
    const int64_t n = x_seq.size(0);
    const int64_t valid = window - rejected;
    at::Tensor inputs = at::cat(
        {at::zeros_like(old).to(at::kFloat), old.narrow(-1, 0, valid).to(at::kFloat), x_seq.to(at::kFloat).t()},
        -1
    );
    at::Tensor convolved = at::conv1d(
        inputs.narrow(-1, window + valid - history, history + n).unsqueeze(0),
        weight.to(at::kFloat),
        std::nullopt,
        at::IntArrayRef{1},
        at::IntArrayRef{0},
        at::IntArrayRef{dilation},
        weight.size(0)
    );
    at::Tensor out = at::silu(convolved.squeeze(0).t().to(x_seq.scalar_type()).to(at::kFloat)).to(x_seq.scalar_type());
    auto window_after = [&](int64_t count) { return inputs.narrow(-1, valid + count, window).to(x_seq.scalar_type()); };
    return {
        out,
        window_after(n),
        page_end > 0 ? std::optional<at::Tensor>(window_after(page_end)) : std::nullopt,
    };
}

struct delta_expected {
    at::Tensor out;
    // The state after each token.
    std::vector<at::Tensor> states;
};

// Hugging Face's recurrent gated delta rule, one token at a time.
delta_expected reference_delta_rule(
    const at::Tensor& x_seq,
    const at::Tensor& a_seq,
    const at::Tensor& b_seq,
    const at::Tensor& dt_bias,
    const at::Tensor& neg_a,
    at::Tensor state,
    int64_t key_heads
) {
    const int64_t n = x_seq.size(0);
    const int64_t value_heads = state.size(0);
    const int64_t key_dim = state.size(1);
    const int64_t value_dim = state.size(2);
    const int64_t repeats = value_heads / key_heads;
    auto l2_normalized = [](const at::Tensor& t) { return t * at::rsqrt(t.pow(2).sum(-1, true) + 1e-6); };
    at::Tensor x = x_seq.to(at::kFloat);
    at::Tensor q = x.narrow(1, 0, key_heads * key_dim).view({n, key_heads, key_dim}).repeat_interleave(repeats, 1);
    at::Tensor k = x.narrow(1, key_heads * key_dim, key_heads * key_dim)
                       .view({n, key_heads, key_dim})
                       .repeat_interleave(repeats, 1);
    at::Tensor v = x.narrow(1, 2 * key_heads * key_dim, value_heads * value_dim).view({n, value_heads, value_dim});
    q = l2_normalized(q) / std::sqrt(static_cast<double>(key_dim));
    k = l2_normalized(k);
    at::Tensor beta = at::sigmoid(b_seq).to(at::kFloat);
    at::Tensor decay = at::softplus(a_seq.to(at::kFloat) + dt_bias) * neg_a;

    std::vector<at::Tensor> outputs;
    std::vector<at::Tensor> states;
    for (int64_t t = 0; t < n; t++) {
        state = state * decay[t].exp().view({-1, 1, 1});
        at::Tensor remembered = (state * k[t].unsqueeze(-1)).sum(-2);
        at::Tensor delta = (v[t] - remembered) * beta[t].unsqueeze(-1);
        state = state + k[t].unsqueeze(-1) * delta.unsqueeze(-2);
        outputs.push_back((state * q[t].unsqueeze(-1)).sum(-2));
        states.push_back(state.clone());
    }
    return {at::stack(outputs).to(x_seq.scalar_type()), states};
}

// norm_weight * RMSNorm(o) * activation(z) of each value head's output o [tokens, value heads, value dim], flattened
// to [tokens, value heads * value dim].
at::Tensor reference_output_gate(
    const at::Tensor& o,
    const at::Tensor& z,
    const at::Tensor& norm_weight,
    double eps,
    pulsarlm::gate_activation activation
) {
    at::Tensor normed = o.to(at::kFloat);
    normed = normed * at::rsqrt(normed.pow(2).mean(-1, true) + eps);
    at::Tensor gate_input = z.to(at::kFloat).view(o.sizes());
    at::Tensor gate = activation == pulsarlm::gate_activation::silu ? at::silu(gate_input) : at::sigmoid(gate_input);
    return ((norm_weight * normed.to(o.scalar_type())).to(at::kFloat) * gate).to(o.scalar_type()).flatten(1);
}

double output_tolerance(at::ScalarType dtype) {
    return dtype == at::kFloat ? 1e-4 : 2e-2;
}

}  // namespace

TEST_CASE("causal_conv continues each sequence's valid inputs and keeps its window", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const auto dtype = GENERATE(at::kFloat, at::kBFloat16);
    const ragged_case batch = GENERATE(prefill_case(), decode_case(), verify_case(), short_verify_case());
    const int64_t dilation = GENERATE(1, 3);
    const int64_t channels = 24;
    const int64_t kernel = 4;
    auto opts = at::device(at::kCUDA).dtype(dtype);
    at::Tensor x = at::randn({batch.total(), channels}, opts);
    at::Tensor weight = at::randn({channels, 1, kernel}, opts);
    at::Tensor pool = spaced_pool({channels, (kernel - 1) * dilation + extra_inputs}, dtype);
    at::Tensor before = pool.clone();
    const pulsarlm::state_rows rows = batch.rows(1);

    at::Tensor out = batch.kind == group_kind::verify
        ? pulsarlm::causal_conv_verify_cuda(x, weight, pool, rows, batch.counts.front(), dilation)
        : pulsarlm::causal_conv_cuda(x, weight, pool, rows, batch.cu_seqlens_q(), dilation);

    const double tolerance = output_tolerance(dtype);
    int64_t first = 0;
    for (size_t s = 0; s < batch.counts.size(); s++) {
        const int64_t n = batch.counts[s];
        const int32_t slot = ragged_case::line(s, 1, 0);
        conv_expected expected = reference_conv(
            x.narrow(0, first, n),
            weight,
            before.select(0, slot),
            batch.rejected[s],
            batch.page_end(s),
            dilation
        );
        CHECK(at::allclose(out.narrow(0, first, n), expected.out, tolerance, tolerance));
        CHECK(at::equal(pool.select(0, slot), expected.window_after));
        if (expected.boundary) {
            CHECK(at::equal(pool.select(0, batch.boundary_line(s, 1)), *expected.boundary));
            const int32_t other = batch.other_boundary_line(s, 1);
            CHECK(at::equal(pool.select(0, other), before.select(0, other)));
        } else {
            for (int64_t j = 1; j < 3; j++) {
                const int32_t boundary = ragged_case::line(s, 1, j);
                CHECK(at::equal(pool.select(0, boundary), before.select(0, boundary)));
            }
        }
        first += n;
    }
}

TEST_CASE(
    "gated_delta_rule continues each sequence's state from its slot, keeps it at its page end and gates its outputs",
    "[ops][cuda]"
) {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const auto dtype = GENERATE(at::kFloat, at::kBFloat16);
    const int64_t key_dim = GENERATE(32, 128, 256);
    const ragged_case batch = GENERATE(prefill_case(), decode_case(), verify_case(), short_verify_case());
    const auto activation = GENERATE(pulsarlm::gate_activation::silu, pulsarlm::gate_activation::sigmoid);
    const int64_t key_heads = 2;
    const int64_t value_heads = 4;
    // Four CTAs per value head.
    const int64_t value_dim = 64;
    const double eps = 1e-6;
    const int64_t channels = 2 * key_heads * key_dim + value_heads * value_dim;
    auto opts = at::device(at::kCUDA).dtype(dtype);
    auto fp32 = at::device(at::kCUDA).dtype(at::kFloat);
    at::Tensor x = at::randn({batch.total(), channels}, opts);
    at::Tensor a = at::randn({batch.total(), value_heads}, opts);
    at::Tensor b = at::randn({batch.total(), value_heads}, opts);
    at::Tensor dt_bias = at::randn({value_heads}, fp32);
    at::Tensor neg_a = -(at::randn({value_heads}, fp32) * 0.5).exp();
    at::Tensor pool = spaced_pool({value_heads, key_dim, value_dim}, at::kFloat);
    at::Tensor before = pool.clone();
    const pulsarlm::state_rows rows = batch.rows(slot_count);
    at::Tensor z = at::randn({batch.total(), value_heads * value_dim}, opts);
    at::Tensor norm_weight = at::randn({value_dim}, opts);
    at::Tensor counters = at::zeros(
        {pulsarlm::gated_delta_rule_counters(batch.num_seqs(), value_heads)},
        at::device(at::kCUDA).dtype(at::kInt)
    );

    at::Tensor out = batch.kind == group_kind::verify ? pulsarlm::gated_delta_rule_verify_cuda(
                                                            x,
                                                            a,
                                                            b,
                                                            dt_bias,
                                                            neg_a,
                                                            pool,
                                                            rows,
                                                            batch.counts.front(),
                                                            key_heads,
                                                            z,
                                                            norm_weight,
                                                            eps,
                                                            activation,
                                                            counters
                                                        )
                                                      : pulsarlm::gated_delta_rule_cuda(
                                                            x,
                                                            a,
                                                            b,
                                                            dt_bias,
                                                            neg_a,
                                                            pool,
                                                            rows,
                                                            batch.cu_seqlens_q(),
                                                            key_heads,
                                                            z,
                                                            norm_weight,
                                                            eps,
                                                            activation,
                                                            counters
                                                        );

    REQUIRE(out.sizes() == at::IntArrayRef{batch.total(), value_heads * value_dim});
    const double tolerance = output_tolerance(dtype);
    int64_t first = 0;
    for (size_t s = 0; s < batch.counts.size(); s++) {
        const int64_t n = batch.counts[s];
        const int64_t start = slot_count - 1 - std::min<int64_t>(batch.rejected[s], slot_count - 1);
        delta_expected expected = reference_delta_rule(
            x.narrow(0, first, n),
            a.narrow(0, first, n),
            b.narrow(0, first, n),
            dt_bias,
            neg_a,
            before.select(0, ragged_case::line(s, slot_count, start)),
            key_heads
        );
        at::Tensor gated = reference_output_gate(expected.out, z.narrow(0, first, n), norm_weight, eps, activation);
        CHECK(at::allclose(out.narrow(0, first, n), gated, tolerance, tolerance));
        // A verify group of n tokens writes the state after token t into slot slot_count - n + t.
        const int64_t first_written = batch.kind == group_kind::verify ? slot_count - n : slot_count - 1;
        for (int64_t slot = 0; slot < slot_count; slot++) {
            const int32_t slot_line = ragged_case::line(s, slot_count, slot);
            const at::Tensor& written = slot < first_written ? before.select(0, slot_line)
                                                             : expected.states[n - slot_count + slot];
            CHECK(at::allclose(pool.select(0, slot_line), written, 1e-4, 1e-4));
        }
        const int64_t page_end = batch.page_end(s);
        for (int64_t j = slot_count; j < slot_count + 2; j++) {
            const int32_t boundary = ragged_case::line(s, slot_count, j);
            const bool written = page_end > 0 && boundary == batch.boundary_line(s, slot_count);
            CHECK(
                at::allclose(
                    pool.select(0, boundary),
                    written ? expected.states[page_end - 1] : before.select(0, boundary),
                    1e-4,
                    1e-4
                )
            );
        }
        first += n;
    }
}

TEST_CASE("causal_conv reads x's rows wherever they lie", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const ragged_case batch = GENERATE(prefill_case(), decode_case());
    const int64_t channels = 24;
    const int64_t kernel = 4;
    auto opts = at::device(at::kCUDA).dtype(at::kBFloat16);
    at::Tensor x = at::randn({batch.total(), channels + 5}, opts).narrow(1, 2, channels);
    at::Tensor weight = at::randn({channels, 1, kernel}, opts);
    at::Tensor strided_pool = spaced_pool({channels, kernel - 1 + extra_inputs}, at::kBFloat16);
    at::Tensor packed_pool = strided_pool.clone();
    const pulsarlm::state_rows rows = batch.rows(1);

    at::Tensor strided = pulsarlm::causal_conv_cuda(x, weight, strided_pool, rows, batch.cu_seqlens_q(), 1);
    at::Tensor packed = pulsarlm::causal_conv_cuda(x.contiguous(), weight, packed_pool, rows, batch.cu_seqlens_q(), 1);

    CHECK(at::equal(strided, packed));
    CHECK(at::equal(strided_pool, packed_pool));
}

TEST_CASE("gated_delta_rule refuses fewer arrival counters than a step needs", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const ragged_case batch = decode_case();
    const int64_t key_heads = 2;
    const int64_t key_dim = 32;
    const int64_t value_heads = 4;
    const int64_t value_dim = 32;
    auto opts = at::device(at::kCUDA).dtype(at::kBFloat16);
    auto fp32 = at::device(at::kCUDA).dtype(at::kFloat);
    const int64_t needed = pulsarlm::gated_delta_rule_counters(batch.num_seqs(), value_heads);
    REQUIRE(needed > 0);

    REQUIRE_THROWS(
        pulsarlm::gated_delta_rule_cuda(
            at::randn({batch.total(), 2 * key_heads * key_dim + value_heads * value_dim}, opts),
            at::randn({batch.total(), value_heads}, opts),
            at::randn({batch.total(), value_heads}, opts),
            at::randn({value_heads}, fp32),
            -at::ones({value_heads}, fp32),
            spaced_pool({value_heads, key_dim, value_dim}, at::kFloat),
            batch.rows(slot_count),
            batch.cu_seqlens_q(),
            key_heads,
            at::randn({batch.total(), value_heads * value_dim}, opts),
            at::randn({value_dim}, opts),
            1e-6,
            pulsarlm::gate_activation::silu,
            at::zeros({needed - 1}, at::device(at::kCUDA).dtype(at::kInt))
        )
    );
}

TEST_CASE("gated_delta_rule reads a's, b's and z's rows wherever they lie", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const ragged_case batch = GENERATE(prefill_case(), decode_case());
    const int64_t key_heads = 2;
    const int64_t key_dim = 32;
    const int64_t value_heads = 4;
    const int64_t value_dim = 32;
    auto opts = at::device(at::kCUDA).dtype(at::kBFloat16);
    auto fp32 = at::device(at::kCUDA).dtype(at::kFloat);
    at::Tensor x = at::randn({batch.total(), 2 * key_heads * key_dim + value_heads * value_dim}, opts);
    at::Tensor gates = at::randn({batch.total(), 2 * value_heads + value_heads * value_dim + 3}, opts);
    at::Tensor a = gates.narrow(1, 0, value_heads);
    at::Tensor b = gates.narrow(1, value_heads, value_heads);
    at::Tensor z = gates.narrow(1, 2 * value_heads, value_heads * value_dim);
    at::Tensor norm_weight = at::randn({value_dim}, opts);
    const auto activation = pulsarlm::gate_activation::sigmoid;
    at::Tensor dt_bias = at::randn({value_heads}, fp32);
    at::Tensor neg_a = -(at::randn({value_heads}, fp32) * 0.5).exp();
    at::Tensor strided_pool = spaced_pool({value_heads, key_dim, value_dim}, at::kFloat);
    at::Tensor packed_pool = strided_pool.clone();
    const pulsarlm::state_rows rows = batch.rows(slot_count);
    auto counters = [&] {
        return at::zeros(
            {pulsarlm::gated_delta_rule_counters(batch.num_seqs(), value_heads)},
            at::device(at::kCUDA).dtype(at::kInt)
        );
    };

    at::Tensor strided = pulsarlm::gated_delta_rule_cuda(
        x,
        a,
        b,
        dt_bias,
        neg_a,
        strided_pool,
        rows,
        batch.cu_seqlens_q(),
        key_heads,
        z,
        norm_weight,
        1e-6,
        activation,
        counters()
    );
    at::Tensor packed = pulsarlm::gated_delta_rule_cuda(
        x,
        a.contiguous(),
        b.contiguous(),
        dt_bias,
        neg_a,
        packed_pool,
        rows,
        batch.cu_seqlens_q(),
        key_heads,
        z.contiguous(),
        norm_weight,
        1e-6,
        activation,
        counters()
    );

    CHECK(at::equal(strided, packed));
    CHECK(at::equal(strided_pool, packed_pool));
}
