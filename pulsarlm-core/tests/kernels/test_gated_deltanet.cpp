#include "pulsarlm/ops.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAFunctions.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

namespace {

constexpr int64_t kLines = 10;

at::Tensor int32_cuda(const std::vector<int32_t>& v) {
    return at::tensor(v, at::dtype(at::kInt)).to(at::kCUDA);
}

// Each sequence's running line, its boundary line, then a column the ops do not read.
at::Tensor line_rows(const std::vector<int32_t>& running, const std::vector<int32_t>& boundary) {
    std::vector<int32_t> rows;
    for (size_t s = 0; s < running.size(); s++) {
        rows.insert(rows.end(), {running[s], boundary[s], -1});
    }
    return int32_cuda(rows).view({static_cast<int64_t>(running.size()), 3});
}

// A pool of random lines with 4 elements between each line's end and the next line.
at::Tensor spaced_pool(const std::vector<int64_t>& line_sizes, at::ScalarType dtype) {
    int64_t line_elements = 1;
    for (int64_t size : line_sizes) {
        line_elements *= size;
    }
    std::vector<int64_t> sizes{kLines};
    sizes.insert(sizes.end(), line_sizes.begin(), line_sizes.end());
    return at::randn({kLines, line_elements + 4}, at::device(at::kCUDA).dtype(dtype))
        .narrow(1, 0, line_elements)
        .view(sizes);
}

struct ragged_case {
    std::vector<int32_t> counts;
    std::vector<int32_t> page_ends;
    bool decode;

    int64_t total() const {
        int64_t total = 0;
        for (int32_t count : counts) {
            total += count;
        }
        return total;
    }

    std::optional<at::Tensor> cu_seqlens_q() const {
        if (decode) {
            return std::nullopt;
        }
        std::vector<int32_t> starts{0};
        for (int32_t count : counts) {
            starts.push_back(starts.back() + count);
        }
        return int32_cuda(starts);
    }
};

ragged_case prefill_case() {
    return {.counts = {1, 2, 5, 7, 140}, .page_ends = {1, 2, 0, 3, 128}, .decode = false};
}

ragged_case decode_case() {
    return {.counts = {1, 1, 1, 1, 1}, .page_ends = {0, 1, 0, 1, 1}, .decode = true};
}

const std::vector<int32_t> kRunning{2, 0, 5, 3, 8};
const std::vector<int32_t> kBoundary{4, 1, 6, 7, 9};

struct conv_expected {
    at::Tensor out;
    at::Tensor running;
    at::Tensor boundary;
};

conv_expected reference_conv(
    const at::Tensor& x_seq,
    const at::Tensor& weight,
    const at::Tensor& old,
    int32_t page_end,
    int64_t dilation
) {
    const int64_t history = (weight.size(2) - 1) * dilation;
    const int64_t n = x_seq.size(0);
    at::Tensor window = at::cat({old.to(at::kFloat), x_seq.to(at::kFloat).t()}, -1);
    at::Tensor convolved = at::conv1d(
        window.unsqueeze(0),
        weight.to(at::kFloat),
        std::nullopt,
        at::IntArrayRef{1},
        at::IntArrayRef{0},
        at::IntArrayRef{dilation},
        weight.size(0)
    );
    at::Tensor out = at::silu(convolved.squeeze(0).t().to(x_seq.scalar_type()).to(at::kFloat)).to(x_seq.scalar_type());
    return {
        out,
        window.narrow(-1, n, history).to(x_seq.scalar_type()),
        window.narrow(-1, page_end, history).to(x_seq.scalar_type()),
    };
}

struct delta_expected {
    at::Tensor out;
    at::Tensor running;
    std::optional<at::Tensor> boundary;
};

// Hugging Face's recurrent gated delta rule, one token at a time.
delta_expected reference_delta_rule(
    const at::Tensor& x_seq,
    const at::Tensor& a_seq,
    const at::Tensor& b_seq,
    const at::Tensor& dt_bias,
    const at::Tensor& neg_a,
    at::Tensor state,
    int64_t key_heads,
    int32_t page_end
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
    std::optional<at::Tensor> boundary;
    for (int64_t t = 0; t < n; t++) {
        state = state * decay[t].exp().view({-1, 1, 1});
        at::Tensor remembered = (state * k[t].unsqueeze(-1)).sum(-2);
        at::Tensor delta = (v[t] - remembered) * beta[t].unsqueeze(-1);
        state = state + k[t].unsqueeze(-1) * delta.unsqueeze(-2);
        outputs.push_back((state * q[t].unsqueeze(-1)).sum(-2));
        if (t + 1 == page_end) {
            boundary = state.clone();
        }
    }
    return {at::stack(outputs).to(x_seq.scalar_type()), state, boundary};
}

double output_tolerance(at::ScalarType dtype) {
    return dtype == at::kFloat ? 1e-4 : 2e-2;
}

}  // namespace

TEST_CASE("causal_conv continues each sequence's inputs and keeps its last ones", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const auto dtype = GENERATE(at::kFloat, at::kBFloat16);
    const ragged_case batch = GENERATE(prefill_case(), decode_case());
    const int64_t dilation = GENERATE(1, 3);
    const int64_t channels = 24;
    const int64_t kernel = 4;
    auto opts = at::device(at::kCUDA).dtype(dtype);
    at::Tensor x = at::randn({batch.total(), channels}, opts);
    at::Tensor weight = at::randn({channels, 1, kernel}, opts);
    at::Tensor pool = spaced_pool({channels, (kernel - 1) * dilation}, dtype);
    at::Tensor before = pool.clone();
    at::Tensor rows = line_rows(kRunning, kBoundary);

    at::Tensor out = pulsarlm::causal_conv_cuda(
        x,
        weight,
        pool,
        rows,
        batch.cu_seqlens_q(),
        int32_cuda(batch.page_ends),
        dilation
    );

    const double tolerance = output_tolerance(dtype);
    int64_t first = 0;
    for (size_t s = 0; s < batch.counts.size(); s++) {
        const int64_t n = batch.counts[s];
        conv_expected expected = reference_conv(
            x.narrow(0, first, n),
            weight,
            before.select(0, kRunning[s]),
            batch.page_ends[s],
            dilation
        );
        CHECK(at::allclose(out.narrow(0, first, n), expected.out, tolerance, tolerance));
        CHECK(at::equal(pool.select(0, kRunning[s]), expected.running));
        CHECK(
            at::equal(
                pool.select(0, kBoundary[s]),
                batch.page_ends[s] > 0 ? expected.boundary : before.select(0, kBoundary[s])
            )
        );
        first += n;
    }
}

TEST_CASE("gated_delta_rule continues each sequence's state and keeps it at its page end", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const auto dtype = GENERATE(at::kFloat, at::kBFloat16);
    const int64_t key_dim = GENERATE(32, 128, 256);
    const ragged_case batch = GENERATE(prefill_case(), decode_case());
    const int64_t key_heads = 2;
    const int64_t value_heads = 4;
    const int64_t value_dim = 16;
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
    at::Tensor rows = line_rows(kRunning, kBoundary);

    at::Tensor out = pulsarlm::gated_delta_rule_cuda(
        x,
        a,
        b,
        dt_bias,
        neg_a,
        pool,
        rows,
        batch.cu_seqlens_q(),
        int32_cuda(batch.page_ends),
        key_heads
    );

    REQUIRE(out.sizes() == at::IntArrayRef{batch.total(), value_heads, value_dim});
    const double tolerance = output_tolerance(dtype);
    int64_t first = 0;
    for (size_t s = 0; s < batch.counts.size(); s++) {
        const int64_t n = batch.counts[s];
        delta_expected expected = reference_delta_rule(
            x.narrow(0, first, n),
            a.narrow(0, first, n),
            b.narrow(0, first, n),
            dt_bias,
            neg_a,
            before.select(0, kRunning[s]),
            key_heads,
            batch.page_ends[s]
        );
        CHECK(at::allclose(out.narrow(0, first, n), expected.out, tolerance, tolerance));
        CHECK(at::allclose(pool.select(0, kRunning[s]), expected.running, 1e-4, 1e-4));
        CHECK(
            at::allclose(
                pool.select(0, kBoundary[s]),
                expected.boundary.value_or(before.select(0, kBoundary[s])),
                1e-4,
                1e-4
            )
        );
        first += n;
    }
}

TEST_CASE("gated_rmsnorm scales the normalized x by weight and the gate of z", "[ops][cuda]") {
    if (c10::cuda::device_count() == 0) {
        SKIP("no CUDA device");
    }
    const auto dtype = GENERATE(at::kFloat, at::kBFloat16);
    const auto activation = GENERATE(pulsarlm::gate_activation::silu, pulsarlm::gate_activation::sigmoid);
    const double eps = 1e-6;
    auto opts = at::device(at::kCUDA).dtype(dtype);
    at::Tensor x = at::randn({6, 4, 128}, opts);
    at::Tensor z = at::randn({6, 4, 128}, opts);
    at::Tensor weight = at::randn({128}, opts);

    at::Tensor got = pulsarlm::gated_rmsnorm_cuda(x, z, weight, eps, activation);

    at::Tensor normed = x.to(at::kFloat);
    normed = normed * at::rsqrt(normed.pow(2).mean(-1, true) + eps);
    at::Tensor gate = activation == pulsarlm::gate_activation::silu ? at::silu(z.to(at::kFloat))
                                                                    : at::sigmoid(z.to(at::kFloat));
    at::Tensor expected = ((weight * normed.to(dtype)).to(at::kFloat) * gate).to(dtype);
    const double tolerance = output_tolerance(dtype);
    CHECK(at::allclose(got, expected, tolerance, tolerance));

    at::Tensor zeros = at::zeros_like(x);
    CHECK(at::equal(pulsarlm::gated_rmsnorm_cuda(zeros, z, weight, eps, activation), zeros));
}
