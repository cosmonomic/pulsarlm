#include "pulsarlm/sampling.hpp"

#include <ATen/ATen.h>

#undef CHECK
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace {

using pulsarlm::sampling_params;

const at::Device device(at::kCUDA, 0);

// Every row draws with seed, row r as the seed's r-th draw.
at::Tensor draw_tokens(const at::Tensor& logits, const sampling_params& params, std::optional<std::uint64_t> seed) {
    auto rows = static_cast<std::size_t>(logits.size(0));
    std::vector<sampling_params> row_params(rows, params);
    std::uint64_t row_seed = pulsarlm::sampling_seed(seed);
    std::vector<pulsarlm::sampling_draw> draws;
    for (std::size_t row = 0; row < rows; row++) {
        draws.push_back(pulsarlm::sampling_draw{.seed = row_seed, .count = row});
    }
    return pulsarlm::sample(logits, row_params, draws).cpu();
}

TEST_CASE("temperature 0 takes each row's argmax") {
    at::Tensor logits = at::tensor({0.1f, 0.9f, 0.3f, 2.0f, -1.0f, 0.5f}).view({2, 3}).to(device);
    at::Tensor tokens = draw_tokens(logits, sampling_params{}, std::nullopt);
    REQUIRE(tokens[0].item<std::int64_t>() == 1);
    REQUIRE(tokens[1].item<std::int64_t>() == 0);
}

TEST_CASE("top_k 1 takes each row's argmax at any temperature") {
    at::manual_seed(0);
    at::Tensor logits = at::randn({16, 50}).to(device);
    at::Tensor tokens = draw_tokens(logits, sampling_params{.temperature = 2.0, .top_k = 1}, 1);
    REQUIRE(at::equal(tokens, logits.argmax(-1).cpu()));
}

TEST_CASE("top_p keeps a token while the probability ranked above it is at most top_p") {
    at::Tensor probabilities = at::tensor({0.5f, 0.3f, 0.2f});
    at::Tensor logits = probabilities.log().view({1, 3}).expand({512, 3}).contiguous().to(device);
    at::Tensor tokens = draw_tokens(logits, sampling_params{.temperature = 1.0, .top_p = 0.6}, 2);
    REQUIRE(tokens.eq(2).sum().item<std::int64_t>() == 0);
    REQUIRE(tokens.eq(0).sum().item<std::int64_t>() > 0);
    REQUIRE(tokens.eq(1).sum().item<std::int64_t>() > 0);
}

TEST_CASE("temperature 1 draws tokens at their probabilities") {
    at::Tensor probabilities = at::tensor({0.7f, 0.2f, 0.1f});
    at::Tensor logits = probabilities.log().view({1, 3}).expand({4000, 3}).contiguous().to(device);
    at::Tensor tokens = draw_tokens(logits, sampling_params{.temperature = 1.0}, 3);
    for (std::int64_t token = 0; token < 3; token++) {
        double frequency = tokens.eq(token).sum().item<double>() / 4000.0;
        REQUIRE(std::abs(frequency - probabilities[token].item<double>()) < 0.03);
    }
}

TEST_CASE("a token the noise cannot lift never wins, over a large vocabulary") {
    constexpr std::int64_t rows = 256;
    constexpr std::int64_t vocab = 262144;
    at::Tensor logits = at::full({rows, vocab}, -1e4f, at::TensorOptions(device));
    logits.select(1, 0).fill_(0.0f);
    at::Tensor tokens = draw_tokens(logits, sampling_params{.temperature = 1.0}, 4);
    REQUIRE(tokens.eq(0).all().item<bool>());
}

TEST_CASE("the same seed draws the same tokens") {
    at::manual_seed(0);
    at::Tensor logits = at::randn({64, 100}).to(device);
    sampling_params params{.temperature = 1.0};
    at::Tensor first = draw_tokens(logits, params, 7);
    at::Tensor second = draw_tokens(logits, params, 7);
    at::Tensor other = draw_tokens(logits, params, 8);
    REQUIRE(at::equal(first, second));
    REQUIRE(!at::equal(first, other));
}

TEST_CASE("a row draws the same token whatever the other rows of its batch are") {
    at::manual_seed(0);
    at::Tensor logits = at::randn({8, 100}).to(device);
    std::vector<sampling_params> params(8, sampling_params{.temperature = 1.0});
    std::vector<pulsarlm::sampling_draw> draws;
    for (std::uint64_t row = 0; row < 8; row++) {
        draws.push_back(pulsarlm::sampling_draw{.seed = 100 + row, .count = 3});
    }
    at::Tensor batched = pulsarlm::sample(logits, params, draws).cpu();
    for (std::int64_t row = 0; row < 8; row++) {
        at::Tensor alone = pulsarlm::sample(
            logits.narrow(0, row, 1),
            std::span(params).subspan(static_cast<std::size_t>(row), 1),
            std::span(draws).subspan(static_cast<std::size_t>(row), 1)
        );
        REQUIRE(alone.item<std::int64_t>() == batched[row].item<std::int64_t>());
    }
}

TEST_CASE("rows at temperature 0 take their argmax beside rows that draw") {
    at::manual_seed(0);
    at::Tensor logits = at::randn({4, 100}).to(device);
    std::vector<sampling_params> params{
        sampling_params{},
        sampling_params{.temperature = 1.0, .top_k = 5},
        sampling_params{},
        sampling_params{.temperature = 1.0},
    };
    std::vector<pulsarlm::sampling_draw> draws(4, pulsarlm::sampling_draw{.seed = 1, .count = 0});
    at::Tensor tokens = pulsarlm::sample(logits, params, draws).cpu();
    at::Tensor argmax = logits.argmax(-1).cpu();
    REQUIRE(tokens[0].item<std::int64_t>() == argmax[0].item<std::int64_t>());
    REQUIRE(tokens[2].item<std::int64_t>() == argmax[2].item<std::int64_t>());
    at::Tensor top5 = std::get<1>(logits[1].topk(5)).cpu();
    REQUIRE(top5.eq(tokens[1]).any().item<bool>());
}

}  // namespace
