#include "pulsarlm/sampling.hpp"

#include <ATen/ATen.h>

#undef CHECK
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace {

using pulsarlm::sampling_params;

const at::Device device(at::kCUDA, 0);

// One generator shared by every row, so each row draws after the one before.
std::vector<at::Generator> shared_generator(std::size_t rows, std::optional<std::uint64_t> seed) {
    return std::vector<at::Generator>(rows, pulsarlm::make_sampling_generator(device, seed));
}

at::Tensor draw_tokens(const at::Tensor& logits, const sampling_params& params, std::optional<std::uint64_t> seed) {
    auto rows = static_cast<std::size_t>(logits.size(0));
    std::vector<sampling_params> row_params(rows, params);
    std::vector<at::Generator> generators = shared_generator(rows, seed);
    return pulsarlm::sample(logits, row_params, generators).cpu();
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

}  // namespace
