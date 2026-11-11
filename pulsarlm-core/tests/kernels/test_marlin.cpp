#include "pulsarlm/kernels/ops.hpp"

#include <ATen/ATen.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cstdint>

using namespace pulsarlm;

namespace {

// N pads to the 64-column layout block, K to the 32-wide K tile, and K spans a partial last 64-wide chunk.
constexpr int64_t n = 200;
constexpr int64_t k = 144;
constexpr int64_t group_size = 16;

at::TensorOptions cuda(at::ScalarType dtype) {
    return at::TensorOptions().device(at::kCUDA).dtype(dtype);
}

at::Tensor random_scale(int64_t block_n) {
    return at::rand({(n + block_n - 1) / block_n, k / group_size}, cuda(at::kFloat)) * 0.1 + 0.01;
}

at::Tensor expanded_scale(const at::Tensor& scale, int64_t block_n) {
    return scale.repeat_interleave(block_n, 0).slice(0, 0, n).repeat_interleave(group_size, 1);
}

// Every e4m3 code except the NaN codes 0x7f and 0xff.
at::Tensor random_fp8() {
    at::Tensor bytes = at::randint(0, 256, {n, k}, cuda(at::kByte));
    bytes.index_put_({(bytes == 0x7f).logical_or(bytes == 0xff)}, 0);
    return bytes.view(at::kFloat8_e4m3fn);
}

at::Tensor random_int4() {
    return at::randint(-8, 8, {n, k}, cuda(at::kInt));
}

at::Tensor packed_int4(const at::Tensor& values) {
    const at::Tensor shifts = at::arange(0, 32, 4, at::TensorOptions().dtype(at::kLong));
    const at::Tensor nibbles = values.cpu().bitwise_and(0xF).to(at::kLong).view({n, k / 8, 8});
    return nibbles.bitwise_left_shift(shifts).sum(-1).to(at::kInt).to(at::kCUDA);
}

at::Tensor dequantized(const at::Tensor& weight, const at::Tensor& scale, int64_t block_n, at::ScalarType dtype) {
    return weight.to(dtype) * expanded_scale(scale, block_n).to(dtype);
}

}  // namespace

TEST_CASE("fp8 marlin dequantizes every e4m3 code exactly", "[ops][cuda]") {
    const at::ScalarType dtype = GENERATE(at::kBFloat16, at::kHalf);
    const int64_t block_n = GENERATE(int64_t{1}, int64_t{128});
    const at::Tensor weight = random_fp8();
    const at::Tensor scale = random_scale(block_n);
    const auto [weight_marlin, scale_marlin] = repack_fp8_marlin_cuda(weight, scale, block_n, group_size);

    const at::Tensor identity = at::eye(k, cuda(dtype));
    const at::Tensor y = gemm_fp8_marlin_cuda(identity, weight_marlin, scale_marlin, n, block_n, group_size);

    REQUIRE(at::equal(y, dequantized(weight, scale, block_n, dtype).t()));
}

TEST_CASE("int4 marlin dequantizes every int4 value exactly", "[ops][cuda]") {
    const at::ScalarType dtype = GENERATE(at::kBFloat16, at::kHalf);
    const int64_t block_n = GENERATE(int64_t{1}, int64_t{128});
    const at::Tensor values = random_int4();
    const at::Tensor scale = random_scale(block_n);
    const auto [weight_marlin, scale_marlin] = repack_int4_marlin_cuda(packed_int4(values), scale, block_n, group_size);

    const at::Tensor identity = at::eye(k, cuda(dtype));
    const at::Tensor y = gemm_int4_marlin_cuda(identity, weight_marlin, scale_marlin, n, block_n, group_size);

    REQUIRE(at::equal(y, dequantized(values, scale, block_n, dtype).t()));
}

TEST_CASE("fp8 marlin matches a dequantize-then-matmul reference", "[ops][cuda]") {
    const at::ScalarType dtype = GENERATE(at::kBFloat16, at::kHalf);
    const int64_t m = GENERATE(int64_t{1}, int64_t{16}, int64_t{100});
    const int64_t block_n = 128;
    const at::Tensor weight = random_fp8();
    const at::Tensor scale = random_scale(block_n);
    const auto [weight_marlin, scale_marlin] = repack_fp8_marlin_cuda(weight, scale, block_n, group_size);

    const at::Tensor x = at::randn({m, k}, cuda(dtype));
    const at::Tensor y = gemm_fp8_marlin_cuda(x, weight_marlin, scale_marlin, n, block_n, group_size);
    const at::Tensor reference = at::matmul(
        x.to(at::kFloat),
        dequantized(weight, scale, block_n, dtype).to(at::kFloat).t()
    );

    REQUIRE(at::allclose(y.to(at::kFloat), reference, 2e-2, 2e-2));
}

TEST_CASE("int4 marlin matches a dequantize-then-matmul reference", "[ops][cuda]") {
    const at::ScalarType dtype = GENERATE(at::kBFloat16, at::kHalf);
    const int64_t m = GENERATE(int64_t{1}, int64_t{16}, int64_t{100});
    const int64_t block_n = 1;
    const at::Tensor values = random_int4();
    const at::Tensor scale = random_scale(block_n);
    const auto [weight_marlin, scale_marlin] = repack_int4_marlin_cuda(packed_int4(values), scale, block_n, group_size);

    const at::Tensor x = at::randn({m, k}, cuda(dtype));
    const at::Tensor y = gemm_int4_marlin_cuda(x, weight_marlin, scale_marlin, n, block_n, group_size);
    const at::Tensor reference = at::matmul(
        x.to(at::kFloat),
        dequantized(values, scale, block_n, dtype).to(at::kFloat).t()
    );

    REQUIRE(at::allclose(y.to(at::kFloat), reference, 2e-2, 2e-2));
}
