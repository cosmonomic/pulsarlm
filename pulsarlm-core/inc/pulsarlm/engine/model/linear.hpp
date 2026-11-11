#pragma once

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <variant>

namespace pulsarlm {

// The weights one fp8 scale covers: n output rows by k input columns.
struct fp8_block {
    std::int64_t n;
    std::int64_t k;
};

// y = x @ W^T over x's last dim, for a weight stored dense or quantized.
class linear {
    struct fp8_weight {
        at::Tensor weight_marlin;
        at::Tensor scale_marlin;
        std::int64_t n;
        std::int64_t block_n;
        std::int64_t group_size;
    };

    std::variant<at::Tensor, fp8_weight> weight;

    explicit linear(std::variant<at::Tensor, fp8_weight> weight);

  public:
    // weight [N, K] in x's dtype.
    static linear dense(at::Tensor weight);

    // weight float8_e4m3fn [N, K]; W[n, k] = weight[n, k] * scale[n / block_n, k / group_size], scale
    // [ceil(N / block_n), K / group_size]. x is bf16 or fp16.
    static linear fp8(const at::Tensor& weight, const at::Tensor& scale, std::int64_t block_n, std::int64_t group_size);

    at::Tensor apply(const at::Tensor& x) const;
};

}  // namespace pulsarlm
