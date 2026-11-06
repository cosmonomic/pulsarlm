#include "pulsarlm/model/linear.hpp"

#include "pulsarlm/ops.hpp"

#include <ATen/ATen.h>

#include <utility>
#include <vector>

namespace pulsarlm {

Linear::Linear(std::variant<at::Tensor, fp8_weight> weight)
: weight(std::move(weight)) {}

Linear Linear::dense(at::Tensor weight) {
    return Linear(std::move(weight));
}

Linear Linear::fp8(const at::Tensor& weight, const at::Tensor& scale, std::int64_t block_n, std::int64_t group_size) {
    auto [weight_marlin, scale_marlin] = repack_fp8_marlin_cuda(weight, scale, block_n, group_size);
    return Linear(
        fp8_weight{
            .weight_marlin = std::move(weight_marlin),
            .scale_marlin = std::move(scale_marlin),
            .n = weight.size(0),
            .block_n = block_n,
            .group_size = group_size,
        }
    );
}

at::Tensor Linear::apply(const at::Tensor& x) const {
    if (const auto* dense = std::get_if<at::Tensor>(&weight)) {
        return at::linear(x, *dense);
    }
    const fp8_weight& w = std::get<fp8_weight>(weight);
    std::vector<std::int64_t> sizes = x.sizes().vec();
    const at::Tensor rows = x.reshape({-1, sizes.back()}).contiguous();
    sizes.back() = w.n;
    return gemm_fp8_marlin_cuda(rows, w.weight_marlin, w.scale_marlin, w.n, w.block_n, w.group_size).view(sizes);
}

}  // namespace pulsarlm
