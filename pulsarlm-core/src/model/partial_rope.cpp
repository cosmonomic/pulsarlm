#include "pulsarlm/model/partial_rope.hpp"

#include <ATen/ATen.h>

#include <vector>

namespace pulsarlm {

at::Tensor rope_permutation(std::int64_t head_dim, std::int64_t rotary_dim, at::Device device) {
    std::vector<std::int64_t> permutation(head_dim);
    std::int64_t half = head_dim / 2;
    std::int64_t next_unrotated = rotary_dim;
    for (std::int64_t i = 0; i < head_dim; i++) {
        std::int64_t pair = i % half;
        permutation[i] = pair < rotary_dim / 2 ? pair + (i >= half ? rotary_dim / 2 : 0) : next_unrotated++;
    }
    return at::tensor(permutation).to(device);
}

at::Tensor permute_heads(const at::Tensor& weight, std::int64_t heads, const at::Tensor& permutation) {
    std::int64_t head_dim = permutation.size(0);
    return weight.view({heads, head_dim, -1}).index_select(1, permutation).reshape(weight.sizes()).contiguous();
}

at::Tensor permute_queries(const at::Tensor& weight, std::int64_t heads, const at::Tensor& permutation) {
    std::int64_t head_dim = permutation.size(0);
    at::Tensor halves = weight.view({heads, 2, head_dim, -1});
    at::Tensor queries = halves.select(1, 0).index_select(1, permutation);
    return at::stack({queries, halves.select(1, 1)}, 1).reshape(weight.sizes()).contiguous();
}

}  // namespace pulsarlm
