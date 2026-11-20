#include "batch.hpp"

#include "pulsarlm/engine/executor.hpp"

#include <ATen/ATen.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace pulsarlm {

at::Tensor zero_page_mass(
    model::batch& g,
    const at::Tensor& host_cu_view_pages,
    const std::vector<std::size_t>& mass_heads,
    at::Device device
) {
    std::int64_t row_size = host_cu_view_pages.size(1);
    const auto* cu_view_pages = host_cu_view_pages.const_data_ptr<std::int32_t>();
    auto layer_pages = [&](std::size_t layer) {
        return cu_view_pages[static_cast<std::int64_t>(layer) * row_size + row_size - 1];
    };
    std::int64_t total = 0;
    for (std::size_t layer = 0; layer < mass_heads.size(); layer++) {
        if (mass_heads[layer] > 0) {
            total += layer_pages(layer) * static_cast<std::int64_t>(mass_heads[layer] + 1);
        }
    }
    at::Tensor mass = at::zeros({total}, at::device(device).dtype(at::kFloat));
    g.page_mass_sum.resize(mass_heads.size());
    g.page_weight_sum.resize(mass_heads.size());
    std::int64_t first = 0;
    for (std::size_t layer = 0; layer < mass_heads.size(); layer++) {
        if (mass_heads[layer] > 0) {
            auto heads = static_cast<std::int64_t>(mass_heads[layer]);
            std::int64_t pages = layer_pages(layer);
            g.page_mass_sum[layer] = mass.narrow(0, first, pages * heads).view({pages, heads});
            first += pages * heads;
            g.page_weight_sum[layer] = mass.narrow(0, first, pages);
            first += pages;
        }
    }
    return mass;
}

std::vector<executor::batch::layer_step_mass>
executor::batch::view_mass(std::size_t view, const view_layout& layout) const {
    const float* values = mass.const_data_ptr<float>();
    auto row_size = static_cast<std::size_t>(inputs.cu_view_pages.size(1));
    const auto* cu_view_pages = inputs.cu_view_pages.const_data_ptr<std::int32_t>();
    std::vector<layer_step_mass> by_layer(layout.layers.size());
    std::size_t first = 0;
    for (std::size_t layer = 0; layer < layout.layers.size(); layer++) {
        std::size_t heads = layout.layers[layer].mass_heads;
        if (heads == 0) {
            continue;
        }
        const std::int32_t* row = cu_view_pages + layer * row_size;
        auto first_page = static_cast<std::size_t>(row[view]);
        auto pages = static_cast<std::size_t>(row[view + 1] - row[view]);
        auto layer_pages = static_cast<std::size_t>(row[row_size - 1]);
        by_layer[layer] = layer_step_mass{
            .mass_sum = std::span(values + first + first_page * heads, pages * heads),
            .weight_sum = std::span(values + first + layer_pages * heads + first_page, pages),
        };
        first += layer_pages * (heads + 1);
    }
    return by_layer;
}

void executor::batch::join_mass(std::span<const shard> shards, const view_layout& layout) {
    if (!mass_by_shard.front().defined() || mass_by_shard.size() == 1) {
        mass = mass_by_shard.front();
        return;
    }
    std::int64_t total = 0;
    auto row_size = static_cast<std::size_t>(inputs.cu_view_pages.size(1));
    const auto* cu_view_pages = inputs.cu_view_pages.const_data_ptr<std::int32_t>();
    for (std::size_t layer = 0; layer < layout.layers.size(); layer++) {
        if (layout.layers[layer].mass_heads > 0) {
            auto pages = static_cast<std::int64_t>(cu_view_pages[layer * row_size + row_size - 1]);
            total += pages * static_cast<std::int64_t>(layout.layers[layer].mass_heads + 1);
        }
    }
    mass = pinned({total}, at::kFloat);
    float* out = mass.mutable_data_ptr<float>();
    std::vector<const float*> in;
    for (const at::Tensor& shard_mass : mass_by_shard) {
        in.push_back(shard_mass.const_data_ptr<float>());
    }
    for (std::size_t layer = 0; layer < layout.layers.size(); layer++) {
        if (layout.layers[layer].mass_heads == 0) {
            continue;
        }
        auto pages = static_cast<std::size_t>(cu_view_pages[layer * row_size + row_size - 1]);
        for (std::size_t page = 0; page < pages; page++) {
            for (std::size_t shard = 0; shard < in.size(); shard++) {
                std::size_t heads = shards[shard].mass_heads[layer];
                out = std::copy_n(in[shard], heads, out);
                in[shard] += heads;
            }
        }
        out = std::copy_n(in.front(), pages, out);
        for (const float*& shard_in : in) {
            shard_in += pages;
        }
    }
}

// An entry of the second run measured each query token's mass with weight 1; its accepted query tokens are weighed as
// one step of accepted tokens.
void executor::batch::add_measured_mass(
    std::size_t index,
    request_state& req,
    std::size_t start_position,
    std::size_t accepted_count,
    const view_layout& layout
) const {
    const batch_entry& e = entries[index];
    const std::size_t n_layers = layout.layers.size();
    const std::size_t first_view = count_views_before(index);
    std::vector<layer_step_mass> step_mass;
    std::vector<std::vector<float>> verified_mass(n_layers);
    std::vector<std::vector<float>> verified_weight(n_layers);
    if (verifies(index)) {
        std::vector<float> weights(accepted_count);
        req.fill_query_weights(start_position, weights);
        step_mass.resize(n_layers);
        for (std::size_t token = 0; token < accepted_count; token++) {
            std::vector<layer_step_mass> token_mass = view_mass(first_view + token, layout);
            for (std::size_t layer = 0; layer < n_layers; layer++) {
                if (layout.layers[layer].mass_heads == 0) {
                    continue;
                }
                verified_mass[layer].resize(token_mass[layer].mass_sum.size());
                verified_weight[layer].resize(token_mass[layer].weight_sum.size());
                for (std::size_t j = 0; j < verified_mass[layer].size(); j++) {
                    verified_mass[layer][j] += weights[token] * token_mass[layer].mass_sum[j];
                }
                for (std::size_t j = 0; j < verified_weight[layer].size(); j++) {
                    verified_weight[layer][j] += weights[token] * token_mass[layer].weight_sum[j];
                }
                step_mass[layer] = layer_step_mass{
                    .mass_sum = verified_mass[layer],
                    .weight_sum = verified_weight[layer],
                };
            }
        }
    } else {
        step_mass = view_mass(first_view, layout);
    }
    std::vector<std::span<const float>> step_mass_sums;
    std::vector<std::span<const float>> step_weight_sums;
    for (const layer_step_mass& layer_mass : step_mass) {
        step_mass_sums.push_back(layer_mass.mass_sum);
        step_weight_sums.push_back(layer_mass.weight_sum);
    }
    req.add_step_mass(step_mass_sums, step_weight_sums, start_position + accepted_count);
}

}  // namespace pulsarlm
