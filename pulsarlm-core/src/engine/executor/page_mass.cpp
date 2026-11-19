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
    model::batch& on_device,
    const at::Tensor& host_cu_view_pages,
    const std::vector<std::size_t>& mass_heads
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
    at::Tensor mass = at::zeros({total}, on_device.tokens.options().dtype(at::kFloat));
    on_device.page_mass_sum.resize(mass_heads.size());
    on_device.page_weight_sum.resize(mass_heads.size());
    std::int64_t first = 0;
    for (std::size_t layer = 0; layer < mass_heads.size(); layer++) {
        if (mass_heads[layer] > 0) {
            auto heads = static_cast<std::int64_t>(mass_heads[layer]);
            std::int64_t pages = layer_pages(layer);
            on_device.page_mass_sum[layer] = mass.narrow(0, first, pages * heads).view({pages, heads});
            first += pages * heads;
            on_device.page_weight_sum[layer] = mass.narrow(0, first, pages);
            first += pages;
        }
    }
    return mass;
}

std::vector<executor::batch::layer_step_mass> executor::batch::view_mass(
    const model::batch& group,
    const at::Tensor& mass,
    std::size_t view,
    const view_layout& layout
) {
    const float* values = mass.const_data_ptr<float>();
    auto row_size = static_cast<std::size_t>(group.cu_view_pages.size(1));
    const auto* cu_view_pages = group.cu_view_pages.const_data_ptr<std::int32_t>();
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
    for (group* g : groups()) {
        if (!g->mass_by_shard.front().defined() || g->mass_by_shard.size() == 1) {
            g->mass = g->mass_by_shard.front();
            continue;
        }
        std::int64_t total = 0;
        auto row_size = static_cast<std::size_t>(g->inputs.cu_view_pages.size(1));
        const auto* cu_view_pages = g->inputs.cu_view_pages.const_data_ptr<std::int32_t>();
        for (std::size_t layer = 0; layer < layout.layers.size(); layer++) {
            if (layout.layers[layer].mass_heads > 0) {
                auto pages = static_cast<std::int64_t>(cu_view_pages[layer * row_size + row_size - 1]);
                total += pages * static_cast<std::int64_t>(layout.layers[layer].mass_heads + 1);
            }
        }
        g->mass = pinned({total}, at::kFloat);
        float* out = g->mass.mutable_data_ptr<float>();
        std::vector<const float*> in;
        for (const at::Tensor& mass : g->mass_by_shard) {
            in.push_back(mass.const_data_ptr<float>());
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
}

// A verify entry measured each row's mass with weight 1; its accepted rows are weighed as one step of accepted tokens.
void executor::batch::add_measured_mass(
    const group& g,
    std::size_t position,
    std::size_t accepted_count,
    const view_layout& layout
) const {
    const entry& e = g.entries[position];
    sequence& seq = *e.seq;
    const std::size_t n_layers = layout.layers.size();
    std::vector<layer_step_mass> step_mass;
    std::vector<std::vector<float>> verified_mass(n_layers);
    std::vector<std::vector<float>> verified_weight(n_layers);
    if (&g == &verify_group) {
        std::vector<float> weights(accepted_count);
        seq.fill_query_weights(e.start_position, weights);
        step_mass.resize(n_layers);
        for (std::size_t row = 0; row < accepted_count; row++) {
            std::vector<layer_step_mass> row_mass = view_mass(g.inputs, g.mass, position * verify_rows + row, layout);
            for (std::size_t layer = 0; layer < n_layers; layer++) {
                if (layout.layers[layer].mass_heads == 0) {
                    continue;
                }
                verified_mass[layer].resize(row_mass[layer].mass_sum.size());
                verified_weight[layer].resize(row_mass[layer].weight_sum.size());
                for (std::size_t j = 0; j < verified_mass[layer].size(); j++) {
                    verified_mass[layer][j] += weights[row] * row_mass[layer].mass_sum[j];
                }
                for (std::size_t j = 0; j < verified_weight[layer].size(); j++) {
                    verified_weight[layer][j] += weights[row] * row_mass[layer].weight_sum[j];
                }
                step_mass[layer] = layer_step_mass{
                    .mass_sum = verified_mass[layer],
                    .weight_sum = verified_weight[layer],
                };
            }
        }
    } else {
        step_mass = view_mass(g.inputs, g.mass, position, layout);
    }
    std::vector<std::span<const float>> step_mass_sums;
    std::vector<std::span<const float>> step_weight_sums;
    for (const layer_step_mass& layer_mass : step_mass) {
        step_mass_sums.push_back(layer_mass.mass_sum);
        step_weight_sums.push_back(layer_mass.weight_sum);
    }
    seq.add_step_mass(step_mass_sums, step_weight_sums, e.end_position);
}

}  // namespace pulsarlm
