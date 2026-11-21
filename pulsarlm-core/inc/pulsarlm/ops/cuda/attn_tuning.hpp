#pragma once

#include <ATen/cuda/CUDAContext.h>

#include <type_traits>
#include <utility>

// Launch shapes for the tensor-core attention kernels. Every value here is
// device specific; the kernels take them as template parameters and name none of
// them themselves.
//
// Each row is a separate instantiation of every kernel that reads it, so a row
// only earns its place once the device has been measured.

namespace pulsarlm::ops::cuda::attn {

struct attn_tuning {
    int sm_arch;  // major * 10 + minor; any_arch matches any device
    int prefill_warps;  // warps per CTA of attn_prefill_tc
    int prefill_key_pages;  // pages one prefill key tile spans
    int prefill_ctas_per_sm;  // prefill CTAs that must stay resident per SM
    int decode_warps;  // warps per CTA of the decode kernels
};

constexpr int any_arch = -1;

// Rows are searched in order and the last one must be any_arch. Its
// prefill_ctas_per_sm is 1, which states no occupancy floor and leaves the
// register budget to ptxas: an unmeasured device inherits the tile shape but not
// a register limit that was fitted to another device's shared-memory budget.
//
// A row that states no occupancy floor must keep prefill_warps at 4: at 4 warps
// the time is flat across every prefill_ctas_per_sm, while 8 warps only reaches
// the same time when a floor of 4 forces its register use down.
constexpr attn_tuning attn_tuning_table[] = {
    // sm_arch, prefill_warps, prefill_key_pages, prefill_ctas_per_sm, decode_warps
    {120, 4, 2, 1, 8},
    {any_arch, 4, 2, 1, 8},
};

constexpr int attn_tuning_rows = static_cast<int>(sizeof(attn_tuning_table) / sizeof(attn_tuning_table[0]));

static_assert(attn_tuning_table[attn_tuning_rows - 1].sm_arch == any_arch, "the last tuning row must match any device");

inline int current_sm_arch() {
    const auto& prop = *at::cuda::getCurrentDeviceProperties();
    return prop.major * 10 + prop.minor;
}

// Row this arch selects: the first exact match, else the fallback row.
inline int attn_tuning_row(int sm_arch) {
    for (int row = 0; row + 1 < attn_tuning_rows; ++row) {
        if (attn_tuning_table[row].sm_arch == sm_arch) {
            return row;
        }
    }
    return attn_tuning_rows - 1;
}

template <int candidate_row, typename Fn> void dispatch_tuning_row(int row, Fn&& fn) {
    if constexpr (candidate_row + 1 < attn_tuning_rows) {
        if (row != candidate_row) {
            dispatch_tuning_row<candidate_row + 1>(row, std::forward<Fn>(fn));
            return;
        }
    }
    fn(std::integral_constant<int, candidate_row>{});
}

// Call fn with the compile-time index of the row the current device selects.
template <typename Fn> void dispatch_attn_tuning(Fn&& fn) {
    dispatch_tuning_row<0>(attn_tuning_row(current_sm_arch()), std::forward<Fn>(fn));
}

}  // namespace pulsarlm::ops::cuda::attn
