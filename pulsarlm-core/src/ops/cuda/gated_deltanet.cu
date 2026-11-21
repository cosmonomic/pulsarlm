#include "pulsarlm/ops/cuda/arrival_counters.cuh"
#include "pulsarlm/ops/cuda/dependent_launch.cuh"
#include "pulsarlm/ops/cuda/device.hpp"
#include "pulsarlm/ops/gated_deltanet.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <type_traits>

namespace pulsarlm::ops::cuda {
namespace {

constexpr int conv_threads = 256;
// The most inputs a convolution line keeps: its history and the draft tokens' inputs.
constexpr int max_conv_window = 16;
// Value columns per delta rule CTA.
constexpr int columns = 16;

}  // namespace

namespace kernel {

// Sequence s holds tokens [cu_seqlens_q[s], cu_seqlens_q[s + 1]).
struct ragged_tokens {
    const int32_t* cu_seqlens_q;
    int num_seqs;

    __device__ int first_token(int s) const {
        return cu_seqlens_q[s];
    }

    __device__ int count(int s) const {
        return cu_seqlens_q[s + 1] - cu_seqlens_q[s];
    }

    __device__ int find_sequence(int token) const {
        int lo = 0;
        int hi = num_seqs - 1;
        while (lo < hi) {
            const int mid = (lo + hi) / 2;
            if (cu_seqlens_q[mid + 1] > token) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }
};

// A sequence's lines in a recurrent layer (state_rows): slots lines holding states, then the boundary lines of its even
// and odd pages.
struct state_lines {
    const int32_t* ids;
    int64_t row_stride;
    int slots;
    const int64_t* history_positions;
    const int32_t* rejected;
    int page_size;

    // The slot holding the state before the sequence's first token in the batch.
    __device__ int64_t start(int s) const {
        return ids[s * row_stride + slots - 1 - min(rejected[s], slots - 1)];
    }

    __device__ int64_t slot(int s, int i) const {
        return ids[s * row_stride + i];
    }

    __device__ int64_t last_slot(int s) const {
        return slot(s, slots - 1);
    }

    __device__ int rejected_tokens(int s) const {
        return rejected[s];
    }

    // The sequence's tokens in the batch up to and including the newest of its count tokens that ends a page, 0 when
    // none does.
    __device__ int page_end_tokens(int s, int count) const {
        const int64_t first = history_positions[s];
        const int64_t end = (first + count) / page_size * page_size;
        return end > first ? static_cast<int>(end - first) : 0;
    }

    // The boundary line of the page that the sequence's first page_end tokens in the batch end.
    __device__ int64_t boundary(int s, int page_end) const {
        const int64_t page = (history_positions[s] + page_end) / page_size - 1;
        return ids[s * row_stride + slots + page % 2];
    }
};

template <typename scalar_t> __device__ __forceinline__ float rounded(float value) {
    return static_cast<float>(static_cast<scalar_t>(value));
}

__device__ __forceinline__ float silu(float value) {
    return value / (1.0f + expf(-value));
}

__device__ __forceinline__ float sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

// Sums value over each aligned group of lanes lanes of the warp, lanes a power of two.
template <int lanes> __device__ __forceinline__ float lane_sum(float value) {
    for (int offset = lanes / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

// weight * RMSNorm(o) * activation(z) of each value head's output o.
template <typename scalar_t> struct output_gate {
    const scalar_t* z;
    int64_t z_row_stride;
    const scalar_t* weight;
    float eps;
    gate_activation activation;
};

// y = weight * RMSNorm(x) * activation(z) over dim elements, the normalized x and its product with weight each rounded
// to scalar_t, the CTA's threads striding them. blockDim.x is a power of two and sums holds blockDim.x floats. y may
// be x.
template <typename scalar_t>
__device__ void gate_row(
    const scalar_t* x,
    const scalar_t* __restrict__ z,
    const output_gate<scalar_t>& gate,
    scalar_t* y,
    int dim,
    float* sums
) {
    float local = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = static_cast<float>(x[i]);
        local += v * v;
    }
    sums[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sums[threadIdx.x] += sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(sums[0] / static_cast<float>(dim) + gate.eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float normed = rounded<scalar_t>(static_cast<float>(x[i]) * inv_rms);
        const float scaled = rounded<scalar_t>(static_cast<float>(gate.weight[i]) * normed);
        const float g = static_cast<float>(z[i]);
        y[i] = static_cast<scalar_t>(scaled * (gate.activation == gate_activation::silu ? silu(g) : sigmoid(g)));
    }
}

// A convolution line keeps the last window inputs before a sequence's next token, the history (kernel - 1) *
// dilation the convolution reads and the draft tokens' inputs before it; after a step that did not accept its last
// rejected tokens, the valid inputs end window - rejected entries in.

// One thread per (token, channel). Reads the lines only: conv_state writes them afterwards.
template <typename scalar_t>
__global__ void causal_conv(
    const scalar_t* __restrict__ x,
    int64_t x_row_stride,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ conv_pool,
    int64_t line_stride,
    state_lines lines,
    ragged_tokens tokens,
    int64_t total,
    int channels,
    int kernel,
    int dilation,
    int window,
    scalar_t* __restrict__ out
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total * channels) {
        return;
    }
    const int token = static_cast<int>(index / channels);
    const int channel = static_cast<int>(index % channels);
    const int s = tokens.find_sequence(token);
    const int first = tokens.first_token(s);
    const int history = (kernel - 1) * dilation;
    const int valid = window - lines.rejected_tokens(s);
    const scalar_t* old = conv_pool + lines.start(s) * line_stride + static_cast<int64_t>(channel) * window;
    float sum = 0.0f;
    for (int j = 0; j < kernel; j++) {
        const int input = token - first - history + j * dilation;
        const scalar_t value = input >= 0 ? x[static_cast<int64_t>(first + input) * x_row_stride + channel]
                                          : old[valid + input];
        sum += static_cast<float>(weight[channel * kernel + j]) * static_cast<float>(value);
    }
    out[index] = static_cast<scalar_t>(silu(rounded<scalar_t>(sum)));
}

// The window entry j of the inputs up to count tokens into the sequence's tokens in the batch: the line's valid
// inputs, then the batch's, zero before the first.
template <typename scalar_t> struct conv_inputs {
    const scalar_t* old;
    int valid;
    const scalar_t* x;
    int64_t x_row_stride;
    int first;
    int channel;
    int window;

    __device__ scalar_t entry(int count, int j) const {
        const int index = valid + count - window + j;
        if (index < 0) {
            return scalar_t(0);
        }
        return index < valid ? old[index] : x[static_cast<int64_t>(first + index - valid) * x_row_stride + channel];
    }
};

// One thread per (sequence, channel). Writes the window after the sequence's last token in the batch, and the one
// after the token ending the newest page into that page's boundary line.
template <typename scalar_t>
__global__ void conv_state(
    const scalar_t* __restrict__ x,
    int64_t x_row_stride,
    scalar_t* __restrict__ conv_pool,
    int64_t line_stride,
    state_lines lines,
    ragged_tokens tokens,
    int channels,
    int window
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<int64_t>(tokens.num_seqs) * channels) {
        return;
    }
    const int s = static_cast<int>(index / channels);
    const int channel = static_cast<int>(index % channels);
    const int64_t channel_offset = static_cast<int64_t>(channel) * window;
    scalar_t old[max_conv_window];
    const scalar_t* start = conv_pool + lines.start(s) * line_stride + channel_offset;
    for (int i = 0; i < window; i++) {
        old[i] = start[i];
    }
    const conv_inputs<scalar_t> inputs{
        .old = old,
        .valid = window - lines.rejected_tokens(s),
        .x = x,
        .x_row_stride = x_row_stride,
        .first = tokens.first_token(s),
        .channel = channel,
        .window = window,
    };
    const int n = tokens.count(s);
    const int page_end = lines.page_end_tokens(s, n);
    if (page_end > 0) {
        scalar_t* boundary = conv_pool + lines.boundary(s, page_end) * line_stride + channel_offset;
        for (int j = 0; j < window; j++) {
            boundary[j] = inputs.entry(page_end, j);
        }
    }
    scalar_t* last = conv_pool + lines.last_slot(s) * line_stride + channel_offset;
    for (int j = 0; j < window; j++) {
        last[j] = inputs.entry(n, j);
    }
}

// One thread per (sequence, channel): causal_conv's output for each of the sequence's tokens in turn, then
// conv_state's update of the sequence's lines.
template <typename scalar_t>
__global__ void conv_step(
    const scalar_t* __restrict__ x,
    int64_t x_row_stride,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ conv_pool,
    int64_t line_stride,
    state_lines lines,
    ragged_tokens tokens,
    int channels,
    int kernel,
    int dilation,
    int window,
    scalar_t* __restrict__ out
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<int64_t>(tokens.num_seqs) * channels) {
        return;
    }
    const int s = static_cast<int>(index / channels);
    const int channel = static_cast<int>(index % channels);
    const int history = (kernel - 1) * dilation;
    const int64_t channel_offset = static_cast<int64_t>(channel) * window;
    scalar_t old[max_conv_window];
    const scalar_t* start = conv_pool + lines.start(s) * line_stride + channel_offset;
    for (int i = 0; i < window; i++) {
        old[i] = start[i];
    }
    const int first = tokens.first_token(s);
    const conv_inputs<scalar_t> inputs{
        .old = old,
        .valid = window - lines.rejected_tokens(s),
        .x = x,
        .x_row_stride = x_row_stride,
        .first = first,
        .channel = channel,
        .window = window,
    };
    const int n = tokens.count(s);
    for (int t = 0; t < n; t++) {
        float sum = 0.0f;
        for (int j = 0; j < kernel; j++) {
            sum += static_cast<float>(weight[channel * kernel + j]) *
                static_cast<float>(inputs.entry(t + 1, window - 1 - history + j * dilation));
        }
        out[static_cast<int64_t>(first + t) * channels + channel] = static_cast<scalar_t>(silu(rounded<scalar_t>(sum)));
    }
    const int page_end = lines.page_end_tokens(s, n);
    if (page_end > 0) {
        scalar_t* boundary = conv_pool + lines.boundary(s, page_end) * line_stride + channel_offset;
        for (int j = 0; j < window; j++) {
            boundary[j] = inputs.entry(page_end, j);
        }
    }
    scalar_t* last = conv_pool + lines.last_slot(s) * line_stride + channel_offset;
    for (int j = 0; j < window; j++) {
        last[j] = inputs.entry(n, j);
    }
}

struct delta_rule_params {
    const float* dt_bias;
    const float* neg_a;
    float* state_pool;
    int64_t line_stride;
    state_lines lines;
    ragged_tokens tokens;
    int key_heads;
    int value_heads;
    int value_dim;
    float key_dim_sqrt;
    int64_t gate_row_stride;
};

// A CTA's place in the batch: sequence s, value head head, value columns [first_column, first_column + columns).
struct delta_rule_block {
    int s;
    int head;
    int first_column;
    int key_head;
    int key_dim;
    int channels;
    int64_t state_offset;

    __device__ explicit delta_rule_block(const delta_rule_params& p, int key_head_dim) {
        const int column_tiles = p.value_dim / columns;
        s = blockIdx.x / (p.value_heads * column_tiles);
        head = blockIdx.x / column_tiles % p.value_heads;
        first_column = blockIdx.x % column_tiles * columns;
        key_head = head / (p.value_heads / p.key_heads);
        key_dim = p.key_heads * key_head_dim;
        channels = 2 * key_dim + p.value_heads * p.value_dim;
        state_offset = static_cast<int64_t>(head) * key_head_dim * p.value_dim + first_column;
    }

    // The block's part of the state in line.
    __device__ float* state(const delta_rule_params& p, int64_t line) const {
        return p.state_pool + line * p.line_stride + state_offset;
    }
};

// log(gain) = neg_a * softplus(a + dt_bias), Hugging Face's softplus threshold 20 included.
__device__ __forceinline__ float compute_log_decay(float a, const delta_rule_params& p, int head) {
    const float shifted = a + p.dt_bias[head];
    return (shifted > 20.0f ? shifted : log1pf(expf(shifted))) * p.neg_a[head];
}

template <typename scalar_t> __device__ __forceinline__ float compute_beta(scalar_t b) {
    return rounded<scalar_t>(sigmoid(static_cast<float>(b)));
}

// fn(i) for each i < COUNT, spread over the CTA's THREADS threads, its loop unrolled so that loads issue together.
template <int count, int threads, typename Fn> __device__ __forceinline__ void for_each_index(Fn&& fn) {
#pragma unroll
    for (int r = 0; r < (count + threads - 1) / threads; r++) {
        const int i = r * threads + static_cast<int>(threadIdx.x);
        if (count % threads == 0 || i < count) {
            fn(i);
        }
    }
}

}  // namespace kernel

namespace {

constexpr int step_lanes = 8;
constexpr int step_threads = columns * step_lanes;

}  // namespace

namespace kernel {

// tokens.count(s) tokens per sequence, one after another, the state after each of the last kept_state_counts[s] tokens
// written into its slot. One CTA per (sequence, value head, run of value columns); step_lanes lanes keep a column, lane
// l of them its rows 32 i + 4 l + e, e < 4. The last CTA of a (sequence, value head) to finish gates the head's
// outputs.
template <typename scalar_t, int key_dim>
__global__ void __launch_bounds__(step_threads) delta_rule_step(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    delta_rule_params p,
    const int32_t* __restrict__ kept_state_counts,
    output_gate<scalar_t> output,
    int32_t* __restrict__ head_counters,
    scalar_t* out
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    constexpr int quads = key_dim / 32;
    const delta_rule_block block(p, key_dim);
    const int block_column = threadIdx.x / step_lanes;
    const int row_lane = threadIdx.x % step_lanes;
    const int column = block.first_column + block_column;
    const int first = p.tokens.first_token(block.s);
    const int count = p.tokens.count(block.s);
    const int kept = kept_state_counts[block.s];
    CUDA_KERNEL_ASSERT(kept >= 1 && kept <= count && kept <= p.lines.slots);
    auto compute_row = [&](int quad, int e) { return 32 * quad + 4 * row_lane + e; };
    auto head_out = [&](int t) {
        return out + (static_cast<int64_t>(first + t) * p.value_heads + block.head) * p.value_dim;
    };
    auto store = [&](float* line, const float (&values)[quads][4]) {
#pragma unroll
        for (int quad = 0; quad < quads; quad++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                line[static_cast<int64_t>(compute_row(quad, e)) * p.value_dim] = values[quad][e];
            }
        }
    };

    float state[quads][4];
    const float* start = block.state(p, p.lines.start(block.s)) + block_column;
#pragma unroll
    for (int quad = 0; quad < quads; quad++) {
#pragma unroll
        for (int e = 0; e < 4; e++) {
            state[quad][e] = start[static_cast<int64_t>(compute_row(quad, e)) * p.value_dim];
        }
    }
    for (int t = 0; t < count; t++) {
        const int64_t token = first + t;
        const scalar_t* row = x + token * block.channels;
        float q[quads][4];
        float k[quads][4];
        float qq = 0.0f;
        float kk = 0.0f;
        float qk = 0.0f;
#pragma unroll
        for (int quad = 0; quad < quads; quad++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int dim = block.key_head * key_dim + compute_row(quad, e);
                q[quad][e] = static_cast<float>(row[dim]);
                k[quad][e] = static_cast<float>(row[block.key_dim + dim]);
                qq += q[quad][e] * q[quad][e];
                kk += k[quad][e] * k[quad][e];
                qk += q[quad][e] * k[quad][e];
            }
        }
        const float v = static_cast<float>(row[2 * block.key_dim + block.head * p.value_dim + column]);
        const int64_t gate = token * p.gate_row_stride + block.head;
        const float gain = expf(compute_log_decay(static_cast<float>(a[gate]), p, block.head));
        const float beta = compute_beta(b[gate]);
        const float q_scale = rsqrtf(lane_sum<step_lanes>(qq) + 1e-6f) / p.key_dim_sqrt;
        const float k_scale = rsqrtf(lane_sum<step_lanes>(kk) + 1e-6f);
        qk = lane_sum<step_lanes>(qk) * q_scale * k_scale;

        float k_state = 0.0f;
        float q_state = 0.0f;
#pragma unroll
        for (int quad = 0; quad < quads; quad++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                k[quad][e] *= k_scale;
                k_state += k[quad][e] * state[quad][e];
                q_state += q[quad][e] * q_scale * state[quad][e];
            }
        }
        k_state = lane_sum<step_lanes>(k_state);
        q_state = lane_sum<step_lanes>(q_state);
        const float delta = (v - gain * k_state) * beta;
#pragma unroll
        for (int quad = 0; quad < quads; quad++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                state[quad][e] = state[quad][e] * gain + k[quad][e] * delta;
            }
        }
        if (t >= count - kept) {
            store(block.state(p, p.lines.slot(block.s, p.lines.slots - count + t)) + block_column, state);
        }
        if (p.lines.page_end_tokens(block.s, t + 1) == t + 1) {
            store(block.state(p, p.lines.boundary(block.s, t + 1)) + block_column, state);
        }
        if (row_lane == 0) {
            head_out(t)[column] = static_cast<scalar_t>(gain * q_state + qk * delta);
        }
    }
    const int64_t head_row = static_cast<int64_t>(block.s) * p.value_heads + block.head;
    if (last_to_arrive(head_counters + head_row, p.value_dim / columns)) {
        __shared__ float sums[step_threads];
        for (int t = 0; t < count; t++) {
            const scalar_t* z = output.z + (first + t) * output.z_row_stride + block.head * p.value_dim;
            gate_row(head_out(t), z, output, head_out(t), p.value_dim, sums);
            __syncthreads();
        }
    }
}

}  // namespace kernel

namespace {

constexpr int chunk_threads = 256;
constexpr int solve_lanes = 8;

}  // namespace

namespace kernel {

// The chunked CTA's shared memory, in floats. Query and key rows lie row_stride apart.
template <int key_dim> struct delta_rule_chunk_tile {
    static constexpr int chunk = key_dim > 128 ? 16 : 32;
    static constexpr int row_stride = key_dim + 4;
    static constexpr int queries = 0;
    static constexpr int keys = queries + chunk * row_stride;
    static constexpr int state = keys + chunk * row_stride;
    static constexpr int deltas = state + key_dim * columns;
    static constexpr int remembered = deltas + chunk * columns;
    static constexpr int outputs = remembered + chunk * columns;
    static constexpr int key_key = outputs + chunk * columns;
    static constexpr int query_key = key_key + chunk * chunk;
    static constexpr int decay = query_key + chunk * chunk;
    static constexpr int beta = decay + chunk;
    static constexpr int tail_decay = beta + chunk;
    static constexpr size_t bytes = (tail_decay + chunk) * sizeof(float);
};

__device__ __forceinline__ float component(const float4& v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

__device__ __forceinline__ float4& vector_at(float* data, int index) {
    return *reinterpret_cast<float4*>(data + index);
}

__device__ __forceinline__ float dot(const float4& l, const float4& r) {
    return l.x * r.x + l.y * r.y + l.z * r.z + l.w * r.w;
}

// One CTA per (sequence, value head, run of value columns), over the sequence's tokens in chunks that end at its
// page end. With cum_i the summed log decay of the chunk's tokens up to i and S the state before the chunk, its deltas
// are u_i = beta_i (v_i - e^cum_i k_i.S - sum_{j < i} e^(cum_i - cum_j) (k_i.k_j) u_j), its outputs
// o_i = e^cum_i q_i.S + sum_{j <= i} e^(cum_i - cum_j) (q_i.k_j) u_j and the state after it
// e^cum_last S + sum_j e^(cum_last - cum_j) k_j u_j.
template <typename scalar_t, int key_dim>
__global__ void __launch_bounds__(chunk_threads) delta_rule_chunk(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    delta_rule_params p,
    scalar_t* __restrict__ out
) {
    wait_for_prior_kernels();
    allow_next_kernel();
    using tile = delta_rule_chunk_tile<key_dim>;
    constexpr int chunk = tile::chunk;
    constexpr int row_stride = tile::row_stride;
    extern __shared__ float4 shared_vectors[];
    float* shared = reinterpret_cast<float*>(shared_vectors);
    float* queries = shared + tile::queries;
    float* keys = shared + tile::keys;
    float* state = shared + tile::state;
    float* deltas = shared + tile::deltas;
    float* remembered = shared + tile::remembered;
    float* outputs = shared + tile::outputs;
    float* key_key = shared + tile::key_key;
    float* query_key = shared + tile::query_key;
    float* decay = shared + tile::decay;
    float* beta = shared + tile::beta;
    float* tail_decay = shared + tile::tail_decay;

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const delta_rule_block block(p, key_dim);
    const int first = p.tokens.first_token(block.s);
    const int n = p.tokens.count(block.s);
    const int page_end = p.lines.page_end_tokens(block.s, n);
    auto line_index = [&](int i) {
        return static_cast<int64_t>(i / (columns / 4)) * p.value_dim + i % (columns / 4) * 4;
    };
    auto store_state = [&](float* line) {
        for_each_index<key_dim * columns / 4, chunk_threads>([&](int i) {
            *reinterpret_cast<float4*>(line + line_index(i)) = vector_at(state, 4 * i);
        });
    };

    constexpr int state_items = chunk / 4 * (columns / 2);
    constexpr int pair_items = chunk / 4 * (chunk / 4 + 1);
    int tokens = 0;
    // 2 tokens x 4 keys of the decayed k_i.k_j (j < i), stored at [j][i], or q_i.k_j (j <= i), stored at [i][j].
    // Row pair r covers tokens 2r and 2r + 1, which reach key quads 0 to (2r + 1) / 4.
    auto decayed_pairs = [&](bool key_rows, int pair) {
        int r = 0;
        while (pair >= (2 * r + 1) / 4 + 1) {
            pair -= (2 * r + 1) / 4 + 1;
            r++;
        }
        const int i0 = 2 * r;
        const int j0 = 4 * pair;
        if (i0 >= tokens) {
            return;
        }
        const float* rows = key_rows ? keys : queries;
        float sum[2][4] = {};
#pragma unroll 4
        for (int d = 0; d < key_dim; d += 4) {
            const float4 row[2] = {
                *reinterpret_cast<const float4*>(rows + i0 * row_stride + d),
                *reinterpret_cast<const float4*>(rows + (i0 + 1) * row_stride + d),
            };
            for (int j = 0; j < 4; j++) {
                const float4 key = *reinterpret_cast<const float4*>(keys + (j0 + j) * row_stride + d);
                sum[0][j] += dot(row[0], key);
                sum[1][j] += dot(row[1], key);
            }
        }
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 4; j++) {
                const int row_token = i0 + i;
                const int key_token = j0 + j;
                const bool kept = key_rows ? key_token < row_token : key_token <= row_token;
                const float value = kept ? expf(decay[row_token] - decay[key_token]) * sum[i][j] : 0.0f;
                if (key_rows) {
                    key_key[key_token * chunk + row_token] = value;
                } else {
                    query_key[row_token * chunk + key_token] = value;
                }
            }
        }
    };

    const float* start = block.state(p, p.lines.start(block.s));
    for_each_index<key_dim * columns / 4, chunk_threads>([&](int i) {
        vector_at(state, 4 * i) = *reinterpret_cast<const float4*>(start + line_index(i));
    });
    for (int chunk_first = 0; chunk_first < n;) {
        const int chunk_end = min(min(chunk_first + chunk, n), chunk_first < page_end ? page_end : n);
        tokens = chunk_end - chunk_first;
        const scalar_t* chunk_x = x + static_cast<int64_t>(first + chunk_first) * block.channels;
        for_each_index<chunk * key_dim, chunk_threads>([&](int i) {
            const int t = i / key_dim;
            const int d = i % key_dim;
            const scalar_t* query = chunk_x + static_cast<int64_t>(t) * block.channels + block.key_head * key_dim + d;
            queries[t * row_stride + d] = t < tokens ? static_cast<float>(query[0]) : 0.0f;
            keys[t * row_stride + d] = t < tokens ? static_cast<float>(query[block.key_dim]) : 0.0f;
        });
        for_each_index<chunk * columns, chunk_threads>([&](int i) {
            const int t = i / columns;
            const int64_t value = static_cast<int64_t>(t) * block.channels + 2 * block.key_dim +
                block.head * p.value_dim + block.first_column + i % columns;
            deltas[i] = t < tokens ? static_cast<float>(chunk_x[value]) : 0.0f;
        });
        if (threadIdx.x < chunk) {
            const int t = threadIdx.x;
            const int64_t gate = static_cast<int64_t>(first + chunk_first + t) * p.gate_row_stride + block.head;
            decay[t] = t < tokens ? compute_log_decay(static_cast<float>(a[gate]), p, block.head) : 0.0f;
            beta[t] = t < tokens ? compute_beta(b[gate]) : 0.0f;
        }
        __syncthreads();

        if (warp == 0) {
            float cum = lane < chunk ? decay[lane] : 0.0f;
            for (int offset = 1; offset < 32; offset <<= 1) {
                const float before = __shfl_up_sync(0xffffffffu, cum, offset);
                if (lane >= offset) {
                    cum += before;
                }
            }
            if (lane < chunk) {
                decay[lane] = cum;
            }
        }
#pragma unroll
        for (int u = 0; u < chunk / (chunk_threads / 32); u++) {
            const int t = u * (chunk_threads / 32) + warp;
            if (t >= tokens) {
                break;
            }
            float qq = 0.0f;
            float kk = 0.0f;
            for (int d = lane; d < key_dim; d += 32) {
                qq += queries[t * row_stride + d] * queries[t * row_stride + d];
                kk += keys[t * row_stride + d] * keys[t * row_stride + d];
            }
            const float q_scale = rsqrtf(lane_sum<32>(qq) + 1e-6f) / p.key_dim_sqrt;
            const float k_scale = rsqrtf(lane_sum<32>(kk) + 1e-6f);
            for (int d = lane; d < key_dim; d += 32) {
                queries[t * row_stride + d] = queries[t * row_stride + d] * q_scale;
                keys[t * row_stride + d] = keys[t * row_stride + d] * k_scale;
            }
        }
        __syncthreads();

        if (threadIdx.x < chunk) {
            const int t = threadIdx.x;
            tail_decay[t] = t < tokens ? expf(decay[tokens - 1] - decay[t]) : 0.0f;
        }
        // Items: 4 tokens x 2 columns of e^cum_i k_i.S or e^cum_i q_i.S, then the decayed k_i.k_j with j < i.
        for (int item = threadIdx.x; item < 2 * state_items + pair_items; item += blockDim.x) {
            if (item >= 2 * state_items) {
                decayed_pairs(true, item - 2 * state_items);
                continue;
            }
            const int t0 = item % state_items / (columns / 2) * 4;
            const int c0 = item % (columns / 2) * 2;
            if (t0 >= tokens) {
                continue;
            }
            const float* rows = item < state_items ? keys : queries;
            float sum[4][2] = {};
#pragma unroll 4
            for (int d = 0; d < key_dim; d += 4) {
                float4 row[4];
                for (int i = 0; i < 4; i++) {
                    row[i] = *reinterpret_cast<const float4*>(rows + (t0 + i) * row_stride + d);
                }
                for (int e = 0; e < 4; e++) {
                    const float2 column = *reinterpret_cast<const float2*>(state + (d + e) * columns + c0);
                    for (int i = 0; i < 4; i++) {
                        sum[i][0] += component(row[i], e) * column.x;
                        sum[i][1] += component(row[i], e) * column.y;
                    }
                }
            }
            float* target = item < state_items ? remembered : outputs;
            for (int i = 0; i < 4; i++) {
                const float gain = expf(decay[t0 + i]);
                target[(t0 + i) * columns + c0] = gain * sum[i][0];
                target[(t0 + i) * columns + c0 + 1] = gain * sum[i][1];
            }
        }
        __syncthreads();

        // solve_lanes lanes per column solve its deltas, lane g of them keeping rows g + solve_lanes m, while the other
        // warps compute the decayed q_i.k_j with j <= i.
        if (threadIdx.x < columns * solve_lanes) {
            const int c = threadIdx.x / solve_lanes;
            const int g = threadIdx.x % solve_lanes;
            constexpr int owned = chunk / solve_lanes;
            float residual[owned];
#pragma unroll
            for (int m = 0; m < owned; m++) {
                const int i = g + solve_lanes * m;
                residual[m] = deltas[i * columns + c] - remembered[i * columns + c];
            }
#pragma unroll
            for (int j = 0; j < chunk; j++) {
                if (j >= tokens) {
                    break;
                }
                const float
                    delta = __shfl_sync(0xffffffffu, beta[j] * residual[j / solve_lanes], j % solve_lanes, solve_lanes);
                if (g == j % solve_lanes) {
                    deltas[j * columns + c] = delta;
                }
#pragma unroll
                for (int m = 0; m < owned; m++) {
                    const int i = g + solve_lanes * m;
                    if (i > j) {
                        residual[m] -= key_key[j * chunk + i] * delta;
                    }
                }
            }
        } else {
            for (int item = threadIdx.x - columns * solve_lanes; item < pair_items;
                 item += chunk_threads - columns * solve_lanes) {
                decayed_pairs(false, item);
            }
        }
        __syncthreads();

        // Items: 4 rows x 4 columns of the state, then 1 token x 4 columns of the outputs.
        constexpr int update_items = key_dim / 4 * (columns / 4);
        const float chunk_decay = expf(decay[tokens - 1]);
        for (int item = threadIdx.x; item < update_items + chunk * (columns / 4); item += blockDim.x) {
            if (item < update_items) {
                const int r0 = item / (columns / 4) * 4;
                const int c0 = item % (columns / 4) * 4;
                float4 sum[4];
                for (int i = 0; i < 4; i++) {
                    const float4 old = vector_at(state, (r0 + i) * columns + c0);
                    sum[i] = make_float4(
                        old.x * chunk_decay,
                        old.y * chunk_decay,
                        old.z * chunk_decay,
                        old.w * chunk_decay
                    );
                }
#pragma unroll 4
                for (int j = 0; j < tokens; j++) {
                    const float4 key = *reinterpret_cast<const float4*>(keys + j * row_stride + r0);
                    const float4 delta = vector_at(deltas, j * columns + c0);
                    for (int i = 0; i < 4; i++) {
                        const float weight = tail_decay[j] * component(key, i);
                        sum[i].x += weight * delta.x;
                        sum[i].y += weight * delta.y;
                        sum[i].z += weight * delta.z;
                        sum[i].w += weight * delta.w;
                    }
                }
                for (int i = 0; i < 4; i++) {
                    vector_at(state, (r0 + i) * columns + c0) = sum[i];
                }
            } else {
                const int i = (item - update_items) / (columns / 4);
                const int c0 = (item - update_items) % (columns / 4) * 4;
                if (i >= tokens) {
                    continue;
                }
                float4 sum = vector_at(outputs, i * columns + c0);
                for (int j = 0; j <= i; j++) {
                    const float weight = query_key[i * chunk + j];
                    const float4 delta = vector_at(deltas, j * columns + c0);
                    sum.x += weight * delta.x;
                    sum.y += weight * delta.y;
                    sum.z += weight * delta.z;
                    sum.w += weight * delta.w;
                }
                vector_at(outputs, i * columns + c0) = sum;
            }
        }
        __syncthreads();

        for (int i = threadIdx.x; i < tokens * columns; i += blockDim.x) {
            const int64_t token = first + chunk_first + i / columns;
            out[(token * p.value_heads + block.head) * p.value_dim + block.first_column +
                i % columns] = static_cast<scalar_t>(outputs[i]);
        }
        if (chunk_end == page_end) {
            store_state(block.state(p, p.lines.boundary(block.s, page_end)));
        }
        chunk_first = chunk_end;
    }
    store_state(block.state(p, p.lines.last_slot(block.s)));
}

// One CTA per (token, value head) of out [tokens, value heads, value dim], gated in place. blockDim.x is a power of
// two.
template <typename scalar_t>
__global__ void gate_outputs(scalar_t* out, output_gate<scalar_t> gate, int value_heads, int value_dim) {
    wait_for_prior_kernels();
    allow_next_kernel();
    extern __shared__ float sums[];
    const int64_t token = blockIdx.x / value_heads;
    const int head = static_cast<int>(blockIdx.x % value_heads);
    scalar_t* row = out + static_cast<int64_t>(blockIdx.x) * value_dim;
    gate_row(row, gate.z + token * gate.z_row_stride + head * value_dim, gate, row, value_dim, sums);
}

}  // namespace kernel

namespace {

int blocks_for(int64_t threads) {
    return static_cast<int>((threads + conv_threads - 1) / conv_threads);
}

void check_rows(
    const char* op,
    const at::Tensor& lines,
    std::int64_t slots,
    const at::Tensor& history_positions,
    const at::Tensor& rejected_tokens,
    std::int64_t page_size
) {
    TORCH_CHECK(
        slots >= 1 && page_size >= 1 && lines.is_cuda() && lines.scalar_type() == at::kInt && lines.dim() == 2 &&
            lines.size(1) >= slots + 2 && lines.stride(1) == 1,
        op,
        ": lines must be CUDA int32 [num_seqs, >= slots + 2] with its columns adjacent, slots and page_size at "
        "least 1"
    );
    const int64_t num_seqs = lines.size(0);
    TORCH_CHECK(
        slots <= int32_max && page_size <= int32_max && num_seqs <= int32_max,
        op,
        ": slots, page_size and the sequences must fit 32 bits"
    );
    TORCH_CHECK(
        history_positions.is_cuda() && history_positions.scalar_type() == at::kLong && history_positions.dim() == 1 &&
            history_positions.size(0) == num_seqs && history_positions.is_contiguous(),
        op,
        ": history_positions must be contiguous CUDA int64 [num_seqs]"
    );
    TORCH_CHECK(
        rejected_tokens.is_cuda() && rejected_tokens.scalar_type() == at::kInt && rejected_tokens.dim() == 1 &&
            rejected_tokens.size(0) == num_seqs && rejected_tokens.is_contiguous(),
        op,
        ": rejected_tokens must be contiguous CUDA int32 [num_seqs]"
    );
}

// x's tokens split into the sequences of lines by cu_seqlens_q.
void check_tokens(const char* op, const at::Tensor& x, const at::Tensor& lines, const at::Tensor& cu_seqlens_q) {
    TORCH_CHECK(
        x.is_cuda() && x.dim() == 2 && x.stride(1) == 1,
        op,
        ": x must be CUDA [tokens, dim], each row contiguous"
    );
    const int64_t num_seqs = lines.size(0);
    TORCH_CHECK(
        cu_seqlens_q.is_cuda() && cu_seqlens_q.scalar_type() == at::kInt && cu_seqlens_q.dim() == 1 &&
            cu_seqlens_q.size(0) == num_seqs + 1 && cu_seqlens_q.is_contiguous(),
        op,
        ": cu_seqlens_q must be contiguous CUDA int32 [num_seqs + 1]"
    );
}

// out when given, checked to be contiguous of sizes in x's dtype on x's device, or a new such tensor.
at::Tensor
make_output(const char* op, const std::optional<at::Tensor>& out, at::IntArrayRef sizes, const at::Tensor& x) {
    if (!out.has_value()) {
        return at::empty(sizes, x.options());
    }
    TORCH_CHECK(
        out->sizes() == sizes && out->scalar_type() == x.scalar_type() && out->device() == x.device() &&
            out->is_contiguous(),
        op,
        ": out must be contiguous ",
        sizes,
        " in x's dtype on x's device"
    );
    return *out;
}

kernel::ragged_tokens ragged(const at::Tensor& lines, const at::Tensor& cu_seqlens_q) {
    return kernel::ragged_tokens{
        cu_seqlens_q.const_data_ptr<int32_t>(),
        static_cast<int>(lines.size(0)),
    };
}

kernel::state_lines make_state_lines(
    const at::Tensor& lines,
    std::int64_t slots,
    const at::Tensor& history_positions,
    const at::Tensor& rejected_tokens,
    std::int64_t page_size
) {
    return kernel::state_lines{
        lines.const_data_ptr<int32_t>(),
        lines.stride(0),
        static_cast<int>(slots),
        history_positions.const_data_ptr<int64_t>(),
        rejected_tokens.const_data_ptr<int32_t>(),
        static_cast<int>(page_size),
    };
}

template <typename Fn> void dispatch_key_dim(int64_t key_dim, Fn&& fn) {
    if (key_dim == 32) {
        fn(std::integral_constant<int, 32>{});
    } else if (key_dim == 64) {
        fn(std::integral_constant<int, 64>{});
    } else if (key_dim == 128) {
        fn(std::integral_constant<int, 128>{});
    } else if (key_dim == 256) {
        fn(std::integral_constant<int, 256>{});
    } else {
        TORCH_CHECK(false, "gated_delta_rule: unsupported key head dim ", key_dim, " (compiled for 32, 64, 128, 256)");
    }
}

}  // namespace

namespace {

enum class recurrence_algorithm { step, chunked };

// The convolution of x's tokens, cu_seqlens_q splitting them into sequences, by kernel::conv_step or by
// kernel::causal_conv and kernel::conv_state.
at::Tensor causal_conv(
    recurrence_algorithm algorithm,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_pool,
    const at::Tensor& lines,
    std::int64_t slots,
    const at::Tensor& history_positions,
    const at::Tensor& rejected_tokens,
    std::int64_t page_size,
    const at::Tensor& cu_seqlens_q,
    int64_t dilation,
    const std::optional<at::Tensor>& out
) {
    check_on_current_device(
        "causal_conv",
        {x, weight, conv_pool, lines, history_positions, rejected_tokens, cu_seqlens_q, out.value_or(at::Tensor())}
    );
    check_rows("causal_conv", lines, slots, history_positions, rejected_tokens, page_size);
    TORCH_CHECK(slots == 1, "causal_conv: slots must be 1");
    check_tokens("causal_conv", x, lines, cu_seqlens_q);
    const int64_t channels = x.size(1);
    TORCH_CHECK(
        weight.scalar_type() == x.scalar_type() && weight.dim() == 3 && weight.size(0) == channels &&
            weight.size(1) == 1 && weight.size(2) >= 1 && weight.is_contiguous(),
        "causal_conv: weight must be contiguous [channels, 1, kernel] in x's dtype"
    );
    const int64_t kernel = weight.size(2);
    const int64_t history = (kernel - 1) * dilation;
    TORCH_CHECK(
        conv_pool.scalar_type() == x.scalar_type() && conv_pool.dim() == 3 && conv_pool.size(1) == channels &&
            conv_pool.stride(2) == 1 && conv_pool.stride(1) == conv_pool.size(2),
        "causal_conv: conv_pool must be [lines, channels, window] in x's dtype, each line packed"
    );
    const int64_t window = conv_pool.size(2);
    TORCH_CHECK(
        dilation >= 1 && history <= window && window <= max_conv_window,
        "causal_conv: the window must hold (kernel - 1) * dilation inputs, dilation at least 1, and at most ",
        max_conv_window
    );
    const int64_t total = x.size(0);
    const int64_t num_seqs = lines.size(0);
    TORCH_CHECK(
        total <= int32_max && channels * kernel <= int32_max && dilation <= int32_max &&
            (std::max(total, num_seqs) * channels + conv_threads - 1) / conv_threads <= int32_max,
        "causal_conv: too many tokens, sequences or channels"
    );
    at::Tensor convolved = make_output("causal_conv", out, x.sizes(), x);
    if (num_seqs == 0) {
        return convolved;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "causal_conv", [&] {
        if (algorithm == recurrence_algorithm::step) {
            launch_dependent(
                kernel::conv_step<scalar_t>,
                blocks_for(num_seqs * channels),
                conv_threads,
                0,
                stream,
                x.const_data_ptr<scalar_t>(),
                x.stride(0),
                weight.const_data_ptr<scalar_t>(),
                conv_pool.mutable_data_ptr<scalar_t>(),
                conv_pool.stride(0),
                make_state_lines(lines, slots, history_positions, rejected_tokens, page_size),
                ragged(lines, cu_seqlens_q),
                static_cast<int>(channels),
                static_cast<int>(kernel),
                static_cast<int>(dilation),
                static_cast<int>(window),
                convolved.mutable_data_ptr<scalar_t>()
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }
        if (total > 0) {
            launch_dependent(
                kernel::causal_conv<scalar_t>,
                blocks_for(total * channels),
                conv_threads,
                0,
                stream,
                x.const_data_ptr<scalar_t>(),
                x.stride(0),
                weight.const_data_ptr<scalar_t>(),
                conv_pool.const_data_ptr<scalar_t>(),
                conv_pool.stride(0),
                make_state_lines(lines, slots, history_positions, rejected_tokens, page_size),
                ragged(lines, cu_seqlens_q),
                total,
                static_cast<int>(channels),
                static_cast<int>(kernel),
                static_cast<int>(dilation),
                static_cast<int>(window),
                convolved.mutable_data_ptr<scalar_t>()
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        launch_dependent(
            kernel::conv_state<scalar_t>,
            blocks_for(num_seqs * channels),
            conv_threads,
            0,
            stream,
            x.const_data_ptr<scalar_t>(),
            x.stride(0),
            conv_pool.mutable_data_ptr<scalar_t>(),
            conv_pool.stride(0),
            make_state_lines(lines, slots, history_positions, rejected_tokens, page_size),
            ragged(lines, cu_seqlens_q),
            static_cast<int>(channels),
            static_cast<int>(window)
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return convolved;
}

// The gated delta rule over x's tokens, cu_seqlens_q splitting them into sequences: by kernel::delta_rule_step,
// which reads kept_state_counts, or by kernel::delta_rule_chunk and kernel::gate_outputs, which do not.
at::Tensor gated_delta_rule(
    recurrence_algorithm algorithm,
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& dt_bias,
    const at::Tensor& neg_a,
    const at::Tensor& state_pool,
    const at::Tensor& lines,
    std::int64_t slots,
    const at::Tensor& history_positions,
    const at::Tensor& rejected_tokens,
    std::int64_t page_size,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& kept_state_counts,
    int64_t key_heads,
    const at::Tensor& z,
    const at::Tensor& norm_weight,
    double eps,
    gate_activation activation,
    const at::Tensor& counters,
    const std::optional<at::Tensor>& out
) {
    check_on_current_device(
        "gated_delta_rule",
        {x,
         a,
         b,
         dt_bias,
         neg_a,
         state_pool,
         lines,
         history_positions,
         rejected_tokens,
         cu_seqlens_q,
         kept_state_counts,
         z,
         norm_weight,
         counters,
         out.value_or(at::Tensor())}
    );
    check_rows("gated_delta_rule", lines, slots, history_positions, rejected_tokens, page_size);
    check_tokens("gated_delta_rule", x, lines, cu_seqlens_q);
    TORCH_CHECK(x.is_contiguous(), "gated_delta_rule: x must be contiguous");
    if (algorithm == recurrence_algorithm::step) {
        TORCH_CHECK(
            kept_state_counts.is_cuda() && kept_state_counts.scalar_type() == at::kInt &&
                kept_state_counts.dim() == 1 && kept_state_counts.size(0) == lines.size(0) &&
                kept_state_counts.is_contiguous(),
            "gated_delta_rule: kept_state_counts must be contiguous CUDA int32 [num_seqs]"
        );
    }
    TORCH_CHECK(
        state_pool.is_cuda() && state_pool.scalar_type() == at::kFloat && state_pool.dim() == 4 &&
            state_pool.stride(3) == 1 && state_pool.stride(2) == state_pool.size(3) &&
            state_pool.stride(1) == state_pool.size(2) * state_pool.size(3) && state_pool.stride(0) % 4 == 0 &&
            reinterpret_cast<uintptr_t>(state_pool.const_data_ptr()) % 16 == 0,
        "gated_delta_rule: state_pool must be fp32 [lines, value heads, key dim, value dim], each line packed and "
        "16-byte aligned"
    );
    const int64_t value_heads = state_pool.size(1);
    const int64_t key_dim = state_pool.size(2);
    const int64_t value_dim = state_pool.size(3);
    TORCH_CHECK(
        key_heads > 0 && value_heads % key_heads == 0,
        "gated_delta_rule: the value heads must be a multiple of the key heads"
    );
    TORCH_CHECK(value_dim % columns == 0, "gated_delta_rule: the value head dim must be a multiple of ", columns);
    TORCH_CHECK(
        x.size(1) == 2 * key_heads * key_dim + value_heads * value_dim,
        "gated_delta_rule: x must be [tokens, 2 * key heads * key dim + value heads * value dim]"
    );
    const int64_t total = x.size(0);
    for (const at::Tensor* gate : {&a, &b}) {
        TORCH_CHECK(
            gate->scalar_type() == x.scalar_type() && gate->dim() == 2 && gate->size(0) == total &&
                gate->size(1) == value_heads && gate->stride(1) == 1 && gate->stride(0) == a.stride(0),
            "gated_delta_rule: a and b must be [tokens, value heads] in x's dtype, each row contiguous, their rows "
            "equally far apart"
        );
    }
    for (const at::Tensor* head_values : {&dt_bias, &neg_a}) {
        TORCH_CHECK(
            head_values->is_cuda() && head_values->scalar_type() == at::kFloat && head_values->dim() == 1 &&
                head_values->size(0) == value_heads && head_values->is_contiguous(),
            "gated_delta_rule: dt_bias and neg_a must be contiguous CUDA fp32 [value heads]"
        );
    }
    TORCH_CHECK(
        z.scalar_type() == x.scalar_type() && z.dim() == 2 && z.size(0) == total &&
            z.size(1) == value_heads * value_dim && z.stride(1) == 1,
        "gated_delta_rule: z must be [tokens, value heads * value dim] in x's dtype, each row contiguous"
    );
    TORCH_CHECK(
        norm_weight.scalar_type() == x.scalar_type() && norm_weight.dim() == 1 && norm_weight.size(0) == value_dim &&
            norm_weight.is_contiguous(),
        "gated_delta_rule: norm_weight must be contiguous [value dim] in x's dtype"
    );
    const int64_t num_seqs = lines.size(0);
    TORCH_CHECK(
        total <= int32_max && x.size(1) <= int32_max && total * value_heads <= int32_max &&
            num_seqs * value_heads * (value_dim / columns) <= int32_max,
        "gated_delta_rule: too many tokens, sequences, heads or channels"
    );
    at::Tensor gated = make_output("gated_delta_rule", out, {total, value_heads * value_dim}, x);
    TORCH_CHECK(
        counters.is_cuda() && counters.device() == x.device() && counters.scalar_type() == at::kInt &&
            counters.is_contiguous() && counters.numel() >= gated_delta_rule_counters(num_seqs, value_heads),
        "gated_delta_rule: counters must be contiguous int32 on x's device, at least gated_delta_rule_counters of them"
    );
    if (num_seqs == 0) {
        return gated;
    }
    const kernel::delta_rule_params params{
        dt_bias.const_data_ptr<float>(),
        neg_a.const_data_ptr<float>(),
        state_pool.mutable_data_ptr<float>(),
        state_pool.stride(0),
        make_state_lines(lines, slots, history_positions, rejected_tokens, page_size),
        ragged(lines, cu_seqlens_q),
        static_cast<int>(key_heads),
        static_cast<int>(value_heads),
        static_cast<int>(value_dim),
        static_cast<float>(std::sqrt(static_cast<double>(key_dim))),
        a.stride(0),
    };
    const auto blocks = static_cast<int>(num_seqs * value_heads * (value_dim / columns));
    int gate_threads = 1;
    while (gate_threads < std::min<int64_t>(1024, value_dim)) {
        gate_threads <<= 1;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "gated_delta_rule", [&] {
        dispatch_key_dim(key_dim, [&](auto key_dim_tag) {
            constexpr int key_dim = decltype(key_dim_tag)::value;
            const scalar_t* x_data = x.const_data_ptr<scalar_t>();
            const scalar_t* a_data = a.const_data_ptr<scalar_t>();
            const scalar_t* b_data = b.const_data_ptr<scalar_t>();
            scalar_t* out_data = gated.mutable_data_ptr<scalar_t>();
            const kernel::output_gate<scalar_t> gate{
                z.const_data_ptr<scalar_t>(),
                z.stride(0),
                norm_weight.const_data_ptr<scalar_t>(),
                static_cast<float>(eps),
                activation,
            };
            if (algorithm == recurrence_algorithm::step) {
                launch_dependent(
                    kernel::delta_rule_step<scalar_t, key_dim>,
                    blocks,
                    step_threads,
                    0,
                    stream,
                    x_data,
                    a_data,
                    b_data,
                    params,
                    kept_state_counts.const_data_ptr<int32_t>(),
                    gate,
                    counters.mutable_data_ptr<int32_t>(),
                    out_data
                );
            } else {
                const auto kernel = kernel::delta_rule_chunk<scalar_t, key_dim>;
                constexpr size_t bytes = kernel::delta_rule_chunk_tile<key_dim>::bytes;
                allow_shared_memory(kernel, bytes);
                launch_dependent(
                    kernel,
                    blocks,
                    chunk_threads,
                    bytes,
                    stream,
                    x_data,
                    a_data,
                    b_data,
                    params,
                    out_data
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                if (total > 0) {
                    launch_dependent(
                        kernel::gate_outputs<scalar_t>,
                        static_cast<int>(total * value_heads),
                        gate_threads,
                        gate_threads * sizeof(float),
                        stream,
                        out_data,
                        gate,
                        static_cast<int>(value_heads),
                        static_cast<int>(value_dim)
                    );
                }
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
    return gated;
}

}  // namespace

at::Tensor causal_conv_step(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_pool,
    const at::Tensor& lines,
    std::int64_t slots,
    const at::Tensor& history_positions,
    const at::Tensor& rejected_tokens,
    std::int64_t page_size,
    const at::Tensor& cu_seqlens_q,
    std::int64_t dilation,
    const std::optional<at::Tensor>& out
) {
    return causal_conv(
        recurrence_algorithm::step,
        x,
        weight,
        conv_pool,
        lines,
        slots,
        history_positions,
        rejected_tokens,
        page_size,
        cu_seqlens_q,
        dilation,
        out
    );
}

at::Tensor causal_conv_chunked(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_pool,
    const at::Tensor& lines,
    std::int64_t slots,
    const at::Tensor& history_positions,
    const at::Tensor& rejected_tokens,
    std::int64_t page_size,
    const at::Tensor& cu_seqlens_q,
    std::int64_t dilation,
    const std::optional<at::Tensor>& out
) {
    return causal_conv(
        recurrence_algorithm::chunked,
        x,
        weight,
        conv_pool,
        lines,
        slots,
        history_positions,
        rejected_tokens,
        page_size,
        cu_seqlens_q,
        dilation,
        out
    );
}

// One per (sequence, value head) of a ops::cuda::gated_delta_rule_step call.
int64_t gated_delta_rule_counters(int64_t num_seqs, int64_t value_heads) {
    return num_seqs * value_heads;
}

at::Tensor gated_delta_rule_step(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& dt_bias,
    const at::Tensor& neg_a,
    const at::Tensor& state_pool,
    const at::Tensor& lines,
    std::int64_t slots,
    const at::Tensor& history_positions,
    const at::Tensor& rejected_tokens,
    std::int64_t page_size,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& kept_state_counts,
    int64_t key_heads,
    const at::Tensor& z,
    const at::Tensor& norm_weight,
    double eps,
    gate_activation activation,
    const at::Tensor& counters,
    const std::optional<at::Tensor>& out
) {
    return gated_delta_rule(
        recurrence_algorithm::step,
        x,
        a,
        b,
        dt_bias,
        neg_a,
        state_pool,
        lines,
        slots,
        history_positions,
        rejected_tokens,
        page_size,
        cu_seqlens_q,
        kept_state_counts,
        key_heads,
        z,
        norm_weight,
        eps,
        activation,
        counters,
        out
    );
}

at::Tensor gated_delta_rule_chunked(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& dt_bias,
    const at::Tensor& neg_a,
    const at::Tensor& state_pool,
    const at::Tensor& lines,
    std::int64_t slots,
    const at::Tensor& history_positions,
    const at::Tensor& rejected_tokens,
    std::int64_t page_size,
    const at::Tensor& cu_seqlens_q,
    int64_t key_heads,
    const at::Tensor& z,
    const at::Tensor& norm_weight,
    double eps,
    gate_activation activation,
    const at::Tensor& counters,
    const std::optional<at::Tensor>& out
) {
    return gated_delta_rule(
        recurrence_algorithm::chunked,
        x,
        a,
        b,
        dt_bias,
        neg_a,
        state_pool,
        lines,
        slots,
        history_positions,
        rejected_tokens,
        page_size,
        cu_seqlens_q,
        at::Tensor(),
        key_heads,
        z,
        norm_weight,
        eps,
        activation,
        counters,
        out
    );
}

}  // namespace pulsarlm::ops::cuda
