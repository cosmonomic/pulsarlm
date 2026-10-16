#include "pulsarlm/ops.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <type_traits>

namespace pulsarlm {
namespace {

constexpr int kConvThreads = 256;
constexpr int kMaxConvHistory = 7;
// Value columns per delta rule CTA.
constexpr int kColumns = 16;
constexpr size_t kSharedMemoryWithoutOptIn = 48 * 1024;

// Sequence s holds tokens [first(s), first(s) + count(s)); without cu_seqlens_q, the one token s.
struct ragged_tokens {
    const int32_t* cu_seqlens_q;
    int num_seqs;

    __device__ int first(int s) const {
        return cu_seqlens_q != nullptr ? cu_seqlens_q[s] : s;
    }

    __device__ int count(int s) const {
        return cu_seqlens_q != nullptr ? cu_seqlens_q[s + 1] - cu_seqlens_q[s] : 1;
    }

    __device__ int sequence_of(int token) const {
        if (cu_seqlens_q == nullptr) {
            return token;
        }
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

// Line ids of the recurrent layers' rows, each sequence's running line then its boundary line.
struct line_rows {
    const int32_t* ids;
    int64_t row_stride;

    __device__ int64_t running(int s) const {
        return ids[s * row_stride];
    }

    __device__ int64_t boundary(int s) const {
        return ids[s * row_stride + 1];
    }
};

template <typename scalar_t> __device__ __forceinline__ float rounded(float value) {
    return static_cast<float>(static_cast<scalar_t>(value));
}

__device__ __forceinline__ float silu(float value) {
    return value / (1.0f + expf(-value));
}

// Sums value over each aligned group of lanes lanes of the warp, lanes a power of two.
template <int lanes> __device__ __forceinline__ float lane_sum(float value) {
    for (int offset = lanes / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

// One thread per (token, channel). Reads the running lines only: conv_state_kernel writes them afterwards.
template <typename scalar_t>
__global__ void causal_conv_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ conv_pool,
    int64_t line_stride,
    line_rows rows,
    ragged_tokens tokens,
    int64_t total,
    int channels,
    int kernel,
    scalar_t* __restrict__ out
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total * channels) {
        return;
    }
    const int token = static_cast<int>(index / channels);
    const int channel = static_cast<int>(index % channels);
    const int s = tokens.sequence_of(token);
    const int first = tokens.first(s);
    const int history = kernel - 1;
    const scalar_t* old = conv_pool + rows.running(s) * line_stride + static_cast<int64_t>(channel) * history;
    float sum = 0.0f;
    for (int j = 0; j < kernel; j++) {
        const int input = token - first - history + j;
        const scalar_t value = input >= 0 ? x[static_cast<int64_t>(first + input) * channels + channel]
                                          : old[history + input];
        sum += static_cast<float>(weight[channel * kernel + j]) * static_cast<float>(value);
    }
    out[index] = static_cast<scalar_t>(silu(rounded<scalar_t>(sum)));
}

// One thread per (sequence, channel).
template <typename scalar_t>
__global__ void conv_state_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ conv_pool,
    int64_t line_stride,
    line_rows rows,
    ragged_tokens tokens,
    const int32_t* __restrict__ page_end_tokens,
    int channels,
    int kernel
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<int64_t>(tokens.num_seqs) * channels) {
        return;
    }
    const int s = static_cast<int>(index / channels);
    const int channel = static_cast<int>(index % channels);
    const int first = tokens.first(s);
    const int history = kernel - 1;
    const int64_t channel_offset = static_cast<int64_t>(channel) * history;
    scalar_t* running = conv_pool + rows.running(s) * line_stride + channel_offset;
    scalar_t old[kMaxConvHistory];
    for (int i = 0; i < history; i++) {
        old[i] = running[i];
    }
    auto window = [&](int i) {
        return i < history ? old[i] : x[static_cast<int64_t>(first + i - history) * channels + channel];
    };
    const int page_end = page_end_tokens[s];
    if (page_end > 0) {
        scalar_t* boundary = conv_pool + rows.boundary(s) * line_stride + channel_offset;
        for (int i = 0; i < history; i++) {
            boundary[i] = window(page_end + i);
        }
    }
    const int n = tokens.count(s);
    for (int i = 0; i < history; i++) {
        running[i] = window(n + i);
    }
}

struct delta_rule_params {
    const float* dt_bias;
    const float* neg_a;
    float* state_pool;
    int64_t line_stride;
    line_rows rows;
    ragged_tokens tokens;
    const int32_t* page_end_tokens;
    int key_heads;
    int value_heads;
    int value_dim;
    float key_dim_sqrt;
};

// A CTA's place in the batch: sequence s, value head head, value columns [first_column, first_column + kColumns).
struct delta_rule_block {
    int s;
    int head;
    int first_column;
    int key_head;
    int key_dim;
    int channels;
    int64_t state_offset;

    __device__ explicit delta_rule_block(const delta_rule_params& p, int key_head_dim) {
        const int column_tiles = p.value_dim / kColumns;
        s = blockIdx.x / (p.value_heads * column_tiles);
        head = blockIdx.x / column_tiles % p.value_heads;
        first_column = blockIdx.x % column_tiles * kColumns;
        key_head = head / (p.value_heads / p.key_heads);
        key_dim = p.key_heads * key_head_dim;
        channels = 2 * key_dim + p.value_heads * p.value_dim;
        state_offset = static_cast<int64_t>(head) * key_head_dim * p.value_dim + first_column;
    }

    __device__ float* running(const delta_rule_params& p) const {
        return p.state_pool + p.rows.running(s) * p.line_stride + state_offset;
    }

    __device__ float* boundary(const delta_rule_params& p) const {
        return p.state_pool + p.rows.boundary(s) * p.line_stride + state_offset;
    }
};

// log(gain) = neg_a * softplus(a + dt_bias), Hugging Face's softplus threshold 20 included.
__device__ __forceinline__ float log_decay_of(float a, const delta_rule_params& p, int head) {
    const float shifted = a + p.dt_bias[head];
    return (shifted > 20.0f ? shifted : log1pf(expf(shifted))) * p.neg_a[head];
}

template <typename scalar_t> __device__ __forceinline__ float beta_of(scalar_t b) {
    return rounded<scalar_t>(1.0f / (1.0f + expf(-static_cast<float>(b))));
}

// fn(i) for each i < COUNT, spread over the CTA's THREADS threads, its loop unrolled so that loads issue together.
template <int COUNT, int THREADS, typename Fn> __device__ __forceinline__ void for_each_index(Fn&& fn) {
#pragma unroll
    for (int r = 0; r < (COUNT + THREADS - 1) / THREADS; r++) {
        const int i = r * THREADS + static_cast<int>(threadIdx.x);
        if (COUNT % THREADS == 0 || i < COUNT) {
            fn(i);
        }
    }
}

constexpr int kStepLanes = 8;
constexpr int kStepThreads = kColumns * kStepLanes;

// One token per sequence. One CTA per (sequence, value head, kColumns value columns); kStepLanes lanes keep a column,
// lane l of them its rows 32 i + 4 l + e, e < 4.
template <typename scalar_t, int KEY_DIM>
__global__ void __launch_bounds__(kStepThreads) delta_rule_step_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    delta_rule_params p,
    scalar_t* __restrict__ out
) {
    constexpr int kQuads = KEY_DIM / 32;
    const delta_rule_block block(p, KEY_DIM);
    const int block_column = threadIdx.x / kStepLanes;
    const int row_lane = threadIdx.x % kStepLanes;
    const int column = block.first_column + block_column;
    const scalar_t* row = x + static_cast<int64_t>(block.s) * block.channels;
    float* running = block.running(p) + block_column;
    auto row_of = [&](int quad, int e) { return 32 * quad + 4 * row_lane + e; };

    float q[kQuads][4];
    float k[kQuads][4];
    float state[kQuads][4];
    float qq = 0.0f;
    float kk = 0.0f;
    float qk = 0.0f;
#pragma unroll
    for (int quad = 0; quad < kQuads; quad++) {
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int dim = block.key_head * KEY_DIM + row_of(quad, e);
            q[quad][e] = static_cast<float>(row[dim]);
            k[quad][e] = static_cast<float>(row[block.key_dim + dim]);
            state[quad][e] = running[static_cast<int64_t>(row_of(quad, e)) * p.value_dim];
            qq += q[quad][e] * q[quad][e];
            kk += k[quad][e] * k[quad][e];
            qk += q[quad][e] * k[quad][e];
        }
    }
    const float v = static_cast<float>(row[2 * block.key_dim + block.head * p.value_dim + column]);
    const int64_t gate = static_cast<int64_t>(block.s) * p.value_heads + block.head;
    const float gain = expf(log_decay_of(static_cast<float>(a[gate]), p, block.head));
    const float beta = beta_of(b[gate]);
    const float q_scale = rsqrtf(lane_sum<kStepLanes>(qq) + 1e-6f) / p.key_dim_sqrt;
    const float k_scale = rsqrtf(lane_sum<kStepLanes>(kk) + 1e-6f);
    qk = lane_sum<kStepLanes>(qk) * q_scale * k_scale;

    float k_state = 0.0f;
    float q_state = 0.0f;
#pragma unroll
    for (int quad = 0; quad < kQuads; quad++) {
#pragma unroll
        for (int e = 0; e < 4; e++) {
            k[quad][e] *= k_scale;
            k_state += k[quad][e] * state[quad][e];
            q_state += q[quad][e] * q_scale * state[quad][e];
        }
    }
    k_state = lane_sum<kStepLanes>(k_state);
    q_state = lane_sum<kStepLanes>(q_state);
    const float delta = (v - gain * k_state) * beta;
    const bool ends_page = p.page_end_tokens[block.s] > 0;
    float* boundary = block.boundary(p) + block_column;
#pragma unroll
    for (int quad = 0; quad < kQuads; quad++) {
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const float updated = state[quad][e] * gain + k[quad][e] * delta;
            const int64_t index = static_cast<int64_t>(row_of(quad, e)) * p.value_dim;
            running[index] = updated;
            if (ends_page) {
                boundary[index] = updated;
            }
        }
    }
    if (row_lane == 0) {
        out[(static_cast<int64_t>(block.s) * p.value_heads + block.head) * p.value_dim + column] =
            static_cast<scalar_t>(gain * q_state + qk * delta);
    }
}

constexpr int kChunkThreads = 256;
constexpr int kSolveLanes = 8;

// The chunked CTA's shared memory, in floats. Query and key rows lie row_stride apart.
template <int KEY_DIM> struct delta_rule_chunk_tile {
    static constexpr int chunk = KEY_DIM > 128 ? 16 : 32;
    static constexpr int row_stride = KEY_DIM + 4;
    static constexpr int queries = 0;
    static constexpr int keys = queries + chunk * row_stride;
    static constexpr int state = keys + chunk * row_stride;
    static constexpr int deltas = state + KEY_DIM * kColumns;
    static constexpr int remembered = deltas + chunk * kColumns;
    static constexpr int outputs = remembered + chunk * kColumns;
    static constexpr int key_key = outputs + chunk * kColumns;
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

// One CTA per (sequence, value head, kColumns value columns), over the sequence's tokens in chunks that end at its
// page end. With cum_i the summed log decay of the chunk's tokens up to i and S the state before the chunk, its deltas
// are u_i = beta_i (v_i - e^cum_i k_i.S - sum_{j < i} e^(cum_i - cum_j) (k_i.k_j) u_j), its outputs
// o_i = e^cum_i q_i.S + sum_{j <= i} e^(cum_i - cum_j) (q_i.k_j) u_j and the state after it
// e^cum_last S + sum_j e^(cum_last - cum_j) k_j u_j.
template <typename scalar_t, int KEY_DIM>
__global__ void __launch_bounds__(kChunkThreads) delta_rule_chunk_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    delta_rule_params p,
    scalar_t* __restrict__ out
) {
    using tile = delta_rule_chunk_tile<KEY_DIM>;
    constexpr int kChunk = tile::chunk;
    constexpr int kRowStride = tile::row_stride;
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
    const delta_rule_block block(p, KEY_DIM);
    const int first = p.tokens.first(block.s);
    const int n = p.tokens.count(block.s);
    const int page_end = p.page_end_tokens[block.s];
    auto line_index = [&](int i) {
        return static_cast<int64_t>(i / (kColumns / 4)) * p.value_dim + i % (kColumns / 4) * 4;
    };
    auto store_state = [&](float* line) {
        for_each_index<KEY_DIM * kColumns / 4, kChunkThreads>([&](int i) {
            *reinterpret_cast<float4*>(line + line_index(i)) = vector_at(state, 4 * i);
        });
    };

    constexpr int kStateItems = kChunk / 4 * (kColumns / 2);
    constexpr int kPairItems = kChunk / 4 * (kChunk / 4 + 1);
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
        for (int d = 0; d < KEY_DIM; d += 4) {
            const float4 row[2] = {
                *reinterpret_cast<const float4*>(rows + i0 * kRowStride + d),
                *reinterpret_cast<const float4*>(rows + (i0 + 1) * kRowStride + d),
            };
            for (int j = 0; j < 4; j++) {
                const float4 key = *reinterpret_cast<const float4*>(keys + (j0 + j) * kRowStride + d);
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
                    key_key[key_token * kChunk + row_token] = value;
                } else {
                    query_key[row_token * kChunk + key_token] = value;
                }
            }
        }
    };

    const float* running = block.running(p);
    for_each_index<KEY_DIM * kColumns / 4, kChunkThreads>([&](int i) {
        vector_at(state, 4 * i) = *reinterpret_cast<const float4*>(running + line_index(i));
    });
    for (int chunk_first = 0; chunk_first < n;) {
        const int chunk_end = min(min(chunk_first + kChunk, n), chunk_first < page_end ? page_end : n);
        tokens = chunk_end - chunk_first;
        const scalar_t* chunk_x = x + static_cast<int64_t>(first + chunk_first) * block.channels;
        for_each_index<kChunk * KEY_DIM, kChunkThreads>([&](int i) {
            const int t = i / KEY_DIM;
            const int d = i % KEY_DIM;
            const scalar_t* query = chunk_x + static_cast<int64_t>(t) * block.channels + block.key_head * KEY_DIM + d;
            queries[t * kRowStride + d] = t < tokens ? static_cast<float>(query[0]) : 0.0f;
            keys[t * kRowStride + d] = t < tokens ? static_cast<float>(query[block.key_dim]) : 0.0f;
        });
        for_each_index<kChunk * kColumns, kChunkThreads>([&](int i) {
            const int t = i / kColumns;
            const int64_t value = static_cast<int64_t>(t) * block.channels + 2 * block.key_dim +
                block.head * p.value_dim + block.first_column + i % kColumns;
            deltas[i] = t < tokens ? static_cast<float>(chunk_x[value]) : 0.0f;
        });
        if (threadIdx.x < kChunk) {
            const int t = threadIdx.x;
            const int64_t gate = static_cast<int64_t>(first + chunk_first + t) * p.value_heads + block.head;
            decay[t] = t < tokens ? log_decay_of(static_cast<float>(a[gate]), p, block.head) : 0.0f;
            beta[t] = t < tokens ? beta_of(b[gate]) : 0.0f;
        }
        __syncthreads();

        if (warp == 0) {
            float cum = lane < kChunk ? decay[lane] : 0.0f;
            for (int offset = 1; offset < 32; offset <<= 1) {
                const float before = __shfl_up_sync(0xffffffffu, cum, offset);
                if (lane >= offset) {
                    cum += before;
                }
            }
            if (lane < kChunk) {
                decay[lane] = cum;
            }
        }
#pragma unroll
        for (int u = 0; u < kChunk / (kChunkThreads / 32); u++) {
            const int t = u * (kChunkThreads / 32) + warp;
            if (t >= tokens) {
                break;
            }
            float qq = 0.0f;
            float kk = 0.0f;
            for (int d = lane; d < KEY_DIM; d += 32) {
                qq += queries[t * kRowStride + d] * queries[t * kRowStride + d];
                kk += keys[t * kRowStride + d] * keys[t * kRowStride + d];
            }
            const float q_scale = rsqrtf(lane_sum<32>(qq) + 1e-6f) / p.key_dim_sqrt;
            const float k_scale = rsqrtf(lane_sum<32>(kk) + 1e-6f);
            for (int d = lane; d < KEY_DIM; d += 32) {
                queries[t * kRowStride + d] = queries[t * kRowStride + d] * q_scale;
                keys[t * kRowStride + d] = keys[t * kRowStride + d] * k_scale;
            }
        }
        __syncthreads();

        if (threadIdx.x < kChunk) {
            const int t = threadIdx.x;
            tail_decay[t] = t < tokens ? expf(decay[tokens - 1] - decay[t]) : 0.0f;
        }
        // Items: 4 tokens x 2 columns of e^cum_i k_i.S or e^cum_i q_i.S, then the decayed k_i.k_j with j < i.
        for (int item = threadIdx.x; item < 2 * kStateItems + kPairItems; item += blockDim.x) {
            if (item >= 2 * kStateItems) {
                decayed_pairs(true, item - 2 * kStateItems);
                continue;
            }
            const int t0 = item % kStateItems / (kColumns / 2) * 4;
            const int c0 = item % (kColumns / 2) * 2;
            if (t0 >= tokens) {
                continue;
            }
            const float* rows = item < kStateItems ? keys : queries;
            float sum[4][2] = {};
#pragma unroll 4
            for (int d = 0; d < KEY_DIM; d += 4) {
                float4 row[4];
                for (int i = 0; i < 4; i++) {
                    row[i] = *reinterpret_cast<const float4*>(rows + (t0 + i) * kRowStride + d);
                }
                for (int e = 0; e < 4; e++) {
                    const float2 column = *reinterpret_cast<const float2*>(state + (d + e) * kColumns + c0);
                    for (int i = 0; i < 4; i++) {
                        sum[i][0] += component(row[i], e) * column.x;
                        sum[i][1] += component(row[i], e) * column.y;
                    }
                }
            }
            float* target = item < kStateItems ? remembered : outputs;
            for (int i = 0; i < 4; i++) {
                const float gain = expf(decay[t0 + i]);
                target[(t0 + i) * kColumns + c0] = gain * sum[i][0];
                target[(t0 + i) * kColumns + c0 + 1] = gain * sum[i][1];
            }
        }
        __syncthreads();

        // kSolveLanes lanes per column solve its deltas, lane g of them keeping rows g + kSolveLanes m, while the other
        // warps compute the decayed q_i.k_j with j <= i.
        if (threadIdx.x < kColumns * kSolveLanes) {
            const int c = threadIdx.x / kSolveLanes;
            const int g = threadIdx.x % kSolveLanes;
            constexpr int kOwned = kChunk / kSolveLanes;
            float residual[kOwned];
#pragma unroll
            for (int m = 0; m < kOwned; m++) {
                const int i = g + kSolveLanes * m;
                residual[m] = deltas[i * kColumns + c] - remembered[i * kColumns + c];
            }
#pragma unroll
            for (int j = 0; j < kChunk; j++) {
                if (j >= tokens) {
                    break;
                }
                const float delta =
                    __shfl_sync(0xffffffffu, beta[j] * residual[j / kSolveLanes], j % kSolveLanes, kSolveLanes);
                if (g == j % kSolveLanes) {
                    deltas[j * kColumns + c] = delta;
                }
#pragma unroll
                for (int m = 0; m < kOwned; m++) {
                    const int i = g + kSolveLanes * m;
                    if (i > j) {
                        residual[m] -= key_key[j * kChunk + i] * delta;
                    }
                }
            }
        } else {
            for (int item = threadIdx.x - kColumns * kSolveLanes; item < kPairItems;
                 item += kChunkThreads - kColumns * kSolveLanes) {
                decayed_pairs(false, item);
            }
        }
        __syncthreads();

        // Items: 4 rows x 4 columns of the state, then 1 token x 4 columns of the outputs.
        constexpr int kUpdateItems = KEY_DIM / 4 * (kColumns / 4);
        const float chunk_decay = expf(decay[tokens - 1]);
        for (int item = threadIdx.x; item < kUpdateItems + kChunk * (kColumns / 4); item += blockDim.x) {
            if (item < kUpdateItems) {
                const int r0 = item / (kColumns / 4) * 4;
                const int c0 = item % (kColumns / 4) * 4;
                float4 sum[4];
                for (int i = 0; i < 4; i++) {
                    const float4 old = vector_at(state, (r0 + i) * kColumns + c0);
                    sum[i] = make_float4(
                        old.x * chunk_decay,
                        old.y * chunk_decay,
                        old.z * chunk_decay,
                        old.w * chunk_decay
                    );
                }
#pragma unroll 4
                for (int j = 0; j < tokens; j++) {
                    const float4 key = *reinterpret_cast<const float4*>(keys + j * kRowStride + r0);
                    const float4 delta = vector_at(deltas, j * kColumns + c0);
                    for (int i = 0; i < 4; i++) {
                        const float weight = tail_decay[j] * component(key, i);
                        sum[i].x += weight * delta.x;
                        sum[i].y += weight * delta.y;
                        sum[i].z += weight * delta.z;
                        sum[i].w += weight * delta.w;
                    }
                }
                for (int i = 0; i < 4; i++) {
                    vector_at(state, (r0 + i) * kColumns + c0) = sum[i];
                }
            } else {
                const int i = (item - kUpdateItems) / (kColumns / 4);
                const int c0 = (item - kUpdateItems) % (kColumns / 4) * 4;
                if (i >= tokens) {
                    continue;
                }
                float4 sum = vector_at(outputs, i * kColumns + c0);
                for (int j = 0; j <= i; j++) {
                    const float weight = query_key[i * kChunk + j];
                    const float4 delta = vector_at(deltas, j * kColumns + c0);
                    sum.x += weight * delta.x;
                    sum.y += weight * delta.y;
                    sum.z += weight * delta.z;
                    sum.w += weight * delta.w;
                }
                vector_at(outputs, i * kColumns + c0) = sum;
            }
        }
        __syncthreads();

        for (int i = threadIdx.x; i < tokens * kColumns; i += blockDim.x) {
            const int64_t token = first + chunk_first + i / kColumns;
            out[(token * p.value_heads + block.head) * p.value_dim + block.first_column + i % kColumns] =
                static_cast<scalar_t>(outputs[i]);
        }
        if (chunk_end == page_end) {
            store_state(block.boundary(p));
        }
        chunk_first = chunk_end;
    }
    store_state(block.running(p));
}

// One CTA per row. blockDim.x is a power of two.
template <typename scalar_t>
__global__ void gated_rmsnorm_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ z,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ y,
    int64_t dim,
    float eps
) {
    const int64_t offset = static_cast<int64_t>(blockIdx.x) * dim;
    float local = 0.0f;
    for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = static_cast<float>(x[offset + i]);
        local += v * v;
    }
    extern __shared__ float shared[];
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(shared[0] / static_cast<float>(dim) + eps);
    for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
        const float normed = rounded<scalar_t>(static_cast<float>(x[offset + i]) * inv_rms);
        const float scaled = rounded<scalar_t>(static_cast<float>(weight[i]) * normed);
        y[offset + i] = static_cast<scalar_t>(scaled * silu(static_cast<float>(z[offset + i])));
    }
}

int blocks_for(int64_t threads) {
    return static_cast<int>((threads + kConvThreads - 1) / kConvThreads);
}

void check_ragged(
    const char* op,
    const at::Tensor& x,
    const at::Tensor& rows,
    const std::optional<at::Tensor>& cu_seqlens_q,
    const at::Tensor& page_end_tokens
) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous(), op, ": x must be a contiguous CUDA [tokens, dim]");
    TORCH_CHECK(
        rows.is_cuda() && rows.scalar_type() == at::kInt && rows.dim() == 2 && rows.size(1) >= 2 && rows.stride(1) == 1,
        op,
        ": rows must be CUDA int32 [num_seqs, >= 2] with its columns adjacent"
    );
    const int64_t num_seqs = rows.size(0);
    TORCH_CHECK(
        page_end_tokens.is_cuda() && page_end_tokens.scalar_type() == at::kInt && page_end_tokens.dim() == 1 &&
            page_end_tokens.size(0) == num_seqs && page_end_tokens.is_contiguous(),
        op,
        ": page_end_tokens must be contiguous CUDA int32 [num_seqs]"
    );
    if (cu_seqlens_q.has_value()) {
        TORCH_CHECK(
            cu_seqlens_q->is_cuda() && cu_seqlens_q->scalar_type() == at::kInt && cu_seqlens_q->dim() == 1 &&
                cu_seqlens_q->size(0) == num_seqs + 1 && cu_seqlens_q->is_contiguous(),
            op,
            ": cu_seqlens_q must be contiguous CUDA int32 [num_seqs + 1]"
        );
    } else {
        TORCH_CHECK(x.size(0) == num_seqs, op, ": without cu_seqlens_q, x holds one token per sequence");
    }
}

ragged_tokens ragged(const at::Tensor& rows, const std::optional<at::Tensor>& cu_seqlens_q) {
    return ragged_tokens{
        cu_seqlens_q.has_value() ? cu_seqlens_q->const_data_ptr<int32_t>() : nullptr,
        static_cast<int>(rows.size(0)),
    };
}

line_rows rows_of(const at::Tensor& rows) {
    return line_rows{rows.const_data_ptr<int32_t>(), rows.stride(0)};
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

at::Tensor causal_conv_cuda(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_pool,
    const at::Tensor& rows,
    const std::optional<at::Tensor>& cu_seqlens_q,
    const at::Tensor& page_end_tokens
) {
    check_ragged("causal_conv", x, rows, cu_seqlens_q, page_end_tokens);
    const int64_t channels = x.size(1);
    TORCH_CHECK(
        weight.scalar_type() == x.scalar_type() && weight.dim() == 3 && weight.size(0) == channels &&
            weight.size(1) == 1 && weight.size(2) >= 1 && weight.size(2) <= kMaxConvHistory + 1 &&
            weight.is_contiguous(),
        "causal_conv: weight must be contiguous [channels, 1, kernel] in x's dtype, kernel at most ",
        kMaxConvHistory + 1
    );
    const int64_t kernel = weight.size(2);
    TORCH_CHECK(
        conv_pool.scalar_type() == x.scalar_type() && conv_pool.dim() == 3 && conv_pool.size(1) == channels &&
            conv_pool.size(2) == kernel - 1 && conv_pool.stride(2) == 1 && conv_pool.stride(1) == kernel - 1,
        "causal_conv: conv_pool must be [lines, channels, kernel - 1] in x's dtype, each line packed"
    );
    at::Tensor out = at::empty_like(x);
    const int64_t total = x.size(0);
    if (rows.size(0) == 0) {
        return out;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "causal_conv_cuda", [&] {
        if (total > 0) {
            causal_conv_kernel<scalar_t><<<blocks_for(total * channels), kConvThreads, 0, stream>>>(
                x.const_data_ptr<scalar_t>(),
                weight.const_data_ptr<scalar_t>(),
                conv_pool.const_data_ptr<scalar_t>(),
                conv_pool.stride(0),
                rows_of(rows),
                ragged(rows, cu_seqlens_q),
                total,
                static_cast<int>(channels),
                static_cast<int>(kernel),
                out.mutable_data_ptr<scalar_t>()
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        conv_state_kernel<scalar_t><<<blocks_for(rows.size(0) * channels), kConvThreads, 0, stream>>>(
            x.const_data_ptr<scalar_t>(),
            conv_pool.mutable_data_ptr<scalar_t>(),
            conv_pool.stride(0),
            rows_of(rows),
            ragged(rows, cu_seqlens_q),
            page_end_tokens.const_data_ptr<int32_t>(),
            static_cast<int>(channels),
            static_cast<int>(kernel)
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return out;
}

at::Tensor gated_delta_rule_cuda(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& dt_bias,
    const at::Tensor& neg_a,
    const at::Tensor& state_pool,
    const at::Tensor& rows,
    const std::optional<at::Tensor>& cu_seqlens_q,
    const at::Tensor& page_end_tokens,
    int64_t key_heads
) {
    check_ragged("gated_delta_rule", x, rows, cu_seqlens_q, page_end_tokens);
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
    TORCH_CHECK(
        value_dim % kColumns == 0,
        "gated_delta_rule: the value head dim must be a multiple of ",
        kColumns
    );
    TORCH_CHECK(
        x.size(1) == 2 * key_heads * key_dim + value_heads * value_dim,
        "gated_delta_rule: x must be [tokens, 2 * key heads * key dim + value heads * value dim]"
    );
    const int64_t total = x.size(0);
    for (const at::Tensor* gate : {&a, &b}) {
        TORCH_CHECK(
            gate->scalar_type() == x.scalar_type() && gate->dim() == 2 && gate->size(0) == total &&
                gate->size(1) == value_heads && gate->is_contiguous(),
            "gated_delta_rule: a and b must be contiguous [tokens, value heads] in x's dtype"
        );
    }
    for (const at::Tensor* head_values : {&dt_bias, &neg_a}) {
        TORCH_CHECK(
            head_values->is_cuda() && head_values->scalar_type() == at::kFloat && head_values->dim() == 1 &&
                head_values->size(0) == value_heads && head_values->is_contiguous(),
            "gated_delta_rule: dt_bias and neg_a must be contiguous CUDA fp32 [value heads]"
        );
    }
    at::Tensor out = at::empty({total, value_heads, value_dim}, x.options());
    const int64_t num_seqs = rows.size(0);
    if (num_seqs == 0) {
        return out;
    }
    const delta_rule_params params{
        dt_bias.const_data_ptr<float>(),
        neg_a.const_data_ptr<float>(),
        state_pool.mutable_data_ptr<float>(),
        state_pool.stride(0),
        rows_of(rows),
        ragged(rows, cu_seqlens_q),
        page_end_tokens.const_data_ptr<int32_t>(),
        static_cast<int>(key_heads),
        static_cast<int>(value_heads),
        static_cast<int>(value_dim),
        static_cast<float>(std::sqrt(static_cast<double>(key_dim))),
    };
    const auto blocks = static_cast<int>(num_seqs * value_heads * (value_dim / kColumns));
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "gated_delta_rule_cuda", [&] {
        dispatch_key_dim(key_dim, [&](auto key_dim_tag) {
            constexpr int kKeyDim = decltype(key_dim_tag)::value;
            const scalar_t* x_data = x.const_data_ptr<scalar_t>();
            const scalar_t* a_data = a.const_data_ptr<scalar_t>();
            const scalar_t* b_data = b.const_data_ptr<scalar_t>();
            scalar_t* out_data = out.mutable_data_ptr<scalar_t>();
            if (!cu_seqlens_q.has_value()) {
                delta_rule_step_kernel<scalar_t, kKeyDim>
                    <<<blocks, kStepThreads, 0, stream>>>(x_data, a_data, b_data, params, out_data);
            } else {
                const auto kernel = delta_rule_chunk_kernel<scalar_t, kKeyDim>;
                constexpr size_t bytes = delta_rule_chunk_tile<kKeyDim>::bytes;
                if (bytes > kSharedMemoryWithoutOptIn) {
                    C10_CUDA_CHECK(cudaFuncSetAttribute(
                        kernel,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(bytes)
                    ));
                }
                kernel<<<blocks, kChunkThreads, bytes, stream>>>(x_data, a_data, b_data, params, out_data);
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
    return out;
}

at::Tensor gated_rmsnorm_cuda(const at::Tensor& x, const at::Tensor& z, const at::Tensor& weight, double eps) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "gated_rmsnorm: x must be a contiguous CUDA tensor");
    TORCH_CHECK(
        z.sizes() == x.sizes() && z.scalar_type() == x.scalar_type() && z.is_contiguous(),
        "gated_rmsnorm: z must be contiguous, of x's shape and dtype"
    );
    const int64_t dim = x.size(-1);
    TORCH_CHECK(
        weight.scalar_type() == x.scalar_type() && weight.dim() == 1 && weight.size(0) == dim &&
            weight.is_contiguous(),
        "gated_rmsnorm: weight must be contiguous [dim] in x's dtype"
    );
    at::Tensor y = at::empty_like(x);
    const int64_t rows = dim == 0 ? 0 : x.numel() / dim;
    if (rows == 0) {
        return y;
    }
    const int64_t max_threads = std::min<int64_t>(1024, dim);
    int threads = 1;
    while (threads < max_threads) {
        threads <<= 1;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "gated_rmsnorm_cuda", [&] {
        gated_rmsnorm_kernel<scalar_t>
            <<<static_cast<int>(rows), threads, static_cast<size_t>(threads) * sizeof(float), stream>>>(
                x.const_data_ptr<scalar_t>(),
                z.const_data_ptr<scalar_t>(),
                weight.const_data_ptr<scalar_t>(),
                y.mutable_data_ptr<scalar_t>(),
                dim,
                static_cast<float>(eps)
            );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return y;
}

}  // namespace pulsarlm
