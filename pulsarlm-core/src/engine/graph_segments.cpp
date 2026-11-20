#include "pulsarlm/engine/graph_segments.hpp"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cstddef>

namespace pulsarlm {

namespace {

class silent_warnings final : public c10::WarningHandler {
  public:
    void process(const c10::Warning&) override {}
};

}  // namespace

graph_segments::graph_segments(std::int64_t token_count)
: token_count(token_count) {}

graph_segments::graph_segments(std::int64_t token_count, segment_capture& capture, at::cuda::MempoolId_t pool)
: token_count(token_count),
  capture(&capture),
  pool(pool) {
    assert(capture.segments.empty() && capture.segment_has_work.empty() && capture.calls.empty());
    begin_segment();
}

at::Tensor graph_segments::narrow_to_tokens(const at::Tensor& x, std::int64_t token_count) {
    if (!x.defined() || x.size(0) <= token_count) {
        return x;
    }
    return x.narrow(0, 0, token_count);
}

// The graph keeps its CUDA graph after the capture so that its nodes can be counted.
void graph_segments::begin_segment() {
    capture->segments.push_back(std::make_unique<at::cuda::CUDAGraph>(/*keep_graph=*/true));
    capture->segments.back()->capture_begin(pool, cudaStreamCaptureModeThreadLocal);
}

void graph_segments::end_segment() {
    std::unique_ptr<at::cuda::CUDAGraph>& segment = capture->segments.back();
    {
        // The only warning capture_end gives is that a segment is empty, which is expected of the last one.
        silent_warnings silent;
        c10::WarningUtils::WarningHandlerGuard guard(&silent);
        segment->capture_end();
    }
    std::size_t node_count = 0;
    C10_CUDA_CHECK(cudaGraphGetNodes(segment->raw_cuda_graph(), nullptr, &node_count));
    capture->segment_has_work.push_back(node_count > 0);
    if (node_count > 0) {
        segment->instantiate();
    }
}

void graph_segments::finish_capture() {
    assert(capture != nullptr && capture->segments.size() == capture->calls.size() + 1);
    end_segment();
    capture = nullptr;
}

void graph_segments::replay(segment_capture& capture, std::int64_t token_count) {
    assert(capture.segments.size() == capture.calls.size() + 1);
    for (std::size_t index = 0; index < capture.segments.size(); index++) {
        if (capture.segment_has_work[index]) {
            capture.segments[index]->replay();
        }
        if (index < capture.calls.size()) {
            capture.calls[index](token_count);
        }
    }
}

at::Tensor graph_segments::make_unowned_view(const at::Tensor& x) {
    if (!x.defined()) {
        return x;
    }
    return at::from_blob(x.data_ptr(), x.sizes(), x.strides(), x.options());
}

}  // namespace pulsarlm
