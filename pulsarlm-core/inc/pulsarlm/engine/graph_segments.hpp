#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAGraph.h>

#include <concepts>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace pulsarlm {

// The segments of a step's computation between calls that depend on the batch's runs. Without piecewise graphs it
// only runs each call; while capturing, it ends the captured segment before a call and begins the next one after it;
// while replaying, it replays the segment before a call and then runs the call.
class graph_segments {
  public:
    // A call between two segments, given the step's query token count.
    using call_type = std::move_only_function<void(std::int64_t)>;
    // The segments of one capture, in order: segments[i] runs before calls[i], and the last segment after the last
    // call. A segment that holds no work is kept, as the memory pool's use ends with its last graph, but not replayed.
    struct segment_capture {
        std::vector<std::unique_ptr<at::cuda::CUDAGraph>> segments;
        std::vector<bool> segment_has_work;
        std::vector<call_type> calls;
    };

  private:
    // The step's query tokens; a padded batch's tensors hold more rows.
    std::int64_t token_count;
    // While capturing, the capture the segments and calls go into; null otherwise.
    segment_capture* capture = nullptr;
    at::cuda::MempoolId_t pool{0, 0};

    void begin_segment();
    void end_segment();
    // The first token_count rows of x, or x when it has fewer rows or is undefined.
    static at::Tensor narrow_to_tokens(const at::Tensor& x, std::int64_t token_count);

  public:
    // Runs every call at once on the current stream, with its tensors narrowed to token_count rows.
    explicit graph_segments(std::int64_t token_count);
    // Captures into capture on the current stream, which is not the default stream, each segment from pool in
    // thread-local capture mode, starting the first segment now; finish_capture ends the last.
    graph_segments(std::int64_t token_count, segment_capture& capture, at::cuda::MempoolId_t pool);
    graph_segments(const graph_segments&) = delete;
    graph_segments& operator=(const graph_segments&) = delete;

    // Runs op(tensors...) eagerly between two captured segments, each tensor narrowed along dim 0 to the step's
    // query token count (or whole when it has fewer rows), undefined tensors passed as they are. op writes only into
    // tensors allocated before the call, and reads the step's tokens only through its tensors. While capturing, the
    // call is kept for each replay with non-owning views of the tensors, so op captures only objects that outlive the
    // capture: the model, the batch the forward was given, the lines, and values.
    template <typename op_type, typename... tensor_types>
        requires(std::same_as<tensor_types, at::Tensor> && ...)
    void run_between_segments(op_type&& op, const tensor_types&... tensors) {
        if (capture == nullptr) {
            op(narrow_to_tokens(tensors, token_count)...);
            return;
        }
        end_segment();
        capture->calls.emplace_back([op = std::forward<op_type>(op),
                                     ... views = make_unowned_view(tensors)](std::int64_t count) mutable {
            op(narrow_to_tokens(views, count)...);
        });
        begin_segment();
    }

    // Ends the capture's last segment.
    void finish_capture();
    // Replays capture's segments and runs its calls between them, for a step of token_count query tokens, on the
    // current stream.
    static void replay(segment_capture& capture, std::int64_t token_count);
    // A tensor over x's memory, sizes and strides that does not keep the memory allocated.
    static at::Tensor make_unowned_view(const at::Tensor& x);
};

}  // namespace pulsarlm
