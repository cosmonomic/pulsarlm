#include "pulsarlm/model/compiled_model.hpp"

#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cstring>
#include <regex>
#include <string>
#include <vector>

namespace pulsarlm {

namespace {

// n_layers = 1 + max i over the package's constant FQNs matching "layers.<i>.".
int64_t derive_n_layers(torch::inductor::AOTIModelPackageLoader& loader) {
    const std::regex re("layers\\.([0-9]+)\\.");
    int64_t max_layer = -1;
    for (const std::string& fqn : loader.get_constant_fqns()) {
        std::smatch m;
        if (std::regex_search(fqn, m, re)) {
            max_layer = std::max<int64_t>(max_layer, std::stoll(m[1].str()));
        }
    }
    TORCH_CHECK(max_layer >= 0, "CompiledModel: no layer constants found in the .pt2 package");
    return max_layer + 1;
}

void* current_stream() {
    return reinterpret_cast<void*>(c10::cuda::getCurrentCUDAStream().stream());
}

at::Tensor i32_to_dev(const std::vector<int32_t>& v, at::Device dev) {
    auto t = at::empty({static_cast<int64_t>(v.size())}, at::TensorOptions().dtype(at::kInt));
    if (!v.empty()) {
        std::memcpy(t.data_ptr<int32_t>(), v.data(), v.size() * sizeof(int32_t));
    }
    return t.to(dev);
}

at::Tensor i64_to_dev(const std::vector<int64_t>& v, at::Device dev) {
    auto t = at::empty({static_cast<int64_t>(v.size())}, at::TensorOptions().dtype(at::kLong));
    if (!v.empty()) {
        std::memcpy(t.data_ptr<int64_t>(), v.data(), v.size() * sizeof(int64_t));
    }
    return t.to(dev);
}

// Row index [0, 1, ..., R-1, 0, 0, ..., 0] of length num_seqs on `dev`: the real
// rows in order, then the padding rows all mapped to row 0. index_select with it
// row-pads a per-seq tensor up to num_seqs by duplicating row 0.
at::Tensor pad_row_index(int64_t R, int64_t num_seqs, at::Device dev) {
    std::vector<int64_t> idx(num_seqs, 0);
    for (int64_t i = 0; i < R; ++i) {
        idx[i] = i;
    }
    return i64_to_dev(idx, dev);
}

// Right-pad a [rows, held] page table's column dim to max_pages with page id 0.
// held <= max_pages (a sequence never holds more than the baked width). Padded
// columns lie past ceil(active_len/page_size) and are never read by the attn kernels.
at::Tensor pad_page_table_cols(at::Tensor bt, int64_t max_pages) {
    int64_t held = bt.size(1);
    TORCH_CHECK(
        held <= max_pages,
        "page_tables has ",
        held,
        " columns but the compiled graph bakes max_pages=",
        max_pages
    );
    if (held < max_pages) {
        bt = at::constant_pad_nd(bt, {0, max_pages - held}, 0);
    }
    return bt.contiguous();
}

}  // namespace

CompiledModel::CompiledModel(
    std::string decode_pt2,
    std::string prefill_pt2,
    int64_t num_seqs,
    int64_t num_pages,
    int64_t page_size
)
    : decode_loader(decode_pt2),
      prefill_loader(prefill_pt2),
      num_seqs(num_seqs),
      num_pages(num_pages),
      page_size(page_size),
      n_layers(derive_n_layers(decode_loader)),
      max_pages(0) {
    TORCH_CHECK(num_seqs > 0, "CompiledModel: num_seqs must be > 0");
    // Paged export lays out num_pages = max_pages * num_seqs, so the baked
    // page-table width is num_pages / num_seqs.
    TORCH_CHECK(
        num_pages % num_seqs == 0,
        "CompiledModel: num_pages (",
        num_pages,
        ") must be divisible by num_seqs (",
        num_seqs,
        ")"
    );
    this->max_pages = num_pages / num_seqs;
}

void CompiledModel::validate(const kv& kv) const {
    const int64_t kv_layers = kv.k.size(0);
    const int64_t kv_pages = kv.k.size(1);
    const int64_t kv_page_size = kv.k.size(2);
    if (kv_pages != this->num_pages || kv_page_size != this->page_size || kv_layers != this->n_layers) {
        TORCH_CHECK(
            false,
            "KV line dims do not match the compiled graph: got "
            "(n_layers=",
            kv_layers,
            ", num_pages=",
            kv_pages,
            ", page_size=",
            kv_page_size,
            "), baked (n_layers=",
            this->n_layers,
            ", num_pages=",
            this->num_pages,
            ", page_size=",
            this->page_size,
            ")"
        );
    }
}

at::Tensor
CompiledModel::run_decode_group(torch::inductor::AOTIModelPackageLoader& loader, const GroupBatch& g, const kv& kv) {
    TORCH_CHECK(g.context_lens.defined(), "decode group missing context_lens");
    const int64_t R = g.context_lens.numel();
    TORCH_CHECK(
        R >= 1 && R <= this->num_seqs,
        "decode batch has ",
        R,
        " sequences; the compiled graph bakes num_seqs=",
        this->num_seqs,
        " (need 1 <= R <= num_seqs)"
    );
    const at::Device dev = g.tokens.device();
    const int64_t pad = this->num_seqs - R;

    at::Tensor tokens = g.tokens, pos = g.pos;
    at::Tensor slot = g.slot_mapping, context_lens = g.context_lens;
    at::Tensor page_tables = g.page_tables;

    if (pad > 0) {
        at::Tensor idx = pad_row_index(R, this->num_seqs, dev);
        tokens = tokens.index_select(0, idx);
        pos = pos.index_select(0, idx);
        slot = slot.index_select(0, idx);
        context_lens = context_lens.index_select(0, idx);
        page_tables = page_tables.index_select(0, idx);
    }
    page_tables = pad_page_table_cols(page_tables, this->max_pages);

    // decode: [tokens, pos, slot_mapping, page_tables, context_lens, k, v]
    std::vector<at::Tensor> inputs{tokens, pos, slot, page_tables, context_lens, kv.k, kv.v};
    at::Tensor logits = loader.run(inputs, current_stream())[0];
    return logits.narrow(0, 0, R);  // real rows only
}

at::Tensor
CompiledModel::run_prefill_group(torch::inductor::AOTIModelPackageLoader& loader, const GroupBatch& g, const kv& kv) {
    TORCH_CHECK(g.cu_seqlens_q.defined() && g.seqlens_k.defined(), "prefill group missing cu_seqlens_q/seqlens_k");
    const int64_t R = g.cu_seqlens_q.numel() - 1;
    TORCH_CHECK(
        R >= 1 && R <= this->num_seqs,
        "prefill batch has ",
        R,
        " sequences; the compiled graph bakes num_seqs=",
        this->num_seqs,
        " (need 1 <= R <= num_seqs)"
    );
    const at::Device dev = g.tokens.device();
    const int64_t total_q = g.tokens.numel();
    const int64_t pad = this->num_seqs - R;

    at::Tensor tokens = g.tokens, pos = g.pos;
    at::Tensor slot = g.slot_mapping;
    at::Tensor cu = g.cu_seqlens_q, seqlens_k = g.seqlens_k;
    at::Tensor page_tables = g.page_tables;

    if (pad > 0) {
        auto cu_cpu = cu.to(at::kCPU, at::kInt).contiguous();
        auto cu_a = cu_cpu.accessor<int32_t, 1>();
        const int64_t total_q0 = cu_a[1] - cu_a[0];  // seq 0's query token count

        // Query-token index: the real total_q rows, then `pad` copies of seq 0's
        // query rows [0, total_q0). Dummies appended after the real rows keep the
        // real logits in [0, total_q).
        std::vector<int64_t> q_idx(total_q);
        for (int64_t i = 0; i < total_q; ++i) {
            q_idx[i] = i;
        }
        for (int64_t d = 0; d < pad; ++d) {
            for (int64_t j = 0; j < total_q0; ++j) {
                q_idx.push_back(j);
            }
        }
        at::Tensor q_idx_dev = i64_to_dev(q_idx, dev);
        tokens = tokens.index_select(0, q_idx_dev);
        pos = pos.index_select(0, q_idx_dev);
        slot = slot.index_select(0, q_idx_dev);

        // cu_seqlens_q: extend the prefix sums by total_q0 per dummy -> [num_seqs+1].
        std::vector<int32_t> cu_v(cu_a.data(), cu_a.data() + R + 1);
        int32_t last = cu_a[R];
        for (int64_t d = 0; d < pad; ++d) {
            last += static_cast<int32_t>(total_q0);
            cu_v.push_back(last);
        }
        cu = i32_to_dev(cu_v, dev);

        at::Tensor ridx = pad_row_index(R, this->num_seqs, dev);
        seqlens_k = seqlens_k.index_select(0, ridx);
        page_tables = page_tables.index_select(0, ridx);
    }
    page_tables = pad_page_table_cols(page_tables, this->max_pages);

    // prefill: [tokens, pos, slot_mapping, page_tables, cu_seqlens_q, seqlens_k, k, v]
    std::vector<at::Tensor> inputs{tokens, pos, slot, page_tables, cu, seqlens_k, kv.k, kv.v};
    at::Tensor logits = loader.run(inputs, current_stream())[0];
    return logits.narrow(0, 0, total_q);  // real query rows only
}

StepLogits CompiledModel::forward(const StepBatch& batch, const kv& kv) {
    this->validate(kv);
    StepLogits out;
    if (batch.prefill.has_value()) {
        out.prefill = this->run_prefill_group(this->prefill_loader, *batch.prefill, kv);
    }
    if (batch.decode.has_value()) {
        out.decode = this->run_decode_group(this->decode_loader, *batch.decode, kv);
    }
    return out;
}

}  // namespace pulsarlm
