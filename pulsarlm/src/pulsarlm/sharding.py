"""Each GPU's part of a checkpoint's weights under tensor parallelism, as vLLM
v0.31.0 splits Qwen3.5 and Qwen4-Exp: GPU rank of size holds the rank-th of
size equal parts of each split dimension, and every other weight whole."""

import re
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import safetensors
import torch
from torch import Tensor

from pulsarlm.checkpoint import PulsarLMModelConfig, PulsarLMModelLoadingError

_SPLIT_ARCHITECTURES = ("qwen3_5", "qwen3_5_text", "qwen4_exp")
_NGRAM_SHARD = re.compile(r"(.*ngram_embedding\.)shard_(\d+)\.weight")
_SCALE_SUFFIX = "_scale_inv"
_VOCAB_MULTIPLE = 64


@dataclass(frozen=True)
class _Ranges:
    """The ranges [first, last) of dim, concatenated in order."""

    dim: int
    ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _VocabRows:
    """Rows [first, first + count) of a table whose rows past its own are zero."""

    first: int
    count: int


@dataclass(frozen=True)
class _Split:
    config: PulsarLMModelConfig
    rank: int
    size: int

    def part(self, count: int, what: str, first: int = 0) -> tuple[int, int]:
        if count % self.size:
            raise PulsarLMModelLoadingError(
                f"{count} {what} do not split across {self.size} GPUs"
            )
        share = count // self.size
        return first + self.rank * share, first + (self.rank + 1) * share

    def heads(
        self, count: int, head_size: int, what: str, first: int = 0
    ) -> tuple[int, int]:
        start, stop = self.part(count, what)
        return first + start * head_size, first + stop * head_size

    def kv_heads(self) -> tuple[int, int]:
        text = self.config.text
        n_kv_heads, head_dim = text["num_key_value_heads"], text["head_dim"]
        if self.size <= n_kv_heads:
            return self.heads(n_kv_heads, head_dim, "KV heads")
        if self.size % n_kv_heads:
            raise PulsarLMModelLoadingError(
                f"{n_kv_heads} KV heads neither split across nor repeat over"
                f" {self.size} GPUs"
            )
        head = self.rank // (self.size // n_kv_heads)
        return head * head_dim, (head + 1) * head_dim

    def vocab_rows(self, rows: int) -> _VocabRows:
        padded = -(-rows // _VOCAB_MULTIPLE) * _VOCAB_MULTIPLE
        first, last = self.part(padded, "padded vocabulary rows")
        return _VocabRows(first=first, count=last - first)

    def cut(self, name: str, shape: list[int]) -> _Ranges | _VocabRows | None:
        """None for a weight every GPU holds whole."""
        text = self.config.text
        n_heads, head_dim = text["num_attention_heads"], text["head_dim"]
        key_heads, key_head_dim = (
            text["linear_num_key_heads"],
            text["linear_key_head_dim"],
        )
        value_heads, value_head_dim = (
            text["linear_num_value_heads"],
            text["linear_value_head_dim"],
        )
        key_dim = key_heads * key_head_dim
        if name.endswith("embed_tokens.weight") or name == "lm_head.weight":
            return self.vocab_rows(shape[0])
        if name.endswith("self_attn.q_proj.weight"):
            return _Ranges(0, (self.heads(n_heads, 2 * head_dim, "query heads"),))
        if name.endswith(("self_attn.k_proj.weight", "self_attn.v_proj.weight")):
            return _Ranges(0, (self.kv_heads(),))
        if name.endswith("self_attn.o_proj.weight"):
            return _Ranges(1, (self.heads(n_heads, head_dim, "query heads"),))
        if name.endswith(
            ("linear_attn.in_proj_qkv.weight", "linear_attn.conv1d.weight")
        ):
            return _Ranges(
                0,
                (
                    self.heads(key_heads, key_head_dim, "key heads"),
                    self.heads(key_heads, key_head_dim, "key heads", first=key_dim),
                    self.heads(
                        value_heads, value_head_dim, "value heads", first=2 * key_dim
                    ),
                ),
            )
        if name.endswith("linear_attn.in_proj_z.weight"):
            return _Ranges(0, (self.heads(value_heads, value_head_dim, "value heads"),))
        if name.endswith(
            (
                "linear_attn.in_proj_b.weight",
                "linear_attn.in_proj_a.weight",
                "linear_attn.dt_bias",
                "linear_attn.A_log",
            )
        ):
            return _Ranges(0, (self.part(value_heads, "value heads"),))
        if name.endswith("linear_attn.out_proj.weight"):
            return _Ranges(1, (self.heads(value_heads, value_head_dim, "value heads"),))
        if name.endswith(
            (
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.shared_expert.gate_proj.weight",
                "mlp.shared_expert.up_proj.weight",
            )
        ):
            return _Ranges(0, (self.part(shape[0], "intermediate rows"),))
        if name.endswith(
            ("mlp.down_proj.weight", "mlp.shared_expert.down_proj.weight")
        ):
            return _Ranges(1, (self.part(shape[1], "intermediate columns"),))
        if name.endswith("mlp.experts.gate_up_proj"):
            intermediate = shape[1] // 2
            return _Ranges(
                1,
                (
                    self.part(intermediate, "expert intermediate rows"),
                    self.part(
                        intermediate, "expert intermediate rows", first=intermediate
                    ),
                ),
            )
        if name.endswith("mlp.experts.down_proj"):
            return _Ranges(2, (self.part(shape[2], "expert intermediate columns"),))
        return None

    def scale_cut(self, weight_cut: _Ranges | _VocabRows | None) -> _Ranges | None:
        """A block-quantized weight's scales cut along its weight's ranges, in
        blocks of the checkpoint's weight_block_size."""
        if weight_cut is None:
            return None
        if not isinstance(weight_cut, _Ranges) or weight_cut.dim > 1:
            raise PulsarLMModelLoadingError("only linear weights carry fp8 scales")
        quantization = self.config.quantization or {}
        block = quantization["weight_block_size"][weight_cut.dim]
        if any(
            bound % block for bound_range in weight_cut.ranges for bound in bound_range
        ):
            raise PulsarLMModelLoadingError(
                f"a GPU's part of an fp8 weight is not whole blocks of {block}"
            )
        return _Ranges(
            weight_cut.dim,
            tuple((first // block, last // block) for first, last in weight_cut.ranges),
        )


def _join(parts: list[Tensor], dim: int) -> Tensor:
    """Concatenates parts along dim, an fp8 tensor as bytes."""
    if len(parts) == 1:
        return parts[0].contiguous()
    if parts[0].dtype == torch.float8_e4m3fn:
        return torch.cat([p.view(torch.uint8) for p in parts], dim).view(
            torch.float8_e4m3fn
        )
    return torch.cat(parts, dim)


def _read_ranges(tensor_slice, cut: _Ranges) -> Tensor:
    leading = (slice(None),) * cut.dim
    parts = [
        tensor_slice[leading + (slice(first, last),)] for first, last in cut.ranges
    ]
    return _join(parts, cut.dim)


def _read_vocab_rows(tensor_slice, rows: int, cut: _VocabRows) -> Tensor:
    start = min(cut.first, rows)
    held = tensor_slice[start : min(cut.first + cut.count, rows)]
    padding = torch.zeros(
        (cut.count - held.shape[0], *held.shape[1:]), dtype=held.dtype
    )
    return torch.cat([held, padding])


def _read_table_rows(
    shards: list[tuple[safetensors.safe_open, str]], first: int, last: int
) -> Tensor:
    """Rows [first, last) of the table the shards make up, in order."""
    parts = []
    offset = 0
    for f, name in shards:
        tensor_slice = f.get_slice(name)
        rows = tensor_slice.get_shape()[0]
        start, stop = max(first, offset), min(last, offset + rows)
        if start < stop:
            parts.append(tensor_slice[start - offset : stop - offset])
        offset += rows
    return _join(parts, 0)


def rank_weights(
    paths: list[Path],
    config: PulsarLMModelConfig,
    rank: int,
    size: int,
    unused_prefixes: tuple[str, ...] = (),
) -> Iterator[tuple[str, Tensor]]:
    """GPU rank's part of each weight in the safetensors files at paths, on the
    host, one at a time. A PLE layer's n-gram table is one shard_0 of its rows
    of the table its shards make up."""
    if config.architecture not in _SPLIT_ARCHITECTURES:
        raise PulsarLMModelLoadingError(
            f"{config.architecture} models do not split across GPUs"
        )
    split = _Split(config=config, rank=rank, size=size)
    with ExitStack() as files:
        index = {}
        for path in paths:
            f = files.enter_context(safetensors.safe_open(path, framework="pt"))
            for name in f.keys():
                if not name.startswith(unused_prefixes):
                    index[name] = f
        tables: dict[str, list[tuple[int, str]]] = {}
        for name, f in index.items():
            if match := _NGRAM_SHARD.fullmatch(name):
                tables.setdefault(match[1], []).append((int(match[2]), name))
                continue
            tensor_slice = f.get_slice(name)
            shape = tensor_slice.get_shape()
            base = name.removesuffix(_SCALE_SUFFIX)
            if base != name and base in index:
                cut = split.scale_cut(
                    split.cut(base, index[base].get_slice(base).get_shape())
                )
            else:
                cut = split.cut(name, shape)
            if cut is None:
                yield name, f.get_tensor(name)
            elif isinstance(cut, _Ranges):
                yield name, _read_ranges(tensor_slice, cut)
            else:
                yield name, _read_vocab_rows(tensor_slice, shape[0], cut)
        for prefix, shards in tables.items():
            ordered = [(index[name], name) for _, name in sorted(shards)]
            total = sum(f.get_slice(name).get_shape()[0] for f, name in ordered)
            first, last = split.part(total, "n-gram table rows")
            yield prefix + "shard_0.weight", _read_table_rows(ordered, first, last)
