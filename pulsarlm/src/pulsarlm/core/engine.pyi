import enum
from concurrent.futures import Future
from typing import Protocol

import torch
from torch import Tensor

type Page = tuple[Tensor, Tensor]

class PageStore(Protocol):
    """Called on the engine's loop thread with the GIL held; both return without
    waiting for the store. A page is one KV layer's (k, v) host tensors, each
    [1, page_size, n_kv_heads, head_dim], in pageable memory the store owns; k is
    v for a layer keeping one tensor. A fetch whose call raises, or whose
    future fails, is cancelled, is dropped unfinished or yields a malformed result,
    misses every page it asked for. An exception raised by put goes to
    sys.unraisablehook.
    """

    def fetch(self, hashes: list[bytes]) -> Future[list[Page | None]]: ...
    def put(
        self, hashes: list[bytes], pages: list[Page], owners: list[tuple[int, int]]
    ) -> None: ...

class ActivePage:
    """A page of a sequence: its 16-byte hash and its eviction density, 0 for a
    page never measured."""

    hash: bytes
    density: float
    def __init__(self, hash: bytes, density: float = 0.0) -> None: ...

class LayerSequence:
    """One KV layer's active pages, in view order."""

    pages: list[ActivePage]
    def __init__(self, pages: list[ActivePage] = []) -> None: ...

class State:
    """head is the 16-byte hash of the newest full page of the history, which every
    non-empty sequence ends with; None before the first full page. layers holds one
    sequence per KV layer, and may be empty when head is None. tail holds the tokens
    after head."""

    head: bytes | None
    layers: list[LayerSequence]
    tail: list[int]
    def __init__(
        self,
        head: bytes | None = None,
        layers: list[LayerSequence] = [],
        tail: list[int] = [],
    ) -> None: ...

class CycleEnd:
    """The state a request reached when an eviction cycle ended, covering the
    request's first token_count tokens."""

    @property
    def reached(self) -> State: ...
    @property
    def token_count(self) -> int: ...

class RequestError(enum.Enum):
    MISS = ...
    OOM = ...
    CANCELLED = ...
    LOOP_EXITED = ...

class SamplingParams:
    temperature: float
    top_p: float
    top_k: int
    seed: int | None
    def __init__(
        self,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        seed: int | None = None,
    ) -> None: ...

class EvictionPolicy:
    n_sink: int
    n_working: int
    cycle_tokens: int
    max_pages: int | None
    evict_tau: float
    attention_mass_decay: float
    def __init__(
        self,
        *,
        n_sink: int = 64,
        n_working: int = 512,
        cycle_tokens: int = 512,
        max_pages: int | None = None,
        evict_tau: float = 0.0,
        attention_mass_decay: float = 1.0 / 512,
    ) -> None: ...

class ModelConfig:
    rope: RopeParameters
    def __init__(
        self,
        *,
        architecture: str,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        hidden: int,
        intermediate: int,
        vocab_size: int,
        rope: RopeParameters,
        rms_eps: float,
    ) -> None: ...

class Llama3RopeScaling:
    factor: float
    low_freq_factor: float
    high_freq_factor: float
    original_max_position_embeddings: float
    def __init__(
        self,
        *,
        factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        original_max_position_embeddings: float,
    ) -> None: ...

class RopeParameters:
    """The first floor(partial_rotary_factor * head_dim / 2) rotate_half pairs
    rotate; the others pass through."""

    theta: float
    partial_rotary_factor: float
    llama3_scaling: Llama3RopeScaling | None
    def __init__(
        self,
        *,
        theta: float,
        partial_rotary_factor: float = 1.0,
        llama3_scaling: Llama3RopeScaling | None = None,
    ) -> None: ...

class Gemma4Config:
    layer_types: list[str]
    global_n_kv_heads: int
    hidden_size_per_layer_input: int
    sliding_rope: RopeParameters
    full_rope: RopeParameters
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        global_n_kv_heads: int,
        global_head_dim: int,
        attention_k_eq_v: bool,
        layer_types: list[str],
        sliding_window: int,
        sliding_rope: RopeParameters,
        full_rope: RopeParameters,
        rms_norm_eps: float,
        final_logit_softcapping: float | None,
        num_kv_shared_layers: int,
        hidden_size_per_layer_input: int,
        enable_moe_block: bool,
        top_k_experts: int,
    ) -> None: ...

class Qwen3_5Config:
    layer_types: list[str]
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        layer_types: list[str],
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        rope_theta: float,
        partial_rotary_factor: float,
        linear_n_key_heads: int,
        linear_n_value_heads: int,
        linear_key_head_dim: int,
        linear_value_head_dim: int,
        linear_conv_kernel: int,
        rms_norm_eps: float,
    ) -> None: ...

class Request:
    def take(
        self,
    ) -> tuple[list[int], list[CycleEnd], State | RequestError | None]:
        """The tokens and cycle ends appended since the last take, then the end:
        set by the take that first finds the request ended, None otherwise."""
    def cancel(self) -> None: ...

class Engine:
    def __init__(
        self,
        store: PageStore,
        model: ModelConfig | Gemma4Config | Qwen3_5Config,
        weights: dict[str, Tensor],
        *,
        model_name: str,
        page_size: int,
        kv_dtype: torch.dtype,
        kv_cache_gib: float = 4.0,
        store_gib: float,
        device: torch.device,
        stop_ids: list[int],
        prefill_tokens_per_step: int,
    ) -> None: ...
    def submit(
        self,
        initial_state: State,
        tokens: list[int],
        *,
        sampling: SamplingParams = ...,
        max_tokens: int | None = None,
        eviction: EvictionPolicy = ...,
        request_id: int = 0,
    ) -> Request:
        """Raises ValueError for a request the engine cannot serve. The pages the
        request's steps put reach the store owned by request_id."""
    def page_digests(self, tokens: list[int]) -> list[bytes]:
        """The hashes of the full pages of a token history from its start; the last
        is the head of a state reaching that history."""
    def page_keys(self, state: State) -> list[bytes]:
        """The store keys of the pages of state's sequences, which a request
        resuming from state reads."""
    def fileno(self) -> int:
        """An eventfd that becomes readable when a request has results to take."""
    def clear_wake(self) -> None:
        """Called before taking the results that made fileno() readable."""
    def exception(self) -> str | None:
        """The message of the exception that panicked the engine's loop; None while
        it runs or when it was stopped."""
