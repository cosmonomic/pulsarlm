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
    def put(self, hashes: list[bytes], pages: list[Page]) -> None: ...

class State:
    """The ordered page hashes of a view, each 16 bytes, then the tokens of its
    partial last page."""

    page_hashes: list[bytes]
    tail: list[int]
    def __init__(self, page_hashes: list[bytes] = [], tail: list[int] = []) -> None: ...

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
        n_sink: int = 0,
        n_working: int = 1,
        cycle_tokens: int = 512,
        max_pages: int | None = None,
        evict_tau: float = 0.0,
        attention_mass_decay: float = 1.0 / 512,
    ) -> None: ...

class ModelConfig:
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
        rope_theta: float,
        rms_eps: float,
        relevance_layers: list[int],
    ) -> None: ...

class RopeParameters:
    """The first floor(partial_rotary_factor * head_dim / 2) rotate_half pairs
    rotate; the others pass through."""

    theta: float
    partial_rotary_factor: float
    def __init__(self, *, theta: float, partial_rotary_factor: float = 1.0) -> None: ...

class Gemma4Config:
    layer_types: list[str]
    global_n_kv_heads: int
    hidden_size_per_layer_input: int
    sliding_rope: RopeParameters
    full_rope: RopeParameters
    relevance_layers: list[int]
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
        relevance_layers: list[int],
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
        model: ModelConfig | Gemma4Config,
        weights: dict[str, Tensor],
        *,
        model_name: str,
        page_size: int,
        kv_dtype: torch.dtype,
        kv_cache_gib: float,
        staging_gib: float,
        max_position_embeddings: int,
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
    ) -> Request:
        """Raises ValueError for a request the engine cannot serve."""
    def page_digests(self, tokens: list[int]) -> list[bytes]:
        """The hashes of the full pages of a token history from its start: the page
        hashes a state reaching that history ends with."""
    def page_keys(self, state: State) -> list[bytes]:
        """The store keys of the pages a request resuming from state reads."""
    def fileno(self) -> int:
        """An eventfd that becomes readable when a request has results to take."""
    def clear_wake(self) -> None:
        """Called before taking the results that made fileno() readable."""
    def exception(self) -> str | None:
        """The message of the exception that panicked the engine's loop; None while
        it runs or when it was stopped."""
