import asyncio
import enum
from collections.abc import Callable
from typing import Literal, Protocol

import torch
from torch import Tensor

type Page = Tensor

class PageStore(Protocol):
    """read_pages and read_state run on the event loop given to the Engine; the
    other methods are called on the engine's loop thread with the GIL held. A page
    is one KV layer's line as a host uint8 tensor of its bytes, laid out as only
    the engine reads it; an owner pairs a lease id with the index of a page its
    request's step produced. read_state gives the newest stored state reaching a
    proper prefix of tokens, or an empty state, and a new lease keeping its pages
    until release. claim takes the pages of a step the store has for the leases
    owning them and gives the indices of those it needs, and write_pages then hands
    it their host copies with those pages' owners. write_state makes a state the
    lease's request reached findable, page_keys being its pages. A read_pages that
    raises or yields a malformed result misses every page it asked for; a
    read_state that does ends its request with MISS, holding no lease, so the
    store cleans up what it took. An exception raised by claim,
    which then claims nothing, or by write_pages, write_state or release goes to
    sys.unraisablehook.
    """

    async def read_pages(self, hashes: list[bytes]) -> list[Page | None]: ...
    async def read_state(
        self, tokens: list[int], page_digests: list[bytes]
    ) -> tuple[State, int]: ...
    def claim(
        self, hashes: list[bytes], owners: list[tuple[int, int]]
    ) -> list[int]: ...
    def write_pages(
        self, hashes: list[bytes], pages: list[Page], owners: list[tuple[int, int]]
    ) -> None: ...
    def write_state(
        self, lease_id: int, state: State, page_keys: list[bytes]
    ) -> None: ...
    def release(self, lease_id: int) -> None: ...

class Moments:
    """The running count, mean and sum of squared deviations from the mean
    (Welford) of a layer's samples."""

    count: int
    mean: float
    m2: float
    def __init__(self, count: int = 0, mean: float = 0.0, m2: float = 0.0) -> None: ...

class ActivePage:
    """A page of a sequence: its 16-byte hash, its index among the history's full
    pages, and the sums its eviction densities come from: mass_sum in each query
    head its layer measures mass in, and weight_sum. A state the engine reaches
    holds a mass_sum per head; an initial state may hold none, which reads as 0,
    with weight_sum 0."""

    hash: bytes
    index: int
    mass_sum: list[float]
    weight_sum: float
    def __init__(
        self,
        hash: bytes,
        index: int,
        mass_sum: list[float] = ...,
        weight_sum: float = 0.0,
    ) -> None: ...

class LayerSequence:
    """One KV layer's active pages, in view order, and the moments of the pages
    it sampled, each once a cycle's tokens followed it: of their densities, one
    per query head the layer measures mass in or none, and of their scores."""

    pages: list[ActivePage]
    density_moments: list[Moments]
    score_moments: Moments
    def __init__(
        self,
        pages: list[ActivePage] = [],
        density_moments: list[Moments] = [],
        score_moments: Moments = ...,
    ) -> None: ...

class State:
    """head is the 16-byte hash of the newest full page of the history, which every
    non-empty sequence ends with; None before the first full page. layers holds one
    sequence per KV layer, and may be empty when head is None. tail holds the tokens
    after head."""

    @property
    def head(self) -> bytes | None: ...
    @property
    def layers(self) -> list[LayerSequence]: ...
    @property
    def tail(self) -> list[int]: ...
    def __init__(
        self,
        head: bytes | None = None,
        layers: list[LayerSequence] = [],
        tail: list[int] = [],
    ) -> None: ...

class CycleEnd:
    """The state a request reached when an eviction cycle ended."""

    @property
    def state(self) -> State: ...

class RequestResult:
    """A request's final state, which leaves out the last sampled token, and its
    token counts. total_tokens: the request's tokens, then the tokens it sampled.
    cached_tokens: the tokens of the full pages of the state it resumed, which the
    request did not forward; that state's tail is forwarded again. device_cached_tokens: those of them on pages every
    KV layer reading the page had resident in the KV cache when the request
    started, not fetched from the store."""

    @property
    def state(self) -> State: ...
    @property
    def total_tokens(self) -> int: ...
    @property
    def cached_tokens(self) -> int: ...
    @property
    def device_cached_tokens(self) -> int: ...

class RequestError(enum.Enum):
    MISS = ...
    OOM = ...
    CANCELLED = ...
    LOOP_EXITED = ...

class RequestFailed(Exception):
    @property
    def error(self) -> RequestError: ...

class EngineStat:
    @property
    def kv_cache_bytes(self) -> int:
        """The bytes of the KV cache's pool that lines in use take: those of
        running requests, not those only caching a page."""

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
    max_context_window: int | None
    eviction_threshold: float
    eviction_warmup: int
    attention_mass_decay: float
    def __init__(
        self,
        *,
        n_sink: int = 1024,
        n_working: int = 8192,
        cycle_tokens: int = 512,
        max_context_window: int | None = None,
        eviction_threshold: float = 0.0,
        eviction_warmup: int = 1,
        attention_mass_decay: float = 1.0 / 256,
    ) -> None: ...

class LayerScores:
    @property
    def layer(self) -> int: ...
    @property
    def threshold(self) -> float:
        """The score at quantile eviction_threshold of a normal distribution with
        the layer's score moments; -inf before the warm-up's samples or at
        threshold 0, +inf at threshold 1. A cycle's end evicts the unprotected
        pages scoring below it."""
    @property
    def page_scores(self) -> list[float | None]:
        """page_scores[i] scores the layer's pages[i]: its highest density
        standardized by each head's density moments; -inf when no query token
        attended it, None when no head has the warm-up's samples and a spread."""

class EvictionReport:
    @property
    def cap_tokens(self) -> int | None:
        """The most full-page tokens a view keeps at a cycle's end; None without
        max_context_window."""
    @property
    def layers(self) -> list[LayerScores]:
        """The layers whose pages hold attention mass."""

class ModelConfig:
    rope: RopeParameters
    def __init__(
        self,
        *,
        architecture: str,
        vocab_size: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        rope: RopeParameters,
        rms_norm_eps: float,
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
    fp8_weight_block: tuple[int, int] | None
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
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
        fp8_weight_block: tuple[int, int] | None = None,
    ) -> None: ...

class Qwen3_5MoeConfig:
    layer_types: list[str]
    num_experts_per_tok: int
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
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
        num_experts_per_tok: int,
    ) -> None: ...

class Qwen4ExpConfig:
    layer_types: list[str]
    ple_layers: list[int]
    eos_token_id: int
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
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
        linear_output_gate: Literal["silu", "sigmoid"],
        rms_norm_eps: float,
        num_experts_per_tok: int,
        hc_count: int,
        ple_layers: list[int],
        ngram_size: int,
        eos_token_id: int,
        indexer_n_heads: int,
        indexer_head_dim: int,
        indexer_budget: int,
        indexer_compress_ratio: int,
    ) -> None: ...

class Request:
    def take(self) -> list[int] | None:
        """The tokens sampled since the last take; None once the request ended and
        every token was taken. Passes each cycle end to on_cycle_end, then the
        result to on_end, before it returns. Raises RequestFailed once every token
        before an error was taken."""
    def cancel(self) -> None: ...

class Engine:
    def __init__(
        self,
        store: PageStore,
        loop: asyncio.AbstractEventLoop,
        config: ModelConfig
        | Gemma4Config
        | Qwen3_5Config
        | Qwen3_5MoeConfig
        | Qwen4ExpConfig,
        load_model_shard: Callable[[int, int], dict[str, Tensor]],
        *,
        devices: list[torch.device],
        model_name: str,
        model_hash: str,
        page_size: int,
        kv_dtype: torch.dtype,
        kv_cache_gib: float = 4.0,
        store_gib: float,
        stop_ids: list[int],
        prefill_tokens_per_step: int,
    ) -> None:
        """Runs the model of config split over devices: load_model_shard(rank,
        len(devices)) gives rank's weights on devices[rank], each rank loaded
        before the next. Each device's KV cache takes kv_cache_gib. Raises
        ValueError for a model the engine does not split over devices."""
    def submit(
        self,
        tokens: list[int],
        *,
        initial_digest: bytes | None = None,
        sampling: SamplingParams = ...,
        max_tokens: int | None = None,
        eviction: EvictionPolicy = ...,
        on_cycle_end: Callable[[CycleEnd], None] | None = None,
        on_end: Callable[[RequestResult], None] | None = None,
    ) -> Request:
        """Serves tokens, a history chained from initial_digest or from the engine's
        root without it, from the newest state the store holds for a prefix of
        them, and writes every state the request reaches to the store. Raises
        ValueError for a request the engine cannot serve. Request.take calls
        on_cycle_end and on_end on its caller's thread."""
    def eviction_report_of(
        self, state: State, eviction: EvictionPolicy
    ) -> EvictionReport:
        """The scores a cycle's end under eviction would evict state's pages by."""
    def stat(self) -> EngineStat: ...
    def fileno(self) -> int:
        """An eventfd that becomes readable when a request has results to take."""
    def clear_wake(self) -> None:
        """Called before taking the results that made fileno() readable."""
    def exception(self) -> str | None:
        """The message of the exception that panicked the engine's loop; None while
        it runs or when it was stopped."""
