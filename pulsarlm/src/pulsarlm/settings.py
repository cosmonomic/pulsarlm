from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    model: str
    data_dir: Path | None
    device: str
    tensor_parallel_size: int
    expert_parallel: bool
    dtype: str
    kv_cache_gib: float
    prefill_tokens_per_step: int
    max_running_requests: int
    decode_graphs: bool
    store_gib: float
    persist: bool
    prefix_cache: bool
    cycle_tokens: int
    mass_tokens: int | None
    n_working: int
    eviction_threshold: float
    eviction_warmup: int
    max_context_window: int | None
    num_draft_tokens: int
