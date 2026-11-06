from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    model: str
    data_dir: Path | None
    device: str
    tensor_parallel_size: int
    dtype: str
    kv_cache_gib: float
    prefill_tokens_per_step: int
    store_gib: float
    persist: bool
    prefix_cache: bool
    cycle_tokens: int
    n_working: int
    eviction_threshold: float
    eviction_warmup: int
    max_context_window: int | None
