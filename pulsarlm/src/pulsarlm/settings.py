from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    model: str
    data_dir: Path | None
    device: str
    dtype: str
    kv_cache_gib: float
    staging_gib: float
    prefill_tokens_per_step: int
    store_gib: float
    persist: bool
    cycle_tokens: int
    n_working: int
    evict_tau: float
    max_pages: int | None
