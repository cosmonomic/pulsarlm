from pathlib import Path

import torch
import uvicorn

from pulsarlm.checkpoint import PulsarLMModelLoadingError, model_path
from pulsarlm.server.app import Settings, create_app, load_server_state

__all__ = ["PulsarLMModelLoadingError", "serve"]


def serve(
    model: str,
    *,
    host: str,
    port: int,
    data_dir: Path | None,
    device: str,
    dtype: str,
    kv_cache_gib: float,
    staging_gib: float,
    prefill_tokens_per_step: int,
    store_gib: float,
    cycle_tokens: int,
    n_working: int,
    evict_tau: float,
    max_pages: int | None,
) -> None:
    """Serves model until interrupted. Raises PulsarLMModelLoadingError, before
    starting, when no data path holds it."""
    settings = Settings(
        model=model,
        model_path=model_path(model, data_dir),
        device=device,
        dtype=getattr(torch, dtype),
        kv_cache_gib=kv_cache_gib,
        staging_gib=staging_gib,
        prefill_tokens_per_step=prefill_tokens_per_step,
        store_gib=store_gib,
        cycle_tokens=cycle_tokens,
        n_working=n_working,
        evict_tau=evict_tau,
        max_pages=max_pages,
    )
    uvicorn.run(create_app(lambda: load_server_state(settings)), host=host, port=port)
