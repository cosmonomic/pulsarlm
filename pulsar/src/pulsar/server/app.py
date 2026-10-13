from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from fastapi import FastAPI

from pulsar import checkpoint
from pulsar.async_engine import AsyncEngine
from pulsar.chat import stop_ids
from pulsar.core.engine import Engine, EvictionPolicy, ModelConfig
from pulsar.server.openai import ServerState
from pulsar.server.openai import router as openai_router
from pulsar.store import LocalStore

_PAGE_SIZE = 16  # the tensor-core paged attention kernel requires 16 or 32


@dataclass
class Settings:
    model: str
    model_path: Path
    device: str
    dtype: torch.dtype
    kv_cache_gib: float
    prefill_tokens_per_step: int
    stored_states: int
    cycle_tokens: int
    n_working: int
    evict_tau: float
    max_pages: int | None


def trained_view_pages(max_positions: int, cycle_tokens: int) -> int | None:
    """The most full pages a view keeps at a cycle's end so that, with a partial tail
    page and the next cycle's tokens, its positions stay within the model's trained
    context. None when the config states no limit."""
    if max_positions <= 0:
        return None
    return max(1, (max_positions - cycle_tokens - (_PAGE_SIZE - 1)) // _PAGE_SIZE)


def relevance_layers(n_layers: int) -> list[int]:
    """The top quarter of the layers."""
    return list(range(n_layers - max(1, n_layers // 4), n_layers))


def load_server_state(settings: Settings) -> ServerState:
    model = checkpoint.load(
        settings.model_path,
        device=torch.device(settings.device),
        dtype=settings.dtype,
    )
    cfg = model.config
    stops = stop_ids(model.tokenizer, cfg.eos_ids)
    store = LocalStore(_PAGE_SIZE, settings.stored_states)
    engine = Engine(
        store,
        ModelConfig(
            architecture=cfg.architecture,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            head_dim=cfg.head_dim,
            hidden=cfg.hidden_size,
            intermediate=cfg.intermediate_size,
            vocab_size=cfg.vocab_size,
            rope_theta=float(cfg.rope_theta),
            rms_eps=float(cfg.rms_norm_eps),
            relevance_layers=relevance_layers(cfg.n_layers),
        ),
        model.weights,
        model_name=settings.model,
        page_size=_PAGE_SIZE,
        kv_dtype=settings.dtype,
        kv_cache_gib=settings.kv_cache_gib,
        device=torch.device(settings.device),
        stop_ids=stops,
        prefill_tokens_per_step=settings.prefill_tokens_per_step,
    )
    return ServerState(
        engine=AsyncEngine(engine),
        store=store,
        tokenizer=model.tokenizer,
        model_id=settings.model,
        page_size=_PAGE_SIZE,
        stop_ids=stops,
        eviction=EvictionPolicy(
            n_working=settings.n_working,
            cycle_tokens=settings.cycle_tokens,
            max_pages=settings.max_pages
            or trained_view_pages(cfg.max_position_embeddings, settings.cycle_tokens),
            evict_tau=settings.evict_tau,
        ),
    )


def create_app(load: Callable[[], ServerState]) -> FastAPI:
    """load runs on the server's event loop at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.server_state = load()
        try:
            yield
        finally:
            app.state.server_state.engine.close()
            del app.state.server_state

    app = FastAPI(lifespan=lifespan)
    app.include_router(openai_router, prefix="/api/openai/v1")
    return app
