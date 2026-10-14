from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI

from pulsarlm import checkpoint
from pulsarlm.async_engine import AsyncEngine
from pulsarlm.chat import stop_ids
from pulsarlm.core.engine import Engine, EvictionPolicy
from pulsarlm.engine_model import engine_model
from pulsarlm.paging import PAGE_SIZE, trained_view_pages
from pulsarlm.server.openai import ServerState
from pulsarlm.server.openai import router as openai_router
from pulsarlm.settings import Settings
from pulsarlm.store import LocalStore


def load_server_state(settings: Settings, path: Path) -> ServerState:
    model = checkpoint.load(
        path,
        device=torch.device(settings.device),
        dtype=getattr(torch, settings.dtype),
    )
    cfg = model.config
    stops = stop_ids(model.tokenizer, cfg.eos_ids)
    store = LocalStore(
        PAGE_SIZE,
        int(settings.store_gib * 2**30),
        checkpoint.store_path(settings.data_dir),
    )
    engine = Engine(
        store,
        engine_model(cfg),
        model.weights,
        model_name=f"{settings.model}@{checkpoint.weights_version(path)}",
        page_size=PAGE_SIZE,
        kv_dtype=getattr(torch, settings.dtype),
        kv_cache_gib=settings.kv_cache_gib,
        staging_gib=settings.staging_gib,
        device=torch.device(settings.device),
        stop_ids=stops,
        prefill_tokens_per_step=settings.prefill_tokens_per_step,
    )
    return ServerState(
        engine=AsyncEngine(engine),
        store=store,
        tokenizer=model.tokenizer,
        chat_template=model.chat_template,
        model_id=settings.model,
        page_size=PAGE_SIZE,
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
