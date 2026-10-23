from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from pulsarlm.async_engine import AsyncEngine
from pulsarlm.eviction_log import EvictionLog
from pulsarlm.served_model import load_served_model
from pulsarlm.server.metrics import router as metrics_router
from pulsarlm.server.openai import ServerState
from pulsarlm.server.openai import router as openai_router
from pulsarlm.settings import Settings


def load_server_state(
    settings: Settings, path: Path, eviction_log: Path | None
) -> ServerState:
    served = load_served_model(settings, path)
    return ServerState(
        engine=AsyncEngine(served.engine),
        store=served.store,
        tokenizer=served.tokenizer,
        chat_template=served.chat_template,
        model_id=settings.model,
        stop_ids=served.stop_ids,
        eviction=served.eviction,
        eviction_log=None
        if eviction_log is None
        else EvictionLog(eviction_log, served.eviction, served.max_context),
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
    app.include_router(metrics_router)
    return app
