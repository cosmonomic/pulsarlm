import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

import torch
from fastapi import FastAPI

from pulsar import checkpoint
from pulsar.async_engine import AsyncEngine
from pulsar.chat import stop_ids
from pulsar.core.engine import Engine, EvictionPolicy, ModelConfig
from pulsar.server.openai import ServerState
from pulsar.server.openai import router as openai_router
from pulsar.store import LocalStore

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_PAGE_SIZE = 16  # the tensor-core paged attention kernel requires 16 or 32


@dataclass
class Settings:
    model: str
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    n_lines: int = 2048
    prefill_tokens_per_step: int = 256
    stored_states: int = 4096
    cycle_tokens: int = 512
    n_working: int = 4
    max_pages: int | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        model = os.environ.get("PULSAR_MODEL")
        if not model:
            raise RuntimeError("PULSAR_MODEL must be set")
        max_pages = os.environ.get("PULSAR_MAX_PAGES")
        return cls(
            model=model,
            device=os.environ.get("PULSAR_DEVICE", cls.device),
            dtype=_DTYPES[os.environ.get("PULSAR_DTYPE", "bfloat16")],
            n_lines=int(os.environ.get("PULSAR_N_LINES", cls.n_lines)),
            prefill_tokens_per_step=int(
                os.environ.get(
                    "PULSAR_PREFILL_TOKENS_PER_STEP", cls.prefill_tokens_per_step
                )
            ),
            stored_states=int(
                os.environ.get("PULSAR_STORED_STATES", cls.stored_states)
            ),
            cycle_tokens=int(os.environ.get("PULSAR_CYCLE_TOKENS", cls.cycle_tokens)),
            n_working=int(os.environ.get("PULSAR_N_WORKING", cls.n_working)),
            max_pages=int(max_pages) if max_pages else None,
        )


def relevance_layers(n_layers: int) -> list[int]:
    """The top quarter of the layers."""
    return list(range(n_layers - max(1, n_layers // 4), n_layers))


def load_server_state(settings: Settings) -> ServerState:
    model = checkpoint.load(
        checkpoint.model_path(settings.model),
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
        n_lines=settings.n_lines,
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
            max_pages=settings.max_pages,
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


app = create_app(lambda: load_server_state(Settings.from_env()))
