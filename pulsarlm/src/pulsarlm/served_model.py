"""A model loaded with its engine and store, as the server and the terminal chat run it."""

from dataclasses import dataclass
from pathlib import Path

import torch
from tokenizers import Tokenizer

from pulsarlm import checkpoint
from pulsarlm.chat import ChatTemplate, stop_ids
from pulsarlm.core.engine import Engine, EvictionPolicy
from pulsarlm.engine_model import engine_model
from pulsarlm.paging import PAGE_SIZE
from pulsarlm.settings import Settings
from pulsarlm.store import LocalStore


@dataclass
class ServedModel:
    engine: Engine
    store: LocalStore
    tokenizer: Tokenizer
    chat_template: ChatTemplate
    stop_ids: list[int]
    eviction: EvictionPolicy


def load_served_model(settings: Settings, path: Path) -> ServedModel:
    """eviction is the settings' policy, its n_sink left at the default."""
    device = torch.device(settings.device)
    dtype = getattr(torch, settings.dtype)
    model = checkpoint.load(path, device=device, dtype=dtype)
    cfg = model.config
    stops = stop_ids(model.tokenizer, cfg.eos_ids)
    store = LocalStore(
        PAGE_SIZE,
        int(settings.store_gib * 2**30),
        checkpoint.store_path(settings.data_dir) if settings.persist else None,
    )
    engine = Engine(
        store,
        engine_model(cfg),
        model.weights,
        model_name=f"{settings.model}@{checkpoint.weights_version(path)}",
        page_size=PAGE_SIZE,
        kv_dtype=dtype,
        kv_cache_gib=settings.kv_cache_gib,
        store_gib=settings.store_gib,
        device=device,
        stop_ids=stops,
        prefill_tokens_per_step=settings.prefill_tokens_per_step,
        max_context=cfg.max_position_embeddings or None,
    )
    return ServedModel(
        engine=engine,
        store=store,
        tokenizer=model.tokenizer,
        chat_template=model.chat_template,
        stop_ids=stops,
        eviction=EvictionPolicy(
            n_working=settings.n_working,
            cycle_tokens=settings.cycle_tokens,
            max_context_window=settings.max_context_window
            or cfg.max_position_embeddings
            or None,
            eviction_target=settings.eviction_target,
        ),
    )
