"""A model loaded with its engine and store, as the server and the terminal chat run it."""

import zipfile
from dataclasses import dataclass
from pathlib import Path

import torch
from tokenizers import Tokenizer

from pulsarlm import checkpoint
from pulsarlm.chat import ChatTemplate, stop_ids
from pulsarlm.core.engine import (
    Communicator,
    Engine,
    EvictionPolicy,
    Model,
    make_model,
)
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
    prefix_cache: bool


def split_models(path: Path, size: int, dtype: torch.dtype) -> list[Model]:
    """The model split across cuda:0 to cuda:size - 1, one GPU's part loaded and
    on its GPU before the next is read."""
    devices = [torch.device("cuda", rank) for rank in range(size)]
    communicators = Communicator.make(devices)
    models = []
    for rank, device in enumerate(devices):
        part = checkpoint.load(path, device=device, dtype=dtype, shard=(rank, size))
        models.append(
            make_model(engine_model(part.config), part.weights, communicators[rank])
        )
        del part
    return models


def load_served_model(settings: Settings, path: Path) -> ServedModel:
    """eviction is the settings' policy, its n_sink left at the default. Above one
    tensor-parallel GPU, the model is split across cuda:0 on."""
    device = torch.device(settings.device)
    dtype = getattr(torch, settings.dtype)
    size = settings.tensor_parallel_size
    if size == 1:
        model = checkpoint.load(path, device=device, dtype=dtype)
    else:
        with zipfile.ZipFile(path, mode="r") as archive:
            root = zipfile.Path(archive)
            model = checkpoint.PulsarLMModel(
                config=checkpoint.load_config(root),
                weights={},
                tokenizer=checkpoint.load_tokenizer(root),
                chat_template=checkpoint.load_chat_template(root),
            )
    cfg = model.config
    max_context = cfg.max_position_embeddings or None
    stops = stop_ids(model.tokenizer, cfg.eos_ids)
    store = LocalStore(
        PAGE_SIZE,
        int(settings.store_gib * 2**30),
        checkpoint.store_path(settings.data_dir) if settings.persist else None,
    )
    model_name = f"{settings.model}@{checkpoint.weights_version(path)}"
    if size == 1:
        engine = Engine(
            store,
            engine_model(cfg),
            model.weights,
            model_name=model_name,
            page_size=PAGE_SIZE,
            kv_dtype=dtype,
            kv_cache_gib=settings.kv_cache_gib,
            store_gib=settings.store_gib,
            device=device,
            stop_ids=stops,
            prefill_tokens_per_step=settings.prefill_tokens_per_step,
        )
    else:
        engine = Engine(
            store,
            models=split_models(path, size, dtype),
            model_name=model_name,
            page_size=PAGE_SIZE,
            kv_dtype=dtype,
            kv_cache_gib=settings.kv_cache_gib,
            store_gib=settings.store_gib,
            stop_ids=stops,
            prefill_tokens_per_step=settings.prefill_tokens_per_step,
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
            max_context_window=settings.max_context_window or max_context,
            eviction_threshold=settings.eviction_threshold,
            eviction_warmup=settings.eviction_warmup,
        ),
        prefix_cache=settings.prefix_cache,
    )
