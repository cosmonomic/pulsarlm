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
    prefix_cache: bool


def load_served_model(settings: Settings, path: Path) -> ServedModel:
    """eviction is the settings' policy, its n_sink left at the default. Above one
    tensor-parallel GPU, the model is split across cuda:0 on, one GPU's part loaded
    and on its GPU before the next is read."""
    dtype = getattr(torch, settings.dtype)
    tensor_parallel_size = settings.tensor_parallel_size
    devices = (
        [torch.device(settings.device)]
        if tensor_parallel_size == 1
        else [torch.device("cuda", rank) for rank in range(tensor_parallel_size)]
    )
    with checkpoint.model_root(path) as root:
        cfg = checkpoint.load_config(root)
        tokenizer = checkpoint.load_tokenizer(root)
        chat_template = checkpoint.load_chat_template(root)

    def load_model_shard(rank: int, size: int) -> dict[str, torch.Tensor]:
        return checkpoint.load(
            path, device=devices[rank], dtype=dtype, shard=(rank, size)
        ).weights

    max_context = cfg.max_position_embeddings or None
    stops = stop_ids(tokenizer, cfg.eos_ids)
    store = LocalStore(
        PAGE_SIZE,
        int(settings.store_gib * 2**30),
        checkpoint.store_path(settings.data_dir) if settings.persist else None,
    )
    model_name = f"{settings.model}@{checkpoint.weights_version(path)}"
    try:
        engine = Engine(
            store,
            engine_model(cfg),
            load_model_shard,
            devices=devices,
            model_name=model_name,
            page_size=PAGE_SIZE,
            kv_dtype=dtype,
            kv_cache_gib=settings.kv_cache_gib,
            store_gib=settings.store_gib,
            stop_ids=stops,
            prefill_tokens_per_step=settings.prefill_tokens_per_step,
        )
    except ValueError as error:
        raise checkpoint.PulsarLMModelLoadingError(str(error)) from error
    return ServedModel(
        engine=engine,
        store=store,
        tokenizer=tokenizer,
        chat_template=chat_template,
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
