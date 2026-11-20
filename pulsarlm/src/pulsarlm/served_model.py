"""A model loaded with its engine and store, as the server and the terminal chat run it."""

import asyncio
from dataclasses import dataclass
from pathlib import Path

import torch
from tokenizers import Tokenizer

from pulsarlm import checkpoint
from pulsarlm.chat import ChatTemplate, stop_ids
from pulsarlm.core.engine import Engine, EvictionPolicy
from pulsarlm.engine_model import engine_model
from pulsarlm.paging import PAGE_SIZE
from pulsarlm.reply import ThinkSyntax
from pulsarlm.settings import Settings
from pulsarlm.store import EmptyStore, LocalStore
from pulsarlm.tools import ToolCallSyntax


@dataclass
class ServedModel:
    engine: Engine
    store: LocalStore | EmptyStore
    tokenizer: Tokenizer
    chat_template: ChatTemplate
    stop_ids: list[int]
    tool_call_syntax: ToolCallSyntax | None
    think_syntax: ThinkSyntax | None
    eviction: EvictionPolicy
    max_context_window: int


def load_served_model(
    settings: Settings, path: Path, loop: asyncio.AbstractEventLoop
) -> ServedModel:
    """eviction is eviction_policy's. Above one
    tensor-parallel GPU, the model is split across cuda:0 on, one GPU's part loaded
    and on its GPU before the next is read. The store's reads run on loop. Without
    the prefix cache, the store keeps nothing."""
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

    draft_tokens = settings.num_draft_tokens

    # Without drafts the model loads no multi-token prediction layer.
    def load_model_shard(rank: int, size: int) -> dict[str, torch.Tensor]:
        return checkpoint.load(
            path,
            device=devices[rank],
            dtype=dtype,
            shard=(rank, size),
            expert_parallel=settings.expert_parallel,
            drafts=draft_tokens > 0,
        ).weights

    max_context_window = settings.max_context_window or cfg.max_position_embeddings
    if not max_context_window:
        raise checkpoint.PulsarLMModelLoadingError(
            f"{settings.model} states no trained context length; set --max_context_window"
        )
    eviction = eviction_policy(settings, max_context_window)
    stops = stop_ids(tokenizer, cfg.eos_ids)
    store = (
        LocalStore(
            PAGE_SIZE,
            int(settings.store_gib * 2**30),
            checkpoint.store_path(settings.data_dir) if settings.persist else None,
        )
        if settings.prefix_cache
        else EmptyStore()
    )
    try:
        engine = Engine(
            store,
            loop,
            engine_model(cfg),
            load_model_shard,
            devices=devices,
            model_name=settings.model,
            model_hash=checkpoint.model_hash(path),
            page_size=PAGE_SIZE,
            kv_dtype=dtype,
            kv_cache_gib=settings.kv_cache_gib,
            store_gib=settings.store_gib,
            stop_ids=stops,
            prefill_tokens_per_step=settings.prefill_tokens_per_step,
            decode_graphs=settings.decode_graphs,
            max_context_window=max_context_window,
            draft_tokens=draft_tokens,
            max_running_requests=settings.max_running_requests,
        )
    except ValueError as error:
        raise checkpoint.PulsarLMModelLoadingError(str(error)) from error
    return ServedModel(
        engine=engine,
        store=store,
        tokenizer=tokenizer,
        chat_template=chat_template,
        stop_ids=stops,
        tool_call_syntax=cfg.tool_call_syntax,
        think_syntax=cfg.think_syntax,
        eviction=eviction,
        max_context_window=max_context_window,
    )


def eviction_policy(settings: Settings, max_context_window: int) -> EvictionPolicy:
    """The settings' policy, its n_sink left at the default and a negative
    max_context_capacity taken as max_context_window. Raises
    PulsarLMModelLoadingError for a capacity above the window, which the engine
    rejects every request for."""
    capacity = settings.max_context_capacity
    if capacity < 0:
        capacity = max_context_window
    if capacity > max_context_window:
        raise checkpoint.PulsarLMModelLoadingError(
            f"--max_context_capacity {capacity} exceeds --max_context_window "
            f"{max_context_window}, which every request would fail on"
        )
    return EvictionPolicy(
        n_working=settings.n_working,
        cycle_tokens=settings.cycle_tokens,
        mass_tokens=settings.mass_tokens,
        max_context_capacity=capacity,
        eviction_threshold=settings.eviction_threshold,
        eviction_warmup=settings.eviction_warmup,
    )
