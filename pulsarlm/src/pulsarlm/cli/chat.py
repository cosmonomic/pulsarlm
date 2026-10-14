"""A terminal chat with one model, keeping the conversation in memory."""

import asyncio
import math
import readline  # noqa: F401  enables line editing in input()

import torch
from tokenizers.decoders import DecodeStream

from pulsarlm import checkpoint
from pulsarlm.async_engine import AsyncEngine, EngineStopped
from pulsarlm.chat import Role, encode_prompt, stop_ids
from pulsarlm.checkpoint import PulsarLMModelLoadingError
from pulsarlm.completion import CompletionError, complete
from pulsarlm.core.engine import Engine, EvictionPolicy, SamplingParams
from pulsarlm.engine_model import engine_model
from pulsarlm.paging import PAGE_SIZE, trained_view_pages
from pulsarlm.settings import Settings
from pulsarlm.store import LocalStore

__all__ = ["EngineStopped", "PulsarLMModelLoadingError", "new_chat"]

PROMPT = ">>> "


def new_chat(settings: Settings, *, system: str | None, temperature: float) -> None:
    """Reads user turns until end of input or an interrupt. Raises
    PulsarLMModelLoadingError when no data path holds the model, EngineStopped
    when the engine's loop exited."""
    loaded = checkpoint.load(
        checkpoint.model_path(settings.model, settings.data_dir),
        device=torch.device(settings.device),
        dtype=getattr(torch, settings.dtype),
    )
    cfg = loaded.config
    stops = stop_ids(loaded.tokenizer, cfg.eos_ids)
    store = LocalStore(PAGE_SIZE, int(settings.store_gib * 2**30))
    engine = Engine(
        store,
        engine_model(cfg),
        loaded.weights,
        model_name=settings.model,
        page_size=PAGE_SIZE,
        kv_dtype=getattr(torch, settings.dtype),
        kv_cache_gib=settings.kv_cache_gib,
        staging_gib=settings.staging_gib,
        device=torch.device(settings.device),
        stop_ids=stops,
        prefill_tokens_per_step=settings.prefill_tokens_per_step,
    )
    eviction = EvictionPolicy(
        n_working=settings.n_working,
        cycle_tokens=settings.cycle_tokens,
        max_pages=settings.max_pages
        or trained_view_pages(cfg.max_position_embeddings, settings.cycle_tokens),
        evict_tau=settings.evict_tau,
    )
    messages: list[tuple[Role, str]] = [("system", system)] if system else []

    async def converse() -> None:
        async_engine = AsyncEngine(engine)
        try:
            while True:
                try:
                    line = input(PROMPT).strip()
                except EOFError:
                    print()
                    return
                if not line:
                    continue
                messages.append(("user", line))
                prompt, system_tokens = encode_prompt(
                    loaded.tokenizer, loaded.chat_template, messages
                )
                if system:
                    eviction.n_sink = math.ceil(system_tokens / PAGE_SIZE)
                decoder = DecodeStream(skip_special_tokens=False)
                reply = []
                try:
                    async for token in complete(
                        async_engine,
                        store,
                        prompt,
                        sampling=SamplingParams(temperature=temperature),
                        max_tokens=None,
                        eviction=eviction,
                    ):
                        if token in stops:
                            continue
                        text = decoder.step(loaded.tokenizer, token) or ""
                        reply.append(text)
                        print(text, end="", flush=True)
                except CompletionError as error:
                    print(f"\nerror: request failed: {error}")
                    messages.pop()
                    continue
                print()
                messages.append(("assistant", "".join(reply)))
        finally:
            async_engine.close()

    try:
        asyncio.run(converse())
    except KeyboardInterrupt:
        print()
