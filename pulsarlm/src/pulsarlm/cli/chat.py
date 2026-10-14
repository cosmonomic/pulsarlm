"""A terminal chat with one model, keeping the conversation in memory."""

import asyncio
import math
import readline  # noqa: F401  enables line editing in input()

import torch
import typer

from pulsarlm import checkpoint
from pulsarlm.async_engine import AsyncEngine, EngineStopped
from pulsarlm.chat import Message, ReasoningEffort, encode_prompt, stop_ids
from pulsarlm.checkpoint import PulsarLMModelLoadingError
from pulsarlm.completion import CompletionError, complete
from pulsarlm.core.engine import Engine, EvictionPolicy, SamplingParams
from pulsarlm.engine_model import engine_model
from pulsarlm.paging import PAGE_SIZE, trained_view_pages
from pulsarlm.reply import ReplyDecoder
from pulsarlm.settings import Settings
from pulsarlm.store import LocalStore

__all__ = ["EngineStopped", "PulsarLMModelLoadingError", "new_chat"]

PROMPT = ">>> "


def new_chat(
    settings: Settings,
    *,
    system: str | None,
    temperature: float,
    top_p: float,
    reasoning_effort: ReasoningEffort | None,
    show_thinking: bool,
) -> None:
    """Reads user turns until end of input or an interrupt. Raises
    PulsarLMModelLoadingError when no data path holds the model, EngineStopped
    when the engine's loop exited."""
    path = checkpoint.model_path(settings.model, settings.data_dir)
    loaded = checkpoint.load(
        path,
        device=torch.device(settings.device),
        dtype=getattr(torch, settings.dtype),
    )
    cfg = loaded.config
    stops = stop_ids(loaded.tokenizer, cfg.eos_ids)
    store = LocalStore(
        PAGE_SIZE,
        int(settings.store_gib * 2**30),
        checkpoint.store_path(settings.data_dir) if settings.persist else None,
    )
    engine = Engine(
        store,
        engine_model(cfg),
        loaded.weights,
        model_name=f"{settings.model}@{checkpoint.weights_version(path)}",
        page_size=PAGE_SIZE,
        kv_dtype=getattr(torch, settings.dtype),
        kv_cache_gib=settings.kv_cache_gib,
        store_gib=settings.store_gib,
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
    messages: list[Message] = [{"role": "system", "content": system}] if system else []

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
                messages.append({"role": "user", "content": line})
                prompt, system_tokens = encode_prompt(
                    loaded.tokenizer,
                    loaded.chat_template,
                    messages,
                    reasoning_effort=reasoning_effort,
                )
                if system:
                    eviction.n_sink = math.ceil(system_tokens / PAGE_SIZE)
                decoder = ReplyDecoder(loaded.tokenizer, stops, prompt=prompt)
                answer = []
                reasoning_shown = False
                try:
                    async for token in complete(
                        async_engine,
                        store,
                        prompt,
                        sampling=SamplingParams(temperature=temperature, top_p=top_p),
                        max_tokens=None,
                        eviction=eviction,
                    ):
                        part, text = decoder.step(token)
                        if not text:
                            continue
                        if part == "content":
                            if reasoning_shown and not answer:
                                print()
                            answer.append(text)
                            print(text, end="", flush=True)
                        elif show_thinking:
                            reasoning_shown = True
                            print(typer.style(text, dim=True), end="", flush=True)
                except CompletionError as error:
                    print(f"\nerror: request failed: {error}")
                    messages.pop()
                    continue
                print()
                messages.append({"role": "assistant", "content": "".join(answer)})
        finally:
            async_engine.close()

    try:
        asyncio.run(converse())
    except KeyboardInterrupt:
        print()
