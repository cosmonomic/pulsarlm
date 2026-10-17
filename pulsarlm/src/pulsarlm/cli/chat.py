"""A terminal chat with one model, keeping the conversation in memory."""

import asyncio
import readline  # noqa: F401  enables line editing in input()

import typer

from pulsarlm import checkpoint
from pulsarlm.async_engine import AsyncEngine, EngineStopped
from pulsarlm.chat import Message, ReasoningEffort, encode_prompt
from pulsarlm.checkpoint import PulsarLMModelLoadingError
from pulsarlm.completion import CompletionError, complete
from pulsarlm.core.engine import SamplingParams
from pulsarlm.paging import sink_pages
from pulsarlm.reply import ReplyDecoder
from pulsarlm.served_model import load_served_model
from pulsarlm.settings import Settings

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
    served = load_served_model(
        settings, checkpoint.model_path(settings.model, settings.data_dir)
    )
    messages: list[Message] = [{"role": "system", "content": system}] if system else []

    async def converse() -> None:
        async_engine = AsyncEngine(served.engine)
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
                    served.tokenizer,
                    served.chat_template,
                    messages,
                    reasoning_effort=reasoning_effort,
                )
                served.eviction.n_sink = sink_pages(system_tokens)
                decoder = ReplyDecoder(served.tokenizer, served.stop_ids, prompt=prompt)
                answer = []
                reasoning_shown = False
                try:
                    async for token in complete(
                        async_engine,
                        served.store,
                        prompt,
                        sampling=SamplingParams(temperature=temperature, top_p=top_p),
                        max_tokens=None,
                        eviction=served.eviction,
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
