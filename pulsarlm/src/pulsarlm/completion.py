"""Completes a token prompt from the newest stored state that reaches a prefix of it."""

from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing
from dataclasses import dataclass
from itertools import count

from pulsarlm.async_engine import AsyncEngine
from pulsarlm.core.engine import (
    EvictionPolicy,
    RequestError,
    RequestFailed,
    RequestResult,
    SamplingParams,
    State,
)
from pulsarlm.store import LocalStore

_request_ids = count(1)


class CompletionError(Exception):
    def __init__(self, error: RequestError) -> None:
        super().__init__(error.name)
        self.error = error


@dataclass(frozen=True)
class CompletionUsage:
    """cached_tokens: the prompt tokens the request did not forward, as they were
    on full pages of the state it resumed. device_cached_tokens: those of them on
    pages the engine had resident, not fetched from the store."""

    prompt_tokens: int
    cached_tokens: int
    device_cached_tokens: int
    completion_tokens: int


async def complete(
    engine: AsyncEngine,
    store: LocalStore,
    prompt: list[int],
    *,
    sampling: SamplingParams,
    max_tokens: int | None,
    eviction: EvictionPolicy,
    prefix_cache: bool,
    on_state: Callable[[State], None] | None = None,
    on_completed: Callable[[CompletionUsage], None] | None = None,
) -> AsyncGenerator[int]:
    """Yields the sampled tokens and passes every state the request reaches to
    on_state, then its usage to on_completed once it ends with its state. With
    prefix_cache, the request resumes from the newest stored state reaching a
    prefix of prompt and writes every state it reaches to store; without it, it
    forwards the whole prompt and takes no lease, so the store deletes the pages
    the engine puts for it. Raises CompletionError when the engine ends the
    request with an error, and EngineStopped when its loop exited. Closing the
    iterator early cancels the request."""
    request_id = next(_request_ids)
    completion_tokens = 0
    result: RequestResult | None = None

    def end(ended: RequestResult, handle_state: Callable[[State], None]) -> None:
        nonlocal result
        result = ended
        handle_state(ended.state)

    def report() -> None:
        assert result is not None
        if on_completed is not None:
            on_completed(
                CompletionUsage(
                    prompt_tokens=len(prompt),
                    cached_tokens=result.cached_tokens,
                    device_cached_tokens=result.device_cached_tokens,
                    completion_tokens=completion_tokens,
                )
            )

    if not prefix_cache:

        def pass_state(state: State) -> None:
            if on_state is not None:
                on_state(state)

        async with aclosing(
            _run_request(
                engine,
                State(),
                prompt,
                sampling=sampling,
                max_tokens=max_tokens,
                eviction=eviction,
                request_id=request_id,
                on_cycle_end=pass_state,
                on_end=lambda ended: end(ended, pass_state),
            )
        ) as tokens:
            async for token in tokens:
                completion_tokens += 1
                yield token
        report()
        return

    digests = engine.page_digests(prompt)
    with store.lease(prompt, digests, request_id=request_id) as (lease, token_count):

        def handle_state(state: State) -> None:
            lease.move_to(state, engine.page_keys(state))
            store.write(lease)
            if on_state is not None:
                on_state(state)

        async with aclosing(
            _run_request(
                engine,
                lease.state,
                prompt[token_count:],
                sampling=sampling,
                max_tokens=max_tokens,
                eviction=eviction,
                request_id=request_id,
                on_cycle_end=handle_state,
                on_end=lambda ended: end(ended, handle_state),
            )
        ) as tokens:
            async for token in tokens:
                completion_tokens += 1
                yield token
    report()


async def _run_request(
    engine: AsyncEngine,
    initial_state: State,
    tokens: list[int],
    *,
    sampling: SamplingParams,
    max_tokens: int | None,
    eviction: EvictionPolicy,
    request_id: int,
    on_cycle_end: Callable[[State], None],
    on_end: Callable[[RequestResult], None],
) -> AsyncGenerator[int]:
    """Yields the sampled tokens, passing the state of every cycle end to
    on_cycle_end and the result to on_end."""
    stream = engine.submit(
        initial_state,
        tokens,
        sampling=sampling,
        max_tokens=max_tokens,
        eviction=eviction,
        request_id=request_id,
        on_cycle_end=lambda ended: on_cycle_end(ended.state),
        on_end=on_end,
    )
    try:
        async for token in stream:
            yield token
    except RequestFailed as failure:
        raise CompletionError(failure.error) from failure
    finally:
        stream.cancel()
