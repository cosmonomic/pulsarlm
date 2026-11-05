"""Completes a token prompt from the newest stored state that reaches a prefix of it."""

from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing
from itertools import count

from pulsarlm.async_engine import AsyncEngine
from pulsarlm.core.engine import EvictionPolicy, RequestError, SamplingParams, State
from pulsarlm.store import LocalStore

_request_ids = count(1)


class CompletionError(Exception):
    def __init__(self, error: RequestError) -> None:
        super().__init__(error.name)
        self.error = error


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
) -> AsyncGenerator[int]:
    """Yields the sampled tokens and passes every state the request reaches to
    on_state. With prefix_cache, the request resumes from the newest stored state
    reaching a prefix of prompt and writes every state it reaches to store;
    without it, it forwards the whole prompt and takes no lease, so the store
    deletes the pages the engine puts for it. Raises CompletionError when the
    engine ends the request with an error, and EngineStopped when its loop exited.
    Closing the iterator early cancels the request."""
    request_id = next(_request_ids)
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
                handle_state=pass_state,
            )
        ) as tokens:
            async for token in tokens:
                yield token
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
                handle_state=handle_state,
            )
        ) as tokens:
            async for token in tokens:
                yield token


async def _run_request(
    engine: AsyncEngine,
    initial_state: State,
    tokens: list[int],
    *,
    sampling: SamplingParams,
    max_tokens: int | None,
    eviction: EvictionPolicy,
    request_id: int,
    handle_state: Callable[[State], None],
) -> AsyncGenerator[int]:
    """Yields the sampled tokens, passing every state the request reaches to
    handle_state."""
    stream = engine.submit(
        initial_state,
        tokens,
        sampling=sampling,
        max_tokens=max_tokens,
        eviction=eviction,
        request_id=request_id,
    )
    try:
        while True:
            sampled, cycle_ends, end = await stream.take()
            for cycle_end in cycle_ends:
                handle_state(cycle_end.state)
            for token in sampled:
                yield token
            if isinstance(end, State):
                handle_state(end)
                return
            if end is not None:
                raise CompletionError(end)
    finally:
        stream.cancel()
