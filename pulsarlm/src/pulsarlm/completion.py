"""Completes a token prompt from the newest stored state that reaches a prefix of it."""

from collections.abc import AsyncGenerator, Callable
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
    on_state: Callable[[State], None] | None = None,
) -> AsyncGenerator[int]:
    """Yields the sampled tokens and writes every state the request reaches to
    store, passing each to on_state too. Raises CompletionError when the engine
    ends the request with an error, and EngineStopped when its loop exited.
    Closing the iterator early cancels the request."""
    request_id = next(_request_ids)
    digests = engine.page_digests(prompt)

    def reach(state: State) -> None:
        store.write(state, engine.page_keys(state), request_id=request_id)
        if on_state is not None:
            on_state(state)

    with store.lease(prompt, digests, request_id=request_id) as (state, covered):
        stream = engine.submit(
            state,
            prompt[covered:],
            sampling=sampling,
            max_tokens=max_tokens,
            eviction=eviction,
            request_id=request_id,
        )
        try:
            while True:
                tokens, cycle_ends, end = await stream.take()
                for cycle_end in cycle_ends:
                    reach(cycle_end.reached)
                for token in tokens:
                    yield token
                if isinstance(end, State):
                    reach(end)
                    return
                if end is not None:
                    raise CompletionError(end)
        finally:
            stream.cancel()
