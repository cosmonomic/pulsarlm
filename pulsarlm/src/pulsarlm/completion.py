"""Completes a token prompt from the newest stored state that reaches a prefix of it."""

from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass

from pulsarlm.async_engine import AsyncEngine
from pulsarlm.core.engine import (
    EvictionPolicy,
    RequestError,
    RequestFailed,
    RequestResult,
    SamplingParams,
    State,
)


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
    prompt: list[int],
    *,
    sampling: SamplingParams,
    max_tokens: int | None,
    eviction: EvictionPolicy,
    on_state: Callable[[State], None] | None = None,
    on_completed: Callable[[CompletionUsage], None] | None = None,
) -> AsyncGenerator[int]:
    """Yields the sampled tokens and passes every state the request reaches to
    on_state, then its usage to on_completed once it ends with its state. The
    engine resumes the newest state its store holds for a prefix of prompt.
    Raises CompletionError when the engine ends the request with an error, and
    EngineStopped when its loop exited. Closing the iterator early cancels the
    request."""
    completion_tokens = 0
    result: RequestResult | None = None

    def pass_state(state: State) -> None:
        if on_state is not None:
            on_state(state)

    def end(request_result: RequestResult) -> None:
        nonlocal result
        result = request_result
        pass_state(request_result.state)

    stream = engine.submit(
        prompt,
        sampling=sampling,
        max_tokens=max_tokens,
        eviction=eviction,
        on_cycle_end=lambda cycle_end: pass_state(cycle_end.state),
        on_end=end,
    )
    try:
        async for token in stream:
            completion_tokens += 1
            yield token
    except RequestFailed as failure:
        raise CompletionError(failure.error) from failure
    finally:
        stream.cancel()
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
