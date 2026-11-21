"""Completes a token prompt from the newest stored state that reaches a prefix of it."""

import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass

from pulsarlm.async_engine import AsyncEngine
from pulsarlm.core.engine import (
    EvictionPolicy,
    RequestError,
    RequestFailed,
    RequestStatistics,
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
    pages the engine had resident, not fetched from the store.
    peak_device_memory: the most bytes of each device's KV cache pool the
    request's lines took at once, lines shared with other requests included.
    preemptions: the times the request gave up its lines to let other requests
    run; recomputed_tokens: the tokens it forwarded again after them.
    time_to_first_token_s runs from the submission to the first token, None
    without one; decode_time_s from the first token to the last."""

    prompt_tokens: int
    cached_tokens: int
    device_cached_tokens: int
    peak_device_memory: int
    preemptions: int
    recomputed_tokens: int
    completion_tokens: int
    time_to_first_token_s: float | None
    decode_time_s: float


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
    first_token_at: float | None = None
    last_token_at = 0.0
    statistics: RequestStatistics | None = None

    def pass_state(state: State) -> None:
        if on_state is not None:
            on_state(state)

    def end(state: State, request_statistics: RequestStatistics) -> None:
        nonlocal statistics
        statistics = request_statistics
        pass_state(state)

    submitted_at = time.monotonic()
    stream = engine.submit(
        prompt,
        sampling=sampling,
        max_tokens=max_tokens,
        eviction=eviction,
        on_cycle=lambda cycle_end: pass_state(cycle_end.state),
        on_end=end,
    )
    try:
        async for token in stream:
            last_token_at = time.monotonic()
            if first_token_at is None:
                first_token_at = last_token_at
            completion_tokens += 1
            yield token
    except RequestFailed as failure:
        raise CompletionError(failure.error) from failure
    finally:
        stream.cancel()
    assert statistics is not None
    if on_completed is not None:
        on_completed(
            CompletionUsage(
                prompt_tokens=len(prompt),
                cached_tokens=statistics.cached_token_count,
                device_cached_tokens=statistics.device_cached_token_count,
                peak_device_memory=statistics.peak_device_memory,
                preemptions=statistics.preemption_count,
                recomputed_tokens=statistics.recomputed_token_count,
                completion_tokens=completion_tokens,
                time_to_first_token_s=None
                if first_token_at is None
                else first_token_at - submitted_at,
                decode_time_s=0.0
                if first_token_at is None
                else last_token_at - first_token_at,
            )
        )
