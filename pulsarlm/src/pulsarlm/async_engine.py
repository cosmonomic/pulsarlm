"""The engine binding on an asyncio loop."""

import asyncio
import weakref
from collections.abc import AsyncIterator, Callable

from pulsarlm.core.engine import (
    CycleEnd,
    Engine,
    EngineStat,
    EvictionPolicy,
    EvictionReport,
    Request,
    RequestError,
    RequestFailed,
    RequestResult,
    SamplingParams,
    State,
)


class EngineStopped(Exception):
    """The engine's loop exited: every request failed and new ones are refused."""


class RequestStream:
    """The tokens of a request as they are sampled. Raises RequestFailed when the
    engine ends the request with an error, and EngineStopped when its loop exited.
    Cancels its request when dropped."""

    def __init__(self, engine: Engine, request: Request) -> None:
        self._engine = engine
        self._request = request
        self._wake_event = asyncio.Event()

    def __aiter__(self) -> AsyncIterator[int]:
        return self._tokens()

    async def _tokens(self) -> AsyncIterator[int]:
        while True:
            self._wake_event.clear()
            try:
                tokens = self._request.take()
            except RequestFailed as failure:
                if failure.error == RequestError.LOOP_EXITED:
                    raise EngineStopped(self._engine.exception()) from failure
                raise
            if tokens is None:
                return
            if not tokens:
                await self._wake_event.wait()
            for token in tokens:
                yield token

    def cancel(self) -> None:
        self._request.cancel()

    def wake(self) -> None:
        self._wake_event.set()


class AsyncEngine:
    """Wakes the streams of live requests when the engine's eventfd is readable.
    Created on the loop it serves."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._loop = asyncio.get_running_loop()
        self._streams: weakref.WeakSet[RequestStream] = weakref.WeakSet()
        self._loop.add_reader(engine.fileno(), self._wake_streams)

    def submit(
        self,
        tokens: list[int],
        *,
        sampling: SamplingParams,
        max_tokens: int | None,
        eviction: EvictionPolicy,
        on_cycle_end: Callable[[CycleEnd], None] | None = None,
        on_end: Callable[[RequestResult], None] | None = None,
    ) -> RequestStream:
        """Raises EngineStopped when the engine's loop exited, ValueError for a
        request the engine cannot serve. The stream calls on_cycle_end and on_end
        on the event loop's thread."""
        if (message := self._engine.exception()) is not None:
            raise EngineStopped(message)
        request = self._engine.submit(
            tokens,
            sampling=sampling,
            max_tokens=max_tokens,
            eviction=eviction,
            on_cycle_end=on_cycle_end,
            on_end=on_end,
        )
        stream = RequestStream(self._engine, request)
        self._streams.add(stream)
        return stream

    def eviction_report_of(
        self, state: State, eviction: EvictionPolicy
    ) -> EvictionReport:
        return self._engine.eviction_report_of(state, eviction)

    def stat(self) -> EngineStat:
        return self._engine.stat()

    def close(self) -> None:
        self._loop.remove_reader(self._engine.fileno())

    def _wake_streams(self) -> None:
        self._engine.clear_wake()
        for stream in self._streams:
            stream.wake()
