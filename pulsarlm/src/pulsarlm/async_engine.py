"""The engine binding on an asyncio loop."""

import asyncio
import weakref

from pulsarlm.core.engine import (
    CycleEnd,
    Engine,
    EngineStat,
    EvictionPolicy,
    Request,
    RequestError,
    SamplingParams,
    State,
)


class EngineStopped(Exception):
    """The engine's loop exited: every request failed and new ones are refused."""


class RequestStream:
    """Cancels its request when dropped."""

    def __init__(self, engine: Engine, request: Request) -> None:
        self._engine = engine
        self._request = request
        self._woken = asyncio.Event()

    async def take(
        self,
    ) -> tuple[list[int], list[CycleEnd], State | RequestError | None]:
        """Waits for results, then returns them as Request.take does. Raises
        EngineStopped when the engine's loop exited."""
        while True:
            self._woken.clear()
            tokens, cycle_ends, end = self._request.take()
            if end == RequestError.LOOP_EXITED:
                raise EngineStopped(self._engine.exception())
            if tokens or cycle_ends or end is not None:
                return tokens, cycle_ends, end
            await self._woken.wait()

    def cancel(self) -> None:
        self._request.cancel()

    def wake(self) -> None:
        self._woken.set()


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
        initial_state: State,
        tokens: list[int],
        *,
        sampling: SamplingParams,
        max_tokens: int | None,
        eviction: EvictionPolicy,
        request_id: int,
    ) -> RequestStream:
        """Raises EngineStopped when the engine's loop exited, ValueError for a
        request the engine cannot serve."""
        if (message := self._engine.exception()) is not None:
            raise EngineStopped(message)
        request = self._engine.submit(
            initial_state,
            tokens,
            sampling=sampling,
            max_tokens=max_tokens,
            eviction=eviction,
            request_id=request_id,
        )
        stream = RequestStream(self._engine, request)
        self._streams.add(stream)
        return stream

    def page_digests(self, tokens: list[int]) -> list[bytes]:
        return self._engine.page_digests(tokens)

    def page_keys(self, state: State) -> list[bytes]:
        return self._engine.page_keys(state)

    def stat(self) -> EngineStat:
        return self._engine.stat()

    def close(self) -> None:
        self._loop.remove_reader(self._engine.fileno())

    def _wake_streams(self) -> None:
        self._engine.clear_wake()
        for stream in self._streams:
            stream.wake()
