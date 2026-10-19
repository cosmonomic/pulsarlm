"""Each request's views, as token ranges of its history, for measuring
eviction."""

import hashlib
import json
from collections.abc import AsyncGenerator
from contextlib import aclosing
from pathlib import Path
from typing import Any

from pulsarlm.async_engine import AsyncEngine
from pulsarlm.core.engine import State
from pulsarlm.paging import PAGE_SIZE


def prompt_sha256(prompt: list[int]) -> str:
    return hashlib.sha256(json.dumps(prompt).encode()).hexdigest()


class EvictionLog:
    """Appends one JSON line per finished request to path: its completion id, its
    prompt's length and prompt_sha256, and each state it reached, in order, with
    the tokens the state covers and, per layer eviction measures, the [start, end)
    token ranges of the history's full pages in view. The tokens after the last
    range are in view too."""

    def __init__(self, path: Path, engine: AsyncEngine) -> None:
        self._path = path
        self._engine = engine

    async def record(
        self,
        completion_id: str,
        prompt: list[int],
        tokens: AsyncGenerator[int],
        states: list[State],
    ) -> AsyncGenerator[int]:
        """Yields tokens, then writes the request's line once they end; states
        holds the states the request reached by then."""
        sampled: list[int] = []
        async with aclosing(tokens):
            async for token in tokens:
                sampled.append(token)
                yield token
        history = prompt + sampled
        pages = {h: i for i, h in enumerate(self._engine.page_digests(history))}
        line = {
            "id": completion_id,
            "prompt_tokens": len(prompt),
            "prompt_sha256": prompt_sha256(prompt),
            "states": [_view(state, pages) for state in states],
        }
        with self._path.open("a") as f:
            f.write(json.dumps(line) + "\n")


def _view(state: State, pages: dict[bytes, int]) -> dict[str, Any]:
    full_pages = 0 if state.head is None else pages[state.head] + 1
    return {
        "tokens": full_pages * PAGE_SIZE + len(state.tail),
        "layers": [
            {
                "layer": layer,
                "ranges": _ranges([pages[page.hash] for page in sequence.pages]),
            }
            for layer, sequence in enumerate(state.layers)
            if sequence.pages and sequence.pages[0].density
        ],
    }


def _ranges(view_pages: list[int]) -> list[list[int]]:
    ranges: list[list[int]] = []
    for page in view_pages:
        if ranges and ranges[-1][1] == page * PAGE_SIZE:
            ranges[-1][1] += PAGE_SIZE
        else:
            ranges.append([page * PAGE_SIZE, (page + 1) * PAGE_SIZE])
    return ranges
