"""Each request's views, page by page with their densities, for measuring
eviction."""

import hashlib
import json
from collections.abc import AsyncGenerator
from contextlib import aclosing
from pathlib import Path
from typing import Any

from pulsarlm.core.engine import EvictionPolicy, State, density_threshold
from pulsarlm.paging import PAGE_SIZE


def prompt_sha256(prompt: list[int]) -> str:
    return hashlib.sha256(json.dumps(prompt).encode()).hexdigest()


class EvictionLog:
    """Appends one JSON line per finished request to path: its completion id, its
    prompt's length and prompt_sha256, cap_tokens, and each state it reached, in
    order, with the tokens the state covers and, per layer eviction measures, the
    layer's threshold and the [start, end, density] of each full page in view, density
    being the page's highest among the layer's heads. The tokens after the last
    page are in view too. cap_tokens is the most full-page tokens a view keeps at
    a cycle's end, None without max_context_window. max_context is the model's
    context in tokens, as the engine has it."""

    def __init__(
        self, path: Path, eviction: EvictionPolicy, max_context: int | None
    ) -> None:
        self._path = path
        self._eviction = eviction
        self._max_context = max_context

    async def record(
        self,
        completion_id: str,
        prompt: list[int],
        tokens: AsyncGenerator[int],
        states: list[State],
    ) -> AsyncGenerator[int]:
        """Yields tokens, then writes the request's line once they end; states
        holds the states the request reached by then."""
        async with aclosing(tokens):
            async for token in tokens:
                yield token
        line = {
            "id": completion_id,
            "prompt_tokens": len(prompt),
            "prompt_sha256": prompt_sha256(prompt),
            "cap_tokens": self._cap_tokens(),
            "states": [self._view(state) for state in states],
        }
        with self._path.open("a") as f:
            f.write(json.dumps(line) + "\n")

    # _cap_tokens and _threshold compute what to_pages, pages_to_evict and the scheduler
    # compute in pulsarlm-core.
    def _cap_tokens(self) -> int | None:
        window = self._eviction.max_context_window
        if window is None:
            return None
        max_pages = window // PAGE_SIZE
        next_cycle_pages = _ceil_div(
            self._eviction.cycle_tokens + PAGE_SIZE - 1, PAGE_SIZE
        )
        return max(max_pages - next_cycle_pages, 0) * PAGE_SIZE

    def _threshold(self, n_q_heads: int) -> float:
        if self._max_context is None:
            return 0.0
        max_context_pages = max(self._max_context // PAGE_SIZE, 1)
        return density_threshold(
            self._eviction.eviction_target, n_q_heads, max_context_pages
        )

    def _view(self, state: State) -> dict[str, Any]:
        full_pages = max(
            (
                sequence.pages[-1].index + 1
                for sequence in state.layers
                if sequence.pages
            ),
            default=0,
        )
        return {
            "tokens": full_pages * PAGE_SIZE + len(state.tail),
            "layers": [
                {
                    "layer": layer,
                    "threshold": self._threshold(len(sequence.pages[0].density)),
                    "pages": [
                        [
                            page.index * PAGE_SIZE,
                            (page.index + 1) * PAGE_SIZE,
                            max(page.density),
                        ]
                        for page in sequence.pages
                    ],
                }
                for layer, sequence in enumerate(state.layers)
                if sequence.pages and sequence.pages[0].density
            ],
        }


def _ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d
