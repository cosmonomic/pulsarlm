"""Each request's views, page by page with their scores, for measuring
eviction."""

import hashlib
import json
from collections.abc import AsyncGenerator
from contextlib import aclosing
from pathlib import Path
from typing import Any

from pulsarlm.async_engine import AsyncEngine
from pulsarlm.core.engine import EvictionPolicy, State
from pulsarlm.paging import PAGE_SIZE


def prompt_sha256(prompt: list[int]) -> str:
    return hashlib.sha256(json.dumps(prompt).encode()).hexdigest()


class EvictionLog:
    """Appends one JSON line per finished request to path: its completion id, its
    prompt's length and prompt_sha256, cap_tokens, and each state it reached, in
    order, with the tokens the state covers and, per layer eviction measures, the
    layer's score_threshold and the [start, end, score] of each full page in view,
    score being the engine's page score (None without one). The tokens after the last page are in
    view too. cap_tokens is the most full-page tokens a view keeps at a cycle's end,
    None without max_context_window."""

    def __init__(
        self, path: Path, engine: AsyncEngine, eviction: EvictionPolicy
    ) -> None:
        self._path = path
        self._engine = engine
        self._eviction = eviction

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
            "cap_tokens": self._engine.eviction_report_of(
                State(), self._eviction
            ).cap_tokens,
            "states": [self._view(state) for state in states],
        }
        with self._path.open("a") as f:
            f.write(json.dumps(line) + "\n")

    def _view(self, state: State) -> dict[str, Any]:
        full_pages = max(
            (
                sequence.pages[-1].index + 1
                for sequence in state.layers
                if sequence.pages
            ),
            default=0,
        )
        report = self._engine.eviction_report_of(state, self._eviction)
        return {
            "tokens": full_pages * PAGE_SIZE + len(state.tail),
            "layers": [
                {
                    "layer": scores.layer,
                    "threshold": scores.threshold,
                    "pages": [
                        [page.index * PAGE_SIZE, (page.index + 1) * PAGE_SIZE, score]
                        for page, score in zip(
                            state.layers[scores.layer].pages, scores.page_scores
                        )
                    ],
                }
                for scores in report.layers
            ],
        }
