"""The page store and state cache of one server, in host RAM (NOTES.md section 7)."""

import math
import threading
from collections import OrderedDict
from collections.abc import Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import count

from torch import Tensor

from pulsarlm.core.engine import State

type Page = tuple[Tensor, Tensor]
type StateKey = tuple[bytes, tuple[int, ...]]

NO_PAGES = b""


def state_key(state: State) -> StateKey:
    """The last page hash and the tail, which identify the token history a state
    reaches: the newest page is never evicted, and its hash chains every page
    before it."""
    last_page = state.page_hashes[-1] if state.page_hashes else NO_PAGES
    return last_page, tuple(state.tail)


@dataclass
class StoredPage:
    page: Page
    last_put: int
    references: int = 0
    pins: int = 0


class LocalStore:
    """The engine's page store and an LRU of states over its pages.

    fetch and put run on the engine's loop thread, the rest on the server's, so a
    lock guards everything. A page is deleted once no stored state references it,
    no lease pins it, and every lease open when it was last put has closed: a
    request references the pages it puts only once its state arrives.
    """

    def __init__(self, page_size: int, capacity: int) -> None:
        self._lock = threading.Lock()
        self._page_size = page_size
        self._capacity = capacity
        self._pages: dict[bytes, StoredPage] = {}
        self._states: OrderedDict[StateKey, State] = OrderedDict()
        self._tails: dict[bytes, set[tuple[int, ...]]] = {}
        self._unreferenced: set[bytes] = set()
        self._clock = count()
        # Insertion order is opening order, so the first is the oldest open lease.
        self._open_leases: dict[int, None] = {}

    def fetch(self, hashes: list[bytes]) -> Future[list[Page | None]]:
        with self._lock:
            pages = [
                stored.page if (stored := self._pages.get(h)) is not None else None
                for h in hashes
            ]
        future: Future[list[Page | None]] = Future()
        future.set_result(pages)
        return future

    def put(self, hashes: list[bytes], pages: list[Page]) -> None:
        with self._lock:
            now = next(self._clock)
            for h, page in zip(hashes, pages, strict=True):
                if (stored := self._pages.get(h)) is not None:
                    stored.last_put = now
                else:
                    self._pages[h] = StoredPage(page, now)
                    self._unreferenced.add(h)

    @contextmanager
    def lease(
        self, prompt: list[int], page_digests: list[bytes]
    ) -> Iterator[tuple[State, int]]:
        """The newest stored state reaching a proper prefix of prompt, and how many
        of prompt's tokens it covers; an empty state and 0 when there is none.
        page_digests are prompt's full page hashes. The state's pages stay pinned,
        and pages put meanwhile stay, until the lease closes."""
        with self._lock:
            state, covered = self._find(prompt, page_digests)
            opened = next(self._clock)
            self._open_leases[opened] = None
            for h in state.page_hashes:
                self._pages[h].pins += 1
        try:
            yield state, covered
        finally:
            with self._lock:
                del self._open_leases[opened]
                for h in state.page_hashes:
                    self._pages[h].pins -= 1
                self._delete_unused_pages()

    def write(self, state: State) -> None:
        """Makes state findable. Its pages are in the store: the engine puts them
        before it reports the state, and the writer's lease keeps them."""
        key = state_key(state)
        with self._lock:
            for h in state.page_hashes:
                self._reference(h)
            if (replaced := self._states.pop(key, None)) is not None:
                self._dereference_all(replaced)
            else:
                self._tails.setdefault(key[0], set()).add(key[1])
            self._states[key] = state
            while len(self._states) > self._capacity:
                (last_page, tail), evicted = self._states.popitem(last=False)
                tails = self._tails[last_page]
                tails.remove(tail)
                if not tails:
                    del self._tails[last_page]
                self._dereference_all(evicted)
            self._delete_unused_pages()

    def __len__(self) -> int:
        with self._lock:
            return len(self._states)

    def page_count(self) -> int:
        with self._lock:
            return len(self._pages)

    def _find(self, prompt: list[int], page_digests: list[bytes]) -> tuple[State, int]:
        for page_count in range(len(page_digests), -1, -1):
            last_page = page_digests[page_count - 1] if page_count else NO_PAGES
            start = page_count * self._page_size
            longest: tuple[int, ...] | None = None
            for tail in self._tails.get(last_page, ()):
                end = start + len(tail)
                if (
                    end < len(prompt)
                    and (longest is None or len(tail) > len(longest))
                    and tuple(prompt[start:end]) == tail
                ):
                    longest = tail
            if longest is not None:
                key = (last_page, longest)
                self._states.move_to_end(key)
                return self._states[key], start + len(longest)
        return State(), 0

    def _reference(self, h: bytes) -> None:
        stored = self._pages[h]
        if stored.references == 0:
            self._unreferenced.discard(h)
        stored.references += 1

    def _dereference_all(self, state: State) -> None:
        for h in state.page_hashes:
            stored = self._pages[h]
            stored.references -= 1
            if stored.references == 0:
                self._unreferenced.add(h)

    def _delete_unused_pages(self) -> None:
        oldest_lease = next(iter(self._open_leases), math.inf)
        unused = [
            h
            for h in self._unreferenced
            if self._pages[h].pins == 0 and self._pages[h].last_put < oldest_lease
        ]
        for h in unused:
            del self._pages[h]
        self._unreferenced.difference_update(unused)
