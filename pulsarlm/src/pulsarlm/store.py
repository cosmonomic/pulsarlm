"""The page store and state cache of one server, in host RAM and optionally a file."""

import asyncio
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path

from pulsarlm.core.engine import State
from pulsarlm.store_file import Page, StateKey, StoreFile

NO_PAGES = b""


def state_key(state: State) -> StateKey:
    """The head and the tail, which identify the token history a state reaches:
    the head's hash chains every page before it."""
    return state.head or NO_PAGES, tuple(state.tail)


@dataclass
class StoredPage:
    page: Page | None
    nbytes: int
    references: int = 0


@dataclass
class StoredState:
    state: State
    page_keys: list[bytes]


@dataclass
class Lease:
    """A running request's references in the store: the pages of its state, the
    one it resumed or last wrote, which its next state builds on, and the pages
    written for it that no such state covers yet."""

    page_keys: list[bytes]
    puts: list[list[bytes]] = field(default_factory=list)


class LocalStore:
    """The engine's page store and an LRU of states over its pages.

    claim, write_pages, write_state and release run on the engine's loop thread,
    read_pages and read_state on the server's event loop, so a lock guards
    everything. A page is deleted once neither a stored state nor a running
    request's lease references it. While the pages take more than
    capacity_bytes, the least recently used states are dropped, down to the newest
    one; pages leases reference can keep the store above its capacity.

    With a path, the file holds a copy of every page and state, read at
    construction; a page only in the file is read into RAM when fetched.
    """

    def __init__(
        self, page_size: int, capacity_bytes: int = 4 * 2**30, path: Path | None = None
    ) -> None:
        self._lock = threading.Lock()
        self._page_size = page_size
        self._capacity_bytes = capacity_bytes
        self._page_bytes = 0
        self._pages: dict[bytes, StoredPage] = {}
        self._states: OrderedDict[StateKey, StoredState] = OrderedDict()
        self._tails: dict[bytes, set[tuple[int, ...]]] = {}
        self._unreferenced_pages: set[bytes] = set()
        self._leases: dict[int, Lease] = {}
        self._lease_ids = count()
        # By lease id, the write of a step's pages that claim began and write_pages ends.
        self._step_puts: dict[int, list[bytes]] = {}
        self._clock = count()
        self._file = None if path is None else StoreFile(path)
        if self._file is not None:
            self._load(self._file)

    def _load(self, file: StoreFile) -> None:
        sizes, saved = file.load()
        for h, nbytes in sizes.items():
            self._pages[h] = StoredPage(None, nbytes)
            self._page_bytes += nbytes
        self._unreferenced_pages.update(sizes)
        incomplete = []
        last_used = -1
        for key, state, page_keys, used in saved:
            last_used = used
            if not all(h in self._pages for h in page_keys):
                incomplete.append(key)
                continue
            for h in page_keys:
                self._reference(h)
            self._tails.setdefault(key[0], set()).add(key[1])
            self._states[key] = StoredState(state, page_keys)
        self._clock = count(last_used + 1)
        file.delete(incomplete, [])
        self._fit()

    async def read_pages(self, hashes: list[bytes]) -> list[Page | None]:
        with self._lock:
            pages = [
                stored_page.page
                if (stored_page := self._pages.get(h)) is not None
                else None
                for h in hashes
            ]
            in_file = [
                i
                for i, h in enumerate(hashes)
                if (stored_page := self._pages.get(h)) is not None
                and stored_page.page is None
            ]
        if self._file is None or not in_file:
            return pages
        loaded = await asyncio.wrap_future(
            self._file.read([hashes[i] for i in in_file])
        )
        with self._lock:
            for i, page in zip(in_file, loaded, strict=True):
                pages[i] = page
                stored_page = self._pages.get(hashes[i])
                if (
                    page is not None
                    and stored_page is not None
                    and stored_page.page is None
                ):
                    stored_page.page = page
        return pages

    def claim(
        self, hashes: list[bytes], owners: Sequence[tuple[int, int]]
    ) -> list[int]:
        """The indices of the pages of hashes that an open lease owns and the store
        lacks, which the engine copies and hands to write_pages. owners pairs a
        lease id with the index of a page its request's step produced. The pages
        the store has are referenced for the leases owning them now, so none is
        deleted before write_pages."""
        with self._lock:
            missing: set[int] = set()
            owned: dict[int, list[bytes]] = {}
            for lease_id, index in owners:
                if lease_id not in self._leases:
                    continue
                keys = owned.setdefault(lease_id, [])
                if hashes[index] in self._pages:
                    self._reference(hashes[index])
                    keys.append(hashes[index])
                else:
                    missing.add(index)
            for lease_id, keys in owned.items():
                self._leases[lease_id].puts.append(keys)
            self._step_puts = owned
            return sorted(missing)

    def write_pages(
        self,
        hashes: list[bytes],
        pages: list[Page],
        owners: Sequence[tuple[int, int]] = (),
    ) -> None:
        """owners pairs a lease id with the index of a page its request's step
        produced; a lease's pages join the write claim began for its step. The store
        then fits with room for another write of these pages' size: the engine takes
        a step's host pages from memory the store's capacity bounds before it writes
        them."""
        with self._lock:
            new = []
            for h, page in zip(hashes, pages, strict=True):
                if h not in self._pages:
                    nbytes = page.nbytes
                    self._pages[h] = StoredPage(page, nbytes)
                    self._page_bytes += nbytes
                    self._unreferenced_pages.add(h)
                    new.append((h, page, nbytes))
            owned: dict[int, list[bytes]] = {}
            for lease_id, index in owners:
                if lease_id in self._leases:
                    owned.setdefault(lease_id, []).append(hashes[index])
            for lease_id, keys in owned.items():
                for h in keys:
                    self._reference(h)
                if lease_id in self._step_puts:
                    self._step_puts[lease_id].extend(keys)
                else:
                    self._leases[lease_id].puts.append(keys)
            self._step_puts = {}
            if self._file is not None and new:
                self._file.put(new)
            self._fit(room=sum(page.nbytes for page in pages))

    async def read_state(
        self, tokens: list[int], page_digests: list[bytes]
    ) -> tuple[State, int]:
        """The newest stored state reaching a proper prefix of tokens, or an empty
        state when there is none, and the id of a new lease on its pages.
        page_digests are tokens' full page hashes."""
        with self._lock:
            stored_state = self._find(tokens, page_digests)
            for h in stored_state.page_keys:
                self._reference(h)
            lease_id = next(self._lease_ids)
            self._leases[lease_id] = Lease(stored_state.page_keys)
        return stored_state.state, lease_id

    def write_state(self, lease_id: int, state: State, page_keys: list[bytes]) -> None:
        """The lease's request reached state, whose pages are page_keys: the lease
        references them in place of its previous state's pages and releases the
        puts that state covers, and the state becomes findable."""
        with self._lock:
            lease = self._leases[lease_id]
            for h in page_keys:
                self._reference(h)
            self._dereference(lease.page_keys)
            lease.page_keys = page_keys
            self._release_covered(lease, set(page_keys))
            key = state_key(state)
            for h in page_keys:
                self._reference(h)
            if (replaced := self._states.pop(key, None)) is not None:
                self._dereference(replaced.page_keys)
            else:
                self._tails.setdefault(key[0], set()).add(key[1])
            self._states[key] = StoredState(state, page_keys)
            if self._file is not None:
                self._file.write(key, state, page_keys, next(self._clock))
            self._fit()

    def release(self, lease_id: int) -> None:
        with self._lock:
            lease = self._leases.pop(lease_id)
            self._dereference(lease.page_keys)
            for keys in lease.puts:
                self._dereference(keys)
            self._fit()

    def __len__(self) -> int:
        with self._lock:
            return len(self._states)

    def page_count(self) -> int:
        with self._lock:
            return len(self._pages)

    def byte_count(self) -> int:
        """The bytes the stored pages take."""
        with self._lock:
            return self._page_bytes

    def _find(self, prompt: list[int], page_digests: list[bytes]) -> StoredState:
        for page_count in range(len(page_digests), -1, -1):
            head = page_digests[page_count - 1] if page_count else NO_PAGES
            start = page_count * self._page_size
            longest: tuple[int, ...] | None = None
            for tail in self._tails.get(head, ()):
                end = start + len(tail)
                if (
                    end < len(prompt)
                    and (longest is None or len(tail) > len(longest))
                    and tuple(prompt[start:end]) == tail
                ):
                    longest = tail
            if longest is not None:
                key = (head, longest)
                self._states.move_to_end(key)
                if self._file is not None:
                    self._file.touch(key, next(self._clock))
                return self._states[key]
        return StoredState(State(), [])

    def _reference(self, h: bytes) -> None:
        stored = self._pages[h]
        if stored.references == 0:
            self._unreferenced_pages.discard(h)
        stored.references += 1

    def _dereference(self, keys: list[bytes]) -> None:
        for h in keys:
            page = self._pages[h]
            page.references -= 1
            if page.references == 0:
                self._unreferenced_pages.add(h)

    def _release_covered(self, lease: Lease, covered: set[bytes]) -> None:
        """Releases the puts up to the last one holding a page of the state moved
        to: its step, and every earlier one, are behind the state, as a request puts
        its pages in step order and a state ends at a step's end."""
        last = max(
            (i for i, keys in enumerate(lease.puts) if not covered.isdisjoint(keys)),
            default=-1,
        )
        for keys in lease.puts[: last + 1]:
            self._dereference(keys)
        del lease.puts[: last + 1]

    def _fit(self, room: int = 0) -> None:
        """Drops the least recently used states, down to the newest one, until the
        pages and room fit the capacity."""
        deleted_states: list[StateKey] = []
        deleted_pages = self._delete_unused_pages()
        while self._page_bytes + room > self._capacity_bytes and len(self._states) > 1:
            key, dropped = self._states.popitem(last=False)
            head, tail = key
            tails = self._tails[head]
            tails.remove(tail)
            if not tails:
                del self._tails[head]
            self._dereference(dropped.page_keys)
            deleted_states.append(key)
            deleted_pages += self._delete_unused_pages()
        if self._file is not None and (deleted_states or deleted_pages):
            self._file.delete(deleted_states, deleted_pages)

    def _delete_unused_pages(self) -> list[bytes]:
        unused = list(self._unreferenced_pages)
        for h in unused:
            self._page_bytes -= self._pages.pop(h).nbytes
        self._unreferenced_pages.clear()
        return unused


class EmptyStore:
    """The page store of an engine without a prefix cache: it never finds or
    writes a state and keeps no page, so every request runs its whole prompt."""

    def __init__(self) -> None:
        self._lease_ids = count()

    async def read_pages(self, hashes: list[bytes]) -> list[Page | None]:
        return [None] * len(hashes)

    async def read_state(
        self, tokens: list[int], page_digests: list[bytes]
    ) -> tuple[State, int]:
        return State(), next(self._lease_ids)

    def claim(
        self, hashes: list[bytes], owners: Sequence[tuple[int, int]]
    ) -> list[int]:
        return []

    def write_pages(
        self,
        hashes: list[bytes],
        pages: list[Page],
        owners: Sequence[tuple[int, int]] = (),
    ) -> None:
        pass

    def write_state(self, lease_id: int, state: State, page_keys: list[bytes]) -> None:
        pass

    def release(self, lease_id: int) -> None:
        pass
