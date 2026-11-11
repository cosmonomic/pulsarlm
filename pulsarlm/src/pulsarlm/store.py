"""The page store and state cache of one server, in host RAM and optionally a file."""

import threading
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
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


class Lease:
    """A running request's references in the store: the pages of its state, the
    one it resumed or last moved to, which its next state builds on, and the pages
    its puts brought that no such state covers yet. Leaving the store's lease
    context releases them all."""

    def __init__(
        self, store: "LocalStore", state: State, page_keys: list[bytes]
    ) -> None:
        self.state = state
        self._store = store
        self._page_keys = page_keys
        self._puts: list[list[bytes]] = []

    def move_to(self, state: State, page_keys: list[bytes]) -> None:
        """The request reached state, whose pages are page_keys: the lease
        references them in place of its previous state's pages, and releases its
        puts that state covers."""
        self._store._move(self, state, page_keys)


class LocalStore:
    """The engine's page store and an LRU of states over its pages.

    fetch and put run on the engine's loop thread, the rest on the server's, so a
    lock guards everything. A page is deleted once neither a stored state nor a
    running request's lease references it. While the pages take more than
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
        # By request id, the put of a step's pages that claim began and put ends.
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

    def fetch(self, hashes: list[bytes]) -> Future[list[Page | None]]:
        with self._lock:
            pages = [
                stored.page if (stored := self._pages.get(h)) is not None else None
                for h in hashes
            ]
            in_file = [
                i
                for i, h in enumerate(hashes)
                if (stored := self._pages.get(h)) is not None and stored.page is None
            ]
        future: Future[list[Page | None]] = Future()
        if self._file is None or not in_file:
            future.set_result(pages)
            return future

        def resolve(read: Future[list[Page | None]]) -> None:
            try:
                loaded = read.result()
            except Exception as error:
                future.set_exception(error)
                return
            with self._lock:
                for i, page in zip(in_file, loaded, strict=True):
                    pages[i] = page
                    stored = self._pages.get(hashes[i])
                    if page is not None and stored is not None and stored.page is None:
                        stored.page = page
            future.set_result(pages)

        self._file.read([hashes[i] for i in in_file]).add_done_callback(resolve)
        return future

    def claim(self, hashes: list[bytes], owners: Sequence[tuple[int, int]]) -> list[int]:
        """The indices of the pages of hashes that an open lease owns and the store
        lacks, which the engine copies and hands to put. owners pairs a request id
        with the index of a page its step produced. The pages the store has are
        referenced for the leases owning them now, so none is deleted before put."""
        with self._lock:
            missing: set[int] = set()
            owned: dict[int, list[bytes]] = {}
            for request_id, index in owners:
                if request_id not in self._leases:
                    continue
                keys = owned.setdefault(request_id, [])
                if hashes[index] in self._pages:
                    self._reference(hashes[index])
                    keys.append(hashes[index])
                else:
                    missing.add(index)
            for request_id, keys in owned.items():
                self._leases[request_id]._puts.append(keys)
            self._step_puts = owned
            return sorted(missing)

    def put(
        self,
        hashes: list[bytes],
        pages: list[Page],
        owners: Sequence[tuple[int, int]] = (),
    ) -> None:
        """owners pairs a request id with the index of a page its step produced; a
        lease's pages join the put claim began for its step. The store then fits with
        room for another put of these pages' size: the engine takes a step's host
        pages from memory the store's capacity bounds before it puts them."""
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
            for request_id, index in owners:
                if request_id in self._leases:
                    owned.setdefault(request_id, []).append(hashes[index])
            for request_id, keys in owned.items():
                for h in keys:
                    self._reference(h)
                if request_id in self._step_puts:
                    self._step_puts[request_id].extend(keys)
                else:
                    self._leases[request_id]._puts.append(keys)
            self._step_puts = {}
            if self._file is not None and new:
                self._file.put(new)
            self._fit(room=sum(page.nbytes for page in pages))

    @contextmanager
    def lease(
        self,
        prompt: list[int],
        page_digests: list[bytes],
        *,
        request_id: int,
    ) -> Iterator[tuple[Lease, int]]:
        """A lease on the newest stored state reaching a proper prefix of prompt,
        and how many of prompt's tokens it covers; on an empty state and 0 when
        there is none. page_digests are prompt's full page
        hashes. request_id, distinct among open leases, names the request whose
        puts the lease references."""
        with self._lock:
            found, covered = self._find(prompt, page_digests)
            for h in found.page_keys:
                self._reference(h)
            lease = Lease(self, found.state, found.page_keys)
            self._leases[request_id] = lease
        try:
            yield lease, covered
        finally:
            with self._lock:
                del self._leases[request_id]
                self._dereference(lease._page_keys)
                for keys in lease._puts:
                    self._dereference(keys)
                self._fit()

    def write(self, lease: Lease) -> None:
        """Makes lease's state findable."""
        with self._lock:
            state = lease.state
            key = state_key(state)
            page_keys = lease._page_keys
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

    def _move(self, lease: Lease, state: State, page_keys: list[bytes]) -> None:
        with self._lock:
            for h in page_keys:
                self._reference(h)
            self._dereference(lease._page_keys)
            lease.state = state
            lease._page_keys = page_keys
            self._release_covered(lease, set(page_keys))
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

    def _find(
        self, prompt: list[int], page_digests: list[bytes]
    ) -> tuple[StoredState, int]:
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
                return self._states[key], start + len(longest)
        return StoredState(State(), []), 0

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
            (i for i, keys in enumerate(lease._puts) if not covered.isdisjoint(keys)),
            default=-1,
        )
        for keys in lease._puts[: last + 1]:
            self._dereference(keys)
        del lease._puts[: last + 1]

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
