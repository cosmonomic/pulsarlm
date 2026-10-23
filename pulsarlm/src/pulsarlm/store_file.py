import importlib.metadata
import json
import sqlite3
import struct
import sys
import traceback
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import Tensor

from pulsarlm.core.engine import ActivePage, LayerSequence, Moments, State

type Page = Tensor
type StateKey = tuple[bytes, tuple[int, ...]]
type SavedState = tuple[StateKey, State, list[bytes], int]

DIGEST_BYTES = 16
INDEX_BYTES = 8
FLOAT_BYTES = 4
MOMENTS = struct.Struct("<Qdd")
VERSION = importlib.metadata.version("pulsarlm")

SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    key BLOB PRIMARY KEY,
    nbytes INTEGER NOT NULL,
    bytes BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS states (
    head BLOB NOT NULL,
    tail TEXT NOT NULL,
    layers BLOB NOT NULL,
    page_keys BLOB NOT NULL,
    used INTEGER NOT NULL,
    PRIMARY KEY (head, tail)
);
"""


class StoreInUseError(RuntimeError):
    pass


def split_digests(joined: bytes) -> list[bytes]:
    return [joined[i : i + DIGEST_BYTES] for i in range(0, len(joined), DIGEST_BYTES)]


def layers_bytes(layers: list[LayerSequence]) -> bytes:
    """Per layer, its page count, its mass sums per page and its density moments
    as 4 little-endian bytes each, its page hashes, its page indices as 8
    little-endian bytes each, its mass sums then its weight sums as little-endian
    fp32, page by page, then its density moments and its score moments, each as
    count (8 little-endian bytes), mean and m2 (little-endian fp64). Every page of
    a layer holds as many mass sums."""
    parts = []
    for layer in layers:
        pages = layer.pages
        per_page = len(pages[0].mass_sum) if pages else 0
        if any(len(page.mass_sum) != per_page for page in pages):
            raise ValueError("every page of a layer must hold as many mass sums")
        parts.append(
            struct.pack("<III", len(pages), per_page, len(layer.density_moments))
        )
        parts.extend(page.hash for page in pages)
        parts.append(struct.pack(f"<{len(pages)}Q", *(page.index for page in pages)))
        mass_sums = [m for page in pages for m in page.mass_sum]
        parts.append(struct.pack(f"<{len(mass_sums)}f", *mass_sums))
        parts.append(
            struct.pack(f"<{len(pages)}f", *(page.weight_sum for page in pages))
        )
        for moments in [*layer.density_moments, layer.score_moments]:
            parts.append(MOMENTS.pack(moments.count, moments.mean, moments.m2))
    return b"".join(parts)


def layers_from(blob: bytes) -> list[LayerSequence]:
    layers = []
    offset = 0
    while offset < len(blob):
        count, per_page, heads = struct.unpack_from("<III", blob, offset)
        offset += 12
        hashes = split_digests(blob[offset : offset + count * DIGEST_BYTES])
        offset += count * DIGEST_BYTES
        indices = struct.unpack_from(f"<{count}Q", blob, offset)
        offset += count * INDEX_BYTES
        mass_sums = struct.unpack_from(f"<{count * per_page}f", blob, offset)
        offset += count * per_page * FLOAT_BYTES
        weight_sums = struct.unpack_from(f"<{count}f", blob, offset)
        offset += count * FLOAT_BYTES
        moments = []
        for _ in range(heads + 1):
            moments.append(Moments(*MOMENTS.unpack_from(blob, offset)))
            offset += MOMENTS.size
        pages = [
            ActivePage(
                h,
                indices[i],
                list(mass_sums[i * per_page : (i + 1) * per_page]),
                weight_sums[i],
            )
            for i, h in enumerate(hashes)
        ]
        layers.append(LayerSequence(pages, moments[:-1], moments[-1]))
    return layers


def tail_text(tail: tuple[int, ...]) -> str:
    return json.dumps(list(tail))


def page_from(blob: bytes) -> Page:
    return torch.frombuffer(bytearray(blob), dtype=torch.uint8)


def report_failure(operation: Future[None]) -> None:
    if (error := operation.exception()) is not None:
        print("pulsarlm store file write failed:", file=sys.stderr)
        traceback.print_exception(error)


@contextmanager
def transaction(connection: sqlite3.Connection) -> Iterator[None]:
    connection.execute("BEGIN")
    try:
        yield
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    connection.execute("COMMIT")


def discard_other_version(connection: sqlite3.Connection) -> None:
    """Drops every page and state of a file another pulsarlm version wrote."""
    connection.execute("CREATE TABLE IF NOT EXISTS version (version TEXT NOT NULL)")
    if connection.execute("SELECT version FROM version").fetchall() == [(VERSION,)]:
        return
    connection.execute("DROP TABLE IF EXISTS states")
    connection.execute("DROP TABLE IF EXISTS pages")
    connection.execute("DELETE FROM version")
    connection.execute("INSERT INTO version VALUES (?)", (VERSION,))
    connection.execute("VACUUM")


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=0, isolation_level=None)
    try:
        connection.execute("PRAGMA locking_mode = EXCLUSIVE")
        connection.execute("PRAGMA journal_mode = WAL")
        discard_other_version(connection)
        connection.executescript(SCHEMA)
        # A write takes the exclusive lock, which EXCLUSIVE locking mode keeps until the connection closes.
        connection.execute("BEGIN EXCLUSIVE")
        connection.execute("COMMIT")
    except sqlite3.OperationalError as error:
        connection.close()
        raise StoreInUseError(f"{path} is open in another process") from error
    return connection


class StoreFile:
    """Runs its operations in submission order on one thread, which owns the
    connection. Each write is one transaction, so after a crash the file holds
    every write up to some point and none after it."""

    def __init__(self, path: Path) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="pulsarlm-store"
        )
        self._connection = self._executor.submit(connect, path).result()

    def load(self) -> tuple[dict[bytes, int], list[SavedState]]:
        """Every page's size, and every state in increasing order of use."""
        return self._executor.submit(self._load).result()

    def put(self, pages: list[tuple[bytes, Page, int]]) -> None:
        self._submit(self._put, pages)

    def write(
        self, key: StateKey, state: State, page_keys: list[bytes], used: int
    ) -> None:
        self._submit(self._write, key, state, page_keys, used)

    def touch(self, key: StateKey, used: int) -> None:
        self._submit(self._touch, key, used)

    def delete(self, state_keys: list[StateKey], page_keys: list[bytes]) -> None:
        self._submit(self._delete, state_keys, page_keys)

    def read(self, keys: list[bytes]) -> Future[list[Page | None]]:
        return self._executor.submit(self._read, keys)

    def _submit(self, operation: Callable[..., None], *args: object) -> None:
        self._executor.submit(operation, *args).add_done_callback(report_failure)

    def _load(self) -> tuple[dict[bytes, int], list[SavedState]]:
        pages = dict(self._connection.execute("SELECT key, nbytes FROM pages"))
        rows = self._connection.execute(
            "SELECT head, tail, layers, page_keys, used FROM states ORDER BY used"
        )
        states = []
        for head, tail, layers, page_keys, used in rows:
            tokens = json.loads(tail)
            state = State(head=head or None, layers=layers_from(layers), tail=tokens)
            states.append(
                ((head, tuple(tokens)), state, split_digests(page_keys), used)
            )
        return pages, states

    def _put(self, pages: list[tuple[bytes, Page, int]]) -> None:
        rows = [(key, nbytes, page.numpy().tobytes()) for key, page, nbytes in pages]
        with transaction(self._connection):
            self._connection.executemany(
                "INSERT OR IGNORE INTO pages VALUES (?, ?, ?)", rows
            )

    def _write(
        self, key: StateKey, state: State, page_keys: list[bytes], used: int
    ) -> None:
        head, tail = key
        with transaction(self._connection):
            self._connection.execute(
                "INSERT OR REPLACE INTO states (head, tail, layers, page_keys, used)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    head,
                    tail_text(tail),
                    layers_bytes(state.layers),
                    b"".join(page_keys),
                    used,
                ),
            )

    def _touch(self, key: StateKey, used: int) -> None:
        head, tail = key
        with transaction(self._connection):
            self._connection.execute(
                "UPDATE states SET used = ? WHERE head = ? AND tail = ?",
                (used, head, tail_text(tail)),
            )

    def _delete(self, state_keys: list[StateKey], page_keys: list[bytes]) -> None:
        with transaction(self._connection):
            self._connection.executemany(
                "DELETE FROM states WHERE head = ? AND tail = ?",
                [(head, tail_text(tail)) for head, tail in state_keys],
            )
            self._connection.executemany(
                "DELETE FROM pages WHERE key = ?", [(key,) for key in page_keys]
            )

    def _read(self, keys: list[bytes]) -> list[Page | None]:
        pages: list[Page | None] = []
        for key in keys:
            row = self._connection.execute(
                "SELECT bytes FROM pages WHERE key = ?", (key,)
            ).fetchone()
            pages.append(None if row is None else page_from(*row))
        return pages
