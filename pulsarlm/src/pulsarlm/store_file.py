import json
import sqlite3
import sys
import traceback
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import Tensor

from pulsarlm.core.engine import State

type Page = tuple[Tensor, Tensor]
type StateKey = tuple[bytes, tuple[int, ...]]
type SavedState = tuple[StateKey, State, list[bytes], int]

DIGEST_BYTES = 16

SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    key BLOB PRIMARY KEY,
    dtype TEXT NOT NULL,
    shape TEXT NOT NULL,
    nbytes INTEGER NOT NULL,
    k BLOB,
    v BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS states (
    last_page BLOB NOT NULL,
    tail TEXT NOT NULL,
    page_hashes BLOB NOT NULL,
    page_keys BLOB NOT NULL,
    used INTEGER NOT NULL,
    PRIMARY KEY (last_page, tail)
);
"""


class StoreInUseError(RuntimeError):
    pass


def split_digests(joined: bytes) -> list[bytes]:
    return [joined[i : i + DIGEST_BYTES] for i in range(0, len(joined), DIGEST_BYTES)]


def tail_text(tail: tuple[int, ...]) -> str:
    return json.dumps(list(tail))


def tensor_bytes(tensor: Tensor) -> bytes:
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def tensor_from(blob: bytes, dtype: str, shape: str) -> Tensor:
    return (
        torch.frombuffer(bytearray(blob), dtype=torch.uint8)
        .view(getattr(torch, dtype))
        .reshape(json.loads(shape))
    )


def page_from(dtype: str, shape: str, k: bytes | None, v: bytes) -> Page:
    values = tensor_from(v, dtype, shape)
    return (values if k is None else tensor_from(k, dtype, shape)), values


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


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=0, isolation_level=None)
    try:
        connection.execute("PRAGMA locking_mode = EXCLUSIVE")
        connection.execute("PRAGMA journal_mode = WAL")
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
            "SELECT last_page, tail, page_hashes, page_keys, used FROM states"
            " ORDER BY used"
        )
        states = []
        for last_page, tail, page_hashes, page_keys, used in rows:
            tokens = json.loads(tail)
            state = State(page_hashes=split_digests(page_hashes), tail=tokens)
            states.append(
                ((last_page, tuple(tokens)), state, split_digests(page_keys), used)
            )
        return pages, states

    def _put(self, pages: list[tuple[bytes, Page, int]]) -> None:
        rows = []
        for key, (k, v), nbytes in pages:
            assert k.shape == v.shape and k.dtype == v.dtype
            rows.append(
                (
                    key,
                    str(v.dtype).removeprefix("torch."),
                    json.dumps(list(v.shape)),
                    nbytes,
                    None if k is v else tensor_bytes(k),
                    tensor_bytes(v),
                )
            )
        with transaction(self._connection):
            self._connection.executemany(
                "INSERT OR IGNORE INTO pages VALUES (?, ?, ?, ?, ?, ?)", rows
            )

    def _write(
        self, key: StateKey, state: State, page_keys: list[bytes], used: int
    ) -> None:
        last_page, tail = key
        with transaction(self._connection):
            self._connection.execute(
                "INSERT OR REPLACE INTO states VALUES (?, ?, ?, ?, ?)",
                (
                    last_page,
                    tail_text(tail),
                    b"".join(state.page_hashes),
                    b"".join(page_keys),
                    used,
                ),
            )

    def _touch(self, key: StateKey, used: int) -> None:
        last_page, tail = key
        with transaction(self._connection):
            self._connection.execute(
                "UPDATE states SET used = ? WHERE last_page = ? AND tail = ?",
                (used, last_page, tail_text(tail)),
            )

    def _delete(self, state_keys: list[StateKey], page_keys: list[bytes]) -> None:
        with transaction(self._connection):
            self._connection.executemany(
                "DELETE FROM states WHERE last_page = ? AND tail = ?",
                [(last_page, tail_text(tail)) for last_page, tail in state_keys],
            )
            self._connection.executemany(
                "DELETE FROM pages WHERE key = ?", [(key,) for key in page_keys]
            )

    def _read(self, keys: list[bytes]) -> list[Page | None]:
        pages: list[Page | None] = []
        for key in keys:
            row = self._connection.execute(
                "SELECT dtype, shape, k, v FROM pages WHERE key = ?", (key,)
            ).fetchone()
            pages.append(None if row is None else page_from(*row))
        return pages
