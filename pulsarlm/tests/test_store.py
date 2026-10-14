"""LocalStore with digests chained in Python, as Engine.page_digests chains them."""

import hashlib
import sqlite3
import subprocess
import sys
import textwrap
from contextlib import closing
from pathlib import Path

import torch

from pulsarlm.core.engine import State
from pulsarlm.store import LocalStore
from pulsarlm.store_file import StoreInUseError

PAGE_SIZE = 4
# page() makes a page of two one-element float32 tensors.
PAGE_BYTES = 8
ROOMY = 1 << 20


def page_digests(tokens: list[int]) -> list[bytes]:
    digests = []
    parent = b"root"
    for start in range(0, len(tokens) - PAGE_SIZE + 1, PAGE_SIZE):
        page = tokens[start : start + PAGE_SIZE]
        parent = hashlib.sha256(parent + bytes(page)).digest()[:16]
        digests.append(parent)
    return digests


def reached(history: list[int], evicted: set[int] = set()) -> State:
    full = len(history) // PAGE_SIZE * PAGE_SIZE
    hashes = [h for i, h in enumerate(page_digests(history)) if i not in evicted]
    return State(page_hashes=hashes, tail=history[full:])


def page(value: float) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.full((1,), value, dtype=torch.float32),
        torch.full((1,), -value, dtype=torch.float32),
    )


def write(store: LocalStore, state: State) -> None:
    store.write(state, state.page_hashes)


def put_pages(store: LocalStore, history: list[int]) -> None:
    hashes = page_digests(history)
    store.put(hashes, [page(i) for i in range(len(hashes))])


def run_request(store: LocalStore, prompt: list[int], *states: list[int]) -> None:
    """A request on prompt that reaches each history in states, in order."""
    with store.lease(prompt, page_digests(prompt)):
        for history in states:
            put_pages(store, history)
            write(store, reached(history))


def stored(store: LocalStore, h: bytes) -> bool:
    return store.fetch([h]).result()[0] is not None


HISTORY = list(range(1, 15))


def test_an_empty_store_gives_an_empty_state():
    store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY)
    with store.lease(HISTORY, page_digests(HISTORY)) as (state, covered):
        assert state.page_hashes == [] and state.tail == []
        assert covered == 0


def test_lookup_takes_the_longest_state_the_prompt_continues():
    store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY)
    run_request(store, HISTORY, HISTORY[:5], HISTORY[:9], HISTORY[:10])
    with store.lease(HISTORY, page_digests(HISTORY)) as (state, covered):
        assert covered == 10
        assert state.page_hashes == page_digests(HISTORY[:8])
        assert state.tail == HISTORY[8:10]


def test_lookup_skips_a_state_whose_tail_diverges_from_the_prompt():
    store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY)
    run_request(store, HISTORY, HISTORY[:5], HISTORY[:9] + [99])
    with store.lease(HISTORY, page_digests(HISTORY)) as (_, covered):
        assert covered == 5


def test_lookup_leaves_a_token_to_forward():
    store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY)
    run_request(store, HISTORY, HISTORY[:5], HISTORY)
    with store.lease(HISTORY, page_digests(HISTORY)) as (_, covered):
        assert covered == 5


def test_a_state_with_evicted_pages_is_found_by_its_last_page():
    store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY)
    history = HISTORY[:13]
    with store.lease(history, page_digests(history)):
        put_pages(store, history)
        write(store, reached(history, evicted={1}))
    with store.lease(HISTORY, page_digests(HISTORY)) as (state, covered):
        assert covered == 13
        assert len(state.page_hashes) == 2


def test_dropping_a_state_from_the_lru_deletes_the_pages_only_it_references():
    store = LocalStore(PAGE_SIZE, capacity_bytes=2 * PAGE_BYTES)
    first = HISTORY[:8]
    other = [50 + t for t in HISTORY[:8]]
    run_request(store, first, first)
    run_request(store, other, other)
    assert len(store) == 1
    assert not any(stored(store, h) for h in page_digests(first))
    assert all(stored(store, h) for h in page_digests(other))


def test_pages_shared_by_a_surviving_state_stay():
    store = LocalStore(PAGE_SIZE, capacity_bytes=PAGE_BYTES)
    run_request(store, HISTORY, HISTORY[:5], HISTORY[:9])
    assert all(stored(store, h) for h in page_digests(HISTORY[:9]))


def test_a_lease_pins_its_state_pages_until_it_closes():
    store = LocalStore(PAGE_SIZE, capacity_bytes=2 * PAGE_BYTES)
    run_request(store, HISTORY, HISTORY[:9])
    pinned = page_digests(HISTORY[:9])
    other = [50 + t for t in HISTORY[:8]]
    with store.lease(HISTORY, page_digests(HISTORY)) as (state, _):
        assert state.page_hashes == pinned
        run_request(store, other, other)
        assert all(stored(store, h) for h in pinned)
    assert not any(stored(store, h) for h in pinned)


def test_pages_put_during_a_lease_stay_until_it_closes():
    store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY)
    with store.lease(HISTORY, page_digests(HISTORY)):
        put_pages(store, HISTORY)
        run_request(store, [7, 7, 7, 7, 7], [7, 7, 7, 7, 7])
        assert all(stored(store, h) for h in page_digests(HISTORY))
    assert not any(stored(store, h) for h in page_digests(HISTORY))


def test_a_page_put_again_during_a_later_lease_stays_for_it():
    store = LocalStore(PAGE_SIZE, capacity_bytes=2 * PAGE_BYTES)
    first = HISTORY[:8]
    run_request(store, first, first)
    with store.lease(first, page_digests(first)) as (state, covered):
        assert covered == 0
        put_pages(store, first)
        run_request(store, [7] * 5, [7] * 5)
        assert all(stored(store, h) for h in page_digests(first))
        write(store, reached(first))
    assert all(stored(store, h) for h in page_digests(first))


def test_rewriting_a_state_keeps_its_pages():
    store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY)
    run_request(store, HISTORY, HISTORY[:9], HISTORY[:9])
    assert len(store) == 1
    assert all(stored(store, h) for h in page_digests(HISTORY[:9]))


def test_the_store_drops_least_recently_used_states_until_its_pages_fit():
    store = LocalStore(PAGE_SIZE, capacity_bytes=4 * PAGE_BYTES)
    used, unused, newest = ([base + t for t in HISTORY[:8]] for base in (10, 20, 30))
    run_request(store, used, used)
    run_request(store, unused, unused)
    with store.lease(used + [1], page_digests(used + [1])) as (_, covered):
        assert covered == len(used)
    run_request(store, newest, newest)

    assert all(stored(store, h) for h in page_digests(used) + page_digests(newest))
    assert not any(stored(store, h) for h in page_digests(unused))
    assert store.byte_count() == 4 * PAGE_BYTES


def test_the_newest_state_stays_above_the_capacity():
    store = LocalStore(PAGE_SIZE, capacity_bytes=PAGE_BYTES)
    run_request(store, HISTORY, HISTORY[:9])
    assert len(store) == 1
    assert store.byte_count() == 2 * PAGE_BYTES


def test_a_page_keeping_one_tensor_counts_its_bytes_once():
    store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY)
    v = torch.full((1,), 1.0, dtype=torch.float32)
    store.put(page_digests(HISTORY[:4]), [(v, v)])
    assert store.byte_count() == PAGE_BYTES // 2


def in_another_process(path: Path, script: str) -> subprocess.CompletedProcess[str]:
    """Runs script with this file's helpers, store_path set to path, and LocalStore
    imported; the process exits once its store wrote everything."""
    prelude = f"""
        import sys
        sys.path.insert(0, {str(Path(__file__).parent)!r})
        from pathlib import Path
        from test_store import *
        store_path = Path({str(path)!r})
    """
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(prelude) + textwrap.dedent(script)],
        capture_output=True,
        text=True,
    )


def test_states_and_pages_survive_a_restart(tmp_path: Path):
    path = tmp_path / "store.sqlite"
    wrote = in_another_process(
        path,
        """
        store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY, path=store_path)
        run_request(store, HISTORY, HISTORY[:9])
        """,
    )
    assert wrote.returncode == 0, wrote.stderr

    store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY, path=path)
    assert len(store) == 1
    assert store.byte_count() == 2 * PAGE_BYTES
    with store.lease(HISTORY, page_digests(HISTORY)) as (state, covered):
        assert covered == 9
        assert state.page_hashes == page_digests(HISTORY[:8])
        assert state.tail == [HISTORY[8]]
        fetched = store.fetch(state.page_hashes[1:]).result()[0]
        assert fetched is not None
        k, v = fetched
        assert k.tolist() == [1.0] and v.tolist() == [-1.0]


def test_a_page_keeping_one_tensor_is_read_back_as_one(tmp_path: Path):
    path = tmp_path / "store.sqlite"
    wrote = in_another_process(
        path,
        """
        store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY, path=store_path)
        v = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
        with store.lease(HISTORY, page_digests(HISTORY)):
            store.put(page_digests(HISTORY[:4]), [(v, v)])
            write(store, reached(HISTORY[:5]))
        """,
    )
    assert wrote.returncode == 0, wrote.stderr

    store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY, path=path)
    fetched = store.fetch(page_digests(HISTORY[:4])).result()[0]
    assert fetched is not None
    k, v = fetched
    assert k is v
    assert v.dtype == torch.bfloat16
    assert v.tolist() == [[0, 1, 2], [3, 4, 5]]


def test_a_second_process_cannot_open_the_store(tmp_path: Path):
    path = tmp_path / "store.sqlite"
    LocalStore(PAGE_SIZE, capacity_bytes=ROOMY, path=path)
    opened = in_another_process(
        path, "LocalStore(PAGE_SIZE, capacity_bytes=ROOMY, path=store_path)"
    )
    assert opened.returncode != 0
    assert StoreInUseError.__name__ in opened.stderr


def test_a_restart_deletes_pages_no_state_references(tmp_path: Path):
    path = tmp_path / "store.sqlite"
    wrote = in_another_process(
        path,
        """
        store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY, path=store_path)
        with store.lease(HISTORY, page_digests(HISTORY)):
            put_pages(store, HISTORY)
        """,
    )
    assert wrote.returncode == 0, wrote.stderr

    store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY, path=path)
    assert store.page_count() == 0
    assert store.byte_count() == 0


def test_a_restart_drops_a_state_missing_a_page(tmp_path: Path):
    path = tmp_path / "store.sqlite"
    wrote = in_another_process(
        path,
        """
        store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY, path=store_path)
        run_request(store, HISTORY, HISTORY[:9])
        """,
    )
    assert wrote.returncode == 0, wrote.stderr
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            "DELETE FROM pages WHERE key = ?", (page_digests(HISTORY)[0],)
        )

    store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY, path=path)
    assert len(store) == 0
    assert store.page_count() == 0


def test_recency_survives_a_restart(tmp_path: Path):
    path = tmp_path / "store.sqlite"
    wrote = in_another_process(
        path,
        """
        store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY, path=store_path)
        used, unused = ([base + t for t in HISTORY[:8]] for base in (10, 20))
        run_request(store, used, used)
        run_request(store, unused, unused)
        with store.lease(used + [1], page_digests(used + [1])):
            pass
        """,
    )
    assert wrote.returncode == 0, wrote.stderr

    store = LocalStore(PAGE_SIZE, capacity_bytes=2 * PAGE_BYTES, path=path)
    used = [10 + t for t in HISTORY[:8]]
    assert len(store) == 1
    with store.lease(used + [1], page_digests(used + [1])) as (_, covered):
        assert covered == len(used)


def test_a_smaller_capacity_after_a_restart_deletes_states_from_the_file(
    tmp_path: Path,
):
    path = tmp_path / "store.sqlite"
    wrote = in_another_process(
        path,
        """
        store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY, path=store_path)
        first, second = ([base + t for t in HISTORY[:8]] for base in (10, 20))
        run_request(store, first, first)
        run_request(store, second, second)
        """,
    )
    assert wrote.returncode == 0, wrote.stderr
    shrunk = in_another_process(
        path,
        """
        store = LocalStore(PAGE_SIZE, capacity_bytes=PAGE_BYTES, path=store_path)
        assert len(store) == 1
        """,
    )
    assert shrunk.returncode == 0, shrunk.stderr

    store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY, path=path)
    second = [20 + t for t in HISTORY[:8]]
    assert len(store) == 1
    assert store.byte_count() == 2 * PAGE_BYTES
    with store.lease(second + [1], page_digests(second + [1])) as (_, covered):
        assert covered == len(second)
