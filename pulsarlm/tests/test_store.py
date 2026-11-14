"""LocalStore with digests chained in Python, as the engine chains them."""

import asyncio
import hashlib
import sqlite3
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

import torch

from pulsarlm.core.engine import ActivePage, LayerSequence, Moments, State
from pulsarlm.store import EmptyStore, LocalStore
from pulsarlm.store_file import StoreInUseError

PAGE_SIZE = 4
# page() makes a page of the bytes of two float32.
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


def current_state(history: list[int], evicted: set[int] = set()) -> State:
    """A state of two layers: one keeping every page not evicted, one the newest."""
    full = len(history) // PAGE_SIZE * PAGE_SIZE
    pages = [
        ActivePage(h, i, [i / 4, -i / 8], i / 2)
        for i, h in enumerate(page_digests(history))
        if i not in evicted
    ]
    density_moments = [Moments(3, -0.5, 2.0), Moments(2, 1.0, 0.25)]
    return State(
        head=pages[-1].hash if pages else None,
        layers=[
            LayerSequence(pages, density_moments, Moments(4, 0.125, 1.5)),
            LayerSequence(pages[-1:]),
        ],
        tail=history[full:],
    )


def state_of(hashes: list[bytes]) -> State:
    """A state of one layer keeping hashes."""
    return State(
        head=hashes[-1],
        layers=[LayerSequence([ActivePage(h, i) for i, h in enumerate(hashes)])],
    )


def hashes_of(state: State) -> list[bytes]:
    return [p.hash for p in state.layers[0].pages] if state.layers else []


def page(value: float) -> torch.Tensor:
    return torch.tensor([value, -value], dtype=torch.float32).view(torch.uint8)


def covered_tokens(state: State, prompt: list[int]) -> int:
    """The tokens of prompt that state covers: its full pages, then its tail."""
    pages = 0 if state.head is None else page_digests(prompt).index(state.head) + 1
    return pages * PAGE_SIZE + len(state.tail)


@contextmanager
def request(store: LocalStore, prompt: list[int]) -> Iterator[tuple[int, State]]:
    """A lease on the state the store finds for prompt, released on exit."""
    state, lease_id = asyncio.run(store.read_state(prompt, page_digests(prompt)))
    try:
        yield lease_id, state
    finally:
        store.release(lease_id)


@contextmanager
def lookup(store: LocalStore, prompt: list[int]) -> Iterator[tuple[State, int]]:
    with request(store, prompt) as (_, state):
        yield state, covered_tokens(state, prompt)


def write(store: LocalStore, state: State, lease_id: int) -> None:
    """Writes state for the lease, as a request reaching it does."""
    store.write_state(lease_id, state, hashes_of(state))


def put_pages(
    store: LocalStore, hashes: list[bytes], lease_id: int | None = None
) -> None:
    """One step's write of hashes, owned by lease_id."""
    owners = [] if lease_id is None else [(lease_id, i) for i in range(len(hashes))]
    store.write_pages(hashes, [page(i) for i in range(len(hashes))], owners)


def run_request(store: LocalStore, prompt: list[int], *states: list[int]) -> None:
    """A request on prompt that reaches each history in states, in order."""
    with request(store, prompt) as (lease_id, _):
        for history in states:
            put_pages(store, page_digests(history), lease_id)
            write(store, current_state(history), lease_id)


def read_page(store: LocalStore, h: bytes) -> torch.Tensor | None:
    return asyncio.run(store.read_pages([h]))[0]


def stored(store: LocalStore, h: bytes) -> bool:
    return read_page(store, h) is not None


HISTORY = list(range(1, 15))


def test_an_empty_store_gives_an_empty_state():
    store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY)
    with lookup(store, HISTORY) as (state, covered):
        assert state.head is None and state.layers == [] and state.tail == []
        assert covered == 0


def test_lookup_takes_the_longest_state_the_prompt_continues():
    store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY)
    run_request(store, HISTORY, HISTORY[:5], HISTORY[:9], HISTORY[:10])
    with lookup(store, HISTORY) as (state, covered):
        assert covered == 10
        assert hashes_of(state) == page_digests(HISTORY[:8])
        assert state.tail == HISTORY[8:10]


def test_lookup_skips_a_state_whose_tail_diverges_from_the_prompt():
    store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY)
    run_request(store, HISTORY, HISTORY[:5], HISTORY[:9] + [99])
    with lookup(store, HISTORY) as (_, covered):
        assert covered == 5


def test_lookup_leaves_a_token_to_forward():
    store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY)
    run_request(store, HISTORY, HISTORY[:5], HISTORY)
    with lookup(store, HISTORY) as (_, covered):
        assert covered == 5


def test_a_state_with_evicted_pages_is_found_by_its_last_page():
    store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY)
    history = HISTORY[:13]
    with request(store, history) as (lease_id, _):
        put_pages(store, page_digests(history), lease_id)
        write(store, current_state(history, evicted={1}), lease_id)
    with lookup(store, HISTORY) as (state, covered):
        assert covered == 13
        assert len(hashes_of(state)) == 2


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


def test_a_request_keeps_the_pages_of_the_state_it_resumed_until_release():
    store = LocalStore(PAGE_SIZE, capacity_bytes=2 * PAGE_BYTES)
    run_request(store, HISTORY, HISTORY[:9])
    resumed = page_digests(HISTORY[:9])
    other = [50 + t for t in HISTORY[:8]]
    with lookup(store, HISTORY) as (state, _):
        assert hashes_of(state) == resumed
        run_request(store, other, other)
        assert all(stored(store, h) for h in resumed)
    assert not any(stored(store, h) for h in resumed)


def test_the_pages_a_request_puts_stay_until_its_lease_is_released():
    store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY)
    with request(store, HISTORY) as (lease_id, _):
        put_pages(store, page_digests(HISTORY), lease_id)
        run_request(store, [7, 7, 7, 7, 7], [7, 7, 7, 7, 7])
        assert all(stored(store, h) for h in page_digests(HISTORY))
    assert not any(stored(store, h) for h in page_digests(HISTORY))


def test_claim_gives_the_pages_an_open_lease_owns_that_the_store_lacks():
    store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY)
    first, second, third = page_digests(HISTORY[:12])
    run_request(store, HISTORY[:4], HISTORY[:4])
    with request(store, HISTORY) as (lease_id, _):
        owners = [(lease_id, 0), (lease_id, 1), (lease_id + 1000, 2)]
        assert store.claim([first, second, third], owners) == [1]


def test_claim_holds_a_page_the_store_has_for_its_lease_until_release():
    store = LocalStore(PAGE_SIZE, capacity_bytes=PAGE_BYTES)
    first = page_digests(HISTORY[:4])[0]
    run_request(store, HISTORY[:4], HISTORY[:4])
    with request(store, [7] * 5) as (lease_id, _):
        assert store.claim([first], [(lease_id, 0)]) == []
        run_request(store, [8] * 5, [8] * 5)
        assert stored(store, first)
    assert not stored(store, first)


def test_an_open_lease_does_not_keep_the_pages_another_request_put():
    store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY)
    other = [7] * 9
    with lookup(store, HISTORY):
        with request(store, other) as (lease_id, _):
            put_pages(store, page_digests(other), lease_id)
        assert not any(stored(store, h) for h in page_digests(other))


def test_writing_a_state_releases_the_pages_of_its_steps_it_does_not_reference():
    store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY)
    first, evicted, last = page_digests(HISTORY[:12])
    with request(store, HISTORY) as (lease_id, _):
        for h in (first, evicted, last):
            put_pages(store, [h], lease_id)
        write(store, state_of([first, last]), lease_id)
        assert not stored(store, evicted)
        assert stored(store, first) and stored(store, last)


def test_the_pages_a_request_put_after_the_step_of_the_state_it_reaches_stay():
    store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY)
    first, second = page_digests(HISTORY[:8])
    with request(store, HISTORY) as (lease_id, _):
        put_pages(store, [first], lease_id)
        put_pages(store, [second], lease_id)
        write(store, state_of([first]), lease_id)
        assert stored(store, second)
    assert stored(store, first)
    assert not stored(store, second)


def test_a_running_request_keeps_its_newest_snapshot_after_the_lru_drops_it():
    store = LocalStore(PAGE_SIZE, capacity_bytes=PAGE_BYTES)
    first, second = page_digests(HISTORY[:8])
    with request(store, HISTORY) as (lease_id, _):
        put_pages(store, [first], lease_id)
        write(store, state_of([first]), lease_id)
        run_request(store, [7] * 5, [7] * 5)
        assert stored(store, first)
        put_pages(store, [second], lease_id)
        write(store, state_of([first, second]), lease_id)
    assert stored(store, first) and stored(store, second)


def test_a_page_put_again_by_a_later_request_stays_for_it():
    store = LocalStore(PAGE_SIZE, capacity_bytes=2 * PAGE_BYTES)
    first = HISTORY[:8]
    run_request(store, first, first)
    with request(store, first) as (lease_id, state):
        assert covered_tokens(state, first) == 0
        put_pages(store, page_digests(first), lease_id)
        run_request(store, [7] * 5, [7] * 5)
        assert all(stored(store, h) for h in page_digests(first))
        write(store, current_state(first), lease_id)
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
    with lookup(store, used + [1]) as (_, covered):
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


def test_a_page_no_request_owns_is_deleted_by_its_put():
    store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY)
    put_pages(store, page_digests(HISTORY))
    assert store.page_count() == 0


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
    with lookup(store, HISTORY) as (state, covered):
        assert covered == 9
        assert state.head == current_state(HISTORY[:9]).head
        assert state.layers == current_state(HISTORY[:9]).layers
        assert state.tail == [HISTORY[8]]
        stored_page = read_page(store, hashes_of(state)[1])
        assert stored_page is not None
        assert torch.equal(stored_page, page(1))


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
        with request(store, HISTORY) as (lease_id, _):
            put_pages(store, page_digests(HISTORY), lease_id)
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


def test_a_file_another_version_wrote_is_emptied(tmp_path: Path):
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
        connection.execute("UPDATE version SET version = '0.0.0'")

    store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY, path=path)
    assert len(store) == 0
    assert store.page_count() == 0
    with lookup(store, HISTORY) as (_, covered):
        assert covered == 0


def test_recency_survives_a_restart(tmp_path: Path):
    path = tmp_path / "store.sqlite"
    wrote = in_another_process(
        path,
        """
        store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY, path=store_path)
        used, unused = ([base + t for t in HISTORY[:8]] for base in (10, 20))
        run_request(store, used, used)
        run_request(store, unused, unused)
        with lookup(store, used + [1]):
            pass
        """,
    )
    assert wrote.returncode == 0, wrote.stderr

    store = LocalStore(PAGE_SIZE, capacity_bytes=2 * PAGE_BYTES, path=path)
    used = [10 + t for t in HISTORY[:8]]
    assert len(store) == 1
    with lookup(store, used + [1]) as (_, covered):
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
    with lookup(store, second + [1]) as (_, covered):
        assert covered == len(second)


def test_release_deletes_the_pages_only_the_lease_references():
    store = LocalStore(PAGE_SIZE, capacity_bytes=ROOMY)
    run_request(store, HISTORY[:4], HISTORY[:4])
    shared = page_digests(HISTORY[:4])
    own = page_digests(HISTORY)[1:]
    state, lease_id = asyncio.run(store.read_state(HISTORY, page_digests(HISTORY)))
    assert hashes_of(state) == shared
    put_pages(store, own, lease_id)
    assert all(stored(store, h) for h in shared + own)
    store.release(lease_id)
    assert all(stored(store, h) for h in shared)
    assert not any(stored(store, h) for h in own)


def test_write_state_moves_the_lease_off_the_pages_of_its_previous_state():
    store = LocalStore(PAGE_SIZE, capacity_bytes=2 * PAGE_BYTES)
    first, second, third = page_digests(HISTORY[:12])
    with request(store, HISTORY) as (lease_id, _):
        put_pages(store, [first, second], lease_id)
        write(store, state_of([first, second]), lease_id)
        put_pages(store, [third], lease_id)
        write(store, state_of([first, third]), lease_id)
        assert len(store) == 1
        assert not stored(store, second)
        assert stored(store, first) and stored(store, third)


def test_the_empty_store_finds_nothing_and_keeps_nothing():
    store = EmptyStore()
    state, lease_id = asyncio.run(store.read_state(HISTORY, page_digests(HISTORY)))
    assert state.head is None and state.layers == [] and state.tail == []
    _, next_lease_id = asyncio.run(store.read_state(HISTORY, page_digests(HISTORY)))
    assert next_lease_id != lease_id
    hashes = page_digests(HISTORY)
    assert store.claim(hashes, [(lease_id, i) for i in range(len(hashes))]) == []
    store.write_state(lease_id, current_state(HISTORY), hashes)
    store.release(lease_id)
    assert asyncio.run(store.read_pages(hashes)) == [None] * len(hashes)
