"""The engine binding on a small random Qwen3, with an in-memory page store."""

import asyncio

import pytest
import torch

from pulsarlm.core.engine import (
    CycleEnd,
    Engine,
    EvictionPolicy,
    ModelConfig,
    Request,
    RequestError,
    RequestFailed,
    RequestResult,
    RopeParameters,
    State,
)

N_LAYERS = 2
N_HEADS = 2
N_KV_HEADS = 1
HEAD_DIM = 128
HIDDEN = 256
INTERMEDIATE = 512
VOCAB_SIZE = 512
PAGE_SIZE = 16
DEVICE = torch.device("cuda", 0)
DTYPE = torch.bfloat16
LINE_BYTES = 2 * N_LAYERS * PAGE_SIZE * N_KV_HEADS * HEAD_DIM * DTYPE.itemsize

MODEL = ModelConfig(
    architecture="qwen3",
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    n_kv_heads=N_KV_HEADS,
    head_dim=HEAD_DIM,
    vocab_size=VOCAB_SIZE,
    rope=RopeParameters(theta=1e6),
    rms_norm_eps=1e-6,
)


def random_weights() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)

    def tensor(*shape: int, scale: float = 0.05) -> torch.Tensor:
        values = torch.randn(*shape, generator=generator) * scale
        return values.to(device=DEVICE, dtype=DTYPE)

    def ones(size: int) -> torch.Tensor:
        return torch.ones(size, device=DEVICE, dtype=DTYPE)

    weights = {
        "model.embed_tokens.weight": tensor(VOCAB_SIZE, HIDDEN, scale=1.0),
        "model.norm.weight": ones(HIDDEN),
    }
    for layer in range(N_LAYERS):
        prefix = f"model.layers.{layer}."
        weights |= {
            prefix + "input_layernorm.weight": ones(HIDDEN),
            prefix + "post_attention_layernorm.weight": ones(HIDDEN),
            prefix + "self_attn.q_proj.weight": tensor(N_HEADS * HEAD_DIM, HIDDEN),
            prefix + "self_attn.k_proj.weight": tensor(N_KV_HEADS * HEAD_DIM, HIDDEN),
            prefix + "self_attn.v_proj.weight": tensor(N_KV_HEADS * HEAD_DIM, HIDDEN),
            prefix + "self_attn.o_proj.weight": tensor(HIDDEN, N_HEADS * HEAD_DIM),
            prefix + "self_attn.q_norm.weight": ones(HEAD_DIM),
            prefix + "self_attn.k_norm.weight": ones(HEAD_DIM),
            prefix + "mlp.gate_proj.weight": tensor(INTERMEDIATE, HIDDEN),
            prefix + "mlp.up_proj.weight": tensor(INTERMEDIATE, HIDDEN),
            prefix + "mlp.down_proj.weight": tensor(HIDDEN, INTERMEDIATE),
        }
    return weights


class MemoryStore:
    """Keeps every page and state; read_state finds the state covering the most
    tokens of a proper prefix, as LocalStore does."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.pages: dict[bytes, torch.Tensor] = {}
        self.states: list[tuple[State, list[bytes]]] = []
        self.read_hashes: list[bytes] = []
        self.state_reads: list[tuple[list[int], list[bytes]]] = []
        self.releases: list[int] = []
        self.lease_count = 0

    async def read_pages(self, hashes: list[bytes]) -> list[torch.Tensor | None]:
        self.read_hashes += hashes
        return [self.pages.get(h) for h in hashes]

    async def read_state(
        self, tokens: list[int], page_digests: list[bytes]
    ) -> tuple[State, int]:
        self.state_reads.append((tokens, page_digests))
        self.lease_count += 1
        best, best_count = State(), 0
        for state, _ in self.states:
            if state.head is None or state.head not in page_digests:
                continue
            start = (page_digests.index(state.head) + 1) * PAGE_SIZE
            end = start + len(state.tail)
            if best_count < end < len(tokens) and tokens[start:end] == state.tail:
                best, best_count = state, end
        return best, self.lease_count

    def claim(self, hashes: list[bytes], owners: list) -> list[int]:
        return [i for i, h in enumerate(hashes) if h not in self.pages]

    def write_pages(self, hashes: list[bytes], pages: list, owners: list) -> None:
        self.pages |= zip(hashes, pages)

    def write_state(self, lease_id: int, state: State, page_keys: list[bytes]) -> None:
        self.states.append((state, page_keys))

    def release(self, lease_id: int) -> None:
        self.releases.append(lease_id)


class FailingStore(MemoryStore):
    async def read_pages(self, hashes: list[bytes]) -> list[torch.Tensor | None]:
        raise ConnectionError("store unreachable")


class FailingLookupStore(MemoryStore):
    async def read_state(
        self, tokens: list[int], page_digests: list[bytes]
    ) -> tuple[State, int]:
        raise ConnectionError("store unreachable")


def make_engine(store: MemoryStore, store_pages: int = 0) -> Engine:
    return Engine(
        store,
        store.loop,
        MODEL,
        lambda rank, size: random_weights(),
        model_name="random-qwen3",
        model_hash="1",
        page_size=PAGE_SIZE,
        kv_dtype=DTYPE,
        kv_cache_gib=64 * LINE_BYTES / 2**30,
        store_gib=store_pages * LINE_BYTES / 2**30,
        devices=[DEVICE],
        stop_ids=[],
        prefill_tokens_per_step=64,
    )


class Recording:
    def __init__(self) -> None:
        self.cycle_ends: list[CycleEnd] = []
        self.result: RequestResult | None = None

    def record_result(self, result: RequestResult) -> None:
        self.result = result


def submit(engine: Engine, tokens: list[int], **options) -> tuple[Request, Recording]:
    recording = Recording()
    request = engine.submit(
        tokens,
        on_cycle_end=recording.cycle_ends.append,
        on_end=recording.record_result,
        **options,
    )
    return request, recording


async def run(
    engine: Engine, request: Request, recording: Recording
) -> tuple[list[int], list[CycleEnd], State | RequestError]:
    loop = asyncio.get_running_loop()
    woke = asyncio.Event()
    loop.add_reader(engine.fileno(), woke.set)
    tokens: list[int] = []
    try:
        while True:
            woke.clear()
            engine.clear_wake()
            try:
                taken = request.take()
            except RequestFailed as failure:
                return tokens, recording.cycle_ends, failure.error
            if taken is None:
                assert recording.result is not None
                return tokens, recording.cycle_ends, recording.result.state
            tokens += taken
            if not taken:
                await woke.wait()
    finally:
        loop.remove_reader(engine.fileno())


PROMPT = [(7 * i + 3) % VOCAB_SIZE for i in range(40)]

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_generates_max_tokens_and_ends_with_a_state():
    async def main():
        store = MemoryStore(asyncio.get_running_loop())
        engine = make_engine(store)
        tokens, _, end = await run(engine, *submit(engine, PROMPT, max_tokens=8))

        assert len(tokens) == 8
        assert all(0 <= t < VOCAB_SIZE for t in tokens)
        assert isinstance(end, State)
        forwarded = len(PROMPT) + len(tokens) - 1
        assert len(end.layers[0].pages) * PAGE_SIZE + len(end.tail) == forwarded
        written_state, page_keys = store.states[-1]
        assert written_state.head == end.head and written_state.tail == end.tail
        assert page_keys and all(k in store.pages for k in page_keys)
        del engine
        assert store.releases == [1]

    asyncio.run(main())


def test_the_end_state_chains_from_the_prompt_digests_the_store_read():
    async def main():
        store = MemoryStore(asyncio.get_running_loop())
        engine = make_engine(store)
        tokens, _, end = await run(engine, *submit(engine, PROMPT, max_tokens=24))

        assert isinstance(end, State)
        [(read_tokens, digests)] = store.state_reads
        assert read_tokens == PROMPT
        assert len(digests) == len(PROMPT) // PAGE_SIZE
        hashes = [p.hash for p in end.layers[0].pages]
        assert hashes[: len(digests)] == digests
        assert len(hashes) == (len(PROMPT) + len(tokens) - 1) // PAGE_SIZE
        assert all([p.hash for p in layer.pages] == hashes for layer in end.layers)

    asyncio.run(main())


@pytest.mark.parametrize("store_pages", [0, 1, 64])
def test_another_engine_continues_a_state_from_the_store(store_pages: int):
    async def main():
        store = MemoryStore(asyncio.get_running_loop())
        first = make_engine(store, store_pages)
        sampled, _, end_state = await run(first, *submit(first, PROMPT, max_tokens=8))
        assert isinstance(end_state, State)
        _, end_state_keys = store.states[-1]

        history = PROMPT + sampled + [11, 12, 13]
        expected, _, _ = await run(first, *submit(first, history, max_tokens=8))

        second = make_engine(store, store_pages)
        continued, _, end = await run(second, *submit(second, history, max_tokens=8))

        assert store.read_hashes == end_state_keys
        assert continued == expected
        assert isinstance(end, State)

    asyncio.run(main())


def test_the_result_counts_the_tokens_resumed_on_resident_and_fetched_pages():
    async def main():
        store = MemoryStore(asyncio.get_running_loop())
        first = make_engine(store, store_pages=64)
        request, recording = submit(first, PROMPT, max_tokens=8)
        sampled, _, end_state = await run(first, request, recording)
        assert isinstance(end_state, State)
        assert recording.result is not None
        assert recording.result.total_tokens == len(PROMPT) + len(sampled)
        assert recording.result.cached_tokens == 0

        history = PROMPT + sampled + [11, 12]
        resumed_tokens = len(end_state.layers[0].pages) * PAGE_SIZE
        request, resident = submit(first, history, max_tokens=2)
        await run(first, request, resident)
        second = make_engine(store, store_pages=64)
        request, fetched = submit(second, history, max_tokens=2)
        await run(second, request, fetched)

        assert resident.result is not None and fetched.result is not None
        assert resident.result.total_tokens == len(history) + 2
        assert resident.result.cached_tokens == resumed_tokens
        assert resident.result.device_cached_tokens == resumed_tokens
        assert fetched.result.cached_tokens == resumed_tokens
        assert fetched.result.device_cached_tokens == 0

    asyncio.run(main())


def test_pages_past_the_room_of_the_store_pool_are_pageable():
    async def main():
        store = MemoryStore(asyncio.get_running_loop())
        engine = make_engine(store, store_pages=1)
        await run(engine, *submit(engine, PROMPT, max_tokens=8))

        pinned = [page.is_pinned() for page in store.pages.values()]
        assert True in pinned
        assert False in pinned

    asyncio.run(main())


def test_a_store_pool_with_room_gives_pinned_pages():
    async def main():
        store = MemoryStore(asyncio.get_running_loop())
        engine = make_engine(store, store_pages=64)
        await run(engine, *submit(engine, PROMPT, max_tokens=8))

        assert store.pages
        assert all(page.is_pinned() for page in store.pages.values())

    asyncio.run(main())


def test_a_page_missing_from_the_store_ends_with_miss():
    async def main():
        store = MemoryStore(asyncio.get_running_loop())
        first = make_engine(store)
        sampled, _, _ = await run(first, *submit(first, PROMPT, max_tokens=8))
        store.pages.clear()

        second = make_engine(store)
        _, _, end = await run(second, *submit(second, PROMPT + sampled, max_tokens=1))

        assert end == RequestError.MISS
        del first, second
        assert store.releases == [1, 2]

    asyncio.run(main())


def test_a_failing_read_misses():
    async def main():
        loop = asyncio.get_running_loop()
        store = MemoryStore(loop)
        first = make_engine(store)
        sampled, _, _ = await run(first, *submit(first, PROMPT, max_tokens=8))
        failing_store = FailingStore(loop)
        failing_store.states = store.states

        second = make_engine(failing_store)
        _, _, end = await run(second, *submit(second, PROMPT + sampled, max_tokens=1))

        assert end == RequestError.MISS
        assert second.exception() is None

    asyncio.run(main())


def test_a_failing_state_lookup_misses():
    async def main():
        store = FailingLookupStore(asyncio.get_running_loop())
        engine = make_engine(store)
        sampled, _, end = await run(engine, *submit(engine, PROMPT, max_tokens=8))

        assert sampled == []
        assert end == RequestError.MISS
        assert engine.exception() is None

    asyncio.run(main())


def test_cancel_ends_the_request_with_cancelled():
    async def main():
        store = MemoryStore(asyncio.get_running_loop())
        engine = make_engine(store)
        request, recording = submit(engine, PROMPT)
        request.cancel()
        _, _, end = await run(engine, request, recording)

        assert end == RequestError.CANCELLED
        del engine
        assert store.releases == [1]

    asyncio.run(main())


def test_submit_rejects_a_token_outside_the_vocabulary():
    async def main():
        engine = make_engine(MemoryStore(asyncio.get_running_loop()))
        with pytest.raises(ValueError, match="outside"):
            engine.submit([VOCAB_SIZE])

    asyncio.run(main())


@pytest.mark.parametrize(
    ("kv_cache_gib", "store_gib", "message"),
    [
        (0.0, 0.0, "must be positive"),
        (LINE_BYTES / 2 / 2**30, 0.0, "holds no line"),
        (LINE_BYTES / 2**30, -1.0, "not negative"),
    ],
)
def test_a_cache_without_a_line_or_a_negative_store_is_rejected(
    kv_cache_gib: float, store_gib: float, message: str
):
    loop = asyncio.new_event_loop()
    with pytest.raises(ValueError, match=message):
        Engine(
            MemoryStore(loop),
            loop,
            MODEL,
            lambda rank, size: random_weights(),
            model_name="random-qwen3",
            model_hash="1",
            page_size=PAGE_SIZE,
            kv_dtype=DTYPE,
            kv_cache_gib=kv_cache_gib,
            store_gib=store_gib,
            devices=[DEVICE],
            stop_ids=[],
            prefill_tokens_per_step=64,
        )


def test_a_digest_is_16_bytes():
    with pytest.raises(ValueError, match="16 bytes"):
        State(head=b"short")


def test_the_eviction_report_scores_every_page_of_a_layer_measuring_mass():
    async def main():
        engine = make_engine(MemoryStore(asyncio.get_running_loop()))
        eviction = EvictionPolicy(
            cycle_tokens=PAGE_SIZE, max_context_window=8 * PAGE_SIZE
        )
        _, _, end = await run(
            engine, *submit(engine, PROMPT, max_tokens=8, eviction=eviction)
        )
        assert isinstance(end, State)

        report = engine.eviction_report_of(end, eviction)
        assert report.cap_tokens == 6 * PAGE_SIZE
        assert [scores.layer for scores in report.layers] == list(range(N_LAYERS))
        for scores in report.layers:
            assert len(scores.page_scores) == len(end.layers[scores.layer].pages)
        assert engine.eviction_report_of(end, EvictionPolicy()).cap_tokens is None

    asyncio.run(main())


def test_submit_rejects_a_request_without_tokens():
    async def main():
        engine = make_engine(MemoryStore(asyncio.get_running_loop()))
        with pytest.raises(ValueError, match="no token"):
            engine.submit([])

    asyncio.run(main())
