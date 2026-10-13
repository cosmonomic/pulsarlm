"""The engine binding on a small random Qwen3, with an in-memory page store."""

import asyncio
from concurrent.futures import Future

import pytest
import torch

from pulsar.core.engine import (
    Engine,
    ModelConfig,
    RequestError,
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

MODEL = ModelConfig(
    architecture="qwen3",
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    n_kv_heads=N_KV_HEADS,
    head_dim=HEAD_DIM,
    hidden=HIDDEN,
    intermediate=INTERMEDIATE,
    vocab_size=VOCAB_SIZE,
    rope_theta=1e6,
    rms_eps=1e-6,
    relevance_layers=[N_LAYERS - 1],
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
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.pages: dict[bytes, tuple[torch.Tensor, torch.Tensor]] = {}
        self.fetched: list[bytes] = []

    def fetch(self, hashes: list[bytes]) -> Future:
        self.fetched += hashes
        return asyncio.run_coroutine_threadsafe(self.lookup(hashes), self.loop)

    async def lookup(self, hashes: list[bytes]) -> list:
        return [self.pages.get(h) for h in hashes]

    def put(self, hashes: list[bytes], pages: list) -> None:
        self.pages |= zip(hashes, pages)


class FailingStore(MemoryStore):
    async def lookup(self, hashes: list[bytes]) -> list:
        raise ConnectionError("store unreachable")


def make_engine(store: MemoryStore, eos_id: int = -1) -> Engine:
    return Engine(
        store,
        MODEL,
        random_weights(),
        model_name="random-qwen3",
        page_size=PAGE_SIZE,
        kv_dtype=DTYPE,
        n_lines=64,
        device=DEVICE,
        eos_id=eos_id,
        prefill_tokens_per_step=64,
    )


async def run(engine: Engine, request) -> tuple[list[int], list, object]:
    loop = asyncio.get_running_loop()
    woke = asyncio.Event()
    loop.add_reader(engine.fileno(), woke.set)
    tokens: list[int] = []
    cycle_ends: list = []
    try:
        while True:
            await woke.wait()
            woke.clear()
            engine.clear_wake()
            taken_tokens, taken_cycle_ends, end = request.take()
            tokens += taken_tokens
            cycle_ends += taken_cycle_ends
            if end is not None:
                return tokens, cycle_ends, end
    finally:
        loop.remove_reader(engine.fileno())


PROMPT = [(7 * i + 3) % VOCAB_SIZE for i in range(40)]

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_generates_max_tokens_and_ends_with_a_state():
    async def main():
        store = MemoryStore(asyncio.get_running_loop())
        engine = make_engine(store)
        tokens, _, end = await run(engine, engine.submit(State(), PROMPT, max_tokens=8))

        assert len(tokens) == 8
        assert all(0 <= t < VOCAB_SIZE for t in tokens)
        assert isinstance(end, State)
        forwarded = len(PROMPT) + len(tokens) - 1
        assert len(end.page_hashes) * PAGE_SIZE + len(end.tail) == forwarded
        assert all(h in store.pages for h in end.page_hashes)

    asyncio.run(main())


def test_another_engine_continues_a_state_from_the_store():
    async def main():
        store = MemoryStore(asyncio.get_running_loop())
        first = make_engine(store)
        sampled, _, reached = await run(
            first, first.submit(State(), PROMPT, max_tokens=8)
        )
        assert isinstance(reached, State)

        follow_up = [sampled[-1], 11, 12, 13]
        expected, _, _ = await run(
            first, first.submit(reached, follow_up, max_tokens=8)
        )

        second = make_engine(store)
        continued, _, end = await run(
            second, second.submit(reached, follow_up, max_tokens=8)
        )

        assert store.fetched == reached.page_hashes
        assert continued == expected
        assert isinstance(end, State)

    asyncio.run(main())


def test_a_page_missing_from_the_store_ends_with_miss():
    async def main():
        engine = make_engine(MemoryStore(asyncio.get_running_loop()))
        missing = State(page_hashes=[bytes(16)])
        _, _, end = await run(engine, engine.submit(missing, [1], max_tokens=1))

        assert end == RequestError.MISS

    asyncio.run(main())


def test_a_failing_fetch_misses():
    async def main():
        engine = make_engine(FailingStore(asyncio.get_running_loop()))
        missing = State(page_hashes=[bytes(16)])
        _, _, end = await run(engine, engine.submit(missing, [1], max_tokens=1))

        assert end == RequestError.MISS
        assert engine.exception() is None

    asyncio.run(main())


def test_cancel_ends_the_request_with_cancelled():
    async def main():
        engine = make_engine(MemoryStore(asyncio.get_running_loop()))
        request = engine.submit(State(), PROMPT)
        request.cancel()
        _, _, end = await run(engine, request)

        assert end == RequestError.CANCELLED

    asyncio.run(main())


def test_submit_rejects_a_token_outside_the_vocabulary():
    async def main():
        engine = make_engine(MemoryStore(asyncio.get_running_loop()))
        with pytest.raises(ValueError, match="outside"):
            engine.submit(State(), [VOCAB_SIZE])

    asyncio.run(main())


def test_a_digest_is_16_bytes():
    with pytest.raises(ValueError, match="16 bytes"):
        State(page_hashes=[b"short"])
