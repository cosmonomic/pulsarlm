"""complete() on a small random Qwen3, with LocalStore as the page store."""

import asyncio

import pytest
import torch

from pulsarlm.async_engine import AsyncEngine
from pulsarlm.completion import CompletionUsage, complete
from pulsarlm.core.engine import (
    Engine,
    EvictionPolicy,
    ModelConfig,
    RopeParameters,
    SamplingParams,
    State,
)
from pulsarlm.store import EmptyStore, LocalStore

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


class RecordingStore(LocalStore):
    """Records the pages read, every state found with the digests it was found
    by, and the page keys of every state written."""

    def __init__(self) -> None:
        super().__init__(PAGE_SIZE, capacity_bytes=64 * LINE_BYTES)
        self.read_hashes: list[bytes] = []
        self.state_reads: list[tuple[State, list[bytes]]] = []
        self.written_keys: dict[tuple[bytes | None, tuple[int, ...]], list[bytes]] = {}

    async def read_pages(self, hashes: list[bytes]) -> list[torch.Tensor | None]:
        self.read_hashes += hashes
        return await super().read_pages(hashes)

    async def read_state(
        self, tokens: list[int], page_digests: list[bytes]
    ) -> tuple[State, int]:
        state, lease_id = await super().read_state(tokens, page_digests)
        self.state_reads.append((state, page_digests))
        return state, lease_id

    def write_state(self, lease_id: int, state: State, page_keys: list[bytes]) -> None:
        self.written_keys[(state.head, tuple(state.tail))] = page_keys
        super().write_state(lease_id, state, page_keys)


def make_engine(store: LocalStore | EmptyStore) -> AsyncEngine:
    return AsyncEngine(
        Engine(
            store,
            asyncio.get_running_loop(),
            MODEL,
            lambda rank, size: random_weights(),
            model_name="random-qwen3",
            model_hash="1",
            page_size=PAGE_SIZE,
            kv_dtype=DTYPE,
            kv_cache_gib=64 * LINE_BYTES / 2**30,
            store_gib=64 * LINE_BYTES / 2**30,
            devices=[DEVICE],
            stop_ids=[],
            prefill_tokens_per_step=64,
        )
    )


async def collect(
    engine: AsyncEngine, prompt: list[int], usages: list[CompletionUsage] | None = None
) -> list[int]:
    tokens = complete(
        engine,
        prompt,
        sampling=SamplingParams(),
        max_tokens=8,
        eviction=EvictionPolicy(cycle_tokens=16),
        on_completed=None if usages is None else usages.append,
    )
    return [token async for token in tokens]


def covered_tokens(state: State, page_digests: list[bytes]) -> int:
    pages = 0 if state.head is None else page_digests.index(state.head) + 1
    return pages * PAGE_SIZE + len(state.tail)


async def covered_by_lookup(
    store: LocalStore, prompt: list[int], page_digests: list[bytes]
) -> int:
    state, lease_id = await store.read_state(prompt, page_digests)
    store.release(lease_id)
    return covered_tokens(state, page_digests)


PROMPT = [(7 * i + 3) % VOCAB_SIZE for i in range(40)]

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_a_follow_up_continues_from_the_end_state():
    async def main():
        store = RecordingStore()
        engine = make_engine(store)
        reply = await collect(engine, PROMPT)
        follow_up = PROMPT + reply + [11, 12, 13]
        usages: list[CompletionUsage] = []

        assert len(await collect(engine, follow_up, usages)) == 8
        state, digests = store.state_reads[-1]
        assert covered_tokens(state, digests) == len(PROMPT) + len(reply) - 1
        [usage] = usages
        assert usage.prompt_tokens == len(follow_up)
        assert usage.cached_tokens == len(state.layers[0].pages) * PAGE_SIZE

    asyncio.run(main())


def test_every_cycle_end_state_is_findable():
    async def main():
        store = RecordingStore()
        engine = make_engine(store)
        await collect(engine, PROMPT)
        _, digests = store.state_reads[0]

        assert await covered_by_lookup(store, PROMPT[:17], digests[:1]) == 16
        assert await covered_by_lookup(store, PROMPT[:33], digests[:2]) == 32

    asyncio.run(main())


def test_with_the_empty_store_every_request_runs_its_whole_prompt():
    async def main():
        engine = make_engine(EmptyStore())
        usages: list[CompletionUsage] = []
        reply = await collect(engine, PROMPT, usages)
        follow_up = PROMPT + reply + [11, 12, 13]
        assert len(await collect(engine, follow_up, usages)) == 8

        assert [usage.cached_tokens for usage in usages] == [0, 0]
        assert [usage.prompt_tokens for usage in usages] == [
            len(PROMPT),
            len(follow_up),
        ]

    asyncio.run(main())


def test_another_engine_fetches_the_state_pages_from_the_store():
    async def main():
        store = RecordingStore()
        first = make_engine(store)
        reply = await collect(first, PROMPT)
        follow_up = PROMPT + reply + [11]

        second = make_engine(store)
        assert len(await collect(second, follow_up)) == 8
        state, _ = store.state_reads[-1]
        assert len(state.layers[0].pages) == 2
        assert store.read_hashes == store.written_keys[(state.head, tuple(state.tail))]

    asyncio.run(main())


def test_closing_early_cancels_and_the_engine_serves_the_next_request():
    async def main():
        engine = make_engine(RecordingStore())
        tokens = complete(
            engine,
            PROMPT,
            sampling=SamplingParams(),
            max_tokens=None,
            eviction=EvictionPolicy(),
        )
        await anext(tokens)
        await tokens.aclose()

        assert len(await collect(engine, PROMPT)) == 8

    asyncio.run(main())
