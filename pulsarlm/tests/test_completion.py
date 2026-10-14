"""complete() on a small random Qwen3, with LocalStore as the page store."""

import asyncio

import pytest
import torch

from pulsarlm.async_engine import AsyncEngine
from pulsarlm.completion import complete
from pulsarlm.core.engine import Engine, EvictionPolicy, ModelConfig, SamplingParams
from pulsarlm.store import LocalStore

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


class CountingStore(LocalStore):
    def __init__(self) -> None:
        super().__init__(PAGE_SIZE, capacity_bytes=64 * LINE_BYTES)
        self.fetched: list[bytes] = []

    def fetch(self, hashes: list[bytes]):
        self.fetched += hashes
        return super().fetch(hashes)


def make_engine(store: LocalStore) -> AsyncEngine:
    return AsyncEngine(
        Engine(
            store,
            MODEL,
            random_weights(),
            model_name="random-qwen3",
            page_size=PAGE_SIZE,
            kv_dtype=DTYPE,
            kv_cache_gib=64 * LINE_BYTES / 2**30,
            staging_gib=4 * LINE_BYTES / 2**30,
            max_position_embeddings=4096,
            device=DEVICE,
            stop_ids=[],
            prefill_tokens_per_step=64,
        )
    )


async def collect(
    engine: AsyncEngine, store: LocalStore, prompt: list[int]
) -> list[int]:
    tokens = complete(
        engine,
        store,
        prompt,
        sampling=SamplingParams(),
        max_tokens=8,
        eviction=EvictionPolicy(cycle_tokens=16),
    )
    return [token async for token in tokens]


def covered_by_lookup(engine: AsyncEngine, store: LocalStore, prompt: list[int]) -> int:
    with store.lease(prompt, engine.page_digests(prompt)) as (_, covered):
        return covered


PROMPT = [(7 * i + 3) % VOCAB_SIZE for i in range(40)]

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_a_follow_up_continues_from_the_end_state():
    async def main():
        store = CountingStore()
        engine = make_engine(store)
        reply = await collect(engine, store, PROMPT)
        follow_up = PROMPT + reply + [11, 12, 13]

        assert (
            covered_by_lookup(engine, store, follow_up) == len(PROMPT) + len(reply) - 1
        )
        assert len(await collect(engine, store, follow_up)) == 8

    asyncio.run(main())


def test_every_cycle_end_state_is_findable():
    async def main():
        store = CountingStore()
        engine = make_engine(store)
        await collect(engine, store, PROMPT)

        assert covered_by_lookup(engine, store, PROMPT[:17]) == 16
        assert covered_by_lookup(engine, store, PROMPT[:33]) == 32

    asyncio.run(main())


def test_another_engine_fetches_the_state_pages_from_the_store():
    async def main():
        store = CountingStore()
        first = make_engine(store)
        reply = await collect(first, store, PROMPT)
        follow_up = PROMPT + reply + [11]

        second = make_engine(store)
        with store.lease(follow_up, second.page_digests(follow_up)) as (state, _):
            assert len(state.page_hashes) == 2
            keys = second.page_keys(state)
        assert len(await collect(second, store, follow_up)) == 8
        assert store.fetched == keys

    asyncio.run(main())


def test_closing_early_cancels_and_the_engine_serves_the_next_request():
    async def main():
        store = CountingStore()
        engine = make_engine(store)
        tokens = complete(
            engine,
            store,
            PROMPT,
            sampling=SamplingParams(),
            max_tokens=None,
            eviction=EvictionPolicy(),
        )
        await anext(tokens)
        await tokens.aclose()

        assert len(await collect(engine, store, PROMPT)) == 8

    asyncio.run(main())
