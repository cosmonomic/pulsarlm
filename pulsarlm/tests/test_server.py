"""The Chat Completions route on a small random Qwen3 and a word-level tokenizer."""

import json

import pytest
import torch
from fastapi.testclient import TestClient
from tokenizers import Tokenizer, models, pre_tokenizers

from pulsarlm.async_engine import AsyncEngine
from pulsarlm.chat import encode_prompt
from pulsarlm.core.engine import Engine, EvictionPolicy, ModelConfig
from pulsarlm.server import create_app
from pulsarlm.server.app import trained_view_pages
from pulsarlm.server.openai import ServerState
from pulsarlm.server.openai.schema import ChatMessage
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
MODEL_ID = "random-qwen3"
SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
ROLES = ["system", "user", "assistant"]

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


def random_weights(embedding_scale: float) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)

    def tensor(*shape: int, scale: float = 0.05) -> torch.Tensor:
        values = torch.randn(*shape, generator=generator) * scale
        return values.to(device=DEVICE, dtype=DTYPE)

    def ones(size: int) -> torch.Tensor:
        return torch.ones(size, device=DEVICE, dtype=DTYPE)

    weights = {
        "model.embed_tokens.weight": tensor(VOCAB_SIZE, HIDDEN, scale=embedding_scale),
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


def word_tokenizer() -> Tokenizer:
    words = SPECIAL_TOKENS + ROLES
    words += [f"w{i}" for i in range(VOCAB_SIZE - len(words))]
    tokenizer = Tokenizer(
        models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="w0")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    return tokenizer


IM_END = SPECIAL_TOKENS.index("<|im_end|>")
# The random model samples "assistant" first.
FIRST_SAMPLED = len(SPECIAL_TOKENS) + ROLES.index("assistant")


def load(
    stop_ids: list[int] = [IM_END], n_lines: int = 64, embedding_scale: float = 1.0
) -> ServerState:
    tokenizer = word_tokenizer()
    store = LocalStore(PAGE_SIZE, capacity=64)
    engine = Engine(
        store,
        MODEL,
        random_weights(embedding_scale),
        model_name=MODEL_ID,
        page_size=PAGE_SIZE,
        kv_dtype=DTYPE,
        kv_cache_gib=n_lines * LINE_BYTES / 2**30,
        device=DEVICE,
        stop_ids=stop_ids,
        prefill_tokens_per_step=64,
    )
    return ServerState(
        engine=AsyncEngine(engine),
        store=store,
        tokenizer=tokenizer,
        model_id=MODEL_ID,
        page_size=PAGE_SIZE,
        stop_ids=stop_ids,
        eviction=EvictionPolicy(cycle_tokens=PAGE_SIZE),
    )


SYSTEM = " ".join(f"w{i}" for i in range(10, 40))
MESSAGES = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "w7 w8 w9"},
]


def encode(state: ServerState, messages: list[dict[str, str]]) -> list[int]:
    parsed = [ChatMessage.model_validate(m) for m in messages]
    prompt, _ = encode_prompt(state.tokenizer, [(m.role, m.content) for m in parsed])
    return prompt


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture
def client():
    with TestClient(create_app(load)) as client:
        yield client


def post(client: TestClient, **body):
    return client.post(
        "/api/openai/v1/chat/completions", json={"model": MODEL_ID, **body}
    )


def test_a_completion_stops_at_max_tokens(client):
    response = post(client, messages=MESSAGES, max_tokens=4)

    assert response.status_code == 200
    body = response.json()
    assert body["usage"] == {
        "prompt_tokens": 41,
        "completion_tokens": 4,
        "total_tokens": 45,
    }
    assert body["choices"][0]["finish_reason"] == "length"


def test_a_streamed_completion_ends_with_done(client):
    response = post(client, messages=MESSAGES, max_tokens=4, stream=True)

    assert response.status_code == 200
    events = [line for line in response.text.split("\n\n") if line]
    assert events[-1] == "data: [DONE]"
    last = json.loads(events[-2].removeprefix("data: "))
    assert last["choices"][0]["finish_reason"] == "length"


def test_a_follow_up_continues_from_a_stored_state(client):
    first = post(client, messages=MESSAGES, max_tokens=4).json()
    reply = first["choices"][0]["message"]["content"]
    follow_up = MESSAGES + [
        {"role": "assistant", "content": reply},
        {"role": "user", "content": "w11"},
    ]

    state = client.app.state.server_state
    prompt = encode(state, follow_up)
    with state.store.lease(prompt, state.engine.page_digests(prompt)) as (_, covered):
        assert covered >= 2 * PAGE_SIZE
    assert post(client, messages=follow_up, max_tokens=4).status_code == 200


def test_a_reply_ending_at_any_stop_token_leaves_its_end_state_findable():
    with TestClient(create_app(lambda: load([IM_END, FIRST_SAMPLED]))) as client:
        first = post(client, messages=MESSAGES, max_tokens=4).json()
        assert first["choices"][0]["finish_reason"] == "stop"
        assert first["usage"]["completion_tokens"] == 1

        follow_up = MESSAGES + [
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "w11"},
        ]
        state = client.app.state.server_state
        prompt = encode(state, follow_up)
        with state.store.lease(prompt, state.engine.page_digests(prompt)) as (
            _,
            covered,
        ):
            assert covered == first["usage"]["prompt_tokens"]


def test_a_seeded_sampled_completion_repeats():
    def sample(seed: int) -> str:
        # A small embedding scale spreads the next-token distribution.
        with TestClient(create_app(lambda: load(embedding_scale=0.05))) as client:
            body = post(
                client, messages=MESSAGES, max_tokens=8, temperature=1.0, seed=seed
            ).json()
            return body["choices"][0]["message"]["content"]

    assert sample(1) == sample(1)
    assert sample(1) != sample(2)


def test_a_temperature_outside_the_api_range_is_rejected(client):
    assert post(client, messages=MESSAGES, temperature=2.5).status_code == 422


def test_a_request_larger_than_the_cache_fails():
    with TestClient(create_app(lambda: load(n_lines=2))) as client:
        response = post(client, messages=MESSAGES)
        assert response.status_code == 503


def test_an_error_after_a_stream_started_is_an_error_event():
    with TestClient(create_app(lambda: load(n_lines=2))) as client:
        response = post(client, messages=MESSAGES, stream=True)

    assert response.status_code == 200
    events = [line for line in response.text.split("\n\n") if line]
    assert events[-1] == "data: [DONE]"
    error = json.loads(events[-2].removeprefix("data: "))["error"]
    assert error["code"] == 503


def test_the_view_cap_keeps_positions_within_the_trained_context():
    pages = trained_view_pages(40960, cycle_tokens=512)
    assert pages is not None
    assert pages * PAGE_SIZE + (PAGE_SIZE - 1) + 512 <= 40960
    assert (pages + 1) * PAGE_SIZE + (PAGE_SIZE - 1) + 512 > 40960
    assert trained_view_pages(0, cycle_tokens=512) is None


def test_an_unknown_model_is_not_found(client):
    response = client.post(
        "/api/openai/v1/chat/completions",
        json={"model": "other", "messages": MESSAGES},
    )
    assert response.status_code == 404
