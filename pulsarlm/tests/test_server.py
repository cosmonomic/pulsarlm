"""The Chat Completions route on a small random Qwen3 and a word-level tokenizer."""

import asyncio
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from fastapi.testclient import TestClient
from tokenizers import Tokenizer, models, pre_tokenizers

from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind

from pulsarlm.async_engine import AsyncEngine
from pulsarlm.chat import ChatTemplate, encode_prompt
from pulsarlm.core.engine import (
    Engine,
    EvictionPolicy,
    ModelConfig,
    RopeParameters,
    State,
)
from pulsarlm.eviction_log import EvictionLog, prompt_sha256
from pulsarlm.server import create_app
from pulsarlm.server.openai import ServerState
from pulsarlm.server.openai.routes import template_messages
from pulsarlm.server.openai.schema import ChatMessage
from pulsarlm.server.telemetry import start_telemetry
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
BASE_URL = "http://testserver/api/openai/v1"
SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
ROLES = ["system", "user", "assistant"]

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


CHATML = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
IM_END = SPECIAL_TOKENS.index("<|im_end|>")
# The random model samples "assistant" first.
FIRST_SAMPLED = len(SPECIAL_TOKENS) + ROLES.index("assistant")


class RecordingStore(LocalStore):
    """Records every state found with the digests it was found by."""

    def __init__(self) -> None:
        super().__init__(PAGE_SIZE, capacity_bytes=64 * LINE_BYTES)
        self.state_reads: list[tuple[State, list[bytes]]] = []

    async def read_state(
        self, tokens: list[int], page_digests: list[bytes]
    ) -> tuple[State, int]:
        state, lease_id = await super().read_state(tokens, page_digests)
        self.state_reads.append((state, page_digests))
        return state, lease_id


def covered_tokens(state: State, page_digests: list[bytes]) -> int:
    pages = 0 if state.head is None else page_digests.index(state.head) + 1
    return pages * PAGE_SIZE + len(state.tail)


def load(
    stop_ids: list[int] = [IM_END],
    n_lines: int = 64,
    embedding_scale: float = 1.0,
    chat_template: str = CHATML,
    store: LocalStore | None = None,
) -> ServerState:
    tokenizer = word_tokenizer()
    if store is None:
        store = LocalStore(PAGE_SIZE, capacity_bytes=64 * LINE_BYTES)
    engine = Engine(
        store,
        asyncio.get_running_loop(),
        MODEL,
        lambda rank, size: random_weights(embedding_scale),
        model_name=MODEL_ID,
        model_hash="1",
        page_size=PAGE_SIZE,
        kv_dtype=DTYPE,
        kv_cache_gib=n_lines * LINE_BYTES / 2**30,
        store_gib=64 * LINE_BYTES / 2**30,
        devices=[DEVICE],
        stop_ids=stop_ids,
        prefill_tokens_per_step=64,
        decode_graphs=True,
        max_context_window=4096,
        draft_tokens=0,
    )
    return ServerState(
        engine=AsyncEngine(engine),
        tokenizer=tokenizer,
        chat_template=ChatTemplate(chat_template, {}),
        model_id=MODEL_ID,
        stop_ids=stop_ids,
        tool_call_syntax=None,
        think_syntax=None,
        eviction=EvictionPolicy(cycle_tokens=PAGE_SIZE),
        eviction_log=None,
        telemetry=start_telemetry(engine.stat, store.stat, MODEL_ID),
    )


SYSTEM = " ".join(f"w{i}" for i in range(10, 40))
MESSAGES = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "w7 w8 w9"},
]


def encode(state: ServerState, messages: list[dict[str, str]]) -> list[int]:
    parsed = [ChatMessage.model_validate(m) for m in messages]
    prompt, _ = encode_prompt(
        state.tokenizer, state.chat_template, template_messages(parsed)
    )
    return prompt


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def api_client(load_state: Callable[[], ServerState]) -> TestClient:
    return TestClient(create_app(load_state), base_url=BASE_URL)


@pytest.fixture
def client():
    with api_client(load) as client:
        yield client


def post(client: TestClient, **body):
    return client.post("/chat/completions", json={"model": MODEL_ID, **body})


def test_the_models_route_lists_only_the_loaded_model(client):
    response = client.get("/models")
    assert response.status_code == 200
    assert [card["id"] for card in response.json()["data"]] == [MODEL_ID]


def test_a_completion_stops_at_max_tokens(client):
    response = post(client, messages=MESSAGES, max_tokens=4)

    assert response.status_code == 200
    body = response.json()
    assert body["usage"] == {
        "prompt_tokens": 41,
        "completion_tokens": 4,
        "total_tokens": 45,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    assert body["choices"][0]["finish_reason"] == "length"


def test_a_streamed_completion_ends_with_done(client):
    response = post(client, messages=MESSAGES, max_tokens=4, stream=True)

    assert response.status_code == 200
    events = [line for line in response.text.split("\n\n") if line]
    assert events[-1] == "data: [DONE]"
    last = json.loads(events[-2].removeprefix("data: "))
    assert last["choices"][0]["finish_reason"] == "length"


def test_a_stream_with_include_usage_ends_with_a_usage_chunk(client):
    response = post(
        client,
        messages=MESSAGES,
        max_tokens=4,
        stream=True,
        stream_options={"include_usage": True},
    )

    events = [line for line in response.text.split("\n\n") if line]
    assert events[-1] == "data: [DONE]"
    last = json.loads(events[-2].removeprefix("data: "))
    assert last["choices"] == []
    assert last["usage"] == {
        "prompt_tokens": 41,
        "completion_tokens": 4,
        "total_tokens": 45,
        "prompt_tokens_details": {"cached_tokens": 0},
    }


def test_a_follow_up_continues_from_a_stored_state():
    store = RecordingStore()
    with api_client(lambda: load(store=store)) as client:
        first = post(client, messages=MESSAGES, max_tokens=4).json()
        reply = first["choices"][0]["message"]["content"]
        follow_up = MESSAGES + [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": "w11"},
        ]
        response = post(client, messages=follow_up, max_tokens=4)

    assert response.status_code == 200
    state, digests = store.state_reads[-1]
    assert covered_tokens(state, digests) >= 2 * PAGE_SIZE
    full_page_tokens = covered_tokens(state, digests) - len(state.tail)
    details = response.json()["usage"]["prompt_tokens_details"]
    assert details == {"cached_tokens": full_page_tokens}


def test_a_reply_ending_at_any_stop_token_leaves_its_end_state_findable():
    store = RecordingStore()
    with api_client(lambda: load([IM_END, FIRST_SAMPLED], store=store)) as client:
        first = post(client, messages=MESSAGES, max_tokens=4).json()
        assert first["choices"][0]["finish_reason"] == "stop"
        assert first["usage"]["completion_tokens"] == 1

        follow_up = MESSAGES + [
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "w11"},
        ]
        assert post(client, messages=follow_up, max_tokens=1).status_code == 200

    state, digests = store.state_reads[-1]
    assert covered_tokens(state, digests) == first["usage"]["prompt_tokens"]


def test_a_seeded_sampled_completion_repeats():
    def sample(seed: int) -> str:
        # A small embedding scale spreads the next-token distribution.
        with api_client(lambda: load(embedding_scale=0.05)) as client:
            body = post(
                client, messages=MESSAGES, max_tokens=8, temperature=1.0, seed=seed
            ).json()
            return body["choices"][0]["message"]["content"]

    assert sample(1) == sample(1)
    assert sample(1) != sample(2)


def test_a_temperature_outside_the_api_range_is_rejected(client):
    assert post(client, messages=MESSAGES, temperature=2.5).status_code == 422


def test_a_request_larger_than_the_cache_fails():
    with api_client(lambda: load(n_lines=2)) as client:
        response = post(client, messages=MESSAGES)
        assert response.status_code == 400


def test_an_error_after_a_stream_started_is_an_error_event():
    with api_client(lambda: load(n_lines=2)) as client:
        response = post(client, messages=MESSAGES, stream=True)

    assert response.status_code == 200
    events = [line for line in response.text.split("\n\n") if line]
    assert events[-1] == "data: [DONE]"
    error = json.loads(events[-2].removeprefix("data: "))["error"]
    assert error["code"] == 400


def test_a_chat_template_error_is_a_bad_request():
    template = "{{ raise_exception('unsupported effort ' ~ reasoning_effort) }}"
    with api_client(lambda: load(chat_template=template)) as client:
        response = post(client, messages=MESSAGES, reasoning_effort="high")

    assert response.status_code == 400
    assert response.json()["detail"] == "chat template: unsupported effort high"


def test_without_a_system_message_only_the_first_token_is_sunk(client):
    state = client.app.state.server_state
    assert state.eviction_for(0).n_sink == 1
    assert state.eviction_for(PAGE_SIZE + 1).n_sink == PAGE_SIZE + 1


def test_an_unknown_model_is_not_found(client):
    response = client.post(
        "/chat/completions",
        json={"model": "other", "messages": MESSAGES},
    )
    assert response.status_code == 404


def logging_eviction(log: Path, eviction: EvictionPolicy) -> Callable[[], ServerState]:
    def load_state() -> ServerState:
        state = load(stop_ids=[])
        state.eviction = eviction
        state.eviction_log = EvictionLog(log, state.engine, eviction)
        return state

    return load_state


def read_log(log: Path) -> list[dict[str, Any]]:
    with log.open() as f:
        return [json.loads(line) for line in f]


def test_the_eviction_log_without_eviction_views_every_full_page(tmp_path):
    log = tmp_path / "eviction.jsonl"
    eviction = EvictionPolicy(cycle_tokens=PAGE_SIZE)
    with api_client(logging_eviction(log, eviction)) as client:
        body = post(client, messages=MESSAGES, max_tokens=40).json()
        prompt = encode(cast(Any, client.app).state.server_state, MESSAGES)

    [line] = read_log(log)
    assert line["id"] == body["id"]
    assert line["prompt_tokens"] == len(prompt)
    assert line["prompt_sha256"] == prompt_sha256(prompt)
    assert all(state["cap_tokens"] is None for state in line["states"])
    end = line["states"][-1]
    assert [layer["layer"] for layer in end["layers"]] == list(range(N_LAYERS))
    full = end["tokens"] // PAGE_SIZE * PAGE_SIZE
    for layer in end["layers"]:
        assert layer["threshold"] == -math.inf
        assert [page[:2] for page in layer["pages"]] == [
            [start, start + PAGE_SIZE] for start in range(0, full, PAGE_SIZE)
        ]


def test_the_eviction_log_leaves_out_evicted_pages(tmp_path):
    log = tmp_path / "eviction.jsonl"
    eviction = EvictionPolicy(
        n_working=PAGE_SIZE,
        cycle_tokens=PAGE_SIZE,
        max_context_capacity=7 * PAGE_SIZE,
    )
    with api_client(logging_eviction(log, eviction)) as client:
        post(client, messages=MESSAGES, max_tokens=120)

    [line] = read_log(log)
    *cycle_ends, end = line["states"]
    for state in cycle_ends:
        assert state["cap_tokens"] == 6 * PAGE_SIZE
        for layer in state["layers"]:
            in_view = sum(stop - start for start, stop, _ in layer["pages"])
            assert in_view <= state["cap_tokens"]
    full = end["tokens"] // PAGE_SIZE * PAGE_SIZE
    assert [layer["layer"] for layer in end["layers"]] == list(range(N_LAYERS))
    for layer in end["layers"]:
        starts = [start for start, _, _ in layer["pages"]]
        assert starts == sorted(set(starts))
        assert starts[0] == 0
        assert layer["pages"][-1][1] == full
        assert len(starts) < full // PAGE_SIZE


def metric_value(text: str, name: str) -> float:
    [line] = [line for line in text.splitlines() if line.startswith(f"{name}{{")]
    return float(line.rsplit(" ", 1)[1])


def test_metrics_count_no_device_memory_once_requests_end(client):
    post(client, messages=MESSAGES, max_tokens=4)

    response = client.get("http://testserver/metrics")

    assert response.status_code == 200
    assert metric_value(response.text, "pulsarlm_engine_device_memory_bytes") == 0


def test_metrics_count_the_tokens_of_finished_requests(client):
    post(client, messages=MESSAGES, max_tokens=4)
    follow_up = post(client, messages=MESSAGES, max_tokens=4).json()["usage"]

    text = client.get("http://testserver/metrics").text

    cached = follow_up["prompt_tokens_details"]["cached_tokens"]
    assert cached > 0
    assert (
        metric_value(text, "pulsarlm_engine_usage_input_tokens_sum")
        == 2 * follow_up["prompt_tokens"]
    )
    assert (
        metric_value(text, "pulsarlm_engine_usage_input_tokens_on_device_sum")
        + metric_value(text, "pulsarlm_engine_usage_input_tokens_in_store_sum")
        == cached
    )


def test_metrics_report_the_store_s_pages_states_and_capacity(client):
    post(client, messages=MESSAGES, max_tokens=4)

    text = client.get("http://testserver/metrics").text

    stat = client.app.state.server_state.telemetry.store_stat()
    assert stat.pages > 0
    assert metric_value(text, "pulsarlm_store_usage_bytes") == stat.page_bytes
    assert metric_value(text, "pulsarlm_store_limit_bytes") == 64 * LINE_BYTES
    assert metric_value(text, "pulsarlm_store_utilization") == pytest.approx(
        stat.page_bytes / (64 * LINE_BYTES)
    )
    assert metric_value(text, "pulsarlm_store_pages") == stat.pages
    assert metric_value(text, "pulsarlm_store_states") == stat.states
    assert metric_value(text, "pulsarlm_store_leases") == 0
    assert metric_value(text, "pulsarlm_store_dropped_states_total") == 0


def test_metrics_hold_the_gen_ai_server_histograms_of_finished_requests(client):
    post(client, messages=MESSAGES, max_tokens=4)

    text = client.get("http://testserver/metrics").text

    for name in (
        "gen_ai_server_request_duration_seconds",
        "gen_ai_server_time_to_first_token_seconds",
        "gen_ai_server_time_per_output_token_seconds",
    ):
        [count] = [
            line for line in text.splitlines() if line.startswith(f"{name}_count{{")
        ]
        assert 'gen_ai_operation_name="chat"' in count
        assert 'gen_ai_provider_name="pulsarlm"' in count
        assert f'gen_ai_request_model="{MODEL_ID}"' in count
        assert count.endswith(" 1.0")


def test_a_finished_request_ends_one_request_span_with_its_usage(client):
    exporter = InMemorySpanExporter()
    state: ServerState = client.app.state.server_state
    state.telemetry.tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    response = post(client, messages=MESSAGES, max_tokens=4).json()
    usage = response["usage"]

    [span] = exporter.get_finished_spans()
    attributes = span.attributes or {}
    assert span.name == "request"
    assert span.kind == SpanKind.SERVER
    assert attributes["gen_ai.operation.name"] == "chat"
    assert attributes["gen_ai.response.model"] == MODEL_ID
    assert attributes["gen_ai.response.id"] == response["id"]
    assert attributes["gen_ai.response.finish_reasons"] == (
        response["choices"][0]["finish_reason"],
    )
    assert attributes["gen_ai.usage.input_tokens"] == usage["prompt_tokens"]
    assert attributes["gen_ai.usage.output_tokens"] == usage["completion_tokens"]
    assert attributes["gen_ai.request.max_tokens"] == 4
    peak_device_memory = attributes["pulsarlm.engine.usage.peak_device_memory"]
    time_to_first_token = attributes["pulsarlm.engine.time_to_first_token"]
    duration = attributes["pulsarlm.engine.request.duration"]
    assert isinstance(peak_device_memory, int)
    assert isinstance(time_to_first_token, float)
    assert isinstance(duration, float)
    assert peak_device_memory > 0
    assert 0 < time_to_first_token <= duration
    assert attributes["pulsarlm.store.usage"] > 0
    assert 0 < attributes["pulsarlm.store.utilization"] <= 1


def test_the_server_exports_its_unexported_spans_when_it_shuts_down():
    exporter = InMemorySpanExporter()
    app = create_app(load)
    with TestClient(app, base_url=BASE_URL) as client:
        state: ServerState = app.state.server_state
        state.telemetry.tracer_provider.add_span_processor(
            BatchSpanProcessor(exporter, schedule_delay_millis=3_600_000)
        )
        post(client, messages=MESSAGES, max_tokens=4)
        assert exporter.get_finished_spans() == ()

    assert len(exporter.get_finished_spans()) == 1
