"""Prefill throughput, decode throughput and KV cache bytes in use along growing
chat sessions, with and without eviction, against vLLM."""

import asyncio
import importlib
import json
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
import typer
from openai import APIConnectionError, APIError, APIStatusError, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from tokenizers import Tokenizer
from wonderwords import RandomWord

from pulsarlm import checkpoint
from pulsarlm.models import MODELS, hugging_face_repository

INSTRUCTION = "\n\nContinue the story, using as many of the words above as you can."
METRICS_INTERVAL_S = 0.05


@dataclass(frozen=True)
class Setting:
    engine: Literal["pulsarlm", "vllm"]
    max_context_window: int | None = None
    prefix_cache: bool = True


SETTINGS = {
    "full_kv": Setting("pulsarlm"),
    "full_kv_no_prefix_cache": Setting("pulsarlm", prefix_cache=False),
    "cap_4k": Setting("pulsarlm", max_context_window=4096),
    "cap_8k": Setting("pulsarlm", max_context_window=8192),
    "cap_16k": Setting("pulsarlm", max_context_window=16384),
    "cap_16k_no_prefix_cache": Setting(
        "pulsarlm", max_context_window=16384, prefix_cache=False
    ),
    "vllm": Setting("vllm"),
    "vllm_no_prefix_cache": Setting("vllm", prefix_cache=False),
}


@dataclass(frozen=True)
class Server:
    base_url: str
    metrics_url: str
    request_options: dict[str, Any]


def words() -> list[str]:
    vocabulary = RandomWord().filter(include_parts_of_speech=["nouns", "adjectives"])
    return sorted({w.lower() for w in vocabulary if w.isalpha()})


def user_message(
    tokens: int, rng: random.Random, vocabulary: list[str], tokenizer: Tokenizer
) -> str:
    """Random words of tokens tokens, then the instruction."""
    text = " ".join(rng.choices(vocabulary, k=tokens))
    ids = tokenizer.encode(text, add_special_tokens=False).ids[:tokens]
    return tokenizer.decode(ids) + INSTRUCTION


def wait_until_ready(
    process: subprocess.Popen[bytes], base_url: str, log: Path
) -> None:
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"the server exited, see {log}")
        try:
            httpx.get(f"{base_url}/models", timeout=1).raise_for_status()
            return
        except httpx.HTTPError:
            time.sleep(1)


@contextmanager
def run_server(command: list[str], base_url: str, log: Path) -> Iterator[None]:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        try:
            wait_until_ready(process, base_url, log)
            yield
        finally:
            process.terminate()
            process.wait()


@contextmanager
def run_pulsarlm_server(
    model: str,
    setting: Setting,
    data_dir: Path,
    port: int,
    *,
    kv_cache_gib: float,
    store_gib: float,
    n_working: int,
    prefill_tokens_per_step: int,
    max_context: int,
    tensor_parallel_size: int,
) -> Iterator[Server]:
    executable = Path(sys.executable).parent / "pulsarlm"
    command = [str(executable), "serve", model, "--data_dir", str(data_dir)]
    command += ["--port", str(port), "--no_persist"]
    command += ["--tensor_parallel_size", str(tensor_parallel_size)]
    command += ["--kv_cache_gib", str(kv_cache_gib), "--store_gib", str(store_gib)]
    command += ["--prefill_tokens_per_step", str(prefill_tokens_per_step)]
    if not setting.prefix_cache:
        command += ["--no_prefix_cache"]
    if setting.max_context_window is None:
        # The view would otherwise be capped at the model's trained context.
        command += [
            "--eviction_threshold",
            "0",
            "--max_context_window",
            str(max_context),
        ]
    else:
        command += ["--max_context_window", str(setting.max_context_window)]
        command += ["--n_working", str(n_working)]
    root = f"http://127.0.0.1:{port}"
    base_url = f"{root}/api/openai/v1"
    with run_server(command, base_url, data_dir / "serve.log"):
        yield Server(
            base_url=base_url,
            metrics_url=f"{root}/metrics",
            request_options={"extra_body": {"reasoning_effort": "none"}},
        )


@contextmanager
def run_vllm_server(
    model: str,
    setting: Setting,
    data_dir: Path,
    port: int,
    *,
    vllm: str,
    kv_cache_bytes: int,
    prefill_tokens_per_step: int,
    max_context: int,
    tensor_parallel_size: int,
) -> Iterator[Server]:
    command = ["uvx", "--python", "3.12", "--from", vllm, "vllm", "serve"]
    command += [hugging_face_repository(MODELS[model].url)]
    command += ["--served-model-name", model]
    command += ["--port", str(port), "--dtype", "bfloat16"]
    command += ["--tensor-parallel-size", str(tensor_parallel_size)]
    command += ["--kv-cache-memory-bytes", str(kv_cache_bytes)]
    # The longest context up to max_context that the KV pool holds, so a longer
    # request is rejected; auto searches up to max_position_embeddings.
    command += ["--max-model-len", "auto"]
    overrides = {"text_config": {"max_position_embeddings": max_context}}
    command += ["--hf-overrides", json.dumps(overrides)]
    command += ["--max-num-batched-tokens", str(prefill_tokens_per_step)]
    # Without the vision encoder, as pulsarlm serves text only; it also drops the
    # floor vLLM puts on max_num_batched_tokens to fit one video.
    command += ["--language-model-only"]
    if not setting.prefix_cache:
        command += ["--no-enable-prefix-caching"]
    root = f"http://127.0.0.1:{port}"
    base_url = f"{root}/v1"
    with run_server(command, base_url, data_dir / "serve.log"):
        yield Server(
            base_url=base_url,
            metrics_url=f"{root}/metrics",
            request_options={
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
            },
        )


def gauge(text: str, name: str) -> float:
    for line in text.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            return float(line.rsplit(" ", 1)[1])
    raise ValueError(f"no {name} in the metrics")


def kv_cache_bytes(metrics: str, engine: str, pool_bytes: int) -> float:
    if engine == "vllm":
        return gauge(metrics, "vllm:kv_cache_usage_perc") * pool_bytes
    return gauge(metrics, "pulsarlm:kv_cache_bytes")


async def peak_kv_cache_bytes(
    metrics_url: str, engine: str, pool_bytes: int, done: asyncio.Event
) -> float:
    peak = 0.0
    async with httpx.AsyncClient() as client:
        while not done.is_set():
            response = await client.get(metrics_url)
            response.raise_for_status()
            peak = max(peak, kv_cache_bytes(response.text, engine, pool_bytes))
            try:
                await asyncio.wait_for(done.wait(), METRICS_INTERVAL_S)
            except TimeoutError:
                pass
    return peak


def error_code(error: APIError) -> object:
    """The HTTP status of an error response, or the code of an error event
    sent after the stream started."""
    if isinstance(error, APIStatusError):
        return error.status_code
    return error.body.get("code") if isinstance(error.body, dict) else None


def is_oom(error: APIError) -> bool:
    """pulsarlm answers 400 when a request does not fit its KV cache, before or
    after its stream starts; vLLM answers 400 when it is longer than the context
    its KV pool holds."""
    return error_code(error) == 400 and (
        "out of device memory" in error.message or "context length" in error.message
    )


async def stream_request(
    client: AsyncOpenAI,
    server: Server,
    model: str,
    messages: list[ChatCompletionMessageParam],
    max_tokens: int,
    start: float,
) -> dict[str, Any]:
    """Seconds from start to the first and last chunk carrying text, the token
    counts and the reply."""
    first_token_s: float | None = None
    last_token_s: float | None = None
    usage: dict[str, int] | None = None
    reply: list[str] = []
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True},
            **server.request_options,
        )
        async for chunk in stream:
            if chunk.usage is not None:
                usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                }
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            extra = delta.model_extra or {}
            if delta.content:
                reply.append(delta.content)
            if (
                delta.content
                or extra.get("reasoning")
                or extra.get("reasoning_content")
            ):
                last_token_s = time.perf_counter() - start
                if first_token_s is None:
                    first_token_s = last_token_s
    except APIConnectionError:
        raise
    except APIError as error:
        if is_oom(error):
            return {"error": "oom"}
        return {"error": f"{error_code(error)}: {error.message}"}
    if usage is None or first_token_s is None:
        return {"error": "the stream ended without text or usage"}
    return {
        **usage,
        "first_token_s": first_token_s,
        "last_token_s": last_token_s,
        "reply": "".join(reply),
        "error": None,
    }


async def measure_turn(
    server: Server,
    engine: str,
    model: str,
    sessions: list[list[ChatCompletionMessageParam]],
    *,
    max_tokens: int,
    pool_bytes: int,
) -> dict[str, Any]:
    """Sends every session's next request at once."""
    done = asyncio.Event()
    async with AsyncOpenAI(
        base_url=server.base_url, api_key="none", timeout=3600
    ) as client:
        start = time.perf_counter()
        peak = asyncio.create_task(
            peak_kv_cache_bytes(server.metrics_url, engine, pool_bytes, done)
        )
        requests = await asyncio.gather(
            *(
                stream_request(client, server, model, messages, max_tokens, start)
                for messages in sessions
            )
        )
        done.set()
        return {"requests": requests, "peak_kv_cache_bytes": await peak}


def load_tokenizer(model: str) -> Tokenizer:
    with checkpoint.model_root(checkpoint.model_path(model)) as root:
        return checkpoint.load_tokenizer(root)


def ended(last: dict[str, Any], *, max_context: int, max_tokens: int) -> bool:
    """Whether a run's last turn failed or no further turn fits max_context."""
    context = last["context_tokens"]
    return context is None or context + last["turn_tokens"] + max_tokens > max_context


RunKey = tuple[str, int, int, int]


def run_key(row: dict[str, Any]) -> RunKey:
    """A run's setting, concurrency, tensor-parallel size and index."""
    return row["setting"], row["concurrency"], row["tensor_parallel_size"], row["run"]


def finished_runs(results: Path, *, max_context: int, max_tokens: int) -> set[RunKey]:
    """The runs whose sessions ended. Removes the rows of the others from
    results, as their sessions cannot be resumed, and a last line cut short by a
    job killed while writing it."""
    if not results.exists():
        return set()
    with results.open() as f:
        lines = f.read().splitlines()
    rows = [json.loads(line) for line in lines[:-1]]
    if lines:
        try:
            rows.append(json.loads(lines[-1]))
        except json.JSONDecodeError:
            pass
    last: dict[RunKey, dict[str, Any]] = {}
    for row in rows:
        last[run_key(row)] = row
    finished = {
        key
        for key, row in last.items()
        if ended(row, max_context=max_context, max_tokens=max_tokens)
    }
    kept = [row for row in rows if run_key(row) in finished]
    if len(kept) < len(lines):
        with results.open("w") as f:
            f.writelines(json.dumps(row) + "\n" for row in kept)
    return finished


@contextmanager
def run_server_for(
    name: str,
    model: str,
    data_dir: Path,
    port: int,
    *,
    kv_cache_gib: float,
    store_gib: float,
    n_working: int,
    prefill_tokens_per_step: int,
    max_context: int,
    vllm: str,
    tensor_parallel_size: int,
) -> Iterator[Server]:
    setting = SETTINGS[name]
    if setting.engine == "vllm":
        with run_vllm_server(
            model,
            setting,
            data_dir,
            port,
            vllm=vllm,
            kv_cache_bytes=int(kv_cache_gib * 2**30),
            prefill_tokens_per_step=prefill_tokens_per_step,
            max_context=max_context,
            tensor_parallel_size=tensor_parallel_size,
        ) as server:
            yield server
    else:
        with run_pulsarlm_server(
            model,
            setting,
            data_dir,
            port,
            kv_cache_gib=kv_cache_gib,
            store_gib=store_gib,
            n_working=n_working,
            prefill_tokens_per_step=prefill_tokens_per_step,
            max_context=max_context,
            tensor_parallel_size=tensor_parallel_size,
        ) as server:
            yield server


def context_tokens(requests: list[dict[str, Any]]) -> int:
    """The mean context of the sessions after a turn, prompt and reply."""
    return round(
        statistics.mean(r["prompt_tokens"] + r["completion_tokens"] for r in requests)
    )


app = typer.Typer(no_args_is_help=True)


@app.command()
def run(
    setting: Annotated[str, typer.Option()],
    model: str = "gemma-4-12b-it",
    out: Path = Path("bench/results/perf"),
    concurrency: int = 1,
    runs: int = 5,
    max_context: int = 1048576,
    turn_tokens: int = 4096,
    max_tokens: int = 256,
    seed: int = 0,
    kv_cache_gib: float = 4.0,
    store_gib: float = 8.0,
    n_working: int = 1024,
    prefill_tokens_per_step: int = 256,
    vllm: str = "vllm==0.31.0",
    port: int = 18091,
    tensor_parallel_size: int = 1,
) -> None:
    """Runs setting until runs of its runs have finished, each on a fresh server: grows
    concurrency chat sessions side by side, each turn adding a user message of
    turn_tokens random words and a reply of at most max_tokens, until a session
    would pass max_context or a request fails, and appends a row per turn to
    out/results.jsonl, which accumulates every setting's, concurrency's and
    tensor-parallel size's runs. Runs 0 to runs - 1 that have not finished run, so a
    job restarted after preemption only finishes the rest; run r draws its messages
    from seed and r, the same in every setting. A request that does not fit the KV
    cache fails with oom. A run whose sessions did not end there is dropped and run
    again under its index. Both engines forward at most prefill_tokens_per_step
    prompt tokens per step, and full KV and vLLM may run past the model's trained
    context up to max_context. Both engines split the model across
    tensor_parallel_size GPUs, kv_cache_gib sizing each GPU's KV cache."""
    if setting not in SETTINGS:
        raise typer.BadParameter(f"unknown setting: {setting}")
    tokenizer = load_tokenizer(model)
    vocabulary = words()
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    done = finished_runs(results, max_context=max_context, max_tokens=max_tokens)
    pool_bytes = int(kv_cache_gib * 2**30)
    engine = SETTINGS[setting].engine

    finished = {
        key[3]
        for key in done
        if key[:3] == (setting, concurrency, tensor_parallel_size)
    }
    new_runs = [i for i in range(runs) if i not in finished]
    for run_index in new_runs:

        def message(session: int, turn: int) -> str:
            rng = random.Random(f"{seed}/{run_index}/{concurrency}/{session}/{turn}")
            return user_message(turn_tokens, rng, vocabulary, tokenizer)

        with run_server_for(
            setting,
            model,
            out / f"{setting}_x{concurrency}_run{run_index}",
            port,
            kv_cache_gib=kv_cache_gib,
            store_gib=store_gib,
            n_working=n_working,
            prefill_tokens_per_step=prefill_tokens_per_step,
            max_context=max_context,
            vllm=vllm,
            tensor_parallel_size=tensor_parallel_size,
        ) as server:
            warmup = random.Random(f"{seed}/{run_index}/warmup")
            asyncio.run(
                measure_turn(
                    server,
                    engine,
                    model,
                    [
                        [
                            {
                                "role": "user",
                                "content": user_message(
                                    turn_tokens, warmup, vocabulary, tokenizer
                                ),
                            }
                        ]
                    ],
                    max_tokens=16,
                    pool_bytes=pool_bytes,
                )
            )
            sessions: list[list[ChatCompletionMessageParam]] = [
                [] for _ in range(concurrency)
            ]
            context = 0
            turn = 0
            while context + turn_tokens + max_tokens <= max_context:
                for session, messages in enumerate(sessions):
                    messages.append({"role": "user", "content": message(session, turn)})
                measured = asyncio.run(
                    measure_turn(
                        server,
                        engine,
                        model,
                        sessions,
                        max_tokens=max_tokens,
                        pool_bytes=pool_bytes,
                    )
                )
                requests = measured["requests"]
                failed = any(r["error"] is not None for r in requests)
                previous_context = context
                if not failed:
                    context = context_tokens(requests)
                row = {
                    "model": model,
                    "setting": setting,
                    "concurrency": concurrency,
                    "tensor_parallel_size": tensor_parallel_size,
                    "run": run_index,
                    "turn": turn,
                    "previous_context_tokens": previous_context,
                    "turn_tokens": turn_tokens,
                    "context_tokens": None if failed else context,
                    "kv_cache_pool_bytes": pool_bytes,
                    "peak_kv_cache_bytes": measured["peak_kv_cache_bytes"],
                    "requests": [
                        {k: v for k, v in r.items() if k != "reply"} for r in requests
                    ],
                }
                with results.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(
                    f"{setting} x{concurrency} run {run_index} turn {turn}: "
                    f"{summary(row)}",
                    flush=True,
                )
                if failed:
                    break
                for messages, request in zip(sessions, requests):
                    messages.append({"role": "assistant", "content": request["reply"]})
                turn += 1


@dataclass(frozen=True)
class TurnMetrics:
    context_tokens: int
    prefill_rate: float
    ttft_s: float
    decode_rate: float
    kv_gib: float


def turn_metrics(row: dict[str, Any]) -> TurnMetrics | None:
    """Prefill: the prompt tokens past the previous turn's prompt over the time
    to the last session's first token. TTFT: the mean over the sessions of the
    time to their first token. Decode: each session's tokens
    after its first over the time from its first token to its last, summed over
    the sessions. KV: the most the engine's running requests held while the turn
    ran. None when a request failed."""
    requests = row["requests"]
    if row["context_tokens"] is None:
        return None
    new_tokens = sum(r["prompt_tokens"] for r in requests) - row[
        "previous_context_tokens"
    ] * len(requests)
    first_tokens = [r["first_token_s"] for r in requests]
    prefill = new_tokens / max(first_tokens)
    decode = sum(
        (r["completion_tokens"] - 1) / (r["last_token_s"] - r["first_token_s"])
        for r in requests
        if r["last_token_s"] > r["first_token_s"]
    )
    return TurnMetrics(
        row["context_tokens"],
        prefill,
        statistics.mean(first_tokens),
        decode,
        row["peak_kv_cache_bytes"] / 2**30,
    )


@dataclass(frozen=True)
class OomTurn:
    """A turn failing with oom: the context it tried to reach, the throughput and
    mean time to first token of its sessions whose request finished, None when
    none did, and the most KV the engine held while it ran."""

    context_tokens: int
    finished_sessions: int
    prefill_rate: float | None
    ttft_s: float | None
    decode_rate: float | None
    kv_gib: float


def oom_turn(row: dict[str, Any]) -> OomTurn:
    """turn_metrics over the turn's finished requests."""
    finished = [
        r
        for r in row["requests"]
        if r["error"] is None and r.get("first_token_s") is not None
    ]
    previous = row["previous_context_tokens"]
    prefill = (
        sum(r["prompt_tokens"] - previous for r in finished)
        / max(r["first_token_s"] for r in finished)
        if finished
        else None
    )
    decoding = [r for r in finished if r["last_token_s"] > r["first_token_s"]]
    decode = (
        sum(
            (r["completion_tokens"] - 1) / (r["last_token_s"] - r["first_token_s"])
            for r in decoding
        )
        if decoding
        else None
    )
    return OomTurn(
        previous + row["turn_tokens"],
        len(finished),
        prefill,
        statistics.mean(r["first_token_s"] for r in finished) if finished else None,
        decode,
        row["peak_kv_cache_bytes"] / 2**30,
    )


def summary(row: dict[str, Any]) -> str:
    metrics = turn_metrics(row)
    if metrics is None:
        errors = {r["error"] for r in row["requests"] if r["error"] is not None}
        return ", ".join(sorted(errors))
    return (
        f"context {metrics.context_tokens}, prefill {metrics.prefill_rate:.0f} tok/s, "
        f"decode {metrics.decode_rate:.0f} tok/s, KV {metrics.kv_gib:.2f} GiB"
    )


@dataclass(frozen=True)
class Series:
    label: str
    color: str
    filled: bool = True


SERIES = {
    "full_kv": Series("Full KV", "#52514e"),
    "full_kv_no_prefix_cache": Series(
        "Full KV, no prefix cache", "#52514e", filled=False
    ),
    "cap_4k": Series("Capped @ 4k", "#86b6ef"),
    "cap_8k": Series("Capped @ 8k", "#2a78d6"),
    "cap_16k": Series("Capped @ 16k", "#104281"),
    "cap_16k_no_prefix_cache": Series(
        "Capped @ 16k, no prefix cache", "#104281", filled=False
    ),
    "vllm": Series("vLLM", "#d95926"),
    "vllm_no_prefix_cache": Series("vLLM, no prefix cache", "#d95926", filled=False),
}

PANELS = {
    "ttft_s": "Time to first token (s)",
    "decode_rate": "Decode (tok/s)",
    "kv_gib": "KV cache in use (GiB)",
}


@dataclass(frozen=True)
class Run:
    turns: list[TurnMetrics]
    oom: OomTurn | None
    error: str | None


# A series: its setting, tensor-parallel size and concurrency.
SeriesKey = tuple[str, int, int]


def load_runs(results: Path) -> dict[RunKey, Run]:
    """Each run in results, which holds one model's runs."""
    with results.open() as f:
        rows = [json.loads(line) for line in f]
    models = {row["model"] for row in rows}
    if len(models) > 1:
        raise typer.BadParameter(f"{results} mixes models: {sorted(models)}")
    grouped: dict[RunKey, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[run_key(row)].append(row)
    runs = {}
    for key, run_rows in grouped.items():
        run_rows.sort(key=lambda row: row["turn"])
        turns = [m for row in run_rows if (m := turn_metrics(row)) is not None]
        last = run_rows[-1]
        errors = {r["error"] for r in last["requests"] if r["error"] is not None}
        oom = oom_turn(last) if "oom" in errors else None
        other = sorted(errors - {"oom"})
        runs[key] = Run(turns, oom, ", ".join(other) or None)
    return runs


@dataclass(frozen=True)
class Estimate:
    """A metric's mean over n runs, and the half-width of its 95% Student t
    interval, None below two runs."""

    mean: float
    half_width: float | None
    n: int


def estimate(values: list[float]) -> Estimate:
    n = len(values)
    if n < 2:
        return Estimate(values[0], None, n)
    stats = importlib.import_module("scipy.stats")
    t = float(stats.t.ppf(0.975, n - 1))
    return Estimate(statistics.mean(values), t * statistics.stdev(values) / n**0.5, n)


@dataclass(frozen=True)
class TurnEstimate:
    """A turn's metrics over the runs that reached it."""

    context_tokens: Estimate
    ttft_s: Estimate
    prefill_rate: Estimate
    decode_rate: Estimate
    kv_gib: Estimate


def turn_estimates(runs: list[Run]) -> list[TurnEstimate]:
    turns = max(len(run.turns) for run in runs)
    estimates = []
    for turn in range(turns):
        reached = [run.turns[turn] for run in runs if len(run.turns) > turn]
        estimates.append(
            TurnEstimate(
                **{
                    field: estimate([float(getattr(m, field)) for m in reached])
                    for field in (
                        "context_tokens",
                        "ttft_s",
                        "prefill_rate",
                        "decode_rate",
                        "kv_gib",
                    )
                }
            )
        )
    return estimates


def shown(e: Estimate, digits: int) -> str:
    if e.half_width is None:
        return f"{e.mean:.{digits}f}"
    return f"{e.mean:.{digits}f}±{e.half_width:.{digits}f}"


LINESTYLES = {1: "-", 2: "--", 4: ":", 8: "-."}


@app.command()
def plot(
    results: Path = Path("bench/results/perf/results.jsonl"),
    out: Path = Path("bench/perf.svg"),
    concurrency: Annotated[list[int] | None, typer.Option()] = None,
) -> None:
    """Prints each series' turns, the mean over its runs that reached a turn and
    its 95% Student t interval, and how each run ended, and plots each metric's
    mean and interval against the mean context, one row of panels per
    concurrency (every one in results unless given), a series per setting and
    tensor-parallel size, marking with a cross each run's turn failing with oom
    at the context it tried to reach: its peak KV, and the metrics of the
    sessions whose request finished. The time to first token is the sessions'
    mean on a log scale."""
    pyplot = importlib.import_module("matplotlib.pyplot")
    by_series: dict[SeriesKey, dict[int, Run]] = defaultdict(dict)
    for (name, sessions, size, index), run in load_runs(results).items():
        if concurrency is None or sessions in concurrency:
            by_series[(name, size, sessions)][index] = run
    estimates = {
        key: turn_estimates(list(runs.values())) for key, runs in by_series.items()
    }

    print(
        f"{'setting':<30}{'GPUs':>5}{'sessions':>9}{'runs':>5}{'context':>9}"
        f"{'TTFT s':>13}{'prefill':>14}{'decode':>12}{'KV GiB':>8}"
    )
    for key in sorted(by_series):
        name, size, sessions = key
        for e in estimates[key]:
            print(
                f"{name:<30}{size:>5}{sessions:>9}{e.ttft_s.n:>5}"
                f"{e.context_tokens.mean:>9.0f}{shown(e.ttft_s, 2):>13}"
                f"{shown(e.prefill_rate, 0):>14}{shown(e.decode_rate, 0):>12}"
                f"{e.kv_gib.mean:>8.2f}"
            )
        for index, run in sorted(by_series[key].items()):
            prefix = f"{name:<30}{size:>5}{sessions:>9}  run {index}"
            if run.oom is not None:
                oom = run.oom
                rates = (
                    f"prefill {oom.prefill_rate:.0f} tok/s, "
                    if oom.prefill_rate is not None
                    else ""
                ) + (
                    f"decode {oom.decode_rate:.0f} tok/s, "
                    if oom.decode_rate is not None
                    else ""
                )
                print(
                    f"{prefix}: oom at {oom.context_tokens} tokens: "
                    f"{rates}KV {oom.kv_gib:.2f} GiB, "
                    f"{oom.finished_sessions} of {sessions} sessions finished"
                )
            if run.error is not None:
                print(f"{prefix}: error: {run.error}")

    surface, ink, muted, grid = "#ffffff", "#0b0b0b", "#52514e", "#e4e3df"
    concurrencies = sorted({sessions for _, _, sessions in by_series})
    sizes = sorted({size for _, size, _ in by_series})
    figure, axes_grid = pyplot.subplots(
        len(concurrencies),
        len(PANELS),
        figsize=(4 * len(PANELS), 3.2 * len(concurrencies)),
        dpi=150,
        squeeze=False,
    )
    figure.patch.set_facecolor(surface)
    for axes_row, sessions in zip(axes_grid, concurrencies):
        for axes, (panel, title) in zip(axes_row, PANELS.items()):
            axes.set_facecolor(surface)
            for name, series in SERIES.items():
                for size in sizes:
                    key = (name, size, sessions)
                    if key not in by_series:
                        continue
                    turns = estimates[key]
                    x = [e.context_tokens.mean / 1024 for e in turns]
                    y = [getattr(e, panel).mean for e in turns]
                    gpus = "1 GPU" if size == 1 else f"{size} GPUs"
                    label = series.label if sizes == [1] else f"{series.label}, {gpus}"
                    axes.plot(
                        x,
                        y,
                        color=series.color,
                        linewidth=1.1,
                        linestyle=LINESTYLES.get(size, "-"),
                        marker="o",
                        markersize=2.6,
                        markeredgewidth=0.8,
                        markerfacecolor=series.color if series.filled else surface,
                        label=label,
                    )
                    banded = [
                        (xi, getattr(e, panel))
                        for xi, e in zip(x, turns)
                        if getattr(e, panel).half_width is not None
                    ]
                    if banded:
                        axes.fill_between(
                            [xi for xi, _ in banded],
                            [m.mean - m.half_width for _, m in banded],
                            [m.mean + m.half_width for _, m in banded],
                            color=series.color,
                            alpha=0.15,
                            linewidth=0,
                        )
                    for run in by_series[key].values():
                        if (
                            run.oom is not None
                            and (oom_y := getattr(run.oom, panel)) is not None
                        ):
                            axes.plot(
                                run.oom.context_tokens / 1024,
                                oom_y,
                                color=series.color,
                                marker="x",
                                markersize=6,
                                markeredgewidth=1.4,
                                linestyle="none",
                            )
            noun = "session" if sessions == 1 else "sessions"
            axes.set_title(f"{title}, {sessions} {noun}", color=ink, fontsize=10)
            axes.set_xlim(left=0)
            if panel == "ttft_s":
                axes.set_yscale("log")
            else:
                axes.set_ylim(bottom=0)
            axes.set_xlabel("Context (k tokens)", color=muted)
            axes.tick_params(colors=muted)
            axes.grid(axis="y", color=grid, linewidth=1)
            for side in ("top", "right"):
                axes.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                axes.spines[side].set_color(grid)
    handles, labels = axes_grid[0][0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        labelcolor=ink,
        loc="lower center",
        ncol=min(len(labels), 3),
    )
    figure.text(
        0.99,
        0.01,
        "band: 95% interval over runs; × cache oom",
        color=muted,
        fontsize=8,
        ha="right",
    )
    legend_inches = 0.8
    figure.tight_layout(rect=(0, legend_inches / figure.get_figheight(), 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, facecolor=surface)
    print(f"wrote {out}")


if __name__ == "__main__":
    app()
