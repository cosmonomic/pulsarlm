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
import zipfile
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
import typer
from openai import APIStatusError, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from tokenizers import Tokenizer
from wonderwords import RandomWord

from pulsarlm import checkpoint
from pulsarlm.models import REPOSITORIES

INSTRUCTION = "\n\nContinue the story, using as many of the words above as you can."
METRICS_INTERVAL_S = 0.05


@dataclass(frozen=True)
class Setting:
    engine: Literal["pulsarlm", "vllm"]
    max_context_window: int | None = None


SETTINGS = {
    "full_kv": Setting("pulsarlm"),
    "cap_4k": Setting("pulsarlm", max_context_window=4096),
    "cap_8k": Setting("pulsarlm", max_context_window=8192),
    "cap_16k": Setting("pulsarlm", max_context_window=16384),
    "vllm": Setting("vllm"),
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
def serving(command: list[str], base_url: str, log: Path) -> Iterator[None]:
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
def pulsarlm_server(
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
) -> Iterator[Server]:
    executable = Path(sys.executable).parent / "pulsarlm"
    command = [str(executable), "serve", model, "--data_dir", str(data_dir)]
    command += ["--port", str(port), "--no_persist"]
    command += ["--kv_cache_gib", str(kv_cache_gib), "--store_gib", str(store_gib)]
    command += ["--prefill_tokens_per_step", str(prefill_tokens_per_step)]
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
    with serving(command, base_url, data_dir / "serve.log"):
        yield Server(
            base_url=base_url,
            metrics_url=f"{root}/metrics",
            request_options={"extra_body": {"reasoning_effort": "none"}},
        )


@contextmanager
def vllm_server(
    model: str,
    data_dir: Path,
    port: int,
    *,
    vllm: str,
    kv_cache_bytes: int,
    prefill_tokens_per_step: int,
    max_context: int,
) -> Iterator[Server]:
    command = ["uvx", "--python", "3.12", "--from", vllm, "vllm", "serve"]
    command += [REPOSITORIES[model], "--served-model-name", model]
    command += ["--port", str(port), "--dtype", "bfloat16"]
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
    root = f"http://127.0.0.1:{port}"
    base_url = f"{root}/v1"
    with serving(command, base_url, data_dir / "serve.log"):
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


def is_oom(error: APIStatusError) -> bool:
    """pulsarlm answers 503 when a request does not fit its KV cache; vLLM
    answers 400 when it is longer than the context its KV pool holds."""
    if error.status_code == 503:
        return "does not fit in the KV cache" in error.message
    return error.status_code == 400 and "context length" in error.message


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
    except APIStatusError as error:
        if is_oom(error):
            return {"error": "oom"}
        return {"error": f"{error.status_code}: {error.message}"}
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
    with zipfile.ZipFile(checkpoint.model_path(model)) as archive:
        return checkpoint.load_tokenizer(zipfile.Path(archive))


def done_runs(results: Path) -> set[tuple[str, int]]:
    if not results.exists():
        return set()
    with results.open() as f:
        rows = [json.loads(line) for line in f]
    return {(r["setting"], r["concurrency"]) for r in rows}


@contextmanager
def server_for(
    name: str,
    model: str,
    out: Path,
    port: int,
    *,
    kv_cache_gib: float,
    store_gib: float,
    n_working: int,
    prefill_tokens_per_step: int,
    max_context: int,
    vllm: str,
) -> Iterator[Server]:
    setting = SETTINGS[name]
    if setting.engine == "vllm":
        with vllm_server(
            model,
            out / name,
            port,
            vllm=vllm,
            kv_cache_bytes=int(kv_cache_gib * 2**30),
            prefill_tokens_per_step=prefill_tokens_per_step,
            max_context=max_context,
        ) as server:
            yield server
    else:
        with pulsarlm_server(
            model,
            setting,
            out / name,
            port,
            kv_cache_gib=kv_cache_gib,
            store_gib=store_gib,
            n_working=n_working,
            prefill_tokens_per_step=prefill_tokens_per_step,
            max_context=max_context,
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
    model: str = "gemma-4-12b-it",
    out: Path = Path("bench/results/perf"),
    settings: Annotated[list[str], typer.Option()] = list(SETTINGS),
    concurrency: Annotated[list[int], typer.Option()] = [1, 4],
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
) -> None:
    """Grows concurrency chat sessions side by side in every setting, each turn
    adding a user message of turn_tokens random words and a reply of at most
    max_tokens, until a session would pass max_context or a request fails, and
    appends a row per turn to out/results.jsonl. A request that does not fit the
    KV cache fails with oom. Skips the settings and concurrencies already there.
    Every setting is sent the same user messages. Both engines forward at most
    prefill_tokens_per_step prompt tokens per step, and full KV and vLLM may run
    past the model's trained context up to max_context."""
    unknown = set(settings) - set(SETTINGS)
    if unknown:
        raise typer.BadParameter(f"unknown settings: {sorted(unknown)}")
    tokenizer = load_tokenizer(model)
    vocabulary = words()
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    done = done_runs(results)
    pool_bytes = int(kv_cache_gib * 2**30)

    def message(sessions: int, session: int, turn: int) -> str:
        rng = random.Random(f"{seed}/{sessions}/{session}/{turn}")
        return user_message(turn_tokens, rng, vocabulary, tokenizer)

    for name in settings:
        todo = [n for n in concurrency if (name, n) not in done]
        if not todo:
            continue
        engine = SETTINGS[name].engine
        with server_for(
            name,
            model,
            out,
            port,
            kv_cache_gib=kv_cache_gib,
            store_gib=store_gib,
            n_working=n_working,
            prefill_tokens_per_step=prefill_tokens_per_step,
            max_context=max_context,
            vllm=vllm,
        ) as server:
            warmup = message(0, 0, 0)
            asyncio.run(
                measure_turn(
                    server,
                    engine,
                    model,
                    [[{"role": "user", "content": warmup}]],
                    max_tokens=16,
                    pool_bytes=pool_bytes,
                )
            )
            for n in todo:
                sessions: list[list[ChatCompletionMessageParam]] = [
                    [] for _ in range(n)
                ]
                context = 0
                turn = 0
                while context + turn_tokens + max_tokens <= max_context:
                    for session, messages in enumerate(sessions):
                        messages.append(
                            {"role": "user", "content": message(n, session, turn)}
                        )
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
                        "setting": name,
                        "concurrency": n,
                        "turn": turn,
                        "previous_context_tokens": previous_context,
                        "turn_tokens": turn_tokens,
                        "context_tokens": None if failed else context,
                        "kv_cache_pool_bytes": pool_bytes,
                        "peak_kv_cache_bytes": measured["peak_kv_cache_bytes"],
                        "requests": [
                            {k: v for k, v in r.items() if k != "reply"}
                            for r in requests
                        ],
                    }
                    with results.open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(f"{name} x{n} turn {turn}: {summary(row)}", flush=True)
                    if failed:
                        break
                    for messages, request in zip(sessions, requests):
                        messages.append(
                            {"role": "assistant", "content": request["reply"]}
                        )
                    turn += 1


@dataclass(frozen=True)
class TurnMetrics:
    context_tokens: int
    prefill: float
    decode: float
    kv_gib: float


def turn_metrics(row: dict[str, Any]) -> TurnMetrics | None:
    """Prefill: the prompt tokens past the previous turn's prompt over the time
    to the last session's first token. Decode: each session's tokens after its
    first over the time from its first token to its last, summed over the
    sessions. KV: the most the engine's running requests held while the turn
    ran. None when a request failed."""
    requests = row["requests"]
    if row["context_tokens"] is None:
        return None
    new_tokens = sum(r["prompt_tokens"] for r in requests) - row[
        "previous_context_tokens"
    ] * len(requests)
    prefill = new_tokens / max(r["first_token_s"] for r in requests)
    decode = sum(
        (r["completion_tokens"] - 1) / (r["last_token_s"] - r["first_token_s"])
        for r in requests
        if r["last_token_s"] > r["first_token_s"]
    )
    return TurnMetrics(
        row["context_tokens"], prefill, decode, row["peak_kv_cache_bytes"] / 2**30
    )


def summary(row: dict[str, Any]) -> str:
    metrics = turn_metrics(row)
    if metrics is None:
        errors = {r["error"] for r in row["requests"] if r["error"] is not None}
        return ", ".join(sorted(errors))
    return (
        f"context {metrics.context_tokens}, prefill {metrics.prefill:.0f} tok/s, "
        f"decode {metrics.decode:.0f} tok/s, KV {metrics.kv_gib:.2f} GiB"
    )


@dataclass(frozen=True)
class Series:
    label: str
    color: str
    linestyle: str = "-"


SERIES = {
    "full_kv": Series("Full KV", "#52514e", "--"),
    "cap_4k": Series("Capped @ 4k", "#86b6ef"),
    "cap_8k": Series("Capped @ 8k", "#2a78d6"),
    "cap_16k": Series("Capped @ 16k", "#104281"),
    "vllm": Series("vLLM", "#d95926", ":"),
}

PANELS = {
    "prefill": "Prefill (tok/s)",
    "decode": "Decode (tok/s)",
    "kv_gib": "KV cache in use (GiB)",
}


@dataclass(frozen=True)
class Run:
    turns: list[TurnMetrics]
    # The context a turn failing with oom tried to reach, with its user
    # message.
    oom_at: int | None
    error: str | None


def load_runs(results: Path) -> dict[tuple[str, int], Run]:
    with results.open() as f:
        rows = [json.loads(line) for line in f]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["setting"], row["concurrency"])].append(row)
    runs = {}
    for key, run_rows in grouped.items():
        run_rows.sort(key=lambda row: row["turn"])
        turns = [m for row in run_rows if (m := turn_metrics(row)) is not None]
        last = run_rows[-1]
        errors = {r["error"] for r in last["requests"] if r["error"] is not None}
        oom_at = (
            last["previous_context_tokens"] + last["turn_tokens"]
            if "oom" in errors
            else None
        )
        other = sorted(errors - {"oom"})
        runs[key] = Run(turns, oom_at, ", ".join(other) or None)
    return runs


@app.command()
def plot(
    results: Path = Path("bench/results/perf/results.jsonl"),
    out: Path = Path("bench/perf.svg"),
) -> None:
    """Prints each run's turns and how it ended, and plots each metric against
    the context, one row of panels per concurrency, marking with a cross the
    context at which a run failed with oom."""
    pyplot = importlib.import_module("matplotlib.pyplot")
    runs = load_runs(results)

    print(
        f"{'setting':<10}{'sessions':>10}{'context':>9}{'prefill':>10}"
        f"{'decode':>9}{'KV GiB':>8}"
    )
    for (name, concurrency), run in sorted(runs.items()):
        for m in run.turns:
            print(
                f"{name:<10}{concurrency:>10}{m.context_tokens:>9}"
                f"{m.prefill:>10.0f}{m.decode:>9.0f}{m.kv_gib:>8.2f}"
            )
        if run.oom_at is not None:
            print(f"{name:<10}{concurrency:>10}  oom at {run.oom_at} tokens")
        if run.error is not None:
            print(f"{name:<10}{concurrency:>10}  error: {run.error}")

    surface, ink, muted, grid = "#ffffff", "#0b0b0b", "#52514e", "#e4e3df"
    concurrencies = sorted({c for _, c in runs})
    figure, axes_grid = pyplot.subplots(
        len(concurrencies),
        len(PANELS),
        figsize=(4 * len(PANELS), 3.2 * len(concurrencies)),
        dpi=150,
        squeeze=False,
    )
    figure.patch.set_facecolor(surface)
    for axes_row, concurrency in zip(axes_grid, concurrencies):
        for axes, (panel, title) in zip(axes_row, PANELS.items()):
            axes.set_facecolor(surface)
            for name, series in SERIES.items():
                run = runs.get((name, concurrency))
                if run is None or not run.turns:
                    continue
                xs = [m.context_tokens / 1024 for m in run.turns]
                ys = [getattr(m, panel) for m in run.turns]
                axes.plot(
                    xs,
                    ys,
                    color=series.color,
                    linestyle=series.linestyle,
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    label=series.label,
                )
                if run.oom_at is not None:
                    axes.plot(
                        run.oom_at / 1024,
                        ys[-1],
                        color=series.color,
                        marker="x",
                        markersize=9,
                        markeredgewidth=2,
                        linestyle="none",
                    )
            sessions = "session" if concurrency == 1 else "sessions"
            axes.set_title(f"{title}, {concurrency} {sessions}", color=ink, fontsize=10)
            axes.set_xlim(left=0)
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
        ncol=len(labels),
    )
    figure.text(0.99, 0.01, "× out of KV cache", color=muted, fontsize=8, ha="right")
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, facecolor=surface)
    print(f"wrote {out}")


if __name__ == "__main__":
    app()
