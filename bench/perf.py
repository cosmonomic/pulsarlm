"""Prefill throughput, decode throughput and peak KV cache bytes in use, with and
without eviction, against vLLM."""

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
from tokenizers import Tokenizer
from wonderwords import RandomWord

from pulsarlm import checkpoint
from pulsarlm.models import REPOSITORIES

INSTRUCTION = "\n\nWrite a long story that uses as many of the words above as you can."
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


@dataclass(frozen=True)
class Measurement:
    length: int
    concurrency: int
    repeat: int


def words() -> list[str]:
    vocabulary = RandomWord().filter(include_parts_of_speech=["nouns", "adjectives"])
    return sorted({w.lower() for w in vocabulary if w.isalpha()})


def prompt(
    length: int, rng: random.Random, vocabulary: list[str], tokenizer: Tokenizer
) -> str:
    """Random words of length tokens, then the instruction."""
    text = " ".join(rng.choices(vocabulary, k=length))
    ids = tokenizer.encode(text, add_special_tokens=False).ids[:length]
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
    n_working: int,
) -> Iterator[Server]:
    executable = Path(sys.executable).parent / "pulsarlm"
    command = [str(executable), "serve", model, "--data_dir", str(data_dir)]
    command += ["--port", str(port), "--no_persist"]
    command += ["--kv_cache_gib", str(kv_cache_gib)]
    if setting.max_context_window is None:
        command += ["--eviction_target", "1"]
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
    max_model_len: int,
) -> Iterator[Server]:
    command = ["uvx", "--python", "3.12", "--from", vllm, "vllm", "serve"]
    command += [REPOSITORIES[model], "--served-model-name", model]
    command += ["--port", str(port), "--dtype", "bfloat16"]
    command += ["--kv-cache-memory-bytes", str(kv_cache_bytes)]
    command += ["--max-model-len", str(max_model_len)]
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


async def stream_request(
    client: AsyncOpenAI,
    server: Server,
    model: str,
    text: str,
    max_tokens: int,
    start: float,
) -> dict[str, Any]:
    """Seconds from start to the first and last chunk carrying text, and the
    token counts."""
    first_token_s: float | None = None
    last_token_s: float | None = None
    usage: dict[str, int] | None = None
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": text}],
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
            if (
                delta.content
                or extra.get("reasoning")
                or extra.get("reasoning_content")
            ):
                last_token_s = time.perf_counter() - start
                if first_token_s is None:
                    first_token_s = last_token_s
    except APIStatusError as error:
        return {"error": f"{error.status_code}: {error.message}"}
    if usage is None or first_token_s is None:
        return {"error": "the stream ended without text or usage"}
    return {
        **usage,
        "first_token_s": first_token_s,
        "last_token_s": last_token_s,
        "error": None,
    }


async def measure(
    server: Server,
    engine: str,
    model: str,
    prompts: list[str],
    *,
    max_tokens: int,
    pool_bytes: int,
) -> dict[str, Any]:
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
                stream_request(client, server, model, text, max_tokens, start)
                for text in prompts
            )
        )
        done.set()
        return {"requests": requests, "peak_kv_cache_bytes": await peak}


def load_tokenizer(model: str) -> Tokenizer:
    with zipfile.ZipFile(checkpoint.model_path(model)) as archive:
        return checkpoint.load_tokenizer(zipfile.Path(archive))


def done_keys(results: Path) -> set[tuple[str, int, int, int]]:
    if not results.exists():
        return set()
    with results.open() as f:
        rows = [json.loads(line) for line in f]
    return {(r["setting"], r["length"], r["concurrency"], r["repeat"]) for r in rows}


@contextmanager
def server_for(
    name: str,
    model: str,
    out: Path,
    port: int,
    *,
    kv_cache_gib: float,
    n_working: int,
    vllm: str,
    max_model_len: int,
) -> Iterator[Server]:
    setting = SETTINGS[name]
    if setting.engine == "vllm":
        with vllm_server(
            model,
            out / name,
            port,
            vllm=vllm,
            kv_cache_bytes=int(kv_cache_gib * 2**30),
            max_model_len=max_model_len,
        ) as server:
            yield server
    else:
        with pulsarlm_server(
            model,
            setting,
            out / name,
            port,
            kv_cache_gib=kv_cache_gib,
            n_working=n_working,
        ) as server:
            yield server


app = typer.Typer(no_args_is_help=True)


@app.command()
def run(
    model: str = "gemma-4-12b-it",
    out: Path = Path("bench/results/perf"),
    settings: Annotated[list[str], typer.Option()] = list(SETTINGS),
    lengths: Annotated[list[int], typer.Option()] = [8192, 16384, 32768],
    concurrency: Annotated[list[int], typer.Option()] = [1, 4],
    max_tokens: int = 512,
    repeats: int = 3,
    seed: int = 0,
    kv_cache_gib: float = 4.0,
    n_working: int = 1024,
    vllm: str = "vllm==0.31.0",
    port: int = 18091,
) -> None:
    """Measures every length and concurrency in every setting, repeats times,
    and appends the results to out/results.jsonl, skipping measurements already
    there. Each setting gets its own server and every setting is sent the same
    prompts; no two requests share a prefix past the chat template's."""
    unknown = set(settings) - set(SETTINGS)
    if unknown:
        raise typer.BadParameter(f"unknown settings: {sorted(unknown)}")
    tokenizer = load_tokenizer(model)
    vocabulary = words()
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    done = done_keys(results)
    measurements = [
        Measurement(length, n, repeat)
        for length in lengths
        for n in concurrency
        for repeat in range(repeats)
    ]
    pool_bytes = int(kv_cache_gib * 2**30)

    def prompts(m: Measurement) -> list[str]:
        return [
            prompt(
                m.length,
                random.Random(f"{seed}/{m.length}/{m.concurrency}/{m.repeat}/{i}"),
                vocabulary,
                tokenizer,
            )
            for i in range(m.concurrency)
        ]

    for name in settings:
        todo = [
            m
            for m in measurements
            if (name, m.length, m.concurrency, m.repeat) not in done
        ]
        if not todo:
            continue
        engine = SETTINGS[name].engine
        with server_for(
            name,
            model,
            out,
            port,
            kv_cache_gib=kv_cache_gib,
            n_working=n_working,
            vllm=vllm,
            max_model_len=max(lengths) + max_tokens + 1024,
        ) as server:
            warmup = prompt(
                1024, random.Random(f"{seed}/warmup"), vocabulary, tokenizer
            )
            asyncio.run(
                measure(
                    server,
                    engine,
                    model,
                    [warmup],
                    max_tokens=16,
                    pool_bytes=pool_bytes,
                )
            )
            for m in todo:
                measured = asyncio.run(
                    measure(
                        server,
                        engine,
                        model,
                        prompts(m),
                        max_tokens=max_tokens,
                        pool_bytes=pool_bytes,
                    )
                )
                row = {
                    "setting": name,
                    "length": m.length,
                    "concurrency": m.concurrency,
                    "repeat": m.repeat,
                    "kv_cache_pool_bytes": pool_bytes,
                    **measured,
                }
                with results.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                metrics = row_metrics(row)
                print(
                    f"{name} {m.length} x{m.concurrency} #{m.repeat}: "
                    + (
                        "error"
                        if metrics is None
                        else f"prefill {metrics.prefill:.0f} tok/s, "
                        f"decode {metrics.decode:.0f} tok/s, "
                        f"peak KV {metrics.peak_kv_gib:.2f} GiB"
                    ),
                    flush=True,
                )


@dataclass(frozen=True)
class RowMetrics:
    prefill: float
    decode: float
    peak_kv_gib: float


def row_metrics(row: dict[str, Any]) -> RowMetrics | None:
    """Prefill: the prompt tokens over the time to the last request's first
    token. Decode: each request's tokens after its first over the time from its
    first token to its last, summed over the requests. None when a request
    failed."""
    requests = row["requests"]
    if any(r["error"] is not None for r in requests):
        return None
    prefill = sum(r["prompt_tokens"] for r in requests) / max(
        r["first_token_s"] for r in requests
    )
    decode = sum(
        (r["completion_tokens"] - 1) / (r["last_token_s"] - r["first_token_s"])
        for r in requests
        if r["last_token_s"] > r["first_token_s"]
    )
    return RowMetrics(prefill, decode, row["peak_kv_cache_bytes"] / 2**30)


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
    "peak_kv_gib": "Peak KV cache (GiB)",
}


@app.command()
def plot(
    results: Path = Path("bench/results/perf/results.jsonl"),
    out: Path = Path("bench/perf.svg"),
) -> None:
    """Prints the median of each metric over the repeats per setting, length
    and concurrency, and plots each against prompt length, one row of panels
    per concurrency."""
    pyplot = importlib.import_module("matplotlib.pyplot")
    with results.open() as f:
        rows = [json.loads(line) for line in f]
    groups: dict[tuple[str, int, int], list[RowMetrics]] = defaultdict(list)
    errors: dict[tuple[str, int, int], int] = defaultdict(int)
    for row in rows:
        key = (row["setting"], row["concurrency"], row["length"])
        if (metrics := row_metrics(row)) is None:
            errors[key] += 1
        else:
            groups[key].append(metrics)

    def median(key: tuple[str, int, int], panel: str) -> float:
        return statistics.median(getattr(m, panel) for m in groups[key])

    print(
        f"{'setting':<10}{'requests':>10}{'length':>8}{'prefill':>10}"
        f"{'decode':>9}{'peak KV GiB':>13}{'errors':>8}"
    )
    for key in sorted(groups.keys() | errors.keys()):
        name, concurrency, length = key
        values = (
            f"{median(key, 'prefill'):>10.0f}{median(key, 'decode'):>9.0f}"
            f"{median(key, 'peak_kv_gib'):>13.2f}"
            if groups[key]
            else f"{'-':>10}{'-':>9}{'-':>13}"
        )
        print(f"{name:<10}{concurrency:>10}{length:>8}{values}{errors[key]:>8}")

    surface, ink, muted, grid = "#ffffff", "#0b0b0b", "#52514e", "#e4e3df"
    concurrencies = sorted({c for _, c, _ in groups})
    lengths = sorted({n for _, _, n in groups})
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
                points = [
                    (length, median(key, panel))
                    for length in lengths
                    if groups[key := (name, concurrency, length)]
                ]
                if not points:
                    continue
                xs, ys = zip(*points)
                axes.plot(
                    xs,
                    ys,
                    color=series.color,
                    linestyle=series.linestyle,
                    linewidth=2,
                    marker="o",
                    markersize=8,
                    markeredgecolor=surface,
                    markeredgewidth=2,
                    clip_on=False,
                    label=series.label,
                )
            requests = "request" if concurrency == 1 else "requests"
            axes.set_title(f"{title}, {concurrency} {requests}", color=ink, fontsize=10)
            axes.set_xscale("log", base=2)
            axes.set_xticks(lengths, [f"{length // 1024}k" for length in lengths])
            axes.set_ylim(bottom=0)
            axes.set_xlabel("Prompt length (tokens)", color=muted)
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
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, facecolor=surface)
    print(f"wrote {out}")


if __name__ == "__main__":
    app()
