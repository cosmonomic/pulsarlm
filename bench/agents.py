"""Prefill throughput, decode throughput and KV cache bytes in use while a
Terminal-Bench agent drives the server, with and without eviction and the prefix
cache, against vLLM."""

import asyncio
import importlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
import typer
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from pulsarlm.models import MODELS, hugging_face_repository

METRICS_INTERVAL_S = 0.5


@dataclass(frozen=True)
class Setting:
    engine: Literal["pulsarlm", "vllm"]
    max_context_window: int | None = None
    prefix_cache: bool = True


SETTINGS = {
    "full_kv": Setting("pulsarlm"),
    "full_kv_no_prefix_cache": Setting("pulsarlm", prefix_cache=False),
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
    body_options: dict[str, Any]


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
            body_options={"reasoning_effort": "none"},
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
    # Without it vLLM leaves usage.prompt_tokens_details out.
    command += ["--enable-prompt-tokens-details"]
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
            body_options={"chat_template_kwargs": {"enable_thinking": False}},
        )


@contextmanager
def run_server_for(
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
    tensor_parallel_size: int,
) -> Iterator[Server]:
    setting = SETTINGS[name]
    if setting.engine == "vllm":
        with run_vllm_server(
            model,
            setting,
            out / name,
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
            out / name,
            port,
            kv_cache_gib=kv_cache_gib,
            store_gib=store_gib,
            n_working=n_working,
            prefill_tokens_per_step=prefill_tokens_per_step,
            max_context=max_context,
            tensor_parallel_size=tensor_parallel_size,
        ) as server:
            yield server


def gauge(text: str, name: str) -> float:
    for line in text.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            return float(line.rsplit(" ", 1)[1])
    raise ValueError(f"no {name} in the metrics")


def kv_cache_bytes(metrics: str, engine: str, pool_bytes: int) -> float:
    if engine == "vllm":
        return gauge(metrics, "vllm:kv_cache_usage_perc") * pool_bytes
    return gauge(metrics, "pulsarlm:kv_cache_bytes")


async def record_kv_cache_bytes(
    server: Server,
    engine: str,
    pool_bytes: int,
    row: dict[str, Any],
    out: Path,
    done: asyncio.Event,
) -> None:
    """Appends the KV cache bytes in use, with the seconds since start, to out
    until done is set."""
    start = time.monotonic()
    async with httpx.AsyncClient() as client:
        while not done.is_set():
            response = await client.get(server.metrics_url)
            response.raise_for_status()
            sample = {
                **row,
                "t_s": time.monotonic() - start,
                "kv_cache_bytes": kv_cache_bytes(response.text, engine, pool_bytes),
            }
            with out.open("a") as f:
                f.write(json.dumps(sample) + "\n")
            try:
                await asyncio.wait_for(done.wait(), METRICS_INTERVAL_S)
            except TimeoutError:
                pass


def completion_from_stream(
    model: str, content: list[str], finish_reason: str | None, usage: dict[str, Any]
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-agents-{time.monotonic_ns()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "".join(content)},
                "finish_reason": finish_reason or "stop",
            }
        ],
        "usage": usage,
    }


def measuring_proxy(
    server: Server,
    *,
    max_tokens: int,
    temperature: float,
    row: dict[str, Any],
    out: Path,
) -> Starlette:
    """Forwards each chat completion to server as a stream with usage, answers it
    whole, and appends to out its seconds from the job's start to the request,
    its first token and its last, its token counts and its error."""
    start = time.monotonic()
    client = httpx.AsyncClient(base_url=server.base_url, timeout=None)

    async def models(_: Request) -> Response:
        upstream = await client.get("/models")
        return Response(upstream.content, upstream.status_code, upstream.headers)

    async def chat_completions(request: Request) -> Response:
        body = await request.json()
        body |= server.body_options
        body["temperature"] = temperature
        body.setdefault("max_tokens", max_tokens)
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        record: dict[str, Any] = {
            **row,
            "start_s": time.monotonic() - start,
            "first_token_s": None,
            "end_s": None,
            "prompt_tokens": None,
            "cached_tokens": None,
            "completion_tokens": None,
            "error": None,
        }
        content: list[str] = []
        finish_reason = None
        usage: dict[str, Any] = {}
        response: Response
        async with client.stream("POST", "/chat/completions", json=body) as upstream:
            if upstream.status_code != 200:
                error = await upstream.aread()
                record["error"] = upstream.status_code
                response = Response(error, upstream.status_code, upstream.headers)
            else:
                async for line in upstream.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line.removeprefix("data: "))
                    if "error" in chunk:
                        record["error"] = chunk["error"].get("code", "error")
                        break
                    for choice in chunk.get("choices", []):
                        text = choice.get("delta", {}).get("content")
                        if text:
                            if record["first_token_s"] is None:
                                record["first_token_s"] = time.monotonic() - start
                            content.append(text)
                        finish_reason = choice.get("finish_reason") or finish_reason
                    usage = chunk.get("usage") or usage
                response = JSONResponse(
                    completion_from_stream(body["model"], content, finish_reason, usage)
                    if record["error"] is None
                    else {"error": {"code": record["error"]}},
                    status_code=200 if record["error"] is None else 500,
                )
        record["end_s"] = time.monotonic() - start
        record["prompt_tokens"] = usage.get("prompt_tokens")
        record["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get(
            "cached_tokens"
        )
        record["completion_tokens"] = usage.get("completion_tokens")
        with out.open("a") as f:
            f.write(json.dumps(record) + "\n")
        return response

    @asynccontextmanager
    async def closing_client(_: Starlette) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await client.aclose()

    return Starlette(
        routes=[
            Route("/v1/models", models),
            Route("/v1/chat/completions", chat_completions, methods=["POST"]),
        ],
        lifespan=closing_client,
    )


def harbor_command(
    model: str,
    proxy_url: str,
    *,
    job_name: str,
    jobs_dir: Path,
    dataset: str,
    tasks: int,
    concurrency: int,
    max_context: int,
    agent_timeout_multiplier: float,
) -> list[str]:
    model_info = {
        "max_input_tokens": max_context,
        "max_output_tokens": max_context,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    }
    executable = Path(sys.executable).parent / "harbor"
    command = [str(executable), "run", "--agent", "terminus-2", "--model"]
    command += [f"openai/{model}", "--env", "podman", "--dataset", dataset]
    command += ["--n-tasks", str(tasks), "--n-concurrent", str(concurrency)]
    command += ["--jobs-dir", str(jobs_dir), "--job-name", job_name, "--yes"]
    command += ["--agent-timeout-multiplier", str(agent_timeout_multiplier)]
    command += ["--ak", f"api_base={proxy_url}"]
    command += ["--ak", f"model_info={json.dumps(model_info)}"]
    return command


async def run_job(
    server: Server,
    engine: str,
    command: list[str],
    *,
    row: dict[str, Any],
    out: Path,
    proxy_port: int,
    max_tokens: int,
    temperature: float,
    pool_bytes: int,
) -> int:
    """Serves the measuring proxy while harbor runs command; returns harbor's exit
    code."""
    proxy = uvicorn.Server(
        uvicorn.Config(
            measuring_proxy(
                server,
                max_tokens=max_tokens,
                temperature=temperature,
                row=row,
                out=out / "requests.jsonl",
            ),
            host="127.0.0.1",
            port=proxy_port,
            log_level="warning",
        )
    )
    serving = asyncio.create_task(proxy.serve())
    while not proxy.started:
        await asyncio.sleep(0.1)
    done = asyncio.Event()
    sampling = asyncio.create_task(
        record_kv_cache_bytes(server, engine, pool_bytes, row, out / "kv.jsonl", done)
    )
    env = os.environ | {"OPENAI_API_KEY": "unused"}
    with (out / f"{row['setting']}_x{row['concurrency']}.harbor.log").open("w") as log:
        harbor = await asyncio.create_subprocess_exec(
            *command, stdout=log, stderr=subprocess.STDOUT, env=env
        )
        code = await harbor.wait()
    done.set()
    await sampling
    proxy.should_exit = True
    await serving
    return code


app = typer.Typer(no_args_is_help=True)


@app.command()
def run(
    model: str = "gemma-4-12b-it",
    out: Path = Path("bench/results/agents"),
    settings: Annotated[list[str], typer.Option()] = [
        "full_kv",
        "full_kv_no_prefix_cache",
        "cap_16k",
        "cap_16k_no_prefix_cache",
    ],
    concurrency: Annotated[list[int], typer.Option()] = [4, 16],
    tasks: int = 16,
    dataset: str = "terminal-bench@2.0",
    max_context: int = 131072,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    kv_cache_gib: float = 4.0,
    store_gib: float = 8.0,
    n_working: int = 1024,
    prefill_tokens_per_step: int = 256,
    agent_timeout_multiplier: float = 1.0,
    vllm: str = "vllm==0.31.0",
    tensor_parallel_size: int = 1,
    port: int = 18091,
    proxy_port: int = 18092,
) -> None:
    """Runs a Harbor job of terminus-2 on the first tasks tasks of dataset, at
    each concurrency, against one server per setting, through a proxy that
    measures every request and appends it to out/requests.jsonl; the KV cache
    bytes in use go to out/kv.jsonl. The agent's context limit is max_context. A
    request without max_tokens is given max_tokens, and every request is sampled
    at temperature. The model is split across tensor_parallel_size GPUs,
    kv_cache_gib sizing each GPU's KV cache."""
    unknown = set(settings) - set(SETTINGS)
    if unknown:
        raise typer.BadParameter(f"unknown settings: {sorted(unknown)}")
    out.mkdir(parents=True, exist_ok=True)
    proxy_url = f"http://127.0.0.1:{proxy_port}/v1"
    for name in settings:
        engine = SETTINGS[name].engine
        with run_server_for(
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
            tensor_parallel_size=tensor_parallel_size,
        ) as server:
            for n in concurrency:
                command = harbor_command(
                    model,
                    proxy_url,
                    job_name=f"{name}_x{n}",
                    jobs_dir=out / "jobs",
                    dataset=dataset,
                    tasks=tasks,
                    concurrency=n,
                    max_context=max_context,
                    agent_timeout_multiplier=agent_timeout_multiplier,
                )
                code = asyncio.run(
                    run_job(
                        server,
                        engine,
                        command,
                        row={
                            "setting": name,
                            "concurrency": n,
                            "tensor_parallel_size": tensor_parallel_size,
                        },
                        out=out,
                        proxy_port=proxy_port,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        pool_bytes=int(kv_cache_gib * 2**30),
                    )
                )
                print(f"{name} x{n}: harbor exited with {code}", flush=True)


@dataclass(frozen=True)
class Series:
    label: str
    color: str
    linestyle: str = "-"
    filled: bool = True


SERIES = {
    "full_kv": Series("Full KV", "#52514e", "--"),
    "full_kv_no_prefix_cache": Series(
        "Full KV, no prefix cache", "#52514e", "--", filled=False
    ),
    "cap_16k": Series("Capped @ 16k", "#104281"),
    "cap_16k_no_prefix_cache": Series(
        "Capped @ 16k, no prefix cache", "#104281", filled=False
    ),
    "vllm": Series("vLLM", "#d95926", ":"),
    "vllm_no_prefix_cache": Series(
        "vLLM, no prefix cache", "#d95926", ":", filled=False
    ),
}


def load_rows(path: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    if path.exists():
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                grouped[(row["setting"], row["concurrency"])].append(row)
    return grouped


def answered(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r
        for r in rows
        if r["error"] is None
        and r["first_token_s"] is not None
        and r["prompt_tokens"] is not None
        and r["completion_tokens"] is not None
    ]


@app.command()
def plot(
    results: Path = Path("bench/results/agents"),
    out: Path = Path("bench/agents.svg"),
) -> None:
    """Prints each job's totals and plots, one row of panels per concurrency,
    each request's prefill throughput (its prompt tokens over its time to first
    token) and decode throughput against its prompt tokens, and the KV cache
    bytes in use over the job."""
    pyplot = importlib.import_module("matplotlib.pyplot")
    requests = load_rows(results / "requests.jsonl")
    kv = load_rows(results / "kv.jsonl")

    print(
        f"{'setting':<30}{'sessions':>10}{'requests':>10}{'errors':>8}"
        f"{'prompt tok/s':>14}{'output tok/s':>14}{'cached %':>10}"
    )
    for (name, concurrency), rows in sorted(requests.items()):
        ok = answered(rows)
        wall = max(r["end_s"] for r in rows) - min(r["start_s"] for r in rows)
        prompt = sum(r["prompt_tokens"] for r in ok) / wall
        output = sum(r["completion_tokens"] for r in ok) / wall
        reported = [r for r in ok if r.get("cached_tokens") is not None]
        reported_prompt = sum(r["prompt_tokens"] for r in reported)
        cached = (
            f"{100 * sum(r['cached_tokens'] for r in reported) / reported_prompt:>10.1f}"
            if reported_prompt
            else f"{'-':>10}"
        )
        print(
            f"{name:<30}{concurrency:>10}{len(rows):>10}{len(rows) - len(ok):>8}"
            f"{prompt:>14.0f}{output:>14.0f}{cached}"
        )

    surface, ink, muted, grid = "#ffffff", "#0b0b0b", "#52514e", "#e4e3df"
    concurrencies = sorted({c for _, c in requests})
    panels = ["Prefill (tok/s)", "Decode (tok/s)", "KV cache in use (GiB)"]
    figure, axes_grid = pyplot.subplots(
        len(concurrencies),
        len(panels),
        figsize=(4 * len(panels), 3.2 * len(concurrencies)),
        dpi=150,
        squeeze=False,
    )
    figure.patch.set_facecolor(surface)
    for axes_row, concurrency in zip(axes_grid, concurrencies):
        for name, series in SERIES.items():
            ok = answered(requests.get((name, concurrency), []))
            face = series.color if series.filled else surface
            prompts = [r["prompt_tokens"] / 1024 for r in ok]
            prefill = [
                r["prompt_tokens"] / (r["first_token_s"] - r["start_s"]) for r in ok
            ]
            decode_rows = [
                r
                for r in ok
                if r["completion_tokens"] > 1 and r["end_s"] > r["first_token_s"]
            ]
            decode = [
                (r["completion_tokens"] - 1) / (r["end_s"] - r["first_token_s"])
                for r in decode_rows
            ]
            marker = {
                "color": series.color,
                "markerfacecolor": face,
                "linestyle": "none",
            }
            axes_row[0].plot(
                prompts, prefill, "o", markersize=3, label=series.label, **marker
            )
            axes_row[1].plot(
                [r["prompt_tokens"] / 1024 for r in decode_rows],
                decode,
                "o",
                markersize=3,
                **marker,
            )
            samples = kv.get((name, concurrency), [])
            axes_row[2].plot(
                [s["t_s"] / 60 for s in samples],
                [s["kv_cache_bytes"] / 2**30 for s in samples],
                color=series.color,
                linestyle=series.linestyle,
                linewidth=1.5,
            )
        sessions = "session" if concurrency == 1 else "sessions"
        for axes, title, xlabel in zip(
            axes_row,
            panels,
            ["Prompt (k tokens)", "Prompt (k tokens)", "Time (min)"],
        ):
            axes.set_facecolor(surface)
            axes.set_title(f"{title}, {concurrency} {sessions}", color=ink, fontsize=10)
            axes.set_xlim(left=0)
            axes.set_ylim(bottom=0)
            axes.set_xlabel(xlabel, color=muted)
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
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, facecolor=surface)
    print(f"wrote {out}")


if __name__ == "__main__":
    app()
