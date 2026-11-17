"""SWE-bench Verified with mini-swe-agent against one server per setting: the
tasks the agent solves, how it gets there, each request's time to first token,
throughput and cached tokens from the server's OpenTelemetry spans, and the KV
cache bytes in use. The tasks are the 100 that "The Hidden Costs of Forgetting"
(arXiv:2610.35496) takes from AgentDiet."""

import asyncio
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
import typer
import uvicorn
from scipy.stats import binomtest
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
    ExportMetricsServiceRequest,
    ExportMetricsServiceResponse,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from pulsarlm.models import MODELS, hugging_face_repository

# How often the servers export their metrics and vLLM its spans, and vLLM's KV
# cache bytes in use are sampled. vLLM stopped by SIGTERM exits by the signal and
# drops the spans it has not exported, so its runs wait two intervals after the
# agent ends.
EXPORT_INTERVAL_S = 0.5
PROTOBUF = "application/x-protobuf"
DATASET = "princeton-nlp/SWE-bench_Verified"
BOOTSTRAP_RESAMPLES = 10_000
SAMPLE = re.compile(r"([^\s{#]+)(?:\{(.*)\})?\s+(\S+)")
LABEL = re.compile(r'(\w+)="([^"]*)"')

SWE_BENCH_VERIFIED_100 = (
    "astropy__astropy-13398",
    "astropy__astropy-13579",
    "astropy__astropy-14182",
    "astropy__astropy-14369",
    "astropy__astropy-14598",
    "astropy__astropy-7606",
    "django__django-11099",
    "django__django-11133",
    "django__django-11451",
    "django__django-11477",
    "django__django-11490",
    "django__django-11551",
    "django__django-11734",
    "django__django-12039",
    "django__django-12663",
    "django__django-12708",
    "django__django-12858",
    "django__django-12965",
    "django__django-13012",
    "django__django-13112",
    "django__django-13128",
    "django__django-13195",
    "django__django-13344",
    "django__django-13417",
    "django__django-13512",
    "django__django-13568",
    "django__django-13590",
    "django__django-13658",
    "django__django-13741",
    "django__django-14007",
    "django__django-14053",
    "django__django-14376",
    "django__django-14434",
    "django__django-14559",
    "django__django-14580",
    "django__django-14631",
    "django__django-14725",
    "django__django-14765",
    "django__django-14771",
    "django__django-14855",
    "django__django-15098",
    "django__django-15127",
    "django__django-15280",
    "django__django-15561",
    "django__django-15629",
    "django__django-15930",
    "django__django-15987",
    "django__django-16136",
    "django__django-16139",
    "django__django-16255",
    "django__django-16256",
    "django__django-16315",
    "django__django-16454",
    "django__django-16493",
    "django__django-16642",
    "django__django-16661",
    "django__django-16667",
    "django__django-16901",
    "django__django-17087",
    "matplotlib__matplotlib-14623",
    "matplotlib__matplotlib-22871",
    "matplotlib__matplotlib-23412",
    "matplotlib__matplotlib-24149",
    "matplotlib__matplotlib-24177",
    "matplotlib__matplotlib-24570",
    "matplotlib__matplotlib-24627",
    "matplotlib__matplotlib-25311",
    "matplotlib__matplotlib-25332",
    "mwaskom__seaborn-3069",
    "pallets__flask-5014",
    "psf__requests-1766",
    "pydata__xarray-2905",
    "pydata__xarray-4075",
    "pydata__xarray-4629",
    "pydata__xarray-6938",
    "pylint-dev__pylint-4551",
    "pylint-dev__pylint-6903",
    "pylint-dev__pylint-7080",
    "pytest-dev__pytest-6197",
    "pytest-dev__pytest-7324",
    "scikit-learn__scikit-learn-10844",
    "scikit-learn__scikit-learn-13124",
    "scikit-learn__scikit-learn-13779",
    "scikit-learn__scikit-learn-14629",
    "scikit-learn__scikit-learn-14710",
    "scikit-learn__scikit-learn-25102",
    "scikit-learn__scikit-learn-26194",
    "scikit-learn__scikit-learn-9288",
    "sphinx-doc__sphinx-11510",
    "sphinx-doc__sphinx-7440",
    "sphinx-doc__sphinx-7910",
    "sphinx-doc__sphinx-8593",
    "sphinx-doc__sphinx-9281",
    "sphinx-doc__sphinx-9591",
    "sphinx-doc__sphinx-9698",
    "sympy__sympy-13372",
    "sympy__sympy-15349",
    "sympy__sympy-15809",
    "sympy__sympy-17655",
    "sympy__sympy-21596",
)


@dataclass(frozen=True)
class Setting:
    engine: Literal["pulsarlm", "vllm"]
    max_context_window: int | None = None
    eviction_threshold: float = 0.0
    prefix_cache: bool = True
    mass_tokens: int | None = None
    # None takes the run's n_working.
    n_working: int | None = None


SETTINGS = {
    "vllm": Setting("vllm"),
    "full_kv": Setting("pulsarlm"),
    "cap_4k_mass_0.5": Setting(
        "pulsarlm", max_context_window=4096, eviction_threshold=0.5
    ),
    "cap_16k_mass_0.5": Setting(
        "pulsarlm", max_context_window=16384, eviction_threshold=0.5
    ),
    "cap_64k_mass_0.5": Setting(
        "pulsarlm", max_context_window=65536, eviction_threshold=0.5
    ),
    "cap_16k": Setting("pulsarlm", max_context_window=16384),
    "mass_0.5": Setting("pulsarlm", eviction_threshold=0.5),
    "cap_16k_mass_0.25": Setting(
        "pulsarlm", max_context_window=16384, eviction_threshold=0.25
    ),
    "cap_16k_mass_0.75": Setting(
        "pulsarlm", max_context_window=16384, eviction_threshold=0.75
    ),
    "cap_16k_mass_0.5_working_8k": Setting(
        "pulsarlm", max_context_window=16384, eviction_threshold=0.5, n_working=8192
    ),
    "cap_16k_mass_0.5_mass_tokens_128": Setting(
        "pulsarlm", max_context_window=16384, eviction_threshold=0.5, mass_tokens=128
    ),
    "cap_16k_mass_0.5_no_prefix_cache": Setting(
        "pulsarlm", max_context_window=16384, eviction_threshold=0.5, prefix_cache=False
    ),
}


@dataclass(frozen=True)
class Server:
    base_url: str
    metrics_url: str


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
def run_server(
    command: list[str], base_url: str, log: Path, env: dict[str, str]
) -> Iterator[None]:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as output:
        process = subprocess.Popen(
            command, stdout=output, stderr=subprocess.STDOUT, env=os.environ | env
        )
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
    otlp_url: str,
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
    command += ["--eviction_threshold", str(setting.eviction_threshold)]
    working = n_working if setting.n_working is None else setting.n_working
    command += ["--n_working", str(working)]
    if not setting.prefix_cache:
        command += ["--no_prefix_cache"]
    if setting.max_context_window is None:
        # The view would otherwise be capped at the model's trained context.
        command += ["--max_context_window", str(max_context)]
    else:
        command += ["--max_context_window", str(setting.max_context_window)]
    if setting.mass_tokens is not None:
        command += ["--mass_tokens", str(setting.mass_tokens)]
    root = f"http://127.0.0.1:{port}"
    env = {
        "OTEL_EXPORTER_OTLP_ENDPOINT": otlp_url,
        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
        "OTEL_METRIC_EXPORT_INTERVAL": str(int(EXPORT_INTERVAL_S * 1000)),
    }
    with run_server(command, f"{root}/api/openai/v1", data_dir / "serve.log", env):
        yield Server(base_url=f"{root}/api/openai/v1", metrics_url=f"{root}/metrics")


@contextmanager
def run_vllm_server(
    model: str,
    setting: Setting,
    data_dir: Path,
    port: int,
    otlp_url: str,
    *,
    vllm: str,
    kv_cache_bytes: int,
    prefill_tokens_per_step: int,
    max_context: int,
    tensor_parallel_size: int,
) -> Iterator[Server]:
    command = ["uvx", "--python", "3.12", "--from", vllm]
    command += ["--with", "opentelemetry-sdk", "--with", "opentelemetry-exporter-otlp"]
    command += ["vllm", "serve"]
    command += [hugging_face_repository(MODELS[model].url).repository]
    command += ["--served-model-name", model]
    command += ["--port", str(port), "--dtype", "bfloat16"]
    command += ["--tensor-parallel-size", str(tensor_parallel_size)]
    command += ["--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder"]
    command += ["--reasoning-parser", "qwen3"]
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
    if setting.prefix_cache:
        command += ["--mamba-cache-mode", "align"]
    else:
        command += ["--no-enable-prefix-caching"]
    command += ["--otlp-traces-endpoint", f"{otlp_url}/v1/traces"]
    env = {
        "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "http/protobuf",
        "OTEL_BSP_SCHEDULE_DELAY": str(int(EXPORT_INTERVAL_S * 1000)),
    }
    root = f"http://127.0.0.1:{port}"
    with run_server(command, f"{root}/v1", data_dir / "serve.log", env):
        yield Server(base_url=f"{root}/v1", metrics_url=f"{root}/metrics")


def gauge(text: str, name: str) -> float:
    for line in text.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            return float(line.rsplit(" ", 1)[1])
    raise ValueError(f"no {name} in the metrics")


async def record_vllm_kv_cache_bytes(
    server: Server, pool_bytes: int, out: Path, done: asyncio.Event
) -> None:
    """Appends vLLM's KV cache bytes in use, with the Unix time in nanoseconds,
    to out until done is set."""
    async with httpx.AsyncClient() as client:
        while not done.is_set():
            response = await client.get(server.metrics_url)
            response.raise_for_status()
            usage = gauge(response.text, "vllm:kv_cache_usage_perc")
            sample = {"t_ns": time.time_ns(), "kv_cache_bytes": usage * pool_bytes}
            with out.open("a") as f:
                f.write(json.dumps(sample) + "\n")
            try:
                await asyncio.wait_for(done.wait(), EXPORT_INTERVAL_S)
            except TimeoutError:
                pass


def attribute_value(value: AnyValue) -> Any:
    kind = value.WhichOneof("value")
    if kind is None:
        return None
    if kind == "array_value":
        return [attribute_value(item) for item in value.array_value.values]
    return getattr(value, kind)


def attributes_of(pairs: Iterable[KeyValue]) -> dict[str, Any]:
    return {pair.key: attribute_value(pair.value) for pair in pairs}


def otlp_receiver(out: Path) -> Starlette:
    """Takes OTLP/HTTP protobuf exports: appends each span, with its service, to
    out/spans.jsonl, and each point of pulsarlm.device_memory to out/kv.jsonl."""

    async def traces(request: Request) -> Response:
        export = ExportTraceServiceRequest.FromString(await request.body())
        with (out / "spans.jsonl").open("a") as f:
            for resource_spans in export.resource_spans:
                service = attributes_of(resource_spans.resource.attributes).get(
                    "service.name"
                )
                for scope_spans in resource_spans.scope_spans:
                    for span in scope_spans.spans:
                        record = {
                            "service": service,
                            "name": span.name,
                            "start_ns": span.start_time_unix_nano,
                            "end_ns": span.end_time_unix_nano,
                            "attributes": attributes_of(span.attributes),
                        }
                        f.write(json.dumps(record) + "\n")
        return Response(
            ExportTraceServiceResponse().SerializeToString(), media_type=PROTOBUF
        )

    async def metrics(request: Request) -> Response:
        export = ExportMetricsServiceRequest.FromString(await request.body())
        with (out / "kv.jsonl").open("a") as f:
            for resource_metrics in export.resource_metrics:
                for scope_metrics in resource_metrics.scope_metrics:
                    for metric in scope_metrics.metrics:
                        if metric.name != "pulsarlm.device_memory":
                            continue
                        for point in metric.gauge.data_points:
                            sample = {
                                "t_ns": point.time_unix_nano,
                                "kv_cache_bytes": point.as_int,
                            }
                            f.write(json.dumps(sample) + "\n")
        return Response(
            ExportMetricsServiceResponse().SerializeToString(), media_type=PROTOBUF
        )

    return Starlette(
        routes=[
            Route("/v1/traces", traces, methods=["POST"]),
            Route("/v1/metrics", metrics, methods=["POST"]),
        ]
    )


@contextmanager
def serve_otlp(out: Path, port: int) -> Iterator[str]:
    """Runs otlp_receiver(out) on a thread; yields its URL."""
    server = uvicorn.Server(
        uvicorn.Config(
            otlp_receiver(out), host="127.0.0.1", port=port, log_level="warning"
        )
    )
    thread = threading.Thread(target=server.run)
    thread.start()
    try:
        while not server.started:
            if not thread.is_alive():
                raise RuntimeError(f"the OTLP receiver did not start on port {port}")
            time.sleep(0.1)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join()


def mini_command(
    model: str,
    api_base: str,
    out: Path,
    *,
    tasks: tuple[str, ...],
    workers: int,
    max_steps: int,
    max_tokens: int,
    temperature: float,
) -> list[str]:
    executable = Path(sys.executable).parent / "mini-extra"
    command = [str(executable), "swebench", "--subset", "verified", "--split", "test"]
    command += ["--filter", "^(" + "|".join(tasks) + ")$"]
    command += ["--workers", str(workers), "--output", str(out)]
    command += ["--model", f"openai/{model}", "--config", "swebench.yaml"]
    for key, value in (
        ("model.model_kwargs.api_base", api_base),
        ("model.model_kwargs.api_key", "unused"),
        ("model.model_kwargs.temperature", temperature),
        ("model.model_kwargs.max_tokens", max_tokens),
        ("agent.step_limit", max_steps),
    ):
        command += ["--config", f"{key}={value}"]
    return command


async def run_agents(
    command: list[str], *, out: Path, container_executable: str
) -> int:
    """Runs mini-swe-agent's command; returns its exit code."""
    env = os.environ | {
        "MSWEA_COST_TRACKING": "ignore_errors",
        "MSWEA_DOCKER_EXECUTABLE": container_executable,
    }
    with (out / "mini.log").open("a") as log:
        agents = await asyncio.create_subprocess_exec(
            *command, stdout=log, stderr=subprocess.STDOUT, env=env
        )
        return await agents.wait()


async def run_agents_on_vllm(
    server: Server,
    command: list[str],
    *,
    out: Path,
    pool_bytes: int,
    container_executable: str,
) -> int:
    """Samples vLLM's KV cache bytes in use while mini-swe-agent runs command,
    then saves vLLM's metrics to out/metrics.txt; returns its exit code."""
    done = asyncio.Event()
    sampling = asyncio.create_task(
        record_vllm_kv_cache_bytes(server, pool_bytes, out / "kv.jsonl", done)
    )
    code = await run_agents(command, out=out, container_executable=container_executable)
    done.set()
    await sampling
    async with httpx.AsyncClient() as client:
        response = await client.get(server.metrics_url)
        response.raise_for_status()
    (out / "metrics.txt").write_text(response.text)
    return code


def grade(directory: Path, workers: int) -> None:
    """Runs the SWE-bench harness on directory's predictions; its report goes
    to directory/grading."""
    grading = directory / "grading"
    grading.mkdir(exist_ok=True)
    with (grading / "harness.log").open("a") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "swebench.harness.run_evaluation",
                "--dataset_name",
                DATASET,
                "--split",
                "test",
                "--predictions_path",
                str((directory / "agents" / "preds.json").resolve()),
                "--max_workers",
                str(workers),
                "--run_id",
                directory.name,
            ],
            cwd=grading,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


app = typer.Typer(no_args_is_help=True)


@app.command()
def run(
    model: str = "qwen3.8-flash-next",
    out: Path = Path("bench/results/agents"),
    settings: Annotated[list[str], typer.Option()] = list(SETTINGS),
    tasks: Annotated[list[str], typer.Option()] = list(SWE_BENCH_VERIFIED_100),
    repeats: int = 1,
    workers: int = 4,
    max_steps: int = 200,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    max_context: int = 262144,
    kv_cache_gib: float = 18.0,
    store_gib: float = 8.0,
    n_working: int = 1024,
    prefill_tokens_per_step: int = 512,
    vllm: str = "vllm==0.31.0",
    tensor_parallel_size: int = 4,
    container_executable: str = "docker",
    port: int = 18091,
    otlp_port: int = 18092,
) -> None:
    """Runs mini-swe-agent on tasks, the 100 by default, repeats times per setting, against
    one server per setting and repeat, then grades its patches. A rerun resumes:
    the agent skips the tasks it has a patch for. out/<setting>_r<k> holds the
    agent's trajectories and patches (agents/), the server's spans (spans.jsonl),
    the KV cache bytes in use (kv.jsonl), vLLM's metrics at the end (metrics.txt),
    the server's log and the grading."""
    unknown = set(settings) - set(SETTINGS)
    if unknown:
        raise typer.BadParameter(f"unknown settings: {sorted(unknown)}")
    outside = set(tasks) - set(SWE_BENCH_VERIFIED_100)
    if outside:
        raise typer.BadParameter(f"tasks outside the 100: {sorted(outside)}")
    pool_bytes = int(kv_cache_gib * 2**30)
    for name in settings:
        setting = SETTINGS[name]
        for repeat in range(repeats):
            directory = out / f"{name}_r{repeat}"
            directory.mkdir(parents=True, exist_ok=True)
            with serve_otlp(directory, otlp_port) as otlp_url:
                if setting.engine == "vllm":
                    server = run_vllm_server(
                        model,
                        setting,
                        directory,
                        port,
                        otlp_url,
                        vllm=vllm,
                        kv_cache_bytes=pool_bytes,
                        prefill_tokens_per_step=prefill_tokens_per_step,
                        max_context=max_context,
                        tensor_parallel_size=tensor_parallel_size,
                    )
                else:
                    server = run_pulsarlm_server(
                        model,
                        setting,
                        directory,
                        port,
                        otlp_url,
                        kv_cache_gib=kv_cache_gib,
                        store_gib=store_gib,
                        n_working=n_working,
                        prefill_tokens_per_step=prefill_tokens_per_step,
                        max_context=max_context,
                        tensor_parallel_size=tensor_parallel_size,
                    )
                with server as running:
                    command = mini_command(
                        model,
                        running.base_url,
                        directory / "agents",
                        tasks=tuple(tasks),
                        workers=workers,
                        max_steps=max_steps,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    if setting.engine == "vllm":
                        agents = run_agents_on_vllm(
                            running,
                            command,
                            out=directory,
                            pool_bytes=pool_bytes,
                            container_executable=container_executable,
                        )
                    else:
                        agents = run_agents(
                            command,
                            out=directory,
                            container_executable=container_executable,
                        )
                    code = asyncio.run(agents)
                    if setting.engine == "vllm":
                        time.sleep(2 * EXPORT_INTERVAL_S)
            print(f"{directory.name}: mini-swe-agent exited with {code}", flush=True)
            grade(directory, workers)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f]


@dataclass(frozen=True)
class TaskRun:
    solved: bool
    steps: int
    capped: bool
    looped: bool
    reads: int
    refetches: int


def command_of(call: dict[str, Any]) -> str:
    try:
        return json.loads(call["function"]["arguments"]).get("command", "")
    except (json.JSONDecodeError, KeyError, AttributeError):
        return call.get("function", {}).get("arguments", "")


def task_run(trajectory: dict[str, Any], solved: bool) -> TaskRun:
    """The paper's measures of one run: its turns, whether it hit the step
    limit, whether it repeated one command 10 times in a row, and its reads,
    tool outputs of at least 5 lines, of which re-fetches are those with at
    least 80% of their distinct lines in an earlier read."""
    messages = trajectory.get("messages", [])
    turns = [m for m in messages if m.get("role") == "assistant"]
    commands = [command_of(c) for m in turns for c in m.get("tool_calls") or []]
    longest = run = 0
    for i, command in enumerate(commands):
        run = run + 1 if i and command == commands[i - 1] else 1
        longest = max(longest, run)
    seen: set[str] = set()
    reads = refetches = 0
    for message in messages:
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        text = content if isinstance(content, str) else json.dumps(content)
        lines = {line for line in text.splitlines() if line.strip()}
        if len(lines) < 5:
            continue
        reads += 1
        if len(lines & seen) >= 0.8 * len(lines):
            refetches += 1
        seen |= lines
    return TaskRun(
        solved=solved,
        steps=len(turns),
        capped=trajectory.get("info", {}).get("exit_status") == "LimitsExceeded",
        looped=longest >= 10,
        reads=reads,
        refetches=refetches,
    )


def task_runs(directory: Path) -> dict[str, TaskRun]:
    resolved: set[str] = set()
    for report in (directory / "grading").glob("*.json"):
        resolved |= set(json.loads(report.read_text()).get("resolved_ids", []))
    runs = {}
    for task in SWE_BENCH_VERIFIED_100:
        path = directory / "agents" / task / f"{task}.traj.json"
        if path.exists():
            runs[task] = task_run(json.loads(path.read_text()), task in resolved)
    return runs


def paired_difference(
    values: dict[str, float], reference: dict[str, float]
) -> tuple[float, float, float]:
    """The mean difference over the tasks both hold, and its 95% interval from
    a paired bootstrap over those tasks."""
    tasks = sorted(values.keys() & reference.keys())
    differences = [values[t] - reference[t] for t in tasks]
    if not differences:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(0)
    means = sorted(
        statistics.fmean(rng.choices(differences, k=len(differences)))
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return (
        statistics.fmean(differences),
        means[int(0.025 * BOOTSTRAP_RESAMPLES)],
        means[int(0.975 * BOOTSTRAP_RESAMPLES) - 1],
    )


def sign_test(
    values: dict[str, float], reference: dict[str, float]
) -> tuple[int, int, float]:
    """The tasks where values is higher, those where reference is, and the
    two-sided p-value of the exact sign test on those discordant tasks."""
    tasks = values.keys() & reference.keys()
    higher = sum(values[t] > reference[t] for t in tasks)
    lower = sum(values[t] < reference[t] for t in tasks)
    p = binomtest(higher, higher + lower).pvalue if higher + lower else 1.0
    return higher, lower, p


def turns_ratio(
    turns: dict[str, float], reference_runs: list[dict[str, TaskRun]]
) -> tuple[float, float, float]:
    """The median turns over the tasks divided by the median of each reference
    run, averaged over the reference runs, and its 95% interval from a paired
    bootstrap over the tasks every run holds."""
    tasks = sorted(set(turns).intersection(*reference_runs))
    if not tasks:
        return float("nan"), float("nan"), float("nan")
    reference = [[float(run[t].steps) for t in tasks] for run in reference_runs]
    values = [turns[t] for t in tasks]

    def mean_ratio(indices: list[int]) -> float:
        median = statistics.median(values[i] for i in indices)
        return statistics.fmean(
            median / statistics.median(run[i] for i in indices) for run in reference
        )

    rng = random.Random(0)
    ratios = sorted(
        mean_ratio(rng.choices(range(len(tasks)), k=len(tasks)))
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return (
        mean_ratio(list(range(len(tasks)))),
        ratios[int(0.025 * BOOTSTRAP_RESAMPLES)],
        ratios[int(0.975 * BOOTSTRAP_RESAMPLES) - 1],
    )


def per_task(runs: list[dict[str, TaskRun]], measure: str) -> dict[str, float]:
    """Each task's measure averaged over the repeats that ran it."""
    values: dict[str, list[float]] = {}
    for repeat in runs:
        for task, run in repeat.items():
            values.setdefault(task, []).append(float(getattr(run, measure)))
    return {task: statistics.fmean(v) for task, v in values.items()}


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def metric_samples(text: str) -> Iterator[tuple[str, dict[str, str], float]]:
    for line in text.splitlines():
        sample = SAMPLE.fullmatch(line)
        if sample is not None:
            name, labels, value = sample.groups()
            yield name, dict(LABEL.findall(labels or "")), float(value)


def percentile(values: list[float], q: float) -> float:
    """Interpolated linearly between the closest ranks."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    rank = q * (len(ordered) - 1)
    low = math.floor(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def counter_total(metrics: str, name: str) -> float:
    return sum(value for sample, _, value in metric_samples(metrics) if sample == name)


@dataclass(frozen=True)
class ServedRequests:
    """A setting's finished requests, from the server's spans over every run.
    cached_tokens is None when the spans do not carry it."""

    time_to_first_token_s: list[float]
    prompt_tokens: int
    cached_tokens: int | None
    store_tokens: int | None
    decoded_tokens: int
    decode_s: float


@dataclass(frozen=True)
class SpanAttributeNames:
    time_to_first_token: str
    input_tokens: str
    output_tokens: str
    decode_duration: str


# pulsarlm follows the GenAI semantic conventions; vLLM 0.31.0 keeps the
# deprecated token counts and its own latency attributes.
SPAN_ATTRIBUTE_NAMES = {
    "pulsarlm": SpanAttributeNames(
        time_to_first_token="pulsarlm.time_to_first_token",
        input_tokens="gen_ai.usage.input_tokens",
        output_tokens="gen_ai.usage.output_tokens",
        decode_duration="pulsarlm.decode.duration",
    ),
    "vllm": SpanAttributeNames(
        time_to_first_token="gen_ai.latency.time_to_first_token",
        input_tokens="gen_ai.usage.prompt_tokens",
        output_tokens="gen_ai.usage.completion_tokens",
        decode_duration="gen_ai.latency.time_in_model_decode",
    ),
}


def served_requests(directories: list[Path], engine: str) -> ServedRequests:
    names = SPAN_ATTRIBUTE_NAMES[engine]
    spans = [
        span["attributes"]
        for d in directories
        for span in load_jsonl(d / "spans.jsonl")
        if span["service"] == engine and names.time_to_first_token in span["attributes"]
    ]
    decoding = [a for a in spans if a[names.output_tokens] > 1]
    cached_tokens: int | None = None
    store_tokens: int | None = None
    if engine == "pulsarlm":
        store_tokens = sum(a["pulsarlm.usage.input_tokens_in_store"] for a in spans)
        cached_tokens = store_tokens + sum(
            a["pulsarlm.usage.input_tokens_on_device"] for a in spans
        )
    return ServedRequests(
        time_to_first_token_s=[a[names.time_to_first_token] for a in spans],
        prompt_tokens=sum(a[names.input_tokens] for a in spans),
        cached_tokens=cached_tokens,
        store_tokens=store_tokens,
        decoded_tokens=sum(a[names.output_tokens] - 1 for a in decoding),
        decode_s=sum(a[names.decode_duration] for a in decoding),
    )


def vllm_cached_share(directories: list[Path]) -> float:
    texts = [
        (d / "metrics.txt").read_text()
        for d in directories
        if (d / "metrics.txt").exists()
    ]
    return ratio(
        sum(counter_total(t, "vllm:prompt_tokens_cached_total") for t in texts),
        sum(counter_total(t, "vllm:prompt_tokens_total") for t in texts),
    )


@app.command()
def report(
    results: Path = Path("bench/results/agents"),
    reference: str = "full_kv",
) -> None:
    """Prints per setting the tasks solved, against reference's per-task mean
    with a paired bootstrap and an exact sign test over the tasks, the median
    turns and their ratio to reference's runs, the capped and looping runs and
    the re-fetch rate, then from the server's spans its finished requests,
    time to first token percentiles, prefill throughput over the time to first
    token, decode throughput past each request's first token, cached share and,
    for pulsarlm, the share read from the store, and the KV cache bytes in use."""
    by_setting: dict[str, list[Path]] = {}
    for directory in sorted(results.glob("*_r*")):
        by_setting.setdefault(directory.name.rsplit("_r", 1)[0], []).append(directory)
    runs = {name: [task_runs(d) for d in dirs] for name, dirs in by_setting.items()}
    if reference not in runs:
        raise typer.BadParameter(f"no results for the reference setting {reference}")

    print(
        f"{'setting':<34}{'runs':>5}{'solved':>8}{'Δ solved [95% CI]':>24}"
        f"{'+/−':>8}{'sign p':>8}{'turns':>7}{'× turns [95% CI]':>22}"
        f"{'capped':>8}{'looped':>8}{'re-fetch':>10}"
    )
    reference_solved = per_task(runs[reference], "solved")
    for name, repeats in runs.items():
        solved = per_task(repeats, "solved")
        delta, low, high = paired_difference(solved, reference_solved)
        higher, lower, p = sign_test(solved, reference_solved)
        turns, turns_low, turns_high = turns_ratio(
            per_task(repeats, "steps"), runs[reference]
        )
        all_runs = [run for repeat in repeats for run in repeat.values()]
        reads = sum(r.reads for r in all_runs)
        print(
            f"{name:<34}{len(all_runs):>5}{100 * statistics.fmean(solved.values()):>7.0f}%"
            f"{f'{100 * delta:+.0f} [{100 * low:+.0f}, {100 * high:+.0f}]':>24}"
            f"{f'{higher}/{lower}':>8}{p:>8.3f}"
            f"{statistics.median(r.steps for r in all_runs):>7.0f}"
            f"{f'{turns:.2f} [{turns_low:.2f}, {turns_high:.2f}]':>22}"
            f"{sum(r.capped for r in all_runs):>8}{sum(r.looped for r in all_runs):>8}"
            f"{(sum(r.refetches for r in all_runs) / reads if reads else float('nan')):>10.2f}"
        )

    print(
        f"\n{'setting':<34}{'requests':>9}{'TTFT p50':>9}{'p90':>7}"
        f"{'prefill tok/s':>15}{'decode tok/s':>14}{'cached':>8}{'store':>7}"
        f"{'KV peak':>9}{'KV mean':>9}"
    )
    for name, dirs in by_setting.items():
        engine = SETTINGS[name].engine
        served = served_requests(dirs, engine)
        prompt = served.prompt_tokens
        if served.cached_tokens is None:
            cached_share = vllm_cached_share(dirs)
            computed = prompt * (1 - cached_share)
        else:
            cached_share = ratio(served.cached_tokens, prompt)
            computed = prompt - served.cached_tokens
        store_share = (
            ratio(served.store_tokens, prompt)
            if served.store_tokens is not None
            else None
        )
        ttft = served.time_to_first_token_s
        kv = [
            s["kv_cache_bytes"] / 2**30
            for d in dirs
            for s in load_jsonl(d / "kv.jsonl")
        ]
        print(
            f"{name:<34}{len(ttft):>9}"
            f"{percentile(ttft, 0.5):>8.2f}s{percentile(ttft, 0.9):>6.2f}s"
            f"{ratio(computed, sum(ttft)):>15.0f}"
            f"{ratio(served.decoded_tokens, served.decode_s):>14.1f}"
            f"{100 * cached_share:>7.1f}%"
            f"{'–' if store_share is None else f'{100 * store_share:.1f}%':>7}"
            f"{(max(kv) if kv else float('nan')):>8.2f}G{(statistics.fmean(kv) if kv else float('nan')):>8.2f}G"
        )


if __name__ == "__main__":
    app()
