"""Needle in a haystack read through a tool, with and without eviction.

The item generators follow RULER's NIAH tasks (github.com/NVIDIA/RULER,
Apache-2.0).
"""

import asyncio
import importlib
import json
import math
import random
import subprocess
import sys
import time
import uuid
import zipfile
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import httpx
import typer
from openai import APIStatusError, AsyncOpenAI, OpenAIError
from openai.types.chat import (
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from tokenizers import Tokenizer
from wonderwords import RandomWord

from pulsarlm import checkpoint

NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."


@dataclass(frozen=True)
class Variant:
    haystack: Literal["noise", "needle"]
    kind: Literal["numbers", "uuids"]
    keys: int = 1
    values: int = 1
    queries: int = 1


VARIANTS = {
    "single": Variant("noise", "numbers"),
    "single_uuid": Variant("noise", "uuids"),
    "multikey": Variant("noise", "numbers", keys=4),
    "multikey_needle": Variant("needle", "numbers"),
    "multivalue": Variant("noise", "numbers", values=4),
    "multiquery": Variant("noise", "numbers", keys=4, queries=4),
}


@dataclass(frozen=True)
class Setting:
    evict: bool
    recap: bool


SETTINGS = {
    "full_kv": Setting(evict=False, recap=False),
    "eviction": Setting(evict=True, recap=False),
    "eviction_recap": Setting(evict=True, recap=True),
}


@dataclass
class Item:
    variant: str
    length: int
    index: int
    lines: list[str]
    queries: list[str]
    answers: list[str]
    kind: str
    haystack_tokens: int
    seed: int


class Words:
    def __init__(self) -> None:
        words = RandomWord()
        self._adjectives = self._lowercase(
            words.filter(include_parts_of_speech=["adjectives"])
        )
        self._nouns = self._lowercase(words.filter(include_parts_of_speech=["nouns"]))

    @staticmethod
    def _lowercase(words: list[str]) -> list[str]:
        return sorted({w.lower() for w in words if w.isalpha()})

    def key(self, rng: random.Random) -> str:
        return f"{rng.choice(self._adjectives)}-{rng.choice(self._nouns)}"


def needle(kind: str, key: str, value: str) -> str:
    return f"One of the special magic {kind} for {key} is: {value}."


def magic_value(kind: str, rng: random.Random) -> str:
    if kind == "uuids":
        return str(uuid.UUID(int=rng.getrandbits(128), version=4))
    return str(rng.randint(1_000_000, 9_999_999))


def make_item(
    name: str,
    length: int,
    index: int,
    *,
    seed: int,
    words: Words,
    tokenizer: Tokenizer,
) -> Item:
    """A haystack of about length tokens with the variant's needles at random
    lines."""
    variant = VARIANTS[name]
    rng = random.Random(f"{seed}/{name}/{length}/{index}")
    keys: list[str] = []
    while len(keys) < variant.keys:
        if (key := words.key(rng)) not in keys:
            keys.append(key)
    values = {
        key: [magic_value(variant.kind, rng) for _ in range(variant.values)]
        for key in keys
    }
    needles = [needle(variant.kind, k, v) for k in keys for v in values[k]]

    def tokens(line: str) -> int:
        return len(tokenizer.encode(line + "\n", add_special_tokens=False).ids)

    total = sum(tokens(line) for line in needles)
    filler: list[str] = []
    while total < length:
        if variant.haystack == "noise":
            line = NOISE
        else:
            distractor = words.key(rng)
            if distractor in keys:
                continue
            line = needle(variant.kind, distractor, magic_value(variant.kind, rng))
        filler.append(line)
        total += tokens(line)

    lines = filler
    for line in needles:
        lines.insert(rng.randint(0, len(lines)), line)
    queries = keys[: variant.queries]
    return Item(
        variant=name,
        length=length,
        index=index,
        lines=lines,
        queries=queries,
        answers=[v for q in queries for v in values[q]],
        kind=variant.kind,
        haystack_tokens=total,
        seed=rng.getrandbits(31),
    )


READ_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "read",
        "description": "Returns lines start to end - 1 of the document, each prefixed "
        "with its line number.",
        "parameters": {
            "type": "object",
            "properties": {
                "start": {"type": "integer", "description": "First line to read."},
                "end": {"type": "integer", "description": "Line after the last."},
            },
            "required": ["start", "end"],
        },
    },
}


def system_prompt(item: Item, max_read_lines: int, recap: bool) -> str:
    queries = ", ".join(item.queries)
    prompt = (
        f"A document of {len(item.lines)} lines is available through the read tool, "
        f"at most {max_read_lines} lines per call, numbered from 0. Some special "
        f"magic {item.kind} "
        "are hidden within it.\n"
        f"Question: What are all the special magic {item.kind} for {queries} "
        "mentioned in the document?\n"
        "Read the document until you can answer, then answer with the "
        f"{item.kind} only, separated by commas."
    )
    if recap:
        prompt += (
            "\nAfter every tool result, think about the exact "
            f"{item.kind} found so far, each with its key, and the line ranges read "
            "so far."
        )
    return prompt


def read(item: Item, arguments: str, max_read_lines: int) -> str:
    try:
        parsed = json.loads(arguments)
        start, end = int(parsed["start"]), int(parsed["end"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return "error: read takes integer start and end"
    if not 0 <= start < end:
        return "error: read needs 0 <= start < end"
    if end - start > max_read_lines:
        return f"error: read at most {max_read_lines} lines per call"
    return "\n".join(
        f"{i}: {item.lines[i]}" for i in range(start, min(end, len(item.lines)))
    )


def score(answer: str | None, expected: list[str]) -> float:
    if not answer:
        return 0.0
    return sum(value.lower() in answer.lower() for value in expected) / len(expected)


async def run_item(
    client: AsyncOpenAI,
    model: str,
    item: Item,
    *,
    recap: bool,
    max_read_lines: int,
    reasoning_effort: str,
) -> dict[str, Any]:
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt(item, max_read_lines, recap)},
        {"role": "user", "content": "Answer the question."},
    ]
    max_turns = 2 * math.ceil(len(item.lines) / max_read_lines) + 4
    turns = tool_calls = generated_tokens = 0
    answer: str | None = None
    error: str | None = None
    try:
        while turns < max_turns:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=[READ_TOOL],
                    temperature=1.0,
                    top_p=0.95,
                    seed=item.seed,
                    max_tokens=4096,
                    extra_body={"reasoning_effort": reasoning_effort},
                )
            except APIStatusError as failure:
                if failure.status_code != 503 or "KV cache" not in failure.message:
                    raise
                await asyncio.sleep(1)
                continue
            turns += 1
            if response.usage is not None:
                generated_tokens += response.usage.completion_tokens
            message = response.choices[0].message
            calls = [
                call
                for call in message.tool_calls or []
                if isinstance(call, ChatCompletionMessageFunctionToolCall)
            ]
            if not calls:
                answer = message.content
                break
            reasoning = (message.model_extra or {}).get("reasoning")
            messages.append(
                cast(
                    ChatCompletionMessageParam,
                    {
                        "role": "assistant",
                        "content": message.content,
                        "reasoning": reasoning,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.function.name,
                                    "arguments": call.function.arguments,
                                },
                            }
                            for call in calls
                        ],
                    },
                )
            )
            for call in calls:
                tool_calls += 1
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": read(item, call.function.arguments, max_read_lines),
                    }
                )
        else:
            error = "turn limit"
    except OpenAIError as failure:
        error = str(failure)
    return {
        "variant": item.variant,
        "length": item.length,
        "index": item.index,
        "haystack_tokens": item.haystack_tokens,
        "answers": item.answers,
        "answer": answer,
        "score": score(answer, item.answers),
        "tool_calls": tool_calls,
        "generated_tokens": generated_tokens,
        "error": error,
    }


@contextmanager
def server(
    model: str, data_dir: Path, port: int, server_args: list[str]
) -> Iterator[str]:
    data_dir.mkdir(parents=True, exist_ok=True)
    executable = Path(sys.executable).parent / "pulsarlm"
    command = [str(executable), "serve", model, "--data_dir", str(data_dir)]
    command += ["--port", str(port), "--no_persist", *server_args]
    base_url = f"http://127.0.0.1:{port}/api/openai/v1"
    with open(data_dir / "serve.log", "w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"the server exited, see {data_dir / 'serve.log'}"
                    )
                try:
                    httpx.get(f"{base_url}/models", timeout=1).raise_for_status()
                    break
                except httpx.HTTPError:
                    time.sleep(1)
            yield base_url
        finally:
            process.terminate()
            process.wait()


def load_tokenizer(model: str) -> Tokenizer:
    with zipfile.ZipFile(checkpoint.model_path(model)) as archive:
        return checkpoint.load_tokenizer(zipfile.Path(archive))


def done_keys(results: Path) -> set[tuple[str, str, int, int]]:
    if not results.exists():
        return set()
    with results.open() as f:
        rows = [json.loads(line) for line in f]
    return {(r["setting"], r["variant"], r["length"], r["index"]) for r in rows}


app = typer.Typer(no_args_is_help=True)


@app.command()
def run(
    model: str = "gemma-4-12b-it",
    out: Path = Path("bench/results/niah"),
    settings: Annotated[list[str], typer.Option()] = list(SETTINGS),
    variants: Annotated[list[str], typer.Option()] = list(VARIANTS),
    lengths: Annotated[list[int], typer.Option()] = [8192, 16384, 32768],
    items: int = 10,
    shard: int = 0,
    shards: int = 1,
    seed: int = 0,
    max_read_lines: int = 100,
    view_pages: int = 256,
    n_working: int = 64,
    kv_cache_gib: float | None = None,
    store_gib: float = 4.0,
    concurrency: int = 4,
    reasoning_effort: str = "medium",
    port: int = 18090,
) -> None:
    """Runs every item in every setting and appends the results to
    out/results.jsonl, skipping items already there. With shards, runs only
    the items whose index is shard modulo shards, so the shards together cover
    every item and each covers every variant and length."""
    unknown = set(settings) - set(SETTINGS) | set(variants) - set(VARIANTS)
    if unknown:
        raise typer.BadParameter(f"unknown settings or variants: {sorted(unknown)}")
    if not 0 <= shard < shards:
        raise typer.BadParameter("shard must be in [0, shards)")
    tokenizer = load_tokenizer(model)
    words = Words()
    all_items = [
        make_item(v, n, i, seed=seed, words=words, tokenizer=tokenizer)
        for v in variants
        for n in lengths
        for i in range(shard, items, shards)
    ]
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"
    done = done_keys(results)

    for name in settings:
        setting = SETTINGS[name]
        todo = [
            item
            for item in all_items
            if (name, item.variant, item.length, item.index) not in done
        ]
        if not todo:
            continue
        server_args = ["--evict_tau", "0"]
        if setting.evict:
            server_args = [
                "--max_pages",
                str(view_pages),
                "--n_working",
                str(n_working),
            ]
        if kv_cache_gib is not None:
            server_args += ["--kv_cache_gib", str(kv_cache_gib)]
        server_args += ["--store_gib", str(store_gib)]
        with server(model, out / name, port, server_args) as base_url:
            asyncio.run(
                run_setting(
                    base_url,
                    model,
                    name,
                    setting,
                    todo,
                    results,
                    max_read_lines=max_read_lines,
                    reasoning_effort=reasoning_effort,
                    concurrency=concurrency,
                )
            )


async def run_setting(
    base_url: str,
    model: str,
    name: str,
    setting: Setting,
    todo: list[Item],
    results: Path,
    *,
    max_read_lines: int,
    reasoning_effort: str,
    concurrency: int,
) -> None:
    client = AsyncOpenAI(base_url=base_url, api_key="none", timeout=3600)
    slots = asyncio.Semaphore(concurrency)
    finished = 0

    async def one(item: Item) -> None:
        nonlocal finished
        async with slots:
            row = await run_item(
                client,
                model,
                item,
                recap=setting.recap,
                max_read_lines=max_read_lines,
                reasoning_effort=reasoning_effort,
            )
        row = {"setting": name, **row}
        with results.open("a") as f:
            f.write(json.dumps(row) + "\n")
        finished += 1
        print(
            f"{name} {finished}/{len(todo)} {item.variant} {item.length} "
            f"#{item.index}: score {row['score']:.2f}"
            + (f" ({row['error']})" if row["error"] else ""),
            flush=True,
        )

    await asyncio.gather(*(one(item) for item in todo))


SERIES_COLORS = {
    "full_kv": "#2a78d6",
    "eviction": "#eb6834",
    "eviction_recap": "#1baf7a",
}
SERIES_LABELS = {
    "full_kv": "Full KV",
    "eviction": "Eviction",
    "eviction_recap": "Eviction + recap",
}


@app.command()
def plot(
    results: Path = Path("bench/results/niah/results.jsonl"),
    out: Path = Path("bench/niah.svg"),
) -> None:
    """Prints accuracy, tool calls and generated tokens per setting and length,
    and plots accuracy against haystack length."""
    pyplot = importlib.import_module("matplotlib.pyplot")
    with results.open() as f:
        rows = [json.loads(line) for line in f]
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["setting"], row["length"])].append(row)

    print(
        f"{'setting':<16}{'length':>8}{'items':>7}{'accuracy':>10}"
        f"{'tool calls':>12}{'generated':>11}{'errors':>8}"
    )
    for (name, length), group in sorted(groups.items()):
        n = len(group)
        print(
            f"{name:<16}{length:>8}{n:>7}"
            f"{sum(r['score'] for r in group) / n:>10.3f}"
            f"{sum(r['tool_calls'] for r in group) / n:>12.1f}"
            f"{sum(r['generated_tokens'] for r in group) / n:>11.0f}"
            f"{sum(r['error'] is not None for r in group):>8}"
        )

    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    figure, axes = pyplot.subplots(figsize=(7, 4), dpi=150)
    figure.patch.set_facecolor(surface)
    axes.set_facecolor(surface)
    lengths = sorted({length for _, length in groups})
    for name, color in SERIES_COLORS.items():
        points = [
            (length, 100 * sum(r["score"] for r in group) / len(group))
            for length in lengths
            if (group := groups.get((name, length)))
        ]
        if not points:
            continue
        xs, ys = zip(*points)
        axes.plot(
            xs,
            ys,
            color=color,
            linewidth=2,
            marker="o",
            markersize=8,
            markeredgecolor=surface,
            markeredgewidth=2,
            label=SERIES_LABELS[name],
        )
        axes.annotate(
            SERIES_LABELS[name],
            (xs[-1], ys[-1]),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            color=ink,
            fontsize=9,
        )
    axes.set_xscale("log", base=2)
    axes.set_xticks(lengths, [f"{length // 1024}k" for length in lengths])
    axes.set_ylim(0, 100)
    axes.set_xlabel("Haystack tokens", color=muted)
    axes.set_ylabel("Accuracy (%)", color=muted)
    axes.tick_params(colors=muted)
    axes.grid(axis="y", color=grid, linewidth=1)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(grid)
    axes.legend(frameon=False, labelcolor=ink, loc="lower left")
    axes.set_title("NIAH through a read tool", color=ink, loc="left")
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, facecolor=surface)
    print(f"wrote {out}")


if __name__ == "__main__":
    app()
