"""Needle in a haystack read through a tool, with and without eviction.

The item generators follow RULER's NIAH tasks (github.com/NVIDIA/RULER,
Apache-2.0).
"""

import asyncio
import hashlib
import importlib
import itertools
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
from pulsarlm.chat import ChatTemplate, ReasoningEffort

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
    max_context_window: int | None
    recap: bool


SETTINGS = {
    "full_kv": Setting(max_context_window=None, recap=False),
    "cap_4k": Setting(max_context_window=4096, recap=True),
    "cap_8k": Setting(max_context_window=8192, recap=True),
    "cap_16k": Setting(max_context_window=16384, recap=True),
}


@dataclass
class Item:
    variant: str
    length: int
    index: int
    lines: list[str]
    queries: list[str]
    answers: list[str]
    answer_needles: list[str]
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
        answer_needles=[needle(variant.kind, q, v) for q in queries for v in values[q]],
        kind=variant.kind,
        haystack_tokens=total,
        seed=rng.getrandbits(31),
    )


def read_tool(max_read_lines: int) -> ChatCompletionToolParam:
    return {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Returns lines start to end - 1 of the document, each "
            f"prefixed with its line number, at most {max_read_lines} lines per call.",
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


def system_prompt(item: Item, recap: bool) -> str:
    singular = item.kind.removesuffix("s")
    per_key = len(item.answers) // len(item.queries)
    per_key_kind = singular if per_key == 1 else item.kind
    if len(item.queries) == 1:
        wanted = f"{per_key} special magic {per_key_kind} for the key {item.queries[0]}"
        keys = "that key"
    else:
        keys = ", ".join(item.queries)
        wanted = f"{per_key} special magic {per_key_kind} for each of the keys {keys}"
        keys = "those keys"
    if len(item.answers) == 1:
        question = f"What is the special magic {singular} for {keys}? Answer with the {singular} only."
    else:
        question = (
            f"What are the special magic {item.kind} for {keys}? Answer with the "
            f"{item.kind} only, separated by commas."
        )
    format = needle(item.kind, "<key>", f"<{singular}>")
    prompt = (
        f"You are given a document of {len(item.lines)} lines, accessible through the "
        "read tool. It contains special "
        f'magic {item.kind}, each in this format: "{format}"\n'
        f"The document has {wanted}. Read it in order from line 0.\n"
        f"Question: {question}"
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


@dataclass(frozen=True)
class Prompt:
    sha256: str
    needle_spans: list[list[tuple[int, int]]]


def template_messages(
    messages: list[ChatCompletionMessageParam],
) -> list[dict[str, Any]]:
    """The messages as the server passes them to the chat template."""
    rendered = []
    for m in cast(list[dict[str, Any]], messages):
        message: dict[str, Any] = {"role": m["role"], "content": m["content"] or ""}
        if m.get("reasoning") is not None:
            message["reasoning"] = message["reasoning_content"] = m["reasoning"]
        if m.get("tool_calls"):
            message["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": call["type"],
                    "function": {
                        "name": call["function"]["name"],
                        "arguments": json.loads(call["function"]["arguments"]),
                    },
                }
                for call in m["tool_calls"]
            ]
        if m.get("tool_call_id") is not None:
            message["tool_call_id"] = m["tool_call_id"]
        rendered.append(message)
    return rendered


def locate_needles(
    messages: list[ChatCompletionMessageParam],
    item: Item,
    *,
    max_read_lines: int,
    reasoning_effort: str,
    tokenizer: Tokenizer,
    template: ChatTemplate,
) -> Prompt:
    """The prompt's SHA-256 as the eviction log takes it, and per answer the
    [start, end) token span of each occurrence of its needle in the prompt."""
    text = template.render(
        template_messages(messages),
        add_generation_prompt=True,
        reasoning_effort=cast(ReasoningEffort, reasoning_effort),
        tools=[cast(dict[str, Any], read_tool(max_read_lines))],
    )
    encoding = tokenizer.encode(text, add_special_tokens=False)
    needle_spans: list[list[tuple[int, int]]] = []
    for line in item.answer_needles:
        spans = []
        start = text.find(line)
        while start >= 0:
            first = encoding.char_to_token(start)
            last = encoding.char_to_token(start + len(line) - 1)
            assert first is not None and last is not None
            spans.append((first, last + 1))
            start = text.find(line, start + len(line))
        needle_spans.append(spans)
    sha256 = hashlib.sha256(json.dumps(encoding.ids).encode()).hexdigest()
    return Prompt(sha256, needle_spans)


class EvictionLog:
    """A server's --eviction_log, read as the server appends to it."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._offset = 0
        self._lines: dict[str, dict[str, Any]] = {}

    def line(self, completion_id: str) -> dict[str, Any]:
        if completion_id not in self._lines:
            with self._path.open() as f:
                f.seek(self._offset)
                while (raw := f.readline()).endswith("\n"):
                    line = json.loads(raw)
                    self._lines[line["id"]] = line
                    self._offset = f.tell()
        return self._lines.pop(completion_id)


type Span = tuple[int, int]


def eviction_view(
    line: dict[str, Any], prompt: Prompt, previous: dict[str, Any] | None
) -> dict[str, Any]:
    """Over the layers eviction acts on: at the request's end, the fraction of
    the history in view and per answer the fraction of layers holding one of its
    needle's occurrences wholly in view, None for a needle not read yet; and per
    answer each layer's eviction of an occurrence since previous, the state the
    previous request ended with."""
    if line["prompt_sha256"] != prompt.sha256:
        raise RuntimeError(f"{line['id']}: the server rendered another prompt")
    end = line["states"][-1]
    views = [in_view(layer, end["tokens"]) for layer in end["layers"]]

    def kept(spans: list[Span]) -> float | None:
        if not spans:
            return None
        held = sum(any(wholly_in(view, span) for span in spans) for view in views)
        return held / len(views)

    states = ([previous] if previous else []) + line["states"]
    return {
        "in_view": sum(b - a for view in views for a, b in view)
        / (len(views) * end["tokens"]),
        "needles_kept": [kept(spans) for spans in prompt.needle_spans],
        "needle_evictions": [
            evictions(states, spans, line["cap_tokens"])
            for spans in prompt.needle_spans
        ],
    }


def evictions(
    states: list[dict[str, Any]], spans: list[Span], cap_tokens: int | None
) -> list[dict[str, Any]]:
    """Each layer's eviction of a needle occurrence between consecutive states:
    the tokens of the state after it, the highest density the occurrence's
    evicted pages had in the state before it, the layer's threshold, and whether the
    cap trimmed the layer's view."""
    found = []
    for before, after in itertools.pairwise(states):
        layers_after = {layer["layer"]: layer for layer in after["layers"]}
        for layer in before["layers"]:
            if (kept_layer := layers_after.get(layer["layer"])) is None:
                continue
            view_before = in_view(layer, before["tokens"])
            view_after = in_view(kept_layer, after["tokens"])
            kept_starts = {start for start, _, _ in kept_layer["pages"]}
            for a, b in spans:
                if not wholly_in(view_before, (a, b)) or wholly_in(view_after, (a, b)):
                    continue
                densities = [
                    density
                    for start, stop, density in layer["pages"]
                    if start < b and a < stop and start not in kept_starts
                ]
                full_tokens = sum(
                    stop - start for start, stop, _ in kept_layer["pages"]
                )
                found.append(
                    {
                        "layer": layer["layer"],
                        "tokens": after["tokens"],
                        "density": max(densities),
                        "threshold": layer["threshold"],
                        "cap_acted": full_tokens == cap_tokens,
                    }
                )
    return found


def in_view(layer: dict[str, Any], tokens: int) -> list[Span]:
    """A layer's in-view tokens as ascending spans, adjacent pages merged, with
    the tokens after its last page."""
    view: list[Span] = []
    for start, stop, _ in layer["pages"]:
        if view and view[-1][1] == start:
            view[-1] = (view[-1][0], stop)
        else:
            view.append((start, stop))
    if not view:
        return [(0, tokens)]
    view[-1] = (view[-1][0], tokens)
    return view


def wholly_in(view: list[Span], span: Span) -> bool:
    return any(a <= span[0] and span[1] <= b for a, b in view)


async def run_item(
    client: AsyncOpenAI,
    model: str,
    item: Item,
    *,
    recap: bool,
    max_read_lines: int,
    reasoning_effort: str,
    tokenizer: Tokenizer,
    template: ChatTemplate,
    eviction_log: EvictionLog,
) -> dict[str, Any]:
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt(item, recap)},
        {"role": "user", "content": "Answer the question."},
    ]
    max_turns = 2 * math.ceil(len(item.lines) / max_read_lines) + 4
    turns = tool_calls = generated_tokens = 0
    replies: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    answer: str | None = None
    error: str | None = None
    try:
        while turns < max_turns:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=[read_tool(max_read_lines)],
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
            prompt = locate_needles(
                messages,
                item,
                max_read_lines=max_read_lines,
                reasoning_effort=reasoning_effort,
                tokenizer=tokenizer,
                template=template,
            )
            line = eviction_log.line(response.id)
            view = eviction_view(line, prompt, previous)
            previous = line["states"][-1]
            completion_tokens = (
                0 if response.usage is None else response.usage.completion_tokens
            )
            generated_tokens += completion_tokens
            choice = response.choices[0]
            message = choice.message
            calls = [
                call
                for call in message.tool_calls or []
                if isinstance(call, ChatCompletionMessageFunctionToolCall)
            ]
            reasoning = (message.model_extra or {}).get("reasoning")
            replies.append(
                {
                    "id": response.id,
                    **view,
                    "finish_reason": choice.finish_reason,
                    "completion_tokens": completion_tokens,
                    "reasoning": reasoning,
                    "content": message.content,
                    "tool_calls": [call.function.arguments for call in calls],
                }
            )
            if not calls:
                answer = message.content
                if choice.finish_reason == "length":
                    error = "length"
                break
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
        "replies": replies,
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


def load_chat(model: str) -> tuple[Tokenizer, ChatTemplate]:
    with zipfile.ZipFile(checkpoint.model_path(model)) as archive:
        path = zipfile.Path(archive)
        return checkpoint.load_tokenizer(path), checkpoint.load_chat_template(path)


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
    max_read_lines: int = 25,
    n_working: int = 1024,
    kv_cache_gib: float | None = None,
    store_gib: float = 4.0,
    concurrency: int = 4,
    reasoning_effort: str = "low",
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
    tokenizer, template = load_chat(model)
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
        server_args = ["--eviction_target", "1"]
        if setting.max_context_window is not None:
            server_args = [
                "--max_context_window",
                str(setting.max_context_window),
                "--n_working",
                str(n_working),
            ]
        if kv_cache_gib is not None:
            server_args += ["--kv_cache_gib", str(kv_cache_gib)]
        eviction_log = out / name / "eviction.jsonl"
        server_args += ["--store_gib", str(store_gib)]
        server_args += ["--eviction_log", str(eviction_log)]
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
                    tokenizer=tokenizer,
                    template=template,
                    eviction_log=EvictionLog(eviction_log),
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
    tokenizer: Tokenizer,
    template: ChatTemplate,
    eviction_log: EvictionLog,
) -> None:
    slots = asyncio.Semaphore(concurrency)
    finished = 0

    async def one(client: AsyncOpenAI, item: Item) -> None:
        nonlocal finished
        async with slots:
            row = await run_item(
                client,
                model,
                item,
                recap=setting.recap,
                max_read_lines=max_read_lines,
                reasoning_effort=reasoning_effort,
                tokenizer=tokenizer,
                template=template,
                eviction_log=eviction_log,
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

    async with AsyncOpenAI(base_url=base_url, api_key="none", timeout=3600) as client:
        await asyncio.gather(*(one(client, item) for item in todo))


def mean(values: list[float]) -> str:
    return f"{sum(values) / len(values):.2f}" if values else "-"


@dataclass(frozen=True)
class Series:
    label: str
    color: str
    linestyle: str = "-"


SERIES = {
    "full_kv": Series("Full KV", "#52514e", "--"),
    "cap_4k": Series("Cap 4k", "#86b6ef"),
    "cap_8k": Series("Cap 8k", "#2a78d6"),
    "cap_16k": Series("Cap 16k", "#104281"),
}


@app.command()
def plot(
    results: Path = Path("bench/results/niah/results.jsonl"),
    out: Path = Path("bench/niah.svg"),
) -> None:
    """Prints accuracy, tool calls and generated tokens per setting and length,
    and over each item's last reply, the fraction of the history in view and of
    the layers keeping a needle read so far; plots accuracy against haystack
    length."""
    pyplot = importlib.import_module("matplotlib.pyplot")
    with results.open() as f:
        rows = [json.loads(line) for line in f]
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["setting"], row["length"])].append(row)

    print(
        f"{'setting':<16}{'length':>8}{'items':>7}{'accuracy':>10}"
        f"{'tool calls':>12}{'generated':>11}{'errors':>8}"
        f"{'in view':>9}{'kept':>7}"
    )
    for (name, length), group in sorted(groups.items()):
        n = len(group)
        last_replies = [r["replies"][-1] for r in group if r.get("replies")]
        in_view = [reply["in_view"] for reply in last_replies]
        kept = [
            fraction
            for reply in last_replies
            for fraction in reply["needles_kept"]
            if fraction is not None
        ]
        print(
            f"{name:<16}{length:>8}{n:>7}"
            f"{sum(r['score'] for r in group) / n:>10.3f}"
            f"{sum(r['tool_calls'] for r in group) / n:>12.1f}"
            f"{sum(r['generated_tokens'] for r in group) / n:>11.0f}"
            f"{sum(r['error'] is not None for r in group):>8}"
            f"{mean(in_view):>9}{mean(kept):>7}"
        )

    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    figure, axes = pyplot.subplots(figsize=(7, 4), dpi=150)
    figure.patch.set_facecolor(surface)
    axes.set_facecolor(surface)
    lengths = sorted({length for _, length in groups})
    for name, series in SERIES.items():
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
            color=series.color,
            linestyle=series.linestyle,
            linewidth=2,
            marker="o",
            markersize=8,
            markeredgecolor=surface,
            markeredgewidth=2,
            label=series.label,
        )
        axes.annotate(
            series.label,
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
