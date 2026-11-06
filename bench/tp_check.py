"""Greedy replies of a model served whole and split across GPUs, side by side:
a first turn on a fresh server, then a second turn on a restarted one, whose
prefix pages come back from the persisted store into every GPU's part."""

import json
import random
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer
from openai import OpenAI
from wonderwords import RandomWord

PROMPTS = [
    "Explain how a hash table handles collisions, with an example in Python.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "List the planets of the solar system and one fact about each.",
]
FOLLOW_UP = "Now summarize your answer in three sentences."


def long_prompt(words: int, seed: int) -> str:
    vocabulary = sorted(
        {
            w.lower()
            for w in RandomWord().filter(include_parts_of_speech=["nouns"])
            if w.isalpha()
        }
    )
    text = " ".join(random.Random(seed).choices(vocabulary, k=words))
    return text + "\n\nWrite a story using as many of the words above as you can."


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
def serve(
    model: str, size: int, data_dir: Path, port: int, kv_cache_gib: float, log: Path
) -> Iterator[str]:
    executable = Path(sys.executable).parent / "pulsarlm"
    command = [str(executable), "serve", model, "--data_dir", str(data_dir)]
    command += ["--port", str(port), "--tensor_parallel_size", str(size)]
    command += ["--kv_cache_gib", str(kv_cache_gib)]
    base_url = f"http://127.0.0.1:{port}/api/openai/v1"
    with log.open("a") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        try:
            wait_until_ready(process, base_url, log)
            yield base_url
        finally:
            process.terminate()
            process.wait()


def reply(
    base_url: str, model: str, messages: list[dict[str, str]], max_tokens: int
) -> dict[str, Any]:
    with OpenAI(base_url=base_url, api_key="none", timeout=3600) as client:
        response = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=0.0,
            extra_body={"reasoning_effort": "none"},
        )
    assert response.usage is not None
    return {
        "text": response.choices[0].message.content or "",
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }


def common_prefix(a: str, b: str) -> int:
    length = 0
    for x, y in zip(a, b):
        if x != y:
            break
        length += 1
    return length


app = typer.Typer(no_args_is_help=True)


@app.command()
def run(
    model: str = "qwen3.8-27b-fp8",
    out: Path = Path("bench/results/tp_check"),
    sizes: Annotated[list[int], typer.Option()] = [1, 2],
    kv_cache_gib: float = 8.0,
    max_tokens: int = 512,
    long_prompt_words: int = 3000,
    port: int = 18093,
) -> None:
    """Runs every prompt for two turns at each tensor-parallel size, each size
    on a fresh data directory, the second turn after a restart, and writes the
    replies and how far each size's replies match the first size's to
    out/tp_check.json."""
    out.mkdir(parents=True, exist_ok=True)
    prompts = [*PROMPTS, long_prompt(long_prompt_words, seed=0)]
    replies: dict[int, list[list[dict[str, Any]]]] = {}
    for size in sizes:
        data_dir = out / f"data_tp{size}"
        log = out / f"serve_tp{size}.log"
        conversations = [[{"role": "user", "content": p}] for p in prompts]
        turns: list[list[dict[str, Any]]] = [[] for _ in prompts]
        for turn in range(2):
            with serve(model, size, data_dir, port, kv_cache_gib, log) as base_url:
                for i, messages in enumerate(conversations):
                    if turn == 1:
                        messages += [
                            {"role": "assistant", "content": turns[i][0]["text"]},
                            {"role": "user", "content": FOLLOW_UP},
                        ]
                    turns[i].append(reply(base_url, model, messages, max_tokens))
                    print(
                        f"tp{size} prompt {i} turn {turn}: {turns[i][-1]['completion_tokens']} tokens",
                        flush=True,
                    )
        replies[size] = turns

    reference = sizes[0]
    comparison = []
    for size in sizes[1:]:
        for i in range(len(prompts)):
            for turn in range(2):
                a = replies[reference][i][turn]["text"]
                b = replies[size][i][turn]["text"]
                matched = common_prefix(a, b)
                comparison.append(
                    {
                        "size": size,
                        "prompt": i,
                        "turn": turn,
                        "identical": a == b,
                        "matching_chars": matched,
                        "reference_chars": len(a),
                    }
                )
                print(
                    f"tp{size} vs tp{reference} prompt {i} turn {turn}: {matched}/{len(a)} chars match",
                    flush=True,
                )
    (out / "tp_check.json").write_text(
        json.dumps(
            {
                "replies": {str(k): v for k, v in replies.items()},
                "comparison": comparison,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
