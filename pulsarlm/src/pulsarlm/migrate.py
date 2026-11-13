"""Writes PulsarLM model directories from Hugging Face checkpoint directories or
from model directories, the latter to bring them to the current format.

Reads::

    <src>/config.json
    <src>/tokenizer.json
    <src>/tokenizer_config.json
    <src>/chat_template.jinja                   (or tokenizer_config.json's chat_template)
    <src>/model.safetensors                    (or a sharded set named by
                                                  model.safetensors.index.json,
                                                  or a model directory's
                                                  weights/*.safetensors)
    <src>/generation_config.json                (optional)

eos_token_id is resolved here, from generation_config.json falling back to
config.json and its text_config, plus the tokenizer config's eos_token, and
written into the model's config.json as a list of ids; a model never carries
generation_config.json itself. The model's config overrides are then written
over config.json's top-level keys.
"""

import json
import os
import shutil
from pathlib import Path

import safetensors
from safetensors.torch import save_file
from tokenizers import Tokenizer

from pulsarlm.chat import special_token
from pulsarlm.checkpoint import (
    CHECKSUMS,
    GEMMA4,
    QWEN3_5,
    QWEN3_5_TEXT,
    QWEN4_EXP,
    PulsarLMModelConfig,
    PulsarLMModelLoadingError,
)
from pulsarlm.packing import crc32_of, members, read_checksums

UNSHARDED_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "model.safetensors",
    "model.safetensors.index.json",
]


def resolve_eos(src: Path, cfg_raw: dict, tokenizer_config: dict) -> list[int]:
    gen_path = src / "generation_config.json"
    gen_raw = json.loads(gen_path.read_text()) if gen_path.exists() else cfg_raw
    eos = gen_raw.get(
        "eos_token_id",
        cfg_raw.get("eos_token_id", cfg_raw.get("text_config", {}).get("eos_token_id")),
    )
    ids = (
        []
        if eos is None
        else [int(e) for e in eos]
        if isinstance(eos, list)
        else [int(eos)]
    )
    if (token := special_token(tokenizer_config.get("eos_token"))) is not None:
        tokenizer = Tokenizer.from_file(str(src / "tokenizer.json"))
        if (
            token_id := tokenizer.token_to_id(token)
        ) is not None and token_id not in ids:
            ids.append(token_id)
    return ids


def weight_shards(src: Path) -> list[Path]:
    if (weights := src / "weights").is_dir():
        return sorted(weights.glob("*.safetensors"))
    single = src / "model.safetensors"
    if single.exists():
        return [single]
    index = src / "model.safetensors.index.json"
    if not index.exists():
        raise FileNotFoundError(
            f"no model.safetensors or model.safetensors.index.json under {src}"
        )
    weight_map = json.loads(index.read_text())["weight_map"]
    return [src / shard for shard in sorted(set(weight_map.values()))]


def chat_template(src: Path, tokenizer_config: dict) -> str:
    if (path := src / "chat_template.jinja").exists():
        return path.read_text(encoding="utf-8")
    template = tokenizer_config.get("chat_template")
    if isinstance(template, list):
        template = next(
            (t["template"] for t in template if t.get("name") == "default"), None
        )
    if not isinstance(template, str):
        raise PulsarLMModelLoadingError(f"no chat template under {src}")
    return template


def is_text_weight(architecture: str, name: str) -> bool:
    if architecture in GEMMA4:
        return name.startswith("model.language_model.")
    if architecture in QWEN3_5 + QWEN4_EXP:
        return name.startswith("model.language_model.") or name == "lm_head.weight"
    if architecture in QWEN3_5_TEXT:
        return name.startswith("model.") or name == "lm_head.weight"
    return True


def write_weights(shard: Path, dst: Path, architecture: str) -> None:
    """Writes the text model's weights of shard to dst, a hard link to shard when
    they are all of it; writes nothing when it holds none."""
    with safetensors.safe_open(shard, framework="pt") as f:
        names = list(f.keys())
        kept = [name for name in names if is_text_weight(architecture, name)]
        if len(kept) == len(names):
            os.link(shard, dst)
            return
        if not kept:
            return
        tensors = {name: f.get_tensor(name) for name in kept}
    save_file(tensors, dst)


def checksums_of(directory: Path, src: Path) -> dict[str, dict[str, int]]:
    """A member hard-linked to src's same member takes src's checksum instead of
    being read."""
    known = read_checksums(src) if (src / CHECKSUMS).exists() else {}
    checksums = {}
    for name in members(directory):
        file = directory / name
        if name in known and (src / name).exists() and file.samefile(src / name):
            checksums[name] = known[name]
        else:
            checksums[name] = {"size": file.stat().st_size, "crc32": crc32_of(file)}
    return checksums


def migrate(src: Path, dst: Path, config_overrides: dict[str, str]) -> None:
    """Writes the model directory dst of the Hugging Face checkpoint directory or
    model directory src, through a sibling .build- directory renamed into place.
    src and dst may be the same directory. A shard kept whole is a hard link to
    src's."""
    cfg_raw = json.loads((src / "config.json").read_text())
    tokenizer_config = json.loads((src / "tokenizer_config.json").read_text())
    cfg_raw["eos_token_id"] = resolve_eos(src, cfg_raw, tokenizer_config)
    cfg_raw |= config_overrides
    cfg = PulsarLMModelConfig.from_dict(cfg_raw)
    template = chat_template(src, tokenizer_config)

    staging = dst.with_name(f".build-{dst.name}")
    shutil.rmtree(staging, ignore_errors=True)
    (staging / "weights").mkdir(parents=True)
    (staging / "config.json").write_text(json.dumps(cfg_raw))
    shutil.copyfile(src / "tokenizer.json", staging / "tokenizer.json")
    (staging / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    (staging / "chat_template.jinja").write_text(template, encoding="utf-8")
    for shard in weight_shards(src):
        write_weights(shard, staging / "weights" / shard.name, cfg.architecture)
    checksums = checksums_of(staging, src)
    (staging / CHECKSUMS).write_text(json.dumps(checksums, indent=1))
    if not dst.exists():
        staging.rename(dst)
        return
    previous = dst.with_name(f".previous-{dst.name}")
    shutil.rmtree(previous, ignore_errors=True)
    dst.rename(previous)
    staging.rename(dst)
    shutil.rmtree(previous)
