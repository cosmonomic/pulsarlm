"""Packs a Hugging Face checkpoint directory into a Pulsar model archive.

Reads::

    <src>/config.json
    <src>/tokenizer.json
    <src>/model.safetensors                    (or a sharded model-0000N-of-*.safetensors
                                                  set named by model.safetensors.index.json)
    <src>/generation_config.json                (optional)

and writes a single zip archive of config.json, tokenizer.json and
weights/*.safetensors; see pulsar.checkpoint for how that archive is read back.

eos_token_id is resolved here, from generation_config.json falling back to
config.json, and written into the archived config.json as a list of ids; the
archive never carries generation_config.json itself.
"""

import json
import zipfile
from pathlib import Path

from pulsar.checkpoint import PulsarModelConfig

SOURCE_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "model-*.safetensors",
]


def resolve_eos(src: Path, cfg_raw: dict) -> list[int]:
    gen_path = src / "generation_config.json"
    gen_raw = json.loads(gen_path.read_text()) if gen_path.exists() else cfg_raw
    eos = gen_raw.get("eos_token_id", cfg_raw.get("eos_token_id"))
    if eos is None:
        return []
    return [int(e) for e in eos] if isinstance(eos, list) else [int(eos)]


def weight_shards(src: Path) -> list[Path]:
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


def migrate(src: Path, dst: Path) -> None:
    cfg_raw = json.loads((src / "config.json").read_text())
    cfg_raw["eos_token_id"] = resolve_eos(src, cfg_raw)
    PulsarModelConfig.from_dict(cfg_raw)  # raises on a config pulsar cannot load

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w") as zf:
        zf.writestr("config.json", json.dumps(cfg_raw))
        zf.write(src / "tokenizer.json", "tokenizer.json")
        for shard in weight_shards(src):
            zf.write(shard, f"weights/{shard.name}")
    tmp.replace(dst)
