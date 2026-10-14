"""Packs a Hugging Face checkpoint directory into a PulsarLM model archive.

Reads::

    <src>/config.json
    <src>/tokenizer.json
    <src>/tokenizer_config.json
    <src>/chat_template.jinja                   (or tokenizer_config.json's chat_template)
    <src>/model.safetensors                    (or a sharded model-0000N-of-*.safetensors
                                                  set named by model.safetensors.index.json)
    <src>/generation_config.json                (optional)

and writes a single zip archive of config.json, tokenizer.json,
tokenizer_config.json, chat_template.jinja and weights/*.safetensors; see
pulsarlm.checkpoint for how that archive is read back. The weights are the text
model's alone.

eos_token_id is resolved here, from generation_config.json falling back to
config.json, and written into the archived config.json as a list of ids; the
archive never carries generation_config.json itself.
"""

import json
import tempfile
import zipfile
from pathlib import Path

import safetensors
from safetensors.torch import save_file

from pulsarlm.checkpoint import GEMMA4, PulsarLMModelConfig, PulsarLMModelLoadingError

SOURCE_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
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


def text_weight_prefix(architecture: str) -> str:
    return "model.language_model." if architecture in GEMMA4 else ""


def write_shard(zf: zipfile.ZipFile, shard: Path, prefix: str, scratch: Path) -> None:
    with safetensors.safe_open(shard, framework="pt") as f:
        names = list(f.keys())
        kept = [name for name in names if name.startswith(prefix)]
        if len(kept) == len(names):
            zf.write(shard, f"weights/{shard.name}")
            return
        if not kept:
            return
        tensors = {name: f.get_tensor(name) for name in kept}
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        filtered = Path(tmp) / shard.name
        save_file(tensors, filtered)
        zf.write(filtered, f"weights/{shard.name}")


def migrate(src: Path, dst: Path) -> None:
    cfg_raw = json.loads((src / "config.json").read_text())
    cfg_raw["eos_token_id"] = resolve_eos(src, cfg_raw)
    cfg = PulsarLMModelConfig.from_dict(
        cfg_raw
    )  # raises on a config pulsarlm cannot load
    tokenizer_config = json.loads((src / "tokenizer_config.json").read_text())

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w") as zf:
        zf.writestr("config.json", json.dumps(cfg_raw))
        zf.write(src / "tokenizer.json", "tokenizer.json")
        zf.writestr("tokenizer_config.json", json.dumps(tokenizer_config))
        zf.writestr("chat_template.jinja", chat_template(src, tokenizer_config))
        for shard in weight_shards(src):
            write_shard(zf, shard, text_weight_prefix(cfg.architecture), dst.parent)
    tmp.replace(dst)
