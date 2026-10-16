import hashlib
import json
import os
import shutil
import zipfile
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Self

import safetensors
import torch
from tokenizers import Tokenizer

from pulsarlm.chat import ChatTemplate
from torch import Tensor
from xdg_base_dirs import xdg_cache_home, xdg_data_dirs, xdg_data_home


class PulsarLMModelLoadingError(RuntimeError):
    pass


DENSE = ("qwen2", "qwen3", "llama")
GEMMA4 = ("gemma4", "gemma4_unified")
QWEN3_5 = ("qwen3_5",)
# Multimodal checkpoints whose text model's config is their text_config.
MULTIMODAL = GEMMA4 + QWEN3_5
ARCHITECTURES = DENSE + MULTIMODAL


@dataclass
class PulsarLMModelConfig:
    architecture: str
    text: dict
    eos_ids: list[int]

    @classmethod
    def from_dict(cls, raw: dict) -> Self:
        architecture = raw["model_type"]
        if architecture not in ARCHITECTURES:
            raise PulsarLMModelLoadingError(f"unsupported architecture {architecture}")
        eos = raw["eos_token_id"]
        return cls(
            architecture=architecture,
            text=raw["text_config"] if architecture in MULTIMODAL else raw,
            eos_ids=eos if isinstance(eos, list) else [eos],
        )

    @property
    def vocab_size(self) -> int:
        return self.text["vocab_size"]

    @property
    def n_layers(self) -> int:
        return self.text["num_hidden_layers"]

    @property
    def max_position_embeddings(self) -> int:
        return self.text.get("max_position_embeddings", 0)


class PulsarLMModel(NamedTuple):
    config: PulsarLMModelConfig
    weights: dict[str, Tensor]
    tokenizer: Tokenizer
    chat_template: ChatTemplate


def iter_data_paths(data_dir: Path | None = None) -> Generator[Path]:
    if data_dir is not None:
        yield data_dir.expanduser()

    if path := os.environ.get("PULSARLM_DATA_DIR"):
        yield Path(path).expanduser()

    yield xdg_data_home() / "pulsarlm"

    for path in xdg_data_dirs():
        yield path / "pulsarlm"


def model_path(model: str, data_dir: Path | None = None) -> Path:
    filename = f"{model}.pulsarlm"
    for data_path in iter_data_paths(data_dir):
        if (path := (data_path / filename)).exists():
            return path

    raise PulsarLMModelLoadingError(f"model {model} not found")


def new_model_path(model: str, data_dir: Path | None = None) -> Path:
    return next(iter_data_paths(data_dir)) / f"{model}.pulsarlm"


def store_path(data_dir: Path | None = None) -> Path:
    return next(iter_data_paths(data_dir)) / "store.sqlite"


def weights_version(path: Path) -> str:
    with zipfile.ZipFile(path, mode="r") as archive:
        members = sorted(
            (info.filename, info.CRC, info.file_size) for info in archive.infolist()
        )
    return hashlib.sha256(json.dumps(members).encode()).hexdigest()[:16]


def load_config(path: zipfile.Path) -> PulsarLMModelConfig:
    with (path / "config.json").open("r") as f:
        raw = json.load(f)
    return PulsarLMModelConfig.from_dict(raw)


def load_tokenizer(path: zipfile.Path) -> Tokenizer:
    raw = (path / "tokenizer.json").read_text(encoding="utf-8")
    return Tokenizer.from_str(raw)


def load_chat_template(path: zipfile.Path) -> ChatTemplate:
    template = path / "chat_template.jinja"
    if not template.exists():
        raise PulsarLMModelLoadingError(
            "the archive has no chat template; pack it again with pulsarlm download"
            " or pulsarlm migrate"
        )
    tokenizer_config = json.loads((path / "tokenizer_config.json").read_text())
    return ChatTemplate(template.read_text(encoding="utf-8"), tokenizer_config)


def get_shard_cache_dir(archive_path: Path) -> Path:
    "get or create cache dir for shards, including a tmp dir for atomic extraction"
    stat = archive_path.stat()
    key = hashlib.sha256(
        f"{archive_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    ).hexdigest()[:16]
    cache_dir = xdg_cache_home() / "pulsarlm" / "shards" / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir.resolve()


def cache_shard(shard_path: zipfile.Path, cache_dir: Path) -> Path:
    if (cached_path := cache_dir / shard_path.name).exists():
        return cached_path

    tmp_path = cached_path.with_suffix(cached_path.suffix + ".tmp")

    with (
        shard_path.open("rb") as fsrc,
        tmp_path.open("wb") as fdst,
    ):
        shutil.copyfileobj(fsrc, fdst)

    tmp_path.replace(cached_path)

    return cached_path


def load_weights(
    archive: zipfile.ZipFile,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}

    assert archive.filename is not None
    cache_dir = get_shard_cache_dir(Path(archive.filename))

    for shard_path in zipfile.Path(archive, at="weights/").glob("*.safetensors"):
        cached_shard_path = cache_shard(shard_path, cache_dir)

        with safetensors.safe_open(
            cached_shard_path, framework="pt", device=str(device)
        ) as f:
            for name in f.keys():
                out[name] = f.get_tensor(name).to(dtype=dtype, non_blocking=True)

    return out


def load(
    path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> PulsarLMModel:
    with zipfile.ZipFile(path, mode="r") as archive:
        archive_path = zipfile.Path(archive)

        config = load_config(archive_path)
        tokenizer = load_tokenizer(archive_path)
        chat_template = load_chat_template(archive_path)
        weights = load_weights(archive, device=device, dtype=dtype)

    return PulsarLMModel(
        config=config,
        weights=weights,
        tokenizer=tokenizer,
        chat_template=chat_template,
    )
