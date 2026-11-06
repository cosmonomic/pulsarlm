import hashlib
import importlib
import json
import os
import shutil
import zipfile
from collections.abc import Generator, Iterator
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
QWEN3_5 = ("qwen3_5", "qwen3_5_moe")
# Text-only Qwen3.5 checkpoints, whose text weights are named model.* instead of model.language_model.*.
QWEN3_5_TEXT = ("qwen3_5_text", "qwen3_5_moe_text")
QWEN4_EXP = ("qwen4_exp",)
# Multimodal checkpoints whose text model's config is their text_config.
MULTIMODAL = GEMMA4 + QWEN3_5 + QWEN4_EXP
ARCHITECTURES = DENSE + MULTIMODAL + QWEN3_5_TEXT
# Weights the text model doesn't read yet: Qwen3.5's vision tower and multi-token prediction layer.
UNUSED_PREFIXES = dict.fromkeys(QWEN3_5 + QWEN3_5_TEXT, ("model.visual.", "mtp."))


@dataclass
class PulsarLMModelConfig:
    architecture: str
    text: dict
    eos_ids: list[int]
    quantization: dict | None

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
            quantization=raw.get("quantization_config"),
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


def quantized(name: str, tensor: Tensor) -> bool:
    """An fp8 checkpoint's quantized weight or its scales. They keep their
    stored dtype and stay on the host: the model repacks each onto the device,
    so a device copy here would double them there."""
    return tensor.dtype == torch.float8_e4m3fn or name.endswith(".weight_scale_inv")


def load_weights(
    archive: zipfile.ZipFile,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
    unused_prefixes: tuple[str, ...] = (),
    shard: tuple[int, int] = (0, 1),
) -> dict[str, Tensor]:
    """With shard (rank, size) of more than one GPU, GPU rank's part of the
    weights (pulsarlm.sharding), read without the rest. Each tensor reaches the
    device before the next is read."""
    assert archive.filename is not None
    cache_dir = get_shard_cache_dir(Path(archive.filename))
    paths = [
        cache_shard(shard_path, cache_dir)
        for shard_path in zipfile.Path(archive, at="weights/").glob("*.safetensors")
    ]
    rank, size = shard
    if size > 1:
        sharding = importlib.import_module("pulsarlm.sharding")
        host = sharding.rank_weights(
            paths, load_config(zipfile.Path(archive)), rank, size, unused_prefixes
        )
    else:
        host = whole_weights(paths, unused_prefixes)
    return {
        name: tensor
        if quantized(name, tensor)
        else tensor.to(device=device, dtype=dtype, non_blocking=True)
        for name, tensor in host
    }


def whole_weights(
    paths: list[Path], unused_prefixes: tuple[str, ...]
) -> Iterator[tuple[str, Tensor]]:
    for path in paths:
        with safetensors.safe_open(path, framework="pt") as f:
            for name in f.keys():
                if not name.startswith(unused_prefixes):
                    yield name, f.get_tensor(name)


def text_model_weights(
    architecture: str, weights: dict[str, Tensor]
) -> dict[str, Tensor]:
    """A text-only Qwen3.5 checkpoint's weights under the multimodal names the
    model reads."""
    if architecture not in QWEN3_5_TEXT:
        return weights
    return {
        "model.language_model." + name.removeprefix("model.")
        if name.startswith("model.")
        else name: tensor
        for name, tensor in weights.items()
    }


def load(
    path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
    shard: tuple[int, int] = (0, 1),
) -> PulsarLMModel:
    with zipfile.ZipFile(path, mode="r") as archive:
        archive_path = zipfile.Path(archive)

        config = load_config(archive_path)
        tokenizer = load_tokenizer(archive_path)
        chat_template = load_chat_template(archive_path)
        weights = text_model_weights(
            config.architecture,
            load_weights(
                archive,
                device=device,
                dtype=dtype,
                unused_prefixes=UNUSED_PREFIXES.get(config.architecture, ()),
                shard=shard,
            ),
        )

    return PulsarLMModel(
        config=config,
        weights=weights,
        tokenizer=tokenizer,
        chat_template=chat_template,
    )
