import hashlib
import importlib
import json
import os
import zipfile
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Self, get_args

import torch
from tokenizers import Tokenizer

from pulsarlm.chat import ChatTemplate
from pulsarlm.reply import ThinkSyntax
from pulsarlm.safetensors_file import SafetensorsFile
from pulsarlm.tools import ToolCallSyntax
from torch import Tensor
from xdg_base_dirs import xdg_data_dirs, xdg_data_home


class PulsarLMModelLoadingError(RuntimeError):
    pass


CHECKSUMS = "checksums.json"
ARCHIVE_SUFFIX = ".pulsarlm"

type ModelRoot = Path | zipfile.Path


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


def config_syntax(raw: dict, key: str, syntaxes: tuple[str, ...]) -> Any:
    if key not in raw:
        raise PulsarLMModelLoadingError(
            f"config.json has no {key}; run pulsarlm models migrate"
        )
    value = raw[key]
    if value == "none":
        return None
    if value not in syntaxes:
        raise PulsarLMModelLoadingError(f"unknown {key} {value!r}")
    return value


@dataclass
class PulsarLMModelConfig:
    architecture: str
    text: dict
    eos_ids: list[int]
    quantization: dict | None
    tool_call_syntax: ToolCallSyntax | None
    think_syntax: ThinkSyntax | None

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
            tool_call_syntax=config_syntax(
                raw, "tool_call_syntax", get_args(ToolCallSyntax.__value__)
            ),
            think_syntax=config_syntax(
                raw, "think_syntax", get_args(ThinkSyntax.__value__)
            ),
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
    """The model's directory, or its archive, in the first data path holding
    either; the directory when one holds both."""
    for data_path in iter_data_paths(data_dir):
        for path in (data_path / model, data_path / f"{model}{ARCHIVE_SUFFIX}"):
            if path.exists():
                return path

    raise PulsarLMModelLoadingError(f"model {model} not found")


def local_models(data_dir: Path | None = None) -> dict[str, list[Path]]:
    """Each model's directories and archives in the data paths, in search
    order."""
    found: dict[str, list[Path]] = {}
    seen: set[Path] = set()
    for data_path in iter_data_paths(data_dir):
        if data_path in seen or not data_path.is_dir():
            continue
        seen.add(data_path)
        for path in sorted(data_path.iterdir()):
            if path.name.startswith("."):
                continue
            if path.is_dir() and (path / CHECKSUMS).exists():
                found.setdefault(path.name, []).append(path)
            elif path.is_file() and path.suffix == ARCHIVE_SUFFIX:
                found.setdefault(path.stem, []).append(path)
    return found


def new_model_path(model: str, data_dir: Path | None = None) -> Path:
    return next(iter_data_paths(data_dir)) / model


def store_path(data_dir: Path | None = None) -> Path:
    return next(iter_data_paths(data_dir)) / "store.sqlite"


def member_checksums(path: Path) -> list[tuple[str, int, int]]:
    """Each member's name, CRC-32 and size, from a directory's checksums.json or
    an archive's zip records, sorted."""
    if path.is_dir():
        checksums = json.loads((path / CHECKSUMS).read_text())
        return sorted(
            (name, entry["crc32"], entry["size"]) for name, entry in checksums.items()
        )
    with zipfile.ZipFile(path, mode="r") as archive:
        return sorted(
            (info.filename, info.CRC, info.file_size) for info in archive.infolist()
        )


def weights_version(path: Path) -> str:
    """The same for a model directory and the archive packed from it."""
    return hashlib.sha256(json.dumps(member_checksums(path)).encode()).hexdigest()[:16]


@contextmanager
def model_root(path: Path) -> Iterator[ModelRoot]:
    """The root of a model directory or archive, which its members are read
    under."""
    if path.is_dir():
        yield path
        return
    with zipfile.ZipFile(path, mode="r") as archive:
        yield zipfile.Path(archive)


def weight_files(path: Path) -> list[SafetensorsFile]:
    """The weights/*.safetensors of a model directory or archive, read in
    place."""
    if path.is_dir():
        return [
            SafetensorsFile.of_file(file)
            for file in sorted((path / "weights").glob("*.safetensors"))
        ]
    with zipfile.ZipFile(path, mode="r") as archive:
        infos = sorted(archive.infolist(), key=lambda info: info.filename)
    return [
        SafetensorsFile.of_member(path, info)
        for info in infos
        if info.filename.startswith("weights/")
        and info.filename.endswith(".safetensors")
    ]


def load_config(path: ModelRoot) -> PulsarLMModelConfig:
    with (path / "config.json").open("r") as f:
        raw = json.load(f)
    return PulsarLMModelConfig.from_dict(raw)


def load_tokenizer(path: ModelRoot) -> Tokenizer:
    raw = (path / "tokenizer.json").read_text(encoding="utf-8")
    return Tokenizer.from_str(raw)


def load_chat_template(path: ModelRoot) -> ChatTemplate:
    template = path / "chat_template.jinja"
    if not template.exists():
        raise PulsarLMModelLoadingError(
            "the model has no chat template; download it again with pulsarlm models"
            " download"
        )
    tokenizer_config = json.loads((path / "tokenizer_config.json").read_text())
    return ChatTemplate(template.read_text(encoding="utf-8"), tokenizer_config)


def quantized(name: str, tensor: Tensor) -> bool:
    """An fp8 checkpoint's quantized weight or its scales. They keep their
    stored dtype and stay on the host: the model repacks each onto the device,
    so a device copy here would double them there."""
    return tensor.dtype == torch.float8_e4m3fn or name.endswith(".weight_scale_inv")


def load_weights(
    path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
    unused_prefixes: tuple[str, ...] = (),
    shard: tuple[int, int] = (0, 1),
) -> dict[str, Tensor]:
    """With shard (rank, size) of more than one GPU, GPU rank's part of the
    weights (pulsarlm.sharding), read without the rest. Each tensor reaches the
    device before the next is read."""
    files = weight_files(path)
    rank, size = shard
    if size > 1:
        sharding = importlib.import_module("pulsarlm.sharding")
        with model_root(path) as root:
            config = load_config(root)
        host = sharding.rank_weights(files, config, rank, size, unused_prefixes)
    else:
        host = whole_weights(files, unused_prefixes)
    return {
        name: tensor
        if quantized(name, tensor)
        else tensor.to(device=device, dtype=dtype, non_blocking=True)
        for name, tensor in host
    }


def whole_weights(
    files: list[SafetensorsFile], unused_prefixes: tuple[str, ...]
) -> Iterator[tuple[str, Tensor]]:
    for f in files:
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
    """The model directory or archive at path."""
    with model_root(path) as root:
        config = load_config(root)
        tokenizer = load_tokenizer(root)
        chat_template = load_chat_template(root)
    weights = text_model_weights(
        config.architecture,
        load_weights(
            path,
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
