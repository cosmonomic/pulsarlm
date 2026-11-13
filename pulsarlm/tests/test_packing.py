import json
import zipfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import load, save_file
from tokenizers import Tokenizer, models

from pulsarlm.checkpoint import (
    PulsarLMModelLoadingError,
    load as load_model,
    model_hash,
)
from pulsarlm.migrate import migrate
from pulsarlm.packing import pack, verify

TEMPLATE = "{% for m in messages %}{{ m.content }}{% endfor %}"
OVERRIDES = {"tool_call_syntax": "none", "think_syntax": "none"}


def write_source(
    src: Path, model_type: str, tensors: dict[str, torch.Tensor], tokenizer_config: dict
) -> None:
    config = {"model_type": model_type, "eos_token_id": 1}
    text = {"vocab_size": 8, "num_hidden_layers": 1}
    if model_type.endswith("_text"):
        config |= text
    elif model_type.startswith(("gemma4", "qwen3_5")):
        config["text_config"] = text
    (src / "config.json").write_text(json.dumps(config))
    (src / "tokenizer.json").write_text(
        Tokenizer(models.WordLevel({"a": 0, "<end>": 2}, unk_token="a")).to_str()
    )
    (src / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    save_file(tensors, src / "model.safetensors")


def archived_weights(archive: zipfile.ZipFile) -> dict[str, torch.Tensor]:
    weights = {}
    for name in archive.namelist():
        if name.startswith("weights/"):
            weights |= load(archive.read(name))
    return weights


def built_model(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    write_source(
        src,
        "qwen3",
        {"w": torch.arange(6, dtype=torch.float32).view(2, 3), "b": torch.ones(2)},
        {"chat_template": TEMPLATE},
    )
    dst = tmp_path / "model"
    migrate(src, dst, OVERRIDES)
    return dst


def test_a_directory_and_its_archive_load_the_same_model_and_version(tmp_path: Path):
    directory = built_model(tmp_path)
    archive = tmp_path / "model.pulsarlm"
    pack(directory, archive)

    with zipfile.ZipFile(archive) as zf:
        assert all(info.compress_type == zipfile.ZIP_STORED for info in zf.infolist())
        assert sorted(archived_weights(zf)) == ["b", "w"]
    assert model_hash(directory) == model_hash(archive)
    from_directory = load_model(directory, device=torch.device("cpu"))
    from_archive = load_model(archive, device=torch.device("cpu"))
    assert from_directory.config == from_archive.config
    assert from_directory.weights.keys() == from_archive.weights.keys()
    for name, tensor in from_directory.weights.items():
        assert torch.equal(tensor, from_archive.weights[name])
    verify(directory)
    verify(archive)


def test_a_changed_member_fails_verification_and_packing(tmp_path: Path):
    directory = built_model(tmp_path)
    shard = directory / "weights" / "model.safetensors"
    data = bytearray(shard.read_bytes())
    data[-1] ^= 0xFF
    shard.unlink()
    shard.write_bytes(data)

    with pytest.raises(PulsarLMModelLoadingError, match="does not match its checksum"):
        verify(directory)
    with pytest.raises(PulsarLMModelLoadingError, match="does not match its checksum"):
        pack(directory, tmp_path / "model.pulsarlm")
    assert not (tmp_path / "model.pulsarlm").exists()
    assert not (tmp_path / "model.pulsarlm.tmp").exists()


def test_a_corrupted_archive_fails_verification(tmp_path: Path):
    archive = tmp_path / "model.pulsarlm"
    pack(built_model(tmp_path), archive)
    with zipfile.ZipFile(archive) as zf:
        info = zf.getinfo("weights/model.safetensors")
    data = bytearray(archive.read_bytes())
    data[info.header_offset + 30 + len(info.filename) + 20] ^= 0xFF
    archive.write_bytes(data)

    with pytest.raises(PulsarLMModelLoadingError, match="does not match its checksum"):
        verify(archive)
