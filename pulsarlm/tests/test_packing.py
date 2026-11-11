import json
import os
import zipfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import load, load_file, save_file
from tokenizers import Tokenizer, models

from pulsarlm.checkpoint import (
    PulsarLMModelLoadingError,
    load as load_model,
    weights_version,
)
from pulsarlm.packing import build, pack, verify

TEMPLATE = "{% for m in messages %}{{ m.content }}{% endfor %}"


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


def built_weights(directory: Path) -> dict[str, torch.Tensor]:
    weights = {}
    for file in sorted((directory / "weights").glob("*.safetensors")):
        weights |= load_file(file)
    return weights


def archived_weights(archive: zipfile.ZipFile) -> dict[str, torch.Tensor]:
    weights = {}
    for name in archive.namelist():
        if name.startswith("weights/"):
            weights |= load(archive.read(name))
    return weights


def test_a_multimodal_checkpoint_keeps_its_text_weights_and_chat_template(
    tmp_path: Path,
):
    src = tmp_path / "src"
    src.mkdir()
    text = torch.ones(2)
    write_source(
        src,
        "gemma4",
        {
            "model.language_model.norm.weight": text,
            "model.vision_tower.weight": torch.zeros(3),
        },
        {"chat_template": TEMPLATE, "bos_token": "<bos>"},
    )
    dst = tmp_path / "model"
    build(src, dst)

    weights = built_weights(dst)
    assert list(weights) == ["model.language_model.norm.weight"]
    assert torch.equal(weights["model.language_model.norm.weight"], text)
    assert (dst / "chat_template.jinja").read_text() == TEMPLATE
    assert (
        json.loads((dst / "tokenizer_config.json").read_text())["bos_token"] == "<bos>"
    )


def test_a_qwen3_5_checkpoint_keeps_its_text_weights_and_untied_head(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    write_source(
        src,
        "qwen3_5",
        {
            "model.language_model.norm.weight": torch.ones(2),
            "lm_head.weight": torch.ones(3),
            "model.visual.weight": torch.zeros(3),
            "mtp.fc.weight": torch.zeros(3),
        },
        {"chat_template": TEMPLATE},
    )
    dst = tmp_path / "model"
    build(src, dst)

    assert sorted(built_weights(dst)) == [
        "lm_head.weight",
        "model.language_model.norm.weight",
    ]


def test_a_text_only_qwen3_5_checkpoint_loads_under_the_multimodal_names(
    tmp_path: Path,
):
    src = tmp_path / "src"
    src.mkdir()
    write_source(
        src,
        "qwen3_5_moe_text",
        {
            "model.norm.weight": torch.ones(2),
            "lm_head.weight": torch.ones(3),
            "mtp.fc.weight": torch.zeros(3),
        },
        {"chat_template": TEMPLATE},
    )
    dst = tmp_path / "model"
    build(src, dst)

    assert sorted(built_weights(dst)) == ["lm_head.weight", "model.norm.weight"]
    loaded = load_model(dst, device=torch.device("cpu"))
    assert sorted(loaded.weights) == [
        "lm_head.weight",
        "model.language_model.norm.weight",
    ]


def test_eos_ids_fall_back_to_the_text_config_and_take_the_tokenizer_eos(
    tmp_path: Path,
):
    src = tmp_path / "src"
    src.mkdir()
    write_source(
        src,
        "qwen3_5",
        {"model.language_model.norm.weight": torch.ones(2)},
        {"chat_template": TEMPLATE, "eos_token": "<end>"},
    )
    config = json.loads((src / "config.json").read_text())
    del config["eos_token_id"]
    config["text_config"]["eos_token_id"] = 1
    (src / "config.json").write_text(json.dumps(config))
    dst = tmp_path / "model"
    build(src, dst)

    assert json.loads((dst / "config.json").read_text())["eos_token_id"] == [1, 2]


def test_a_chat_template_file_takes_precedence(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    write_source(src, "qwen3", {"w": torch.ones(1)}, {"chat_template": "other"})
    (src / "chat_template.jinja").write_text(TEMPLATE)
    dst = tmp_path / "model"
    build(src, dst)

    assert (dst / "chat_template.jinja").read_text() == TEMPLATE
    assert list(built_weights(dst)) == ["w"]


def test_a_checkpoint_without_a_chat_template_is_rejected(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    write_source(src, "qwen3", {"w": torch.ones(1)}, {})
    with pytest.raises(PulsarLMModelLoadingError, match="no chat template"):
        build(src, tmp_path / "model")
    assert not (tmp_path / "model").exists()


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
    build(src, dst)
    return dst


def test_a_shard_kept_whole_is_a_hard_link_to_the_source(tmp_path: Path):
    dst = built_model(tmp_path)
    assert os.path.samefile(
        dst / "weights" / "model.safetensors", tmp_path / "src" / "model.safetensors"
    )


def test_a_directory_and_its_archive_load_the_same_model_and_version(tmp_path: Path):
    directory = built_model(tmp_path)
    archive = tmp_path / "model.pulsarlm"
    pack(directory, archive)

    with zipfile.ZipFile(archive) as zf:
        assert all(info.compress_type == zipfile.ZIP_STORED for info in zf.infolist())
        assert sorted(archived_weights(zf)) == ["b", "w"]
    assert weights_version(directory) == weights_version(archive)
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
