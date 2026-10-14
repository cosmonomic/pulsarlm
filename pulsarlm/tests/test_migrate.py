import json
import zipfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import load, save_file
from tokenizers import Tokenizer, models

from pulsarlm.checkpoint import PulsarLMModelLoadingError
from pulsarlm.migrate import migrate

TEMPLATE = "{% for m in messages %}{{ m.content }}{% endfor %}"


def write_source(
    src: Path, model_type: str, tensors: dict[str, torch.Tensor], tokenizer_config: dict
) -> None:
    config = {"model_type": model_type, "eos_token_id": 1}
    if model_type.startswith("gemma4"):
        config["text_config"] = {"vocab_size": 8, "num_hidden_layers": 1}
    (src / "config.json").write_text(json.dumps(config))
    (src / "tokenizer.json").write_text(
        Tokenizer(models.WordLevel({"a": 0}, unk_token="a")).to_str()
    )
    (src / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    save_file(tensors, src / "model.safetensors")


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
    dst = tmp_path / "model.pulsarlm"
    migrate(src, dst)

    with zipfile.ZipFile(dst) as archive:
        weights = archived_weights(archive)
        assert list(weights) == ["model.language_model.norm.weight"]
        assert torch.equal(weights["model.language_model.norm.weight"], text)
        assert archive.read("chat_template.jinja").decode() == TEMPLATE
        assert json.loads(archive.read("tokenizer_config.json"))["bos_token"] == "<bos>"


def test_a_chat_template_file_takes_precedence(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    write_source(src, "qwen3", {"w": torch.ones(1)}, {"chat_template": "other"})
    (src / "chat_template.jinja").write_text(TEMPLATE)
    dst = tmp_path / "model.pulsarlm"
    migrate(src, dst)

    with zipfile.ZipFile(dst) as archive:
        assert archive.read("chat_template.jinja").decode() == TEMPLATE
        assert list(archived_weights(archive)) == ["w"]


def test_a_checkpoint_without_a_chat_template_is_rejected(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    write_source(src, "qwen3", {"w": torch.ones(1)}, {})
    with pytest.raises(PulsarLMModelLoadingError, match="no chat template"):
        migrate(src, tmp_path / "model.pulsarlm")
