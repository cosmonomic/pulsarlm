import json
import zipfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import load, save_file
from tokenizers import Tokenizer, models

from pulsarlm.checkpoint import PulsarLMModelLoadingError, load as load_archive
from pulsarlm.migrate import migrate

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
    dst = tmp_path / "model.pulsarlm"
    migrate(src, dst)

    with zipfile.ZipFile(dst) as archive:
        assert sorted(archived_weights(archive)) == [
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
    dst = tmp_path / "model.pulsarlm"
    migrate(src, dst)

    with zipfile.ZipFile(dst) as archive:
        assert sorted(archived_weights(archive)) == [
            "lm_head.weight",
            "model.norm.weight",
        ]
    loaded = load_archive(dst, device=torch.device("cpu"))
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
    dst = tmp_path / "model.pulsarlm"
    migrate(src, dst)

    with zipfile.ZipFile(dst) as archive:
        assert json.loads(archive.read("config.json"))["eos_token_id"] == [1, 2]


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
