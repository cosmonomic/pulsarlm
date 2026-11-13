import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models

from pulsarlm.checkpoint import (
    CHECKSUMS,
    PulsarLMModelLoadingError,
    load as load_model,
    load_config,
)
from pulsarlm.migrate import migrate

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


def built_weights(directory: Path) -> dict[str, torch.Tensor]:
    weights = {}
    for file in sorted((directory / "weights").glob("*.safetensors")):
        weights |= load_file(file)
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
    migrate(src, dst, OVERRIDES)

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
    migrate(src, dst, OVERRIDES)

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
    migrate(src, dst, OVERRIDES)

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
    migrate(src, dst, OVERRIDES)

    assert json.loads((dst / "config.json").read_text())["eos_token_id"] == [1, 2]


def test_a_chat_template_file_takes_precedence(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    write_source(src, "qwen3", {"w": torch.ones(1)}, {"chat_template": "other"})
    (src / "chat_template.jinja").write_text(TEMPLATE)
    dst = tmp_path / "model"
    migrate(src, dst, OVERRIDES)

    assert (dst / "chat_template.jinja").read_text() == TEMPLATE
    assert list(built_weights(dst)) == ["w"]


def test_a_checkpoint_without_a_chat_template_is_rejected(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    write_source(src, "qwen3", {"w": torch.ones(1)}, {})
    with pytest.raises(PulsarLMModelLoadingError, match="no chat template"):
        migrate(src, tmp_path / "model", OVERRIDES)
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
    migrate(src, dst, OVERRIDES)
    return dst


def test_a_shard_kept_whole_is_a_hard_link_to_the_source(tmp_path: Path):
    dst = built_model(tmp_path)
    assert os.path.samefile(
        dst / "weights" / "model.safetensors", tmp_path / "src" / "model.safetensors"
    )


def test_the_config_overrides_are_written_into_the_config(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    write_source(src, "qwen3", {"w": torch.ones(1)}, {"chat_template": TEMPLATE})
    dst = tmp_path / "model"
    migrate(src, dst, {"tool_call_syntax": "qwen_json", "think_syntax": "qwen"})

    config = load_config(dst)
    assert config.tool_call_syntax == "qwen_json"
    assert config.think_syntax == "qwen"


def test_a_config_without_a_syntax_is_rejected(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    write_source(src, "qwen3", {"w": torch.ones(1)}, {"chat_template": TEMPLATE})
    with pytest.raises(PulsarLMModelLoadingError, match="run pulsarlm models migrate"):
        migrate(src, tmp_path / "model", {"think_syntax": "none"})
    assert not (tmp_path / "model").exists()


def test_migrating_a_model_directory_in_place_keeps_its_weights(tmp_path: Path):
    directory = built_model(tmp_path)
    shard = directory / "weights" / "model.safetensors"
    inode = shard.stat().st_ino
    checksum = json.loads((directory / CHECKSUMS).read_text())[
        "weights/model.safetensors"
    ]
    migrate(
        directory, directory, {"tool_call_syntax": "qwen_json", "think_syntax": "qwen"}
    )

    assert shard.stat().st_ino == inode
    assert (
        json.loads((directory / CHECKSUMS).read_text())["weights/model.safetensors"]
        == checksum
    )
    assert load_config(directory).tool_call_syntax == "qwen_json"
    assert sorted(built_weights(directory)) == ["b", "w"]
    assert (directory / "chat_template.jinja").read_text() == TEMPLATE
    assert sorted(path.name for path in tmp_path.iterdir()) == ["model", "src"]
