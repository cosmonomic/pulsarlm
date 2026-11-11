import pytest

from pulsarlm.checkpoint import PulsarLMModelConfig, PulsarLMModelLoadingError


def test_an_unknown_architecture_is_rejected():
    with pytest.raises(PulsarLMModelLoadingError, match="unsupported architecture"):
        PulsarLMModelConfig.from_dict({"model_type": "mistral", "eos_token_id": 2})


def test_a_single_eos_id_becomes_a_list():
    cfg = PulsarLMModelConfig.from_dict(
        {"model_type": "qwen3", "eos_token_id": 151645, "num_hidden_layers": 36}
    )
    assert cfg.eos_ids == [151645]


def test_a_multimodal_checkpoint_reads_its_text_config():
    raw = {
        "model_type": "gemma4",
        "eos_token_id": [1, 106],
        "text_config": {"num_hidden_layers": 12, "max_position_embeddings": 131072},
    }
    cfg = PulsarLMModelConfig.from_dict(raw)
    assert cfg.n_layers == 12
    assert cfg.max_position_embeddings == 131072
    assert cfg.raw == raw
