import pytest

from pulsarlm.checkpoint import PulsarLMModelConfig, PulsarLMModelLoadingError
from pulsarlm.core.engine import (
    Gemma4Config,
    ModelConfig,
    Qwen3_5Config,
    Qwen3_5MoeConfig,
)
from pulsarlm.engine_model import engine_model, qwen4_exp_config

ROPE = {
    "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
    "full_attention": {
        "rope_type": "proportional",
        "partial_rotary_factor": 0.25,
        "rope_theta": 1000000.0,
    },
}


def gemma4(model_type: str, **text) -> PulsarLMModelConfig:
    base = {
        "vocab_size": 262144,
        "hidden_size": 1536,
        "num_hidden_layers": 12,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "global_head_dim": 512,
        "sliding_window": 512,
        "rope_parameters": ROPE,
        "rms_norm_eps": 1e-6,
        "final_logit_softcapping": 30.0,
        "max_position_embeddings": 131072,
    }
    return PulsarLMModelConfig.from_dict(
        {
            "model_type": model_type,
            "eos_token_id": [1, 106],
            "text_config": base | text,
        }
    )


def qwen3_5(**text) -> PulsarLMModelConfig:
    base = {
        "vocab_size": 248320,
        "hidden_size": 1024,
        "intermediate_size": 3584,
        "num_hidden_layers": 8,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "full_attention_interval": 4,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
        },
        "rms_norm_eps": 1e-6,
    }
    return PulsarLMModelConfig.from_dict(
        {
            "model_type": "qwen3_5_moe" if "num_experts" in text else "qwen3_5",
            "eos_token_id": [248044],
            "text_config": base | text,
        }
    )


def test_a_qwen3_5_checkpoint_maps_its_text_config():
    model = engine_model(qwen3_5())

    assert isinstance(model, Qwen3_5Config)
    assert model.layer_types == (["linear_attention"] * 3 + ["full_attention"]) * 2


def test_a_qwen3_5_moe_checkpoint_maps_its_experts_per_token():
    model = engine_model(
        qwen3_5(num_experts=256, num_experts_per_tok=8, moe_intermediate_size=512)
    )

    assert isinstance(model, Qwen3_5MoeConfig)
    assert model.num_experts_per_tok == 8


def test_a_qwen3_5_checkpoint_without_an_output_gate_is_rejected():
    with pytest.raises(PulsarLMModelLoadingError, match="output gate"):
        engine_model(qwen3_5(attn_output_gate=False))


def test_a_qwen3_5_checkpoint_with_a_sigmoid_linear_attention_gate_is_rejected():
    with pytest.raises(PulsarLMModelLoadingError, match="output gate sigmoid"):
        engine_model(qwen3_5(output_gate_type="sigmoid"))


def qwen4_exp_text(**text) -> dict:
    return {
        "vocab_size": 248320,
        "hidden_size": 2560,
        "num_hidden_layers": 8,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 2,
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "hc_count": 4,
        "ple_layer_ids": [2],
        "ngram_size": 3,
        "output_gate_type": "sigmoid",
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
        },
        "rms_norm_eps": 1e-6,
        "eos_token_id": 248044,
    } | text


def test_a_qwen4_exp_text_config_maps_its_ple_layers_and_eos():
    model = qwen4_exp_config(qwen4_exp_text())

    assert model.ple_layers == [1]
    assert model.eos_token_id == 248044


def test_a_qwen4_exp_checkpoint_is_not_loaded_without_sparse_attention():
    cfg = PulsarLMModelConfig.from_dict(
        {
            "model_type": "qwen4_exp",
            "eos_token_id": [248046, 248044],
            "text_config": qwen4_exp_text(),
        }
    )
    with pytest.raises(PulsarLMModelLoadingError, match="sparse attention"):
        engine_model(cfg)


def test_a_gemma4_checkpoint_maps_its_text_config():
    cfg = gemma4("gemma4", num_kv_shared_layers=4, hidden_size_per_layer_input=256)
    model = engine_model(cfg)

    assert isinstance(model, Gemma4Config)
    assert cfg.n_layers == 12 and cfg.max_position_embeddings == 131072
    assert model.layer_types == (["sliding_attention"] * 5 + ["full_attention"]) * 2
    assert model.global_n_kv_heads == 1
    assert model.hidden_size_per_layer_input == 256
    assert model.full_rope.partial_rotary_factor == 0.25
    assert model.full_rope.theta == 1e6
    assert model.sliding_rope.theta == 1e4


def test_k_eq_v_takes_the_global_kv_heads_and_the_last_layer_is_full():
    model = engine_model(
        gemma4(
            "gemma4_unified",
            attention_k_eq_v=True,
            num_global_key_value_heads=2,
            layer_types=["sliding_attention"] * 12,
        )
    )

    assert isinstance(model, Gemma4Config)
    assert model.global_n_kv_heads == 2
    assert model.layer_types[-1] == "full_attention"
    assert model.hidden_size_per_layer_input == 0


def test_global_kv_heads_apply_only_with_k_eq_v():
    model = engine_model(gemma4("gemma4", num_global_key_value_heads=4))
    assert isinstance(model, Gemma4Config)
    assert model.global_n_kv_heads == 1


def test_an_unsupported_gemma4_config_is_rejected():
    with pytest.raises(PulsarLMModelLoadingError):
        engine_model(gemma4("gemma4", use_bidirectional_attention="all"))
    scaled = ROPE | {"full_attention": ROPE["full_attention"] | {"factor": 8.0}}
    with pytest.raises(PulsarLMModelLoadingError):
        engine_model(gemma4("gemma4", rope_parameters=scaled))


def test_a_qwen3_checkpoint_measures_mass_on_its_top_quarter():
    cfg = PulsarLMModelConfig.from_dict(
        {
            "model_type": "qwen3",
            "eos_token_id": 151645,
            "vocab_size": 151936,
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rope_theta": 1000000,
            "rms_norm_eps": 1e-6,
        }
    )
    assert cfg.eos_ids == [151645]
    assert isinstance(engine_model(cfg), ModelConfig)


def llama(**text) -> PulsarLMModelConfig:
    base = {
        "model_type": "llama",
        "eos_token_id": [128001, 128008, 128009],
        "vocab_size": 128256,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "rope_theta": 500000.0,
        "rope_scaling": None,
        "rms_norm_eps": 1e-5,
    }
    return PulsarLMModelConfig.from_dict(base | text)


LLAMA3_SCALING = {
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3",
}


def test_a_llama_checkpoint_without_rope_scaling_takes_plain_rope():
    model = engine_model(llama())

    assert isinstance(model, ModelConfig)
    assert model.rope.theta == 500000.0
    assert model.rope.llama3_scaling is None


@pytest.mark.parametrize(
    "rope",
    [
        {"rope_scaling": LLAMA3_SCALING},
        {"rope_parameters": LLAMA3_SCALING | {"rope_theta": 500000.0}},
    ],
)
def test_llama3_rope_scaling_is_read_from_either_config_layout(rope):
    model = engine_model(llama(**rope))

    assert isinstance(model, ModelConfig)
    assert model.rope.theta == 500000.0
    scaling = model.rope.llama3_scaling
    assert scaling is not None
    assert scaling.factor == 8.0
    assert scaling.low_freq_factor == 1.0
    assert scaling.high_freq_factor == 4.0
    assert scaling.original_max_position_embeddings == 8192


@pytest.mark.parametrize(
    "text",
    [
        {"rope_scaling": {"type": "yarn", "factor": 4.0}},
        {"rope_scaling": {"rope_type": "dynamic", "factor": 2.0}},
        {"attention_bias": True},
        {"mlp_bias": True},
    ],
)
def test_an_unsupported_llama_config_is_rejected(text):
    with pytest.raises(PulsarLMModelLoadingError):
        engine_model(llama(**text))


def test_an_unknown_architecture_is_rejected():
    with pytest.raises(PulsarLMModelLoadingError, match="unsupported architecture"):
        PulsarLMModelConfig.from_dict({"model_type": "mistral", "eos_token_id": 2})
