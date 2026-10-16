"""The engine's model config for a checkpoint's text model."""

from pulsarlm.checkpoint import (
    GEMMA4,
    QWEN3_5,
    PulsarLMModelConfig,
    PulsarLMModelLoadingError,
)
from pulsarlm.core.engine import (
    Gemma4Config,
    Llama3RopeScaling,
    ModelConfig,
    Qwen3_5Config,
    RopeParameters,
)

SLIDING = "sliding_attention"
FULL = "full_attention"


def top_quarter(n_layers: int) -> list[int]:
    return list(range(n_layers - max(1, n_layers // 4), n_layers))


def gemma4_layer_types(text: dict) -> list[str]:
    """Hugging Face's default is five sliding layers to one full layer, and the last
    layer is full in any case."""
    n_layers = text["num_hidden_layers"]
    types = text.get("layer_types") or [
        SLIDING if (i + 1) % 6 else FULL for i in range(n_layers)
    ]
    return types[:-1] + [FULL]


def gemma4_rope(text: dict, layer_type: str) -> RopeParameters:
    parameters = text["rope_parameters"][layer_type]
    if parameters.get("rope_type", "default") not in ("default", "proportional"):
        raise PulsarLMModelLoadingError(f"unsupported RoPE {parameters['rope_type']}")
    if parameters.get("factor", 1.0) != 1.0:
        raise PulsarLMModelLoadingError("unsupported RoPE scaling factor")
    return RopeParameters(
        theta=float(parameters["rope_theta"]),
        partial_rotary_factor=float(parameters.get("partial_rotary_factor", 1.0)),
    )


def gemma4_config(architecture: str, text: dict) -> Gemma4Config:
    if text.get("use_bidirectional_attention") == "all":
        raise PulsarLMModelLoadingError("bidirectional attention is not supported")
    if text.get("hidden_activation", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
        raise PulsarLMModelLoadingError(
            f"unsupported activation {text['hidden_activation']}"
        )
    layer_types = gemma4_layer_types(text)
    k_eq_v = bool(text.get("attention_k_eq_v", False))
    global_kv_heads = text.get("num_global_key_value_heads")
    per_layer_input = text.get(
        "hidden_size_per_layer_input", 256 if architecture == "gemma4" else 0
    )
    full_layers = [i for i, t in enumerate(layer_types) if t == FULL]
    return Gemma4Config(
        vocab_size=text["vocab_size"],
        hidden_size=text["hidden_size"],
        n_heads=text["num_attention_heads"],
        n_kv_heads=text["num_key_value_heads"],
        head_dim=text["head_dim"],
        global_n_kv_heads=global_kv_heads
        if k_eq_v and global_kv_heads is not None
        else text["num_key_value_heads"],
        global_head_dim=text.get("global_head_dim", 512),
        attention_k_eq_v=k_eq_v,
        layer_types=layer_types,
        sliding_window=text["sliding_window"],
        sliding_rope=gemma4_rope(text, SLIDING),
        full_rope=gemma4_rope(text, FULL),
        rms_norm_eps=float(text["rms_norm_eps"]),
        final_logit_softcapping=text.get("final_logit_softcapping"),
        num_kv_shared_layers=text.get("num_kv_shared_layers", 0),
        hidden_size_per_layer_input=per_layer_input or 0,
        enable_moe_block=bool(text.get("enable_moe_block", False)),
        top_k_experts=text.get("top_k_experts") or 0,
        relevance_layers=[i for i in top_quarter(len(layer_types)) if i in full_layers],
    )


def qwen3_5_layer_types(text: dict) -> list[str]:
    """Hugging Face's default is a full-attention layer every
    full_attention_interval layers, the rest linear attention."""
    interval = text.get("full_attention_interval", 4)
    return text.get("layer_types") or [
        FULL if (i + 1) % interval == 0 else "linear_attention"
        for i in range(text["num_hidden_layers"])
    ]


def qwen3_5_config(text: dict) -> Qwen3_5Config:
    if text.get("attention_bias"):
        raise PulsarLMModelLoadingError("unsupported attention bias")
    if text.get("hidden_act", "silu") != "silu":
        raise PulsarLMModelLoadingError(f"unsupported activation {text['hidden_act']}")
    if not text.get("attn_output_gate", True):
        raise PulsarLMModelLoadingError(
            "attention without an output gate is not supported"
        )
    parameters = text.get("rope_parameters") or {}
    if parameters.get("rope_type", "default") != "default":
        raise PulsarLMModelLoadingError(f"unsupported RoPE {parameters['rope_type']}")
    layer_types = qwen3_5_layer_types(text)
    full_layers = [i for i, t in enumerate(layer_types) if t == FULL]
    return Qwen3_5Config(
        vocab_size=text["vocab_size"],
        hidden_size=text["hidden_size"],
        intermediate_size=text["intermediate_size"],
        layer_types=layer_types,
        n_heads=text["num_attention_heads"],
        n_kv_heads=text["num_key_value_heads"],
        head_dim=text["head_dim"],
        rope_theta=float(parameters.get("rope_theta") or text["rope_theta"]),
        partial_rotary_factor=float(
            parameters.get(
                "partial_rotary_factor", text.get("partial_rotary_factor", 1.0)
            )
        ),
        linear_n_key_heads=text["linear_num_key_heads"],
        linear_n_value_heads=text["linear_num_value_heads"],
        linear_key_head_dim=text["linear_key_head_dim"],
        linear_value_head_dim=text["linear_value_head_dim"],
        linear_conv_kernel=text["linear_conv_kernel_dim"],
        rms_norm_eps=float(text["rms_norm_eps"]),
        relevance_layers=[i for i in top_quarter(len(layer_types)) if i in full_layers],
    )


def rope(text: dict) -> RopeParameters:
    """Reads Hugging Face's rope_parameters, or the rope_scaling and rope_theta
    that preceded it."""
    parameters = text.get("rope_parameters") or text.get("rope_scaling") or {}
    theta = float(parameters.get("rope_theta", text.get("rope_theta")))
    match parameters.get("rope_type", parameters.get("type", "default")):
        case "default":
            return RopeParameters(theta=theta)
        case "llama3":
            return RopeParameters(
                theta=theta,
                llama3_scaling=Llama3RopeScaling(
                    factor=float(parameters["factor"]),
                    low_freq_factor=float(parameters["low_freq_factor"]),
                    high_freq_factor=float(parameters["high_freq_factor"]),
                    original_max_position_embeddings=float(
                        parameters["original_max_position_embeddings"]
                    ),
                ),
            )
        case rope_type:
            raise PulsarLMModelLoadingError(f"unsupported RoPE {rope_type}")


def dense_config(architecture: str, text: dict) -> ModelConfig:
    if text.get("attention_bias") or text.get("mlp_bias"):
        raise PulsarLMModelLoadingError("unsupported attention or MLP bias")
    n_heads = text["num_attention_heads"]
    return ModelConfig(
        architecture=architecture,
        n_layers=text["num_hidden_layers"],
        n_heads=n_heads,
        n_kv_heads=text["num_key_value_heads"],
        head_dim=text.get("head_dim") or text["hidden_size"] // n_heads,
        hidden=text["hidden_size"],
        intermediate=text["intermediate_size"],
        vocab_size=text["vocab_size"],
        rope=rope(text),
        rms_eps=float(text["rms_norm_eps"]),
        relevance_layers=top_quarter(text["num_hidden_layers"]),
    )


def engine_model(
    cfg: PulsarLMModelConfig,
) -> ModelConfig | Gemma4Config | Qwen3_5Config:
    """Measures attention mass on the full-attention layers of the top quarter."""
    if cfg.architecture in GEMMA4:
        return gemma4_config(cfg.architecture, cfg.text)
    if cfg.architecture in QWEN3_5:
        return qwen3_5_config(cfg.text)
    return dense_config(cfg.architecture, cfg.text)
