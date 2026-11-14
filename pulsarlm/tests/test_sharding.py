"""pulsarlm.sharding on tiny checkpoints written as safetensors, and a Qwen3.5
engine split over two ranks on one GPU against the same model on one GPU."""

import asyncio
import json
import zipfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import Tensor

from pulsarlm import checkpoint, sharding
from pulsarlm.checkpoint import PulsarLMModelConfig, PulsarLMModelLoadingError
from pulsarlm.safetensors_file import SafetensorsFile
from pulsarlm.core.engine import Engine, RequestResult, State
from pulsarlm.engine_model import engine_model

PREFIX = "model.language_model."
SYNTAXES = {"tool_call_syntax": "none", "think_syntax": "none"}
QWEN3_5_TEXT = {
    "vocab_size": 300,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "layer_types": ["linear_attention", "full_attention"],
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "rope_theta": 1e6,
    "partial_rotary_factor": 0.25,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 32,
    "linear_conv_kernel_dim": 4,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 4096,
}


def model_config(
    model_type: str = "qwen3_5",
    quantization: dict | None = None,
    **text,
) -> PulsarLMModelConfig:
    raw = {
        "model_type": model_type,
        "eos_token_id": 0,
        "text_config": QWEN3_5_TEXT | text,
    } | SYNTAXES
    if quantization is not None:
        raw["quantization_config"] = quantization
    return PulsarLMModelConfig.from_dict(raw)


def qwen3_5_weights(cfg: PulsarLMModelConfig) -> dict[str, Tensor]:
    text = cfg.text
    generator = torch.Generator().manual_seed(0)

    def randn(*shape: int, scale: float = 0.1) -> Tensor:
        return torch.randn(*shape, generator=generator) * scale

    hidden, intermediate, vocab = (
        text["hidden_size"],
        text["intermediate_size"],
        text["vocab_size"],
    )
    n_heads, n_kv_heads, head_dim = (
        text["num_attention_heads"],
        text["num_key_value_heads"],
        text["head_dim"],
    )
    key_dim = text["linear_num_key_heads"] * text["linear_key_head_dim"]
    value_heads = text["linear_num_value_heads"]
    value_dim = value_heads * text["linear_value_head_dim"]
    channels = 2 * key_dim + value_dim
    weights = {
        PREFIX + "embed_tokens.weight": randn(vocab, hidden, scale=1.0),
        PREFIX + "norm.weight": randn(hidden),
        "lm_head.weight": randn(vocab, hidden, scale=0.5),
    }
    for layer, layer_type in enumerate(text["layer_types"]):
        p = f"{PREFIX}layers.{layer}."
        weights |= {
            p + "input_layernorm.weight": randn(hidden),
            p + "post_attention_layernorm.weight": randn(hidden),
            p + "mlp.gate_proj.weight": randn(intermediate, hidden),
            p + "mlp.up_proj.weight": randn(intermediate, hidden),
            p + "mlp.down_proj.weight": randn(hidden, intermediate),
        }
        if layer_type == "full_attention":
            weights |= {
                p + "self_attn.q_proj.weight": randn(n_heads * 2 * head_dim, hidden),
                p + "self_attn.k_proj.weight": randn(n_kv_heads * head_dim, hidden),
                p + "self_attn.v_proj.weight": randn(n_kv_heads * head_dim, hidden),
                p + "self_attn.o_proj.weight": randn(hidden, n_heads * head_dim),
                p + "self_attn.q_norm.weight": randn(head_dim),
                p + "self_attn.k_norm.weight": randn(head_dim),
            }
        else:
            weights |= {
                p + "linear_attn.in_proj_qkv.weight": randn(channels, hidden),
                p + "linear_attn.in_proj_z.weight": randn(value_dim, hidden),
                p + "linear_attn.in_proj_b.weight": randn(value_heads, hidden),
                p + "linear_attn.in_proj_a.weight": randn(value_heads, hidden),
                p + "linear_attn.conv1d.weight": randn(channels, 1, 4, scale=0.5),
                p + "linear_attn.dt_bias": randn(value_heads, scale=1.0),
                p + "linear_attn.A_log": torch.log(
                    torch.rand(value_heads, generator=generator) * 15.0 + 0.5
                ),
                p + "linear_attn.norm.weight": 1.0
                + randn(text["linear_value_head_dim"]),
                p + "linear_attn.out_proj.weight": randn(hidden, value_dim),
            }
    return weights


def write_files(directory: Path, weights: dict[str, Tensor]) -> list[Path]:
    """The weights in two files, every other name in each."""
    names = sorted(weights)
    paths = [directory / "a.safetensors", directory / "b.safetensors"]
    for i, path in enumerate(paths):
        save_file({name: weights[name].contiguous() for name in names[i::2]}, path)
    return paths


def rank_parts(
    paths: list[Path], cfg: PulsarLMModelConfig, size: int
) -> list[dict[str, Tensor]]:
    files = [SafetensorsFile.of_file(path) for path in paths]
    return [dict(sharding.rank_weights(files, cfg, rank, size)) for rank in range(size)]


def bytes_of(tensor: Tensor) -> Tensor:
    return tensor.view(torch.uint8) if tensor.dtype == torch.float8_e4m3fn else tensor


def joined_q_k_v(parts: list[Tensor], key_dim: int, value_dim: int) -> Tensor:
    """The full in_proj_qkv or conv1d rows from each rank's q, k and v rows."""
    size = len(parts)
    pieces = [
        torch.split(part, [key_dim // size, key_dim // size, value_dim // size])
        for part in parts
    ]
    return torch.cat([piece[i] for i in range(3) for piece in pieces])


def check_joins_back(
    weights: dict[str, Tensor], parts: list[dict[str, Tensor]], cfg: PulsarLMModelConfig
) -> None:
    text = cfg.text
    key_dim = text["linear_num_key_heads"] * text["linear_key_head_dim"]
    value_dim = text["linear_num_value_heads"] * text["linear_value_head_dim"]
    by_rows = (
        "q_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "in_proj_z.weight",
        "in_proj_b.weight",
        "in_proj_a.weight",
        "dt_bias",
        "A_log",
    )
    by_columns = ("o_proj.weight", "mlp.down_proj.weight", "out_proj.weight")
    assert all(part.keys() == weights.keys() for part in parts)
    for name, full in weights.items():
        held = [part[name] for part in parts]
        if name.endswith(("embed_tokens.weight", "lm_head.weight")):
            joined = torch.cat(held)
            assert joined.shape[0] == 320
            assert torch.equal(joined[:300], full)
            assert not joined[300:].any()
        elif name.endswith(by_rows) or (
            name.endswith(("k_proj.weight", "v_proj.weight"))
            and text["num_key_value_heads"] >= len(parts)
        ):
            assert torch.equal(torch.cat(held), full), name
        elif name.endswith(by_columns):
            assert torch.equal(torch.cat(held, 1), full), name
        elif name.endswith(("in_proj_qkv.weight", "conv1d.weight")):
            assert torch.equal(joined_q_k_v(held, key_dim, value_dim), full), name
        else:
            assert all(torch.equal(part, full) for part in held), name


def test_each_rank_s_part_joins_back_to_the_checkpoint(tmp_path: Path):
    cfg = model_config()
    weights = qwen3_5_weights(cfg)
    check_joins_back(weights, rank_parts(write_files(tmp_path, weights), cfg, 2), cfg)


def test_a_kv_head_fewer_than_the_ranks_is_whole_on_each(tmp_path: Path):
    cfg = model_config(num_key_value_heads=1)
    weights = qwen3_5_weights(cfg)
    parts = rank_parts(write_files(tmp_path, weights), cfg, 2)
    check_joins_back(weights, parts, cfg)
    k_proj = PREFIX + "layers.1.self_attn.k_proj.weight"
    assert all(torch.equal(part[k_proj], weights[k_proj]) for part in parts)


FP8 = {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [32, 32]}


def fp8_weights(cfg: PulsarLMModelConfig) -> dict[str, Tensor]:
    """The MLP of layer 0 quantized: gate_proj [128, 64] and down_proj [64, 128]
    in 32 x 32 blocks."""
    weights = qwen3_5_weights(cfg)
    p = PREFIX + "layers.0.mlp."
    for name in ("gate_proj.weight", "down_proj.weight"):
        weight = weights[p + name]
        weights[p + name] = weight.to(torch.float8_e4m3fn)
        weights[p + name + "_scale_inv"] = torch.rand(
            weight.shape[0] // 32, weight.shape[1] // 32
        )
    return weights


def test_an_fp8_weight_s_scales_split_along_its_blocks(tmp_path: Path):
    cfg = model_config(quantization=FP8)
    weights = fp8_weights(cfg)
    parts = rank_parts(write_files(tmp_path, weights), cfg, 2)
    p = PREFIX + "layers.0.mlp."
    gate, down = p + "gate_proj.weight", p + "down_proj.weight"
    assert all(part[gate].dtype == torch.float8_e4m3fn for part in parts)
    assert torch.equal(
        torch.cat([bytes_of(part[gate]) for part in parts]), bytes_of(weights[gate])
    )
    assert torch.equal(
        torch.cat([bytes_of(part[down]) for part in parts], 1), bytes_of(weights[down])
    )
    assert torch.equal(
        torch.cat([part[gate + "_scale_inv"] for part in parts]),
        weights[gate + "_scale_inv"],
    )
    assert torch.equal(
        torch.cat([part[down + "_scale_inv"] for part in parts], 1),
        weights[down + "_scale_inv"],
    )


def test_an_fp8_part_of_partial_blocks_is_rejected(tmp_path: Path):
    cfg = model_config(quantization=FP8 | {"weight_block_size": [128, 128]})
    paths = write_files(tmp_path, fp8_weights(cfg))
    with pytest.raises(PulsarLMModelLoadingError, match="whole blocks"):
        list(
            sharding.rank_weights(
                [SafetensorsFile.of_file(p) for p in paths], cfg, 0, 2
            )
        )


def test_qwen4_exp_splits_experts_and_the_n_gram_table_across_its_shards(
    tmp_path: Path,
):
    cfg = model_config("qwen4_exp")
    generator = torch.Generator().manual_seed(1)
    p = PREFIX + "layers.0."
    table = p + "ple.ple_embedding.ngram_embedding."
    gate_up = torch.randn(2, 2 * 8, 64, generator=generator)
    down = torch.randn(2, 64, 8, generator=generator)
    shards = [
        torch.randn(6, 16, generator=generator),
        torch.randn(4, 16, generator=generator),
    ]
    router = torch.randn(2, 64, generator=generator)
    save_file(
        {table + "shard_0.weight": shards[0], p + "mlp.experts.gate_up_proj": gate_up},
        tmp_path / "a.safetensors",
    )
    save_file(
        {
            table + "shard_1.weight": shards[1],
            p + "mlp.experts.down_proj": down,
            p + "mlp.gate.weight": router,
        },
        tmp_path / "b.safetensors",
    )
    parts = rank_parts([tmp_path / "a.safetensors", tmp_path / "b.safetensors"], cfg, 2)
    assert all(part[table + "shard_0.weight"].shape == (5, 16) for part in parts)
    assert all(table + "shard_1.weight" not in part for part in parts)
    assert torch.equal(
        torch.cat([part[table + "shard_0.weight"] for part in parts]), torch.cat(shards)
    )
    held = [part[p + "mlp.experts.gate_up_proj"] for part in parts]
    assert torch.equal(torch.cat([h[:, :4] for h in held], 1), gate_up[:, :8])
    assert torch.equal(torch.cat([h[:, 4:] for h in held], 1), gate_up[:, 8:])
    assert torch.equal(
        torch.cat([part[p + "mlp.experts.down_proj"] for part in parts], 2), down
    )
    assert all(torch.equal(part[p + "mlp.gate.weight"], router) for part in parts)


def test_an_architecture_without_a_split_is_rejected(tmp_path: Path):
    cfg = PulsarLMModelConfig.from_dict(
        {"model_type": "qwen3", "eos_token_id": 0} | QWEN3_5_TEXT | SYNTAXES
    )
    paths = write_files(tmp_path, {"model.norm.weight": torch.ones(4)})
    with pytest.raises(PulsarLMModelLoadingError, match="do not split"):
        list(
            sharding.rank_weights(
                [SafetensorsFile.of_file(p) for p in paths], cfg, 0, 2
            )
        )


def write_archive(path: Path, cfg_raw: dict, weights: dict[str, Tensor]) -> list[Path]:
    """The paths of the safetensors files the archive holds."""
    files = path.parent / "files"
    files.mkdir()
    paths = write_files(files, weights)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("config.json", json.dumps(cfg_raw))
        for file in paths:
            archive.write(file, f"weights/{file.name}")
    return paths


@pytest.mark.parametrize("form", ["archive", "directory"])
def test_load_weights_reads_one_rank_s_part_of_a_model(tmp_path: Path, form: str):
    cfg = model_config()
    weights = qwen3_5_weights(cfg)
    raw = {
        "model_type": "qwen3_5",
        "eos_token_id": 0,
        "text_config": QWEN3_5_TEXT,
    } | SYNTAXES
    if form == "archive":
        path = tmp_path / "model.pulsarlm"
        paths = write_archive(path, raw, weights)
    else:
        path = tmp_path / "model"
        (path / "weights").mkdir(parents=True)
        paths = write_files(path / "weights", weights)
        (path / "config.json").write_text(json.dumps(raw))
    loaded = checkpoint.load_weights(path, device=torch.device("cpu"), shard=(1, 2))
    whole = checkpoint.load_weights(path, device=torch.device("cpu"))
    expected = rank_parts(paths, cfg, 2)[1]
    assert loaded.keys() == expected.keys()
    assert all(torch.equal(loaded[name], expected[name]) for name in expected)
    assert whole.keys() == weights.keys()
    assert all(torch.equal(whole[name], weights[name]) for name in weights)


class MemoryStore:
    """Keeps every page and the page keys of every state written; finds no state."""

    def __init__(self) -> None:
        self.pages: dict[bytes, Tensor] = {}
        self.written_keys: list[list[bytes]] = []
        self.lease_count = 0

    async def read_pages(self, hashes: list[bytes]) -> list[Tensor | None]:
        return [self.pages.get(h) for h in hashes]

    async def read_state(
        self, tokens: list[int], page_digests: list[bytes]
    ) -> tuple[State, int]:
        self.lease_count += 1
        return State(), self.lease_count

    def claim(self, hashes: list[bytes], owners: list) -> list[int]:
        return [i for i, h in enumerate(hashes) if h not in self.pages]

    def write_pages(self, hashes: list[bytes], pages: list, owners: list) -> None:
        self.pages |= zip(hashes, pages)

    def write_state(self, lease_id: int, state: State, page_keys: list[bytes]) -> None:
        self.written_keys.append(page_keys)

    def release(self, lease_id: int) -> None:
        pass


async def generate(
    engine: Engine, prompt: list[int], max_tokens: int
) -> tuple[list[int], State]:
    results: list[RequestResult] = []
    request = engine.submit(prompt, max_tokens=max_tokens, on_end=results.append)
    loop = asyncio.get_running_loop()
    woke = asyncio.Event()
    loop.add_reader(engine.fileno(), woke.set)
    tokens: list[int] = []
    try:
        while True:
            woke.clear()
            engine.clear_wake()
            taken = request.take()
            if taken is None:
                return tokens, results[0].state
            tokens += taken
            if not taken:
                await woke.wait()
    finally:
        loop.remove_reader(engine.fileno())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("n_kv_heads", [2, 1])
def test_an_engine_split_over_two_ranks_generates_what_one_gpu_does(
    tmp_path: Path, n_kv_heads: int
):
    cfg = model_config(num_key_value_heads=n_kv_heads)
    weights = qwen3_5_weights(cfg)
    parts = rank_parts(write_files(tmp_path, weights), cfg, 2)
    device = torch.device("cuda", 0)
    prompt = [(7 * i + 3) % 300 for i in range(40)]

    async def main():
        loop = asyncio.get_running_loop()
        settings = dict(
            model_name="split-qwen3.5",
            model_hash="1",
            page_size=16,
            kv_dtype=torch.float32,
            kv_cache_gib=1 / 64,
            store_gib=0.0,
            stop_ids=[],
            prefill_tokens_per_step=32,
        )
        one_gpu_store = MemoryStore()
        one_gpu = Engine(
            one_gpu_store,
            loop,
            engine_model(cfg),
            lambda rank, size: {name: t.to(device) for name, t in weights.items()},
            devices=[device],
            **settings,
        )
        loaded: list[int] = []

        def load_model_shard(rank: int, size: int) -> dict[str, Tensor]:
            assert size == 2
            loaded.append(rank)
            return {name: t.to(device) for name, t in parts[rank].items()}

        split_store = MemoryStore()
        split = Engine(
            split_store,
            loop,
            engine_model(cfg),
            load_model_shard,
            devices=[device, device],
            **settings,
        )
        assert loaded == [0, 1]
        expected, expected_end = await generate(one_gpu, prompt, 12)
        tokens, end = await generate(split, prompt, 12)
        assert len(expected) == 12
        assert tokens == expected
        assert end.head == expected_end.head
        assert split_store.written_keys[-1] == one_gpu_store.written_keys[-1]

    asyncio.run(main())
