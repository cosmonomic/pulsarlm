"""The default eviction policy built from the server's settings."""

import pytest

from pulsarlm.checkpoint import PulsarLMModelLoadingError
from pulsarlm.served_model import eviction_policy
from pulsarlm.settings import Settings

MAX_CONTEXT_WINDOW = 4096


def settings(max_context_capacity: int) -> Settings:
    return Settings(
        model="random-qwen3",
        data_dir=None,
        device="cuda",
        tensor_parallel_size=1,
        expert_parallel=False,
        dtype="bfloat16",
        kv_cache_gib=4.0,
        prefill_tokens_per_step=512,
        max_running_requests=64,
        decode_graphs=True,
        store_gib=4.0,
        persist=False,
        prefix_cache=True,
        cycle_tokens=512,
        mass_tokens=None,
        n_working=1024,
        eviction_threshold=0.5,
        eviction_warmup=1,
        max_context_window=MAX_CONTEXT_WINDOW,
        max_context_capacity=max_context_capacity,
        num_draft_tokens=0,
    )


def test_a_capacity_of_0_turns_capacity_eviction_off():
    assert eviction_policy(settings(0), MAX_CONTEXT_WINDOW).max_context_capacity == 0


def test_a_negative_capacity_is_the_window():
    policy = eviction_policy(settings(-1), MAX_CONTEXT_WINDOW)
    assert policy.max_context_capacity == MAX_CONTEXT_WINDOW


def test_a_positive_capacity_is_passed_through():
    policy = eviction_policy(settings(2048), MAX_CONTEXT_WINDOW)
    assert policy.max_context_capacity == 2048
    assert policy.n_working == 1024
    assert policy.eviction_threshold == 0.5


def test_a_capacity_above_the_window_fails_at_start():
    with pytest.raises(PulsarLMModelLoadingError, match="--max_context_capacity"):
        eviction_policy(settings(MAX_CONTEXT_WINDOW + 1), MAX_CONTEXT_WINDOW)
