"""Imports pulsar_core.so, the engine binding (see engine.py)."""

import importlib

import torch  # noqa: F401  (pulsar_core.so links against libtorch and its Python casters)

pulsar_core = importlib.import_module("pulsar.pulsar_core")
