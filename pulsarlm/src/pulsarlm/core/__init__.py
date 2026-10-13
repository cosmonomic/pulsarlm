"""Imports pulsarlm_core.so, the engine binding (see engine.py)."""

import importlib

import torch  # noqa: F401  (pulsarlm_core.so links against libtorch and its Python casters)

pulsarlm_core = importlib.import_module("pulsarlm.pulsarlm_core")
