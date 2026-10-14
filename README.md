# Cosmonomic / PulsarLM

An LLM inference engine for persistent chat bots and agents.
It allows models to forget old and unimportant tokens
to keep context small and inference fast.
Best when paired with a memory system
for recalling forgotten context.

States are kept in an LRU cache (default 16 GiB),
which `--persist` also saves to disk.

## Quickstart

```sh
pip install pulsarlm

# list and download implemented models
pulsarlm models
pulsarlm download gemma-4-e2b-it

# start a chat
pulsarlm chat gemma-4-e2b-it

# or start an OpenAI compatible API server
pulsarlm serve gemma-4-e2b-it
```

Support for newer models is on the roadmap!

## Requirements

- Linux, x86_64 or aarch64, glibc >= 2.34 (Ubuntu 22.04+, Debian 12+, RHEL 9+)
- NVIDIA GPU with compute capability >= 8.0 (Ampere or newer)
- NVIDIA driver >= 580 (CUDA 13)
- Python 3.13

If installing from sdist or building from source:

- nvcc >= 13.4
- gcc >= 15
- xmake
- OpenSSL 3 headers

## Development

Requires `uv` and the source build requirements above.
To get started:

```sh
uv sync
```

## Disclaimer

This project has only been tested on x86_64 with an RTX 5090.
Wheels for other GPUs and for aarch64 are built but untested.

