# Cosmonomic / PulsarLM

An LLM inference engine for persistent chat bots and agents.
It allows models to forget old and unimportant tokens
to keep context small and inference fast.
Best when paired with a memory system
for recalling forgotten context.

States are kept in an LRU cache in pinned host memory
(`--store_gib`, default 4 GiB),
which `--persist` also saves to disk.

## How it works

PulsarLM is inspired by how short-term memory forgets.
During inference, custom attention kernels
accumulate the attention mass of each token page.
At regular intervals, pages that received little attention
are removed from the context,
while the system prompt and the most recent pages are always kept.
Forgotten information can be recovered
by pairing PulsarLM with an agent harness that recalls memories.
This keeps the context window bounded,
reducing the hardware needed for very long sessions.

Removing past tokens from the context breaks the prefix cache
that most engines rely on.
To work around this, PulsarLM stores snapshots of the context
at regular intervals in an external store,
and resumes a conversation from its latest snapshot.

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

