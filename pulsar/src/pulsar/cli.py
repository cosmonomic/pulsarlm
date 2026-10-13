import importlib
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    help="LLM inference engine with unbounded context.",
    no_args_is_help=True,
    add_completion=False,
)


class DType(StrEnum):
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"


@app.callback()
def main() -> None:
    pass


@app.command()
def serve(
    model: Annotated[str, typer.Argument(help="The name of a .pulsar model.")],
    host: Annotated[str, typer.Option(help="Address to listen on.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data_dir",
            help="Searched for the model first, then PULSAR_DATA_DIR, then the XDG "
            "data directories.",
        ),
    ] = None,
    device: Annotated[str, typer.Option(help="Torch device.")] = "cuda",
    dtype: Annotated[
        DType, typer.Option(help="Weight and KV cache dtype.")
    ] = DType.bfloat16,
    kv_cache_gib: Annotated[
        float,
        typer.Option(
            "--kv_cache_gib", help="KV cache size in GiB.", rich_help_panel="Memory"
        ),
    ] = 8.0,
    prefill_tokens_per_step: Annotated[
        int,
        typer.Option(
            "--prefill_tokens_per_step",
            help="Prefill tokens a step forwards across requests.",
            rich_help_panel="Memory",
        ),
    ] = 256,
    stored_states: Annotated[
        int,
        typer.Option(
            "--stored_states",
            help="States the local store keeps.",
            rich_help_panel="Memory",
        ),
    ] = 4096,
    cycle_tokens: Annotated[
        int,
        typer.Option(
            "--cycle_tokens",
            help="Tokens per eviction cycle.",
            rich_help_panel="Eviction",
        ),
    ] = 512,
    n_working: Annotated[
        int,
        typer.Option(
            "--n_working",
            help="Newest pages eviction never touches.",
            rich_help_panel="Eviction",
        ),
    ] = 4,
    evict_tau: Annotated[
        float,
        typer.Option(
            "--evict_tau",
            help="Eviction bar tau / (1 - tau), a multiple of uniform attention.",
            rich_help_panel="Eviction",
        ),
    ] = 0.3,
    max_pages: Annotated[
        int | None,
        typer.Option(
            "--max_pages",
            help="View cap in pages. Defaults to the model's trained context.",
            rich_help_panel="Eviction",
        ),
    ] = None,
) -> None:
    """Serve a model over the OpenAI Chat Completions API."""
    # Imported here: torch takes seconds to load, which --help does not need.
    server = importlib.import_module("pulsar.server.serve")
    try:
        server.serve(
            model,
            host=host,
            port=port,
            data_dir=data_dir,
            device=device,
            dtype=dtype,
            kv_cache_gib=kv_cache_gib,
            prefill_tokens_per_step=prefill_tokens_per_step,
            stored_states=stored_states,
            cycle_tokens=cycle_tokens,
            n_working=n_working,
            evict_tau=evict_tau,
            max_pages=max_pages,
        )
    except server.PulsarModelLoadingError as error:
        typer.secho(f"error: {error}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from error


@app.command()
def download(
    repo_id: Annotated[
        str, typer.Argument(help="A Hugging Face model, such as Qwen/Qwen3-8B.")
    ],
    name: Annotated[
        str | None,
        typer.Option(
            help="The model's name in pulsar. Defaults to the repository's name, lowercased."
        ),
    ] = None,
    revision: Annotated[
        str | None, typer.Option(help="A branch, tag or commit of the repository.")
    ] = None,
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data_dir",
            help="Where the archive goes. Defaults to PULSAR_DATA_DIR, then the XDG data home.",
        ),
    ] = None,
) -> None:
    """Download a Hugging Face model and pack it into a .pulsar archive."""
    # Imported here: torch takes seconds to load, which --help does not need.
    downloading = importlib.import_module("pulsar.download")
    archive = downloading.download(
        repo_id, name=name, revision=revision, data_dir=data_dir
    )
    typer.secho(f"wrote {archive}", fg=typer.colors.GREEN)


@app.command()
def migrate(
    source: Annotated[
        Path,
        typer.Argument(
            help="A Hugging Face checkpoint directory.",
            exists=True,
            file_okay=False,
        ),
    ],
    name: Annotated[
        str | None,
        typer.Option(
            help="The model's name in pulsar. Defaults to the directory's name, lowercased."
        ),
    ] = None,
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data_dir",
            help="Where the archive goes. Defaults to PULSAR_DATA_DIR, then the XDG data home.",
        ),
    ] = None,
) -> None:
    """Pack a Hugging Face checkpoint directory into a .pulsar archive."""
    # Imported here: torch takes seconds to load, which --help does not need.
    checkpoint = importlib.import_module("pulsar.checkpoint")
    migrating = importlib.import_module("pulsar.migrate")
    archive = checkpoint.new_model_path(name or source.name.lower(), data_dir)
    migrating.migrate(source, archive)
    typer.secho(f"wrote {archive}", fg=typer.colors.GREEN)
