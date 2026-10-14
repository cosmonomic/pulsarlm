import importlib
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from pulsarlm.models import REPOSITORIES, repository_url
from pulsarlm.settings import Settings

app = typer.Typer(
    help="LLM inference engine with unbounded context.",
    no_args_is_help=True,
    add_completion=False,
)


class ReasoningEffort(StrEnum):
    none = "none"
    minimal = "minimal"
    low = "low"
    medium = "medium"
    high = "high"


class DType(StrEnum):
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"


def _positive(value: float) -> float:
    if value <= 0.0:
        raise typer.BadParameter(f"{value} is not positive.")
    return value


def _known_model(model: str) -> str:
    if model not in REPOSITORIES:
        raise typer.BadParameter(
            f"{model} is not a known model; `pulsarlm models` lists them."
        )
    return model


@app.callback()
def main() -> None:
    pass


@app.command()
def serve(
    model: Annotated[str, typer.Argument(help="The name of a .pulsarlm model.")],
    host: Annotated[str, typer.Option(help="Address to listen on.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data_dir",
            help="Searched for the model first, then PULSARLM_DATA_DIR, then the XDG "
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
    ] = 4.0,
    staging_gib: Annotated[
        float,
        typer.Option(
            "--staging_gib",
            help="Pinned host buffer for transfers between the KV cache and the store, in GiB.",
            rich_help_panel="Memory",
        ),
    ] = 0.25,
    prefill_tokens_per_step: Annotated[
        int,
        typer.Option(
            "--prefill_tokens_per_step",
            help="Prefill tokens a step forwards across requests.",
            rich_help_panel="Memory",
        ),
    ] = 256,
    store_gib: Annotated[
        float,
        typer.Option(
            "--store_gib",
            help="Host memory the local store keeps pages in, in GiB.",
            rich_help_panel="Memory",
        ),
    ] = 16.0,
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
    ] = 512,
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
    server = importlib.import_module("pulsarlm.cli.serve")
    try:
        server.serve(
            Settings(
                model=model,
                data_dir=data_dir,
                device=device,
                dtype=dtype,
                kv_cache_gib=kv_cache_gib,
                staging_gib=staging_gib,
                prefill_tokens_per_step=prefill_tokens_per_step,
                store_gib=store_gib,
                cycle_tokens=cycle_tokens,
                n_working=n_working,
                evict_tau=evict_tau,
                max_pages=max_pages,
            ),
            host=host,
            port=port,
        )
    except server.PulsarLMModelLoadingError as error:
        typer.secho(f"error: {error}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from error


@app.command()
def chat(
    model: Annotated[str, typer.Argument(help="The name of a .pulsarlm model.")],
    system: Annotated[
        str | None,
        typer.Option(
            envvar="PULSARLM_SYSTEM_PROMPT",
            help="A system message opening the conversation; its pages are the sink.",
        ),
    ] = None,
    temperature: Annotated[
        float,
        typer.Option(
            min=0.0, max=2.0, help="Sampling temperature; 0 takes the argmax."
        ),
    ] = 1.0,
    top_p: Annotated[
        float,
        typer.Option(
            "--top_p",
            max=1.0,
            callback=_positive,
            help="Samples from the smallest set of tokens whose probability reaches top_p.",
        ),
    ] = 1.0,
    reasoning_effort: Annotated[
        ReasoningEffort,
        typer.Option(
            "--reasoning_effort",
            help="Passed to the chat template as reasoning_effort, and as "
            "enable_thinking unless none.",
        ),
    ] = ReasoningEffort.medium,
    show_thinking: Annotated[
        bool,
        typer.Option(
            "--show_thinking",
            help="Print the model's think blocks instead of hiding them.",
        ),
    ] = False,
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data_dir",
            help="Searched for the model first, then PULSARLM_DATA_DIR, then the XDG "
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
    ] = 4.0,
    staging_gib: Annotated[
        float,
        typer.Option(
            "--staging_gib",
            help="Pinned host buffer for transfers between the KV cache and the store, in GiB.",
            rich_help_panel="Memory",
        ),
    ] = 0.25,
    prefill_tokens_per_step: Annotated[
        int,
        typer.Option(
            "--prefill_tokens_per_step",
            help="Prefill tokens a step forwards across requests.",
            rich_help_panel="Memory",
        ),
    ] = 256,
    store_gib: Annotated[
        float,
        typer.Option(
            "--store_gib",
            help="Host memory the local store keeps pages in, in GiB.",
            rich_help_panel="Memory",
        ),
    ] = 16.0,
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
    ] = 512,
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
    """Chat with a model in the terminal. End input to quit."""
    # Imported here: torch takes seconds to load, which --help does not need.
    chat = importlib.import_module("pulsarlm.cli.chat")
    try:
        chat.new_chat(
            Settings(
                model=model,
                data_dir=data_dir,
                device=device,
                dtype=dtype,
                kv_cache_gib=kv_cache_gib,
                staging_gib=staging_gib,
                prefill_tokens_per_step=prefill_tokens_per_step,
                store_gib=store_gib,
                cycle_tokens=cycle_tokens,
                n_working=n_working,
                evict_tau=evict_tau,
                max_pages=max_pages,
            ),
            system=system,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort.value,
            show_thinking=show_thinking,
        )
    except chat.PulsarLMModelLoadingError as error:
        typer.secho(f"error: {error}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from error
    except chat.EngineStopped as error:
        typer.secho(f"error: engine stopped: {error}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from error


@app.command()
def models() -> None:
    """List the models pulsarlm can download and their Hugging Face repositories."""
    width = max(map(len, REPOSITORIES))
    for model in REPOSITORIES:
        typer.echo(f"{model:<{width}}  {repository_url(model)}")


@app.command()
def download(
    model: Annotated[
        str,
        typer.Argument(
            help="A model from `pulsarlm models`, such as qwen3-8b.",
            callback=_known_model,
        ),
    ],
    name: Annotated[
        str | None,
        typer.Option(help="The model's name in pulsarlm. Defaults to the model."),
    ] = None,
    revision: Annotated[
        str | None, typer.Option(help="A branch, tag or commit of the repository.")
    ] = None,
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data_dir",
            help="Where the archive goes. Defaults to PULSARLM_DATA_DIR, then the XDG data home.",
        ),
    ] = None,
) -> None:
    """Download a Hugging Face model and pack it into a .pulsarlm archive."""
    # Imported here: torch takes seconds to load, which --help does not need.
    downloading = importlib.import_module("pulsarlm.download")
    archive = downloading.download(
        model, name=name, revision=revision, data_dir=data_dir
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
            help="The model's name in pulsarlm. Defaults to the directory's name, lowercased."
        ),
    ] = None,
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data_dir",
            help="Where the archive goes. Defaults to PULSARLM_DATA_DIR, then the XDG data home.",
        ),
    ] = None,
) -> None:
    """Pack a Hugging Face checkpoint directory into a .pulsarlm archive."""
    # Imported here: torch takes seconds to load, which --help does not need.
    checkpoint = importlib.import_module("pulsarlm.checkpoint")
    migrating = importlib.import_module("pulsarlm.migrate")
    archive = checkpoint.new_model_path(name or source.name.lower(), data_dir)
    migrating.migrate(source, archive)
    typer.secho(f"wrote {archive}", fg=typer.colors.GREEN)
