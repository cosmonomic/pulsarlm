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
    xhigh = "xhigh"


class DType(StrEnum):
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"


ModelName = Annotated[str, typer.Argument(help="The name of a .pulsarlm model.")]
SearchedDataDir = Annotated[
    Path | None,
    typer.Option(
        "--data_dir",
        help="Searched for the model first, then PULSARLM_DATA_DIR, then the XDG "
        "data directories.",
    ),
]
ArchiveDataDir = Annotated[
    Path | None,
    typer.Option(
        "--data_dir",
        help="Where the archive goes. Defaults to PULSARLM_DATA_DIR, then the XDG data home.",
    ),
]
Device = Annotated[str, typer.Option(help="Torch device.")]
WeightDType = Annotated[DType, typer.Option(help="Weight and KV cache dtype.")]
KvCacheGib = Annotated[
    float,
    typer.Option(
        "--kv_cache_gib", help="KV cache size in GiB.", rich_help_panel="Memory"
    ),
]
PrefillTokensPerStep = Annotated[
    int,
    typer.Option(
        "--prefill_tokens_per_step",
        help="Prefill tokens a step forwards across requests.",
        rich_help_panel="Memory",
    ),
]
StoreGib = Annotated[
    float,
    typer.Option(
        "--store_gib",
        help="Pinned host memory the store keeps pages in, in GiB.",
        rich_help_panel="Memory",
    ),
]
Persist = Annotated[
    bool,
    typer.Option(
        "--persist/--no_persist",
        help="Keep the store in store.sqlite under the data directory.",
        rich_help_panel="Memory",
    ),
]
CycleTokens = Annotated[
    int,
    typer.Option(
        "--cycle_tokens",
        help="Tokens per eviction cycle.",
        rich_help_panel="Eviction",
    ),
]
NWorking = Annotated[
    int,
    typer.Option(
        "--n_working",
        help="Newest pages eviction never touches.",
        rich_help_panel="Eviction",
    ),
]
EvictionTarget = Annotated[
    float,
    typer.Option(
        "--eviction_target",
        help="Fraction of the model's context a layer's pages settle under; 1 is off.",
        rich_help_panel="Eviction",
    ),
]
MaxPages = Annotated[
    int | None,
    typer.Option(
        "--max_pages",
        help="View cap in pages. Defaults to the model's trained context.",
        rich_help_panel="Eviction",
    ),
]


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
    model: ModelName,
    host: Annotated[str, typer.Option(help="Address to listen on.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
    data_dir: SearchedDataDir = None,
    device: Device = "cuda",
    dtype: WeightDType = DType.bfloat16,
    kv_cache_gib: KvCacheGib = 4.0,
    prefill_tokens_per_step: PrefillTokensPerStep = 256,
    store_gib: StoreGib = 4.0,
    persist: Persist = False,
    cycle_tokens: CycleTokens = 512,
    n_working: NWorking = 512,
    eviction_target: EvictionTarget = 0.8,
    max_pages: MaxPages = None,
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
                prefill_tokens_per_step=prefill_tokens_per_step,
                store_gib=store_gib,
                persist=persist,
                cycle_tokens=cycle_tokens,
                n_working=n_working,
                eviction_target=eviction_target,
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
    model: ModelName,
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
    top_k: Annotated[
        int,
        typer.Option(
            "--top_k",
            min=0,
            help="Samples from the top_k most probable tokens; 0 keeps every token.",
        ),
    ] = 0,
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
    data_dir: SearchedDataDir = None,
    device: Device = "cuda",
    dtype: WeightDType = DType.bfloat16,
    kv_cache_gib: KvCacheGib = 4.0,
    prefill_tokens_per_step: PrefillTokensPerStep = 256,
    store_gib: StoreGib = 4.0,
    persist: Persist = False,
    cycle_tokens: CycleTokens = 512,
    n_working: NWorking = 512,
    eviction_target: EvictionTarget = 0.8,
    max_pages: MaxPages = None,
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
                prefill_tokens_per_step=prefill_tokens_per_step,
                store_gib=store_gib,
                persist=persist,
                cycle_tokens=cycle_tokens,
                n_working=n_working,
                eviction_target=eviction_target,
                max_pages=max_pages,
            ),
            system=system,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
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
    data_dir: ArchiveDataDir = None,
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
    data_dir: ArchiveDataDir = None,
) -> None:
    """Pack a Hugging Face checkpoint directory into a .pulsarlm archive."""
    # Imported here: torch takes seconds to load, which --help does not need.
    checkpoint = importlib.import_module("pulsarlm.checkpoint")
    migrating = importlib.import_module("pulsarlm.migrate")
    archive = checkpoint.new_model_path(name or source.name.lower(), data_dir)
    migrating.migrate(source, archive)
    typer.secho(f"wrote {archive}", fg=typer.colors.GREEN)
