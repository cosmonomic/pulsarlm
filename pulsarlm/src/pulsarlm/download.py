import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

from pulsarlm.checkpoint import new_model_path
from pulsarlm.migrate import SOURCE_FILES, migrate


def default_name(repo_id: str) -> str:
    """Qwen/Qwen3-8B is qwen3-8b."""
    return repo_id.rsplit("/", 1)[-1].lower()


def download(
    repo_id: str, *, name: str | None, revision: str | None, data_dir: Path | None
) -> Path:
    """Downloads a Hugging Face model into a directory next to the archive, packs it
    into the archive and deletes the directory. Returns the archive's path."""
    dst = new_model_path(name or default_name(repo_id), data_dir)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dst.parent, prefix=".download-") as src:
        snapshot_download(
            repo_id, revision=revision, allow_patterns=SOURCE_FILES, local_dir=src
        )
        migrate(Path(src), dst)
    return dst
