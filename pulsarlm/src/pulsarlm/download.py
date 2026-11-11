import tempfile
from pathlib import Path

from huggingface_hub import model_info, snapshot_download

from pulsarlm.checkpoint import new_model_path
from pulsarlm.migrate import UNSHARDED_FILES, migrate, weight_shards
from pulsarlm.models import REPOSITORIES


def download(
    model: str, *, name: str | None, revision: str | None, data_dir: Path | None
) -> Path:
    """Downloads a Hugging Face model into a directory next to the archive, packs it
    into the archive and deletes the directory. Returns the archive's path."""
    dst = new_model_path(name or model, data_dir)
    dst.parent.mkdir(parents=True, exist_ok=True)
    repository = REPOSITORIES[model]
    commit = model_info(repository, revision=revision).sha
    with tempfile.TemporaryDirectory(dir=dst.parent, prefix=".download-") as src:
        snapshot_download(
            repository,
            revision=commit,
            allow_patterns=UNSHARDED_FILES,
            local_dir=src,
        )
        snapshot_download(
            repository,
            revision=commit,
            allow_patterns=[shard.name for shard in weight_shards(Path(src))],
            local_dir=src,
        )
        migrate(Path(src), dst)
    return dst
