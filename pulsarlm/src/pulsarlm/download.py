import hashlib
import shutil
from pathlib import Path

from huggingface_hub import model_info, snapshot_download

from pulsarlm.checkpoint import PulsarLMModelLoadingError, new_model_path
from pulsarlm.models import REPOSITORIES
from pulsarlm.packing import UNSHARDED_FILES, build, weight_shards


def check_sha256(file: Path, expected: str) -> None:
    """Removes file and raises when its SHA-256 is not expected, so that the next
    download fetches it again."""
    with file.open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != expected:
        file.unlink()
        raise PulsarLMModelLoadingError(
            f"{file.name} does not match the repository's SHA-256; download it again"
        )


def download(
    model: str, *, name: str | None, revision: str | None, data_dir: Path | None
) -> Path:
    """Downloads a Hugging Face model into a .download- directory beside the
    model, checks each weight shard against the SHA-256 the repository records,
    builds the model directory from it and removes it. Returns the model
    directory's path. The download directory keeps its name across attempts, so
    an interrupted download resumes."""
    dst = new_model_path(name or model, data_dir)
    if dst.exists():
        raise PulsarLMModelLoadingError(f"{dst} exists")
    dst.parent.mkdir(parents=True, exist_ok=True)
    repository = REPOSITORIES[model]
    info = model_info(repository, revision=revision, files_metadata=True)
    src = dst.with_name(f".download-{dst.name}")
    snapshot_download(
        repository, revision=info.sha, allow_patterns=UNSHARDED_FILES, local_dir=src
    )
    shards = weight_shards(src)
    snapshot_download(
        repository,
        revision=info.sha,
        allow_patterns=[shard.name for shard in shards],
        local_dir=src,
    )
    sha256s = {
        sibling.rfilename: sibling.lfs.sha256
        for sibling in info.siblings or []
        if sibling.lfs is not None
    }
    for shard in shards:
        check_sha256(shard, sha256s[shard.name])
    build(src, dst)
    shutil.rmtree(src)
    return dst
