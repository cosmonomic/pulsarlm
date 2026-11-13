import hashlib
import shutil
from pathlib import Path

from huggingface_hub import model_info, snapshot_download

from pulsarlm.checkpoint import PulsarLMModelLoadingError, new_model_path
from pulsarlm.migrate import UNSHARDED_FILES, migrate, weight_shards
from pulsarlm.models import MODELS, hugging_face_repository


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
    migrates it into the model directory and removes it. Returns the model
    directory's path. The download directory keeps its name across attempts, so
    an interrupted download resumes."""
    dst = new_model_path(name or model, data_dir)
    if dst.exists():
        raise PulsarLMModelLoadingError(f"{dst} exists")
    dst.parent.mkdir(parents=True, exist_ok=True)
    source = MODELS[model]
    repository = hugging_face_repository(source.url)
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
    migrate(src, dst, source.config_overrides)
    shutil.rmtree(src)
    return dst
