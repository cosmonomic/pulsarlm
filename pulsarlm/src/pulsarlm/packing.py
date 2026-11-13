"""Packs PulsarLM model directories into archives and verifies both.

A model is config.json, tokenizer.json, tokenizer_config.json,
chat_template.jinja and weights/*.safetensors, the text model's weights alone. As
a directory it also holds checksums.json, each member's size and CRC-32; as a
.pulsarlm zip archive it stores the members uncompressed, so that a shard is read
in place, and the zip records the same size and CRC-32 per member. See
pulsarlm.checkpoint for how either is read back, and pulsarlm.migrate for how a
directory is written.
"""

import json
import zipfile
import zlib
from pathlib import Path

from pulsarlm.checkpoint import CHECKSUMS, PulsarLMModelLoadingError

METADATA = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
]


def crc32_of(path: Path) -> int:
    crc = 0
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            crc = zlib.crc32(chunk, crc)
    return crc


def members(directory: Path) -> list[str]:
    weights = sorted(
        path.name for path in (directory / "weights").glob("*.safetensors")
    )
    return METADATA + [f"weights/{name}" for name in weights]


def read_checksums(directory: Path) -> dict[str, dict[str, int]]:
    return json.loads((directory / CHECKSUMS).read_text())


def pack(directory: Path, archive: Path) -> None:
    """Packs the model directory into archive, through a sibling .tmp file
    renamed into place. Raises PulsarLMModelLoadingError when a member does not
    match its checksum."""
    checksums = read_checksums(directory)
    tmp = archive.with_suffix(archive.suffix + ".tmp")
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
            for name, expected in checksums.items():
                zf.write(directory / name, name)
                info = zf.getinfo(name)
                if (info.file_size, info.CRC) != (expected["size"], expected["crc32"]):
                    raise PulsarLMModelLoadingError(
                        f"{directory / name} does not match its checksum"
                    )
        tmp.replace(archive)
    finally:
        tmp.unlink(missing_ok=True)


def verify(path: Path) -> None:
    """Raises PulsarLMModelLoadingError when a member of the model directory or
    archive at path is missing, extra or does not match its size and CRC-32."""
    if path.is_dir():
        checksums = read_checksums(path)
        if sorted(checksums) != sorted(members(path)):
            raise PulsarLMModelLoadingError(
                f"{path}: its files do not match {CHECKSUMS}"
            )
        for name, expected in checksums.items():
            file = path / name
            if (file.stat().st_size, crc32_of(file)) != (
                expected["size"],
                expected["crc32"],
            ):
                raise PulsarLMModelLoadingError(f"{file} does not match its checksum")
        return
    with zipfile.ZipFile(path) as zf:
        if (bad := zf.testzip()) is not None:
            raise PulsarLMModelLoadingError(
                f"{path}: {bad} does not match its checksum"
            )
