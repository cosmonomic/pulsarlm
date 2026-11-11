"""A safetensors file read in place from memory-mapped bytes of a file: a whole
file, or a member stored uncompressed in a zip archive. It offers the part of
safetensors.safe_open's interface that loading uses."""

import json
import mmap
import struct
import zipfile
from pathlib import Path

import torch
from torch import Tensor

DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}

# A zip local file header: signature, versions, flags, method, time, date, CRC,
# sizes, then the lengths of the name and the extra field before the data.
LOCAL_HEADER = struct.Struct("<4s2B4HL2L2H")


class TensorSlice:
    def __init__(self, tensor: Tensor) -> None:
        self._tensor = tensor

    def get_shape(self) -> list[int]:
        return list(self._tensor.shape)

    def __getitem__(self, index) -> Tensor:
        return self._tensor[index]


class SafetensorsFile:
    """The tensors of the safetensors bytes [offset, offset + size) of path. A
    tensor is a view of the mapping, so its bytes are read as it is used, and the
    mapping stays open while a tensor refers to it. The mapping is copy-on-write,
    so writing to a tensor does not change the file."""

    def __init__(self, path: Path, offset: int, size: int) -> None:
        with path.open("rb") as f:
            mapping = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
        view = memoryview(mapping)[offset : offset + size]
        (header_size,) = struct.unpack_from("<Q", view, 0)
        header = json.loads(bytes(view[8 : 8 + header_size]))
        header.pop("__metadata__", None)
        self._entries: dict[str, dict] = header
        self._data = view[8 + header_size :]

    @classmethod
    def of_file(cls, path: Path) -> "SafetensorsFile":
        return cls(path, 0, path.stat().st_size)

    @classmethod
    def of_member(cls, archive: Path, info: zipfile.ZipInfo) -> "SafetensorsFile":
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{archive}: {info.filename} is compressed")
        with archive.open("rb") as f:
            f.seek(info.header_offset)
            fields = LOCAL_HEADER.unpack(f.read(LOCAL_HEADER.size))
        name_size, extra_size = fields[-2], fields[-1]
        offset = info.header_offset + LOCAL_HEADER.size + name_size + extra_size
        return cls(archive, offset, info.file_size)

    def keys(self) -> list[str]:
        return list(self._entries)

    def get_tensor(self, name: str) -> Tensor:
        entry = self._entries[name]
        dtype = DTYPES[entry["dtype"]]
        begin, end = entry["data_offsets"]
        if begin == end:
            return torch.empty(entry["shape"], dtype=dtype)
        return torch.frombuffer(self._data[begin:end], dtype=dtype).view(entry["shape"])

    def get_slice(self, name: str) -> TensorSlice:
        return TensorSlice(self.get_tensor(name))
