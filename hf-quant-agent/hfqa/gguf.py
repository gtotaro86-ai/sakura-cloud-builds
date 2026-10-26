"""Structural GGUF reader. A valid header is not a working model."""

from __future__ import annotations

import struct
import hashlib
from collections import Counter
from pathlib import Path

GGUF_MAGIC = b"GGUF"
VALUE_TYPES = {
    0: "uint8", 1: "int8", 2: "uint16", 3: "int16", 4: "uint32", 5: "int32",
    6: "float32", 7: "bool", 8: "string", 9: "array", 10: "uint64", 11: "int64", 12: "float64",
}

# (elements per block, bytes per block), for the stock formats accepted here.
# Unknown tensor formats fail closed rather than guessing their payload size.
TENSOR_LAYOUTS = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20),
    6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36),
    10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 292),
    16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136),
    24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8),
    28: (1, 8), 29: (256, 56), 30: (1, 2), 34: (256, 54), 35: (256, 66),
    39: (32, 17), 40: (64, 36), 41: (128, 18), 42: (64, 18),
}
TENSOR_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
    13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 30: "BF16",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 29: "IQ1_M",
    34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4", 41: "Q1_0", 42: "Q2_0",
}


class GGUFError(ValueError):
    pass


def _read(handle, size: int) -> bytes:
    blob = handle.read(size)
    if len(blob) != size:
        raise GGUFError("truncated GGUF")
    return blob


def _u32(handle) -> int:
    return struct.unpack("<I", _read(handle, 4))[0]


def _u64(handle) -> int:
    return struct.unpack("<Q", _read(handle, 8))[0]


def _string(handle) -> str:
    length = _u64(handle)
    if length > 8_000_000:
        raise GGUFError("GGUF string too large")
    return _read(handle, length).decode("utf-8", "replace")


def _skip_value(handle, value_type: int, depth: int = 0) -> None:
    if depth > 4:
        raise GGUFError("GGUF metadata nesting too deep")
    if value_type in (0, 1, 7):
        _read(handle, 1)
    elif value_type in (2, 3):
        _read(handle, 2)
    elif value_type in (4, 5, 6):
        _read(handle, 4)
    elif value_type in (10, 11, 12):
        _read(handle, 8)
    elif value_type == 8:
        _string(handle)
    elif value_type == 9:
        item_type = _u32(handle)
        count = _u64(handle)
        if count > 2_000_000:
            raise GGUFError("GGUF array too large to validate safely")
        for _ in range(count):
            _skip_value(handle, item_type, depth + 1)
    else:
        raise GGUFError(f"unknown GGUF value type {value_type}")


def validate_gguf(path: str | Path, expected_architecture: str | None = None) -> dict:
    file_path = Path(path)
    if not file_path.is_file():
        raise GGUFError(f"missing file: {file_path}")
    size = file_path.stat().st_size
    if size < 24:
        raise GGUFError("file smaller than a GGUF header")
    with file_path.open("rb") as handle:
        magic = _read(handle, 4)
        if magic != GGUF_MAGIC:
            raise GGUFError(f"bad magic {magic!r}")
        version = _u32(handle)
        if version < 2 or version > 3:
            raise GGUFError(f"unsupported GGUF version {version}")
        tensor_count = _u64(handle)
        kv_count = _u64(handle)
        if tensor_count > 2_000_000 or kv_count > 100_000:
            raise GGUFError("implausible GGUF counts")
        metadata_keys = []
        seen_metadata = set()
        architecture = None
        name = None
        alignment = 32
        file_type = None
        for _ in range(kv_count):
            key = _string(handle)
            if key in seen_metadata:
                raise GGUFError(f"duplicate metadata key {key}")
            seen_metadata.add(key)
            value_type = _u32(handle)
            metadata_keys.append(key)
            if key in {"general.architecture", "general.name"} and value_type == 8:
                value = _string(handle)
                if key == "general.architecture":
                    architecture = value
                else:
                    name = value
            elif key in {"general.alignment", "general.file_type"} and value_type == 4:
                value = _u32(handle)
                if key == "general.alignment":
                    alignment = value
                else:
                    file_type = value
            else:
                _skip_value(handle, value_type)
        if not alignment or alignment > 65536 or alignment & (alignment - 1):
            raise GGUFError("invalid GGUF alignment")
        tensor_names = []
        names = set()
        type_counts = Counter()
        ranges = []
        for _ in range(tensor_count):
            tensor_name = _string(handle)
            if not tensor_name or tensor_name in names:
                raise GGUFError("empty or duplicate tensor name")
            names.add(tensor_name)
            if len(tensor_names) < 8:
                tensor_names.append(tensor_name)
            dimensions = _u32(handle)
            if not 1 <= dimensions <= 4:
                raise GGUFError("invalid GGUF tensor dimensions")
            shape = [_u64(handle) for _ in range(dimensions)]
            if any(value == 0 or value > 2**40 for value in shape):
                raise GGUFError("invalid GGUF tensor shape")
            tensor_type = _u32(handle)
            offset = _u64(handle)
            if tensor_type not in TENSOR_LAYOUTS:
                raise GGUFError(f"unsupported tensor format {tensor_type}; payload not validated")
            type_counts[TENSOR_TYPES[tensor_type]] += 1
            block, width = TENSOR_LAYOUTS[tensor_type]
            if shape[0] % block:
                raise GGUFError("tensor row does not fit quantization blocks")
            elements = 1
            for dimension in shape:
                elements *= dimension
            nbytes = elements // block * width
            if offset % alignment:
                raise GGUFError("unaligned tensor payload")
            ranges.append((offset, offset + nbytes))
        data_start = (handle.tell() + alignment - 1) // alignment * alignment
        previous_end = 0
        for start, end in sorted(ranges):
            if start < previous_end or data_start + end > size:
                raise GGUFError("overlapping or truncated tensor payload")
            previous_end = end
    name_digest = hashlib.sha256()
    for tensor_name in sorted(names):
        encoded = tensor_name.encode("utf-8")
        name_digest.update(struct.pack("<Q", len(encoded)))
        name_digest.update(encoded)
    result = {
        "path": str(file_path),
        "bytes": size,
        "magic": "GGUF",
        "version": version,
        "tensor_count": tensor_count,
        "tensor_name_set_sha256": name_digest.hexdigest(),
        "tensor_type_histogram": dict(sorted(type_counts.items())),
        "metadata_count": kv_count,
        "architecture": architecture,
        "name": name,
        "file_type": file_type,
        "metadata_keys_sample": metadata_keys[:30],
        "tensor_name_sample": tensor_names[:8],
        "structure_valid": True,
        "runtime_compatible": None,
        "inference_tested": False,
        "note": "Structural validity does not prove runtime compatibility or answer quality.",
    }
    if not architecture:
        result["structure_valid"] = False
        result["note"] = "missing general.architecture"
    if expected_architecture and architecture != expected_architecture:
        result["structure_valid"] = False
        result["note"] = f"architecture {architecture} != expected {expected_architecture}"
    if tensor_count == 0:
        result["structure_valid"] = False
        result["note"] = "GGUF has no tensors"
    return result


def write_minimal_gguf(path: str | Path, architecture: str = "llama", tensor_name: str = "token_embd.weight") -> None:
    """Tiny synthetic GGUF used by unit tests. Not a loadable model."""
    def s(text: str) -> bytes:
        raw = text.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    # metadata: general.architecture, general.name
    meta = b"".join([
        s("general.architecture"), struct.pack("<I", 8), s(architecture),
        s("general.name"), struct.pack("<I", 8), s("synthetic"),
    ])
    # A structurally complete single F32 tensor. Still not a loadable model.
    tensor = s(tensor_name) + struct.pack("<I", 1) + struct.pack("<Q", 1) + struct.pack("<I", 0) + struct.pack("<Q", 0)
    header = GGUF_MAGIC + struct.pack("<I", 3) + struct.pack("<Q", 1) + struct.pack("<Q", 2) + meta + tensor
    Path(path).write_bytes(header + b"\x00" * ((-len(header)) % 32) + struct.pack("<f", 1.0))
