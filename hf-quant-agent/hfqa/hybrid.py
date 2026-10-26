"""Hybrid release engine: keep the backbone of another provider byte-for-byte, re-quantize only a lookup table (for example the
DeepSeek "Engram" n-gram embedding table) from a higher-precision GGUF.

One output file is produced as a stream (no table-sized file on disk):
  header  = the template shard's key/value block copied verbatim, plus new tensor infos (type, shape, offsets)
  payload = rows of the source tensors, converted through the pinned ggml (`to_float` of the source type, then
            `ggml_quantize_chunk` to the target type); every source file is read start to end once and hashed completely.
The result is an integrity-checked file, not an inference-tested one; the receipt says so.
"""
from __future__ import annotations

import ctypes
import hashlib
import math
import re
import struct
from pathlib import Path

WORK_FLOATS_BYTES = 32 * 1024 ** 2
HASH_BLOCK = 8 * 1024 ** 2
MAX_TENSORS = 64
KNOWN_LAYOUTS = {0: (1, 4), 1: (1, 2), 8: (32, 34), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 30: (1, 2)}
# Ids with a live entry in ggml's type table (checked one by one against the pinned build). Any other id makes ggml abort the process.
SAFE_TYPE_IDS = frozenset(set(range(0, 4)) | {6, 7, 8, 9} | set(range(10, 31)) | {34, 35, 39, 40, 41, 42})


class HybridError(ValueError):
    """The hybrid plan or one of its inputs is not safe to build."""


class _Traits(ctypes.Structure):
    _fields_ = [('type_name', ctypes.c_char_p), ('blck_size', ctypes.c_int64), ('blck_size_interleave', ctypes.c_int64),
                ('type_size', ctypes.c_size_t), ('is_quantized', ctypes.c_bool), ('to_float', ctypes.c_void_p), ('from_float_ref', ctypes.c_void_p)]


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


class Ggml:
    """Row conversion through the pinned stock ggml library. The struct layout is checked against known type sizes first."""

    def __init__(self, library_path, runtime_revision: str):
        if not re.fullmatch('[a-f0-9]{40}', runtime_revision):
            raise HybridError('a full runtime commit is required')
        self.path = Path(library_path)
        self.lib = ctypes.CDLL(str(self.path.resolve()))
        self.lib.ggml_get_type_traits.restype = ctypes.POINTER(_Traits)
        self.lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
        self.lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        self.lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        self.lib.ggml_validate_row_data.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.ggml_validate_row_data.restype = ctypes.c_bool
        for type_id, (block, width) in KNOWN_LAYOUTS.items():
            traits = self.lib.ggml_get_type_traits(type_id).contents
            if (traits.blck_size, traits.type_size) != (block, width):
                raise HybridError('this ggml build lays out types differently than expected; refusing to convert')
        self._to_float = {}
        self.identity = {'runtime_revision': runtime_revision, 'ggml_sha256': _sha256_file(self.path)}

    def layout(self, type_id: int) -> tuple[int, int]:
        if type_id not in SAFE_TYPE_IDS:
            raise HybridError(f'tensor type {type_id} is not known to the pinned ggml build')
        traits = self.lib.ggml_get_type_traits(type_id)
        if not traits:
            raise HybridError(f'unknown tensor type {type_id}')
        return int(traits.contents.blck_size), int(traits.contents.type_size)

    def _decoder(self, type_id: int):
        if type_id not in self._to_float:
            address = self.lib.ggml_get_type_traits(type_id).contents.to_float
            if not address:
                raise HybridError(f'tensor type {type_id} has no decoder in this ggml build')
            self._to_float[type_id] = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)(address)
        return self._to_float[type_id]

    def decode(self, raw: bytes, type_id: int, cols: int):
        """Float32 matrix (rows x cols) from raw rows of any supported source type."""
        import numpy as np
        block, width = self.layout(type_id)
        if cols % block or not raw or len(raw) % (cols // block * width):
            raise HybridError('source rows do not match the tensor row layout')
        rows = len(raw) // (cols // block * width)
        if type_id == 0:
            values = np.frombuffer(raw, dtype='<f4').reshape(rows, cols).astype(np.float32)
        elif type_id == 1:
            values = np.frombuffer(raw, dtype='<f2').astype(np.float32).reshape(rows, cols)
        elif type_id == 30:
            values = (np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16).view(np.float32).reshape(rows, cols)
        else:
            values = np.empty((rows, cols), dtype=np.float32)
            source = np.frombuffer(raw, dtype=np.uint8)
            self._decoder(type_id)(source.ctypes.data, values.ctypes.data, rows * cols)
        values = np.ascontiguousarray(values, dtype=np.float32)
        if not np.isfinite(values).all():
            raise HybridError('source rows contain non-finite values')
        return values

    def convert(self, raw: bytes, in_type: int, out_type: int, cols: int) -> bytes:
        import numpy as np
        if in_type == out_type:
            # Same type: the rows are already what is wanted. A second quantization pass would only add error.
            block, width = self.layout(out_type)
            if cols % block or not raw or len(raw) % (cols // block * width):
                raise HybridError('source rows do not match the tensor row layout')
            buffer = np.frombuffer(raw, dtype=np.uint8)
            if not self.lib.ggml_validate_row_data(out_type, buffer.ctypes.data, buffer.size):
                raise HybridError('source row data failed validation')
            return bytes(raw)
        values = self.decode(raw, in_type, cols)
        block, width = self.layout(out_type)
        if cols % block:
            raise HybridError('row length does not fit the target block size')
        rows = values.shape[0]
        out = np.empty(rows * cols // block * width, dtype=np.uint8)
        written = self.lib.ggml_quantize_chunk(out_type, values.ctypes.data, out.ctypes.data, 0, rows, cols, None)
        if written != out.size:
            raise HybridError('native quantizer returned an unexpected byte count')
        if not self.lib.ggml_validate_row_data(out_type, out.ctypes.data, out.size):
            raise HybridError('native output row validation failed')
        return out.tobytes()


def _string(value: str) -> bytes:
    raw = value.encode('utf-8')
    return struct.pack('<Q', len(raw)) + raw


def _field(item, key):
    return item[key] if isinstance(item, dict) else getattr(item, key)


def _tensor_by_name(remote, name: str):
    found = [t for t in remote.tensors if _field(t, 'name') == name]
    if len(found) != 1:
        raise HybridError(f'tensor {name} is not present exactly once')
    return found[0]


def build_plan(template, outputs: list[dict], sources: dict, out_type: int, ggml: Ggml, kv_overrides: dict | None = None) -> dict:
    """Validate and lay out the output file.

    outputs: [{'name': 'blk.1.engram_embd.weight', 'parts': [{'source': 'maya5', 'tensor': 'name in that file'}, ...]}]
    sources: {'maya5': {'remote': RemoteGGUF, 'descriptor': {repo_id, revision, filename, size, sha256}}}
    """
    if not 1 <= len(outputs) <= MAX_TENSORS:
        raise HybridError('between one and 64 output tensors are supported')
    out_block, out_width = ggml.layout(out_type)
    seen_sources, seen_names, tensors, offset = set(), set(), [], 0
    for item in outputs:
        name, parts = item.get('name'), item.get('parts')
        if not isinstance(name, str) or not name or name in seen_names or not isinstance(parts, list) or not parts:
            raise HybridError('every output tensor needs a unique name and at least one source part')
        seen_names.add(name)
        cols, rows, plan_parts = None, 0, []
        for part in parts:
            key = part.get('source')
            if key not in sources or key in seen_sources:
                raise HybridError('each source file may feed one output part only, and must be declared')
            seen_sources.add(key)
            remote, descriptor = sources[key]['remote'], sources[key]['descriptor']
            if not re.fullmatch('[a-f0-9]{64}', str(descriptor.get('sha256', ''))) or not re.fullmatch('[a-f0-9]{40}', str(descriptor.get('revision', ''))) \
                    or type(descriptor.get('size')) is not int or descriptor['size'] != remote.reader.size:
                raise HybridError('every source needs an exact size, SHA-256 and revision')
            tensor = _tensor_by_name(remote, part.get('tensor'))
            shape, in_type = tuple(_field(tensor, 'shape')), int(_field(tensor, 'type'))
            if len(shape) != 2 or min(shape) <= 0:
                raise HybridError('tables must be two-dimensional')
            in_block, in_width = ggml.layout(in_type)
            expected = shape[0] // in_block * in_width * shape[1]
            if shape[0] % in_block or expected != _field(tensor, 'nbytes'):
                raise HybridError('source tensor size and shape disagree')
            if cols is not None and shape[0] != cols:
                raise HybridError('all parts of one table must have the same row length')
            if shape[0] % out_block or shape[0] * 4 > WORK_FLOATS_BYTES:
                raise HybridError('row length does not fit the target type or the work buffer')
            cols = shape[0]
            rows += shape[1]
            plan_parts.append({'source': key, 'tensor': part['tensor'], 'rows': shape[1], 'in_type': in_type,
                               'start': remote.data_offset + _field(tensor, 'offset'), 'nbytes': expected})
        nbytes = rows * cols // out_block * out_width
        tensors.append({'name': name, 'shape': (cols, rows), 'type': out_type, 'offset': offset, 'nbytes': nbytes, 'parts': plan_parts})
        offset += nbytes + (-nbytes % template.alignment)
    header = _header(template, tensors, kv_overrides or {})
    return {'tensors': tensors, 'header': header, 'size': len(header) + offset, 'out_type': out_type}


def _header(template, tensors: list[dict], overrides: dict) -> bytes:
    entries = []
    for key, kind, raw in template.metadata_entries:
        entries.append((key, kind, overrides.pop(key, (kind, raw))[1] if key in overrides else raw))
    if overrides:
        raise HybridError('override names a key the template does not have: ' + ', '.join(sorted(overrides)))
    kv = b''.join(_string(key) + struct.pack('<I', kind) + raw for key, kind, raw in entries)
    infos = b''
    for tensor in tensors:
        infos += _string(tensor['name']) + struct.pack('<I', len(tensor['shape']))
        infos += b''.join(struct.pack('<Q', dim) for dim in tensor['shape']) + struct.pack('<IQ', tensor['type'], tensor['offset'])
    head = b'GGUF' + struct.pack('<IQQ', template.version, len(tensors), len(entries)) + kv + infos
    return head + b'\0' * (-len(head) % template.alignment)


class HybridTableStream:
    """Single-use iterator over the output bytes. `receipt` exists only after every source file hashed to its pinned SHA-256."""

    def __init__(self, template, plan: dict, sources: dict, ggml: Ggml, should_stop=None):
        self.template, self.plan, self.sources, self.ggml = template, plan, sources, ggml
        self.should_stop, self.receipt, self.used = should_stop, None, False
        self.size = plan['size']

    def _check(self):
        if self.should_stop and self.should_stop():
            raise InterruptedError('hybrid build paused or cancelled')

    def __iter__(self):
        if self.used:
            raise RuntimeError('a hybrid stream cannot be reused')
        self.used = True
        output, written, proofs, verified = hashlib.sha256(), 0, [], []

        def emit(raw: bytes) -> bytes:
            nonlocal written
            output.update(raw)
            written += len(raw)
            return raw
        yield emit(self.plan['header'])
        alignment = self.template.alignment
        for tensor in self.plan['tensors']:
            cols = tensor['shape'][0]
            tensor_hash, tensor_bytes = hashlib.sha256(), 0
            for part in tensor['parts']:
                source = self.sources[part['source']]
                reader, descriptor = source['remote'].reader, source['descriptor']
                digest, cursor = hashlib.sha256(), 0

                def read(count: int) -> bytes:
                    nonlocal cursor
                    self._check()
                    raw = reader.read_range(cursor, count)
                    if len(raw) != count:
                        raise HybridError('source reader returned a short range')
                    digest.update(raw)
                    cursor += count
                    return raw
                while cursor < part['start']:
                    read(min(HASH_BLOCK, part['start'] - cursor))
                block, width = self.ggml.layout(part['in_type'])
                row_bytes = cols // block * width
                step = max(1, WORK_FLOATS_BYTES // (cols * 4))
                for first in range(0, part['rows'], step):
                    count = min(step, part['rows'] - first)
                    converted = self.ggml.convert(read(count * row_bytes), part['in_type'], tensor['type'], cols)
                    tensor_hash.update(converted)
                    tensor_bytes += len(converted)
                    yield emit(converted)
                while cursor < descriptor['size']:
                    read(min(HASH_BLOCK, descriptor['size'] - cursor))
                if digest.hexdigest() != descriptor['sha256']:
                    raise HybridError('a source file does not match its pinned SHA-256')
                verified.append({k: descriptor.get(k) for k in ('repo_id', 'revision', 'filename', 'size', 'sha256')} | {'sha256_verified': True,
                                 'tensor': part['tensor'], 'rows': part['rows']})
            if tensor_bytes != tensor['nbytes']:
                raise HybridError('converted table size differs from the plan')
            padding = -tensor_bytes % alignment
            if padding:
                yield emit(b'\0' * padding)
            proofs.append({'name': tensor['name'], 'type': tensor['type'], 'rows': tensor['shape'][1], 'cols': cols, 'bytes': tensor_bytes,
                           'sha256': tensor_hash.hexdigest(), 'parts': [p['source'] for p in tensor['parts']]})
        if written != self.size:
            raise HybridError('streamed size differs from the plan')
        self.receipt = {'schema': 'hfqa-hybrid-table/1', 'sha256': output.hexdigest(), 'size': written, 'tensors': proofs, 'tensor_count': len(proofs),
                        'sources_verified': verified, 'out_type': self.plan['out_type'], 'native': self.ggml.identity,
                        'source_materialized': False, 'output_materialized': False, 'stream_complete': True,
                        'full_inference_test_performed': False, 'integrity_only': True}


def sample_compatibility(ggml: Ggml, own, parts: list, count: int = 48, tolerance: float = 0.30) -> dict:
    """Is the source table the same table as the backbone's own one? Compare sampled rows after decoding both.

    own: (RemoteGGUF, tensor name) of the table inside the backbone provider's files (any quantization).
    parts: [(RemoteGGUF, tensor name), ...] of the replacement source, concatenated along rows.
    """
    import numpy as np
    own_remote, own_name = own
    own_tensor = _tensor_by_name(own_remote, own_name)
    own_shape, own_type = tuple(_field(own_tensor, 'shape')), int(_field(own_tensor, 'type'))
    sizes = []
    for remote, name in parts:
        tensor = _tensor_by_name(remote, name)
        sizes.append((remote, tensor, tuple(_field(tensor, 'shape')), int(_field(tensor, 'type'))))
    if any(shape[0] != own_shape[0] for _, _, shape, _ in sizes) or sum(shape[1] for _, _, shape, _ in sizes) != own_shape[1]:
        return {'passes': False, 'reason': 'row length or total row count differs from the backbone table', 'rows_sampled': 0}
    total, cols = own_shape[1], own_shape[0]
    picks = sorted({0, total - 1, *(int(i * (total - 1) / max(1, count - 1)) for i in range(count))})
    errors = []
    own_block, own_width = ggml.layout(own_type)
    for row in picks:
        own_raw = own_remote.reader.read_range(own_remote.data_offset + _field(own_tensor, 'offset') + row * (cols // own_block * own_width), cols // own_block * own_width)
        base, local = 0, row
        for remote, tensor, shape, kind in sizes:
            if local < shape[1]:
                block, width = ggml.layout(kind)
                source_raw = remote.reader.read_range(remote.data_offset + _field(tensor, 'offset') + local * (cols // block * width), cols // block * width)
                break
            local -= shape[1]
        a, b = ggml.decode(own_raw, own_type, cols)[0], ggml.decode(source_raw, kind, cols)[0]
        errors.append(float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(a ** 2)) + 1e-9)))
    worst, mean = max(errors), sum(errors) / len(errors)
    return {'passes': worst <= tolerance, 'rows_sampled': len(errors), 'max_relative_rms': round(worst, 5), 'mean_relative_rms': round(mean, 5),
            'tolerance': tolerance, 'own_type': own_type, 'source_types': sorted({kind for *_, kind in sizes})}
