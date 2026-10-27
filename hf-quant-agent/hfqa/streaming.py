"""Bounded native GGUF requantization streams; no complete model on local disk.

The HTTP reader/parser lives in remote_gguf. This module preserves the full
catalogue for one GLOBAL stock quantizer type plan, then processes bounded rows
in source byte order. A release is eligible only after generator exhaustion has
verified the complete immutable input SHA-256 and every output tensor size.
"""
from __future__ import annotations
import ctypes
import base64
import hashlib
import math
import os
import re
import struct
from pathlib import Path
from types import SimpleNamespace

from hfqa.quants import FTYPE as QUANTS  # core presets plus the extra stock presets (see hfqa/quants.py)
# GGML type -> (block elements, block bytes)
LAYOUTS = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20),
           6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 40),
           10: (256, 84), 11: (256, 110), 12: (256, 144),
           13: (256, 176), 14: (256, 210), 15: (256, 292), 30: (1, 2)}
SOURCE_TYPES = {0, 1, 8, 30}
ROW_WORK_BYTES = 8 * 1024**2
MAX_SHARD_BYTES = 50_000_000_000  # the Hub's per-file limit
# People want ONE file per quantization. Everything up to this size is one file; only a larger output is split (still below the Hub limit).
DEFAULT_SHARD_BYTES = 45_000_000_000


def _field(value, key):
    return value[key] if isinstance(value, dict) else getattr(value, key)


def _sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(1024**2), b''):
            result.update(data)
    return result.hexdigest()


def _linked_llama_path(bridge):
    """Resolve the library actually supplying the bridge's stock planner API."""
    symbol = ctypes.cast(bridge.llama_model_quantize_default_params, ctypes.c_void_p)
    if os.name == 'nt':
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        module = wintypes.HMODULE()
        kernel.GetModuleHandleExW.argtypes = [wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(wintypes.HMODULE)]
        kernel.GetModuleHandleExW.restype = wintypes.BOOL
        if not kernel.GetModuleHandleExW(0x4 | 0x2, symbol, ctypes.byref(module)):
            raise RuntimeError('cannot identify the loaded stock llama planner DLL')
        buffer = ctypes.create_unicode_buffer(32768)
        kernel.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
        kernel.GetModuleFileNameW.restype = wintypes.DWORD
        size = kernel.GetModuleFileNameW(module, buffer, len(buffer))
        if not size or size >= len(buffer):
            raise RuntimeError('cannot identify the loaded stock llama planner DLL path')
        return Path(buffer.value).resolve(strict=True)
    class DlInfo(ctypes.Structure):
        _fields_ = [('filename', ctypes.c_char_p), ('base', ctypes.c_void_p),
                    ('symbol_name', ctypes.c_char_p), ('symbol_address', ctypes.c_void_p)]
    loader, result = ctypes.CDLL(None), DlInfo()
    loader.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    loader.dladdr.restype = ctypes.c_int
    if not loader.dladdr(symbol, ctypes.byref(result)) or not result.filename:
        raise RuntimeError('cannot identify the loaded stock llama planner library')
    return Path(os.fsdecode(result.filename)).resolve(strict=True)


def _uniform_int(metadata, key, default=None):
    value = metadata.get(key, default)
    if isinstance(value, (tuple, list)):
        if not value or len(set(value)) != 1:
            raise ValueError('native planner requires uniform metadata: ' + key)
        value = value[0]
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 2**32:
        raise ValueError('missing or invalid native planner metadata: ' + key)
    return value


FFN_KEYS = ('feed_forward_length', 'expert_feed_forward_length', 'expert_shared_feed_forward_length')


def model_hparams(metadata):
    arch = metadata.get('general.architecture')
    if not isinstance(arch, str) or not re.fullmatch('[a-z0-9_-]+', arch):
        raise ValueError('missing or invalid stock GGUF architecture')
    prefix = arch + '.'
    embd = _uniform_int(metadata, prefix + 'embedding_length')
    head = _uniform_int(metadata, prefix + 'attention.head_count')
    if not head or not embd or embd % head:
        raise ValueError('invalid native attention dimensions')
    # Mixture-of-experts models without a dense feed-forward block (qwen35moe) carry only the expert widths; the width is a placeholder for the
    # planner, which NativeQuantizer.plan proves irrelevant for the real tensor catalogue before it is used.
    ffn_key = next((prefix + key for key in FFN_KEYS if prefix + key in metadata), prefix + 'feed_forward_length')
    params = [embd, _uniform_int(metadata, ffn_key),
              _uniform_int(metadata, prefix + 'block_count'), head,
              _uniform_int(metadata, prefix + 'attention.head_count_kv', head),
              _uniform_int(metadata, prefix + 'expert_count', 0),
              _uniform_int(metadata, prefix + 'attention.key_length', embd // head),
              _uniform_int(metadata, prefix + 'attention.value_length', embd // head)]
    if not params[1] or not params[2] or params[2] > 4096 or not params[4] or not params[6] or not params[7]:
        raise ValueError('invalid native model dimensions')
    return arch, params


class NativeQuantizer:
    """C adapter around the pinned upstream GLOBAL type planner and row kernels.

    The adapter is separate project code. The llama.cpp runtime remains stock.
    hfqa_plan signature is documented in github/native/generic_quant_bridge.cpp.
    """
    def __init__(self, bridge_path, ggml_path, runtime_revision):
        if not re.fullmatch('[a-fA-F0-9]{40}', runtime_revision):
            raise ValueError('a full immutable stock runtime SHA is required')
        self.bridge_path, self.ggml_path = Path(bridge_path), Path(ggml_path)
        self.bridge = ctypes.CDLL(str(self.bridge_path.resolve()))
        self.lib = ctypes.CDLL(str(self.ggml_path.resolve()))
        self.llama_path = _linked_llama_path(self.bridge)
        self.bridge.hfqa_plan.argtypes = [ctypes.c_int, ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_int, ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        self.bridge.hfqa_plan.restype = ctypes.c_int
        self.bridge.hfqa_error.restype = ctypes.c_char_p
        self.lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        self.lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        self.lib.ggml_validate_row_data.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.ggml_validate_row_data.restype = ctypes.c_bool
        self.identity = {'runtime_revision': runtime_revision.lower(),
                         'bridge_sha256': _sha256(self.bridge_path), 'ggml_sha256': _sha256(self.ggml_path),
                         'llama_sha256': _sha256(self.llama_path), 'llama_library': str(self.llama_path)}

    def plan(self, remote, quant):
        if quant not in QUANTS:
            raise ValueError('unsupported target quant')
        tensors = remote.tensors
        if not tensors or len(tensors) > 100000:
            raise ValueError('native catalogue count outside bounded planner limit')
        arch, values = model_hparams(remote.metadata)
        count = len(tensors)
        names = (ctypes.c_char_p * count)(*(_field(t, 'name').encode('utf-8') for t in tensors))
        dims = (ctypes.c_int * count)(*(len(_field(t, 'shape')) for t in tensors))
        flat = []
        for tensor in tensors:
            shape = list(_field(tensor, 'shape'))
            if not 1 <= len(shape) <= 4:
                raise ValueError('unsupported tensor dimensions')
            flat.extend(shape + [1] * (4 - len(shape)))
        shapes = (ctypes.c_int64 * (count * 4))(*flat)
        inputs = (ctypes.c_int * count)(*(_field(t, 'type') for t in tensors))
        outputs = (ctypes.c_int * count)()
        def planned(hparam_values):
            result = (ctypes.c_int * count)()
            hparams = (ctypes.c_uint32 * 8)(*hparam_values)
            if self.bridge.hfqa_plan(QUANTS[quant], arch.encode(), hparams, count, names, shapes, dims, inputs, result):
                raise RuntimeError((self.bridge.hfqa_error() or b'native planner failed').decode('utf-8', 'replace'))
            return [int(result[i]) for i in range(count)]
        types = planned(values)
        if (arch + '.feed_forward_length') not in remote.metadata:
            # The model has no dense feed-forward width, so a stand-in was used. Accept it only if the plan does not depend on it at all.
            other = list(values)
            other[1] = values[1] * 2 + 1
            if other[1] >= 2 ** 32 or planned(other) != types:
                raise ValueError('native type plan depends on the feed-forward width, which this model does not declare; refusing to guess')
        return {_field(t, 'name'): types[i] for i, t in enumerate(tensors)}

    def quantize_rows(self, raw, input_type, output_type, cols):
        import numpy as np
        if input_type not in SOURCE_TYPES or output_type not in LAYOUTS:
            raise ValueError('unqualified source or output tensor type')
        if not isinstance(cols, int) or isinstance(cols, bool) or cols <= 0 or cols * 4 > ROW_WORK_BYTES:
            raise ValueError('tensor row exceeds the bounded native workspace')
        if not raw or len(raw) > ROW_WORK_BYTES:
            raise ValueError('source chunk exceeds the bounded native workspace')
        block, width = LAYOUTS[input_type]
        if cols % block or len(raw) % (cols // block * width):
            raise ValueError('source row boundary mismatch')
        rows = len(raw) // (cols // block * width)
        if rows * cols * 4 > ROW_WORK_BYTES:
            raise ValueError('dequantized chunk exceeds the bounded native workspace')
        if input_type == output_type:
            result = np.frombuffer(raw, dtype=np.uint8)
        else:
            if input_type == 0:
                values = np.frombuffer(raw, dtype='<f4').reshape(rows, cols)
            elif input_type == 1:
                values = np.frombuffer(raw, dtype='<f2').astype(np.float32).reshape(rows, cols)
            elif input_type == 30:
                values = (np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16).view(np.float32).reshape(rows, cols)
            else:
                blocks = np.frombuffer(raw, dtype=np.uint8).reshape(rows, cols // 32, 34)
                scales = blocks[:, :, :2].copy().view('<f2').reshape(rows, cols // 32).astype(np.float32)
                values = (blocks[:, :, 2:].view(np.int8).astype(np.float32) * scales[:, :, None]).reshape(rows, cols)
            values = np.ascontiguousarray(values, dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError('source contains non-finite values')
            out_block, out_width = LAYOUTS[output_type]
            if cols % out_block:
                raise ValueError('native type plan produced an invalid row layout')
            result = np.empty(rows * cols // out_block * out_width, dtype=np.uint8)
            written = self.lib.ggml_quantize_chunk(output_type, values.ctypes.data, result.ctypes.data, 0, rows, cols, None)
            if written != result.size:
                raise RuntimeError('native quantizer returned an unexpected byte count')
        if not self.lib.ggml_validate_row_data(output_type, result.ctypes.data, result.size):
            raise RuntimeError('native output row validation failed')
        return result.tobytes()


def _string(value):
    raw = value.encode('utf-8')
    return struct.pack('<Q', len(raw)) + raw


def _output_header(remote, quant, tensors):
    entries = [tuple(entry) for entry in remote.metadata_entries]
    replacements = {'general.file_type': (4, struct.pack('<I', QUANTS[quant])),
                    'general.quantization_version': (4, struct.pack('<I', 2))}
    entries = [(key, typ, raw) for key, typ, raw in entries if key not in replacements]
    entries.extend((key, typ, raw) for key, (typ, raw) in replacements.items())
    kv = b''.join(_string(key) + struct.pack('<I', typ) + raw for key, typ, raw in entries)
    catalogue = b''
    for tensor in tensors:
        shape = tensor['shape']
        catalogue += _string(tensor['name']) + struct.pack('<I', len(shape))
        catalogue += b''.join(struct.pack('<Q', dim) for dim in shape)
        catalogue += struct.pack('<IQ', tensor['type'], tensor['offset'])
    result = b'GGUF' + struct.pack('<IQQ', remote.version, len(tensors), len(entries)) + kv + catalogue
    return result + b'\0' * (-len(result) % remote.alignment)


class QuantizedStream:
    """Single-use output iterator. receipt is set ONLY after all source bytes verify."""
    def __init__(self, remote, descriptor, quant, quantizer, should_stop=None):
        if quant not in QUANTS:
            raise ValueError('unsupported target quant')
        sources = descriptor.get('sources')
        if sources is None:
            if not re.fullmatch('[a-fA-F0-9]{64}', descriptor.get('sha256', '')):
                raise ValueError('immutable source SHA-256 is required')
        else:
            if (not isinstance(sources, list) or not sources or len(sources) > 10000
                    or any(not isinstance(item, dict)
                        or not re.fullmatch('[a-fA-F0-9]{64}', item.get('sha256', ''))
                        or not re.fullmatch('[a-f0-9]{40}', item.get('revision', ''))
                        or not isinstance(item.get('size'), int) or isinstance(item['size'], bool) or item['size'] < 24
                        for item in sources)
                    or sum(item['size'] for item in sources) != descriptor.get('size')):
                raise ValueError('split sources require immutable per-file SHA-256, revision and byte size')
            if not callable(getattr(remote.reader, 'begin_verified_pass', None)) or not callable(getattr(remote.reader, 'finish_verified_pass', None)):
                raise ValueError('split sources require an authenticated sequential source reader')
        if not re.fullmatch('[a-fA-F0-9]{40}', descriptor.get('revision', '')):
            raise ValueError('immutable source revision is required')
        if not isinstance(descriptor.get('size'), int) or isinstance(descriptor['size'], bool) or descriptor['size'] != remote.reader.size:
            raise ValueError('source size does not match pinned descriptor')
        if (sources is None and descriptor.get('split_count', 1) != 1) or any(key.startswith('split.') for key in remote.metadata):
            raise ValueError('split GGUF sources require the separate qualified split reader')
        self.remote, self.descriptor, self.quant, self.quantizer = remote, dict(descriptor), quant, quantizer
        if sources is not None:
            self.descriptor['sources'] = [dict(item) for item in sources]
        self.should_stop, self.receipt, self.used = should_stop, None, False
        if not remote.tensors or len(remote.tensors) > 100000:
            raise ValueError('source tensor catalogue exceeds bounded planner limits')
        types = quantizer.plan(remote, quant)
        if set(types) != {_field(t, 'name') for t in remote.tensors}:
            raise ValueError('global native type plan does not cover the complete catalogue')
        self.tensors, offset = [], 0
        end = remote.data_offset
        names = set()
        for source in sorted(remote.tensors, key=lambda t: _field(t, 'offset')):
            name, shape, input_type = _field(source, 'name'), tuple(_field(source, 'shape')), _field(source, 'type')
            if name in names or not name or not 1 <= len(shape) <= 4 or any(not isinstance(d, int) or isinstance(d, bool) or d <= 0 for d in shape):
                raise ValueError('invalid or duplicate source tensor catalogue')
            names.add(name)
            if input_type not in SOURCE_TYPES:
                raise ValueError('streaming source must use F32/F16/BF16/Q8_0 tensors')
            output_type = types[name]
            if output_type not in LAYOUTS:
                raise ValueError('native planner selected an unqualified output format')
            block, width = LAYOUTS[output_type]
            if shape[0] % block or shape[0] * 4 > ROW_WORK_BYTES:
                raise ValueError('tensor row exceeds bounded stream or native block layout')
            input_block, input_width = LAYOUTS[input_type]
            expected_input = math.prod(shape) // input_block * input_width
            if shape[0] % input_block or expected_input != _field(source, 'nbytes'):
                raise ValueError('source tensor dimensions/type disagree with byte count')
            source_start = remote.data_offset + _field(source, 'offset')
            if source_start < end or source_start + expected_input > descriptor['size']:
                raise ValueError('source tensor data overlaps or extends beyond file')
            end = source_start + expected_input
            nbytes = math.prod(shape) // block * width
            self.tensors.append({'name': name, 'shape': shape, 'type': output_type,
                                 'input_type': input_type, 'source_start': source_start,
                                 'source_nbytes': expected_input, 'offset': offset, 'nbytes': nbytes})
            offset += nbytes + (-nbytes % remote.alignment)
        self.header = _output_header(remote, quant, self.tensors)
        self.size = len(self.header) + offset

    def __iter__(self):
        if self.used:
            raise RuntimeError('a quantized stream cannot be reused')
        self.used = True
        source_hash, output_hash = hashlib.sha256(), hashlib.sha256()
        cursor, output_count, proofs = 0, 0, []
        def check():
            if self.should_stop and self.should_stop():
                raise InterruptedError('streaming job paused or cancelled')
        def read_source(offset, count):
            check()
            raw = self.remote.reader.read_range(offset, count)
            if len(raw) != count:
                raise ValueError('source reader returned a short/excess range')
            source_hash.update(raw)
            return raw
        def skip_source(until):
            nonlocal cursor
            while cursor < until:
                size = min(ROW_WORK_BYTES, until - cursor)
                read_source(cursor, size)
                cursor += size
        def emit(raw):
            nonlocal output_count
            output_hash.update(raw)
            output_count += len(raw)
            return raw
        check()
        if self.descriptor.get('sources') is not None:
            self.remote.reader.begin_verified_pass()
        yield emit(self.header)
        for tensor in self.tensors:
            skip_source(tensor['source_start'])
            cols = tensor['shape'][0]
            rows = math.prod(tensor['shape'][1:])
            block, width = LAYOUTS[tensor['input_type']]
            input_rowbytes = cols // block * width
            rows_per_chunk = max(1, ROW_WORK_BYTES // (cols * 4))
            tensor_hash, tensor_bytes = hashlib.sha256(), 0
            for start in range(0, rows, rows_per_chunk):
                count = min(rows_per_chunk, rows - start) * input_rowbytes
                raw = read_source(cursor, count)
                cursor += count
                result = self.quantizer.quantize_rows(raw, tensor['input_type'], tensor['type'], cols)
                output_block, output_width = LAYOUTS[tensor['type']]
                if len(result) != (count // input_rowbytes) * cols // output_block * output_width:
                    raise RuntimeError('native output chunk byte count differs from global plan')
                tensor_hash.update(result)
                tensor_bytes += len(result)
                yield emit(result)
            if tensor_bytes != tensor['nbytes']:
                raise RuntimeError('quantized tensor byte count differs from global plan')
            padding = -tensor_bytes % self.remote.alignment
            if padding:
                yield emit(b'\0' * padding)
            proofs.append({'name': tensor['name'], 'type': tensor['type'], 'bytes': tensor_bytes,
                           'sha256': tensor_hash.hexdigest()})
        skip_source(self.descriptor['size'])
        source_checks = None
        if self.descriptor.get('sources') is not None:
            source_checks = self.remote.reader.finish_verified_pass()
            verified = source_checks.get('sources_verified') if isinstance(source_checks, dict) else None
            if not isinstance(verified, list) or len(verified) != len(self.descriptor['sources']):
                raise ValueError('split reader did not verify every immutable original source')
            for expected, actual in zip(self.descriptor['sources'], verified):
                if actual.get('sha256_verified') is not True or any(actual.get(key) != expected.get(key) for key in ('repo_id', 'revision', 'filename', 'size', 'sha256')):
                    raise ValueError('split reader source verification differs from the pinned manifest')
        elif source_hash.hexdigest() != self.descriptor['sha256'].lower():
            raise ValueError('complete original GGUF source SHA-256 mismatch')
        if output_count != self.size:
            raise RuntimeError('streamed output byte count differs from plan')
        check()
        self.receipt = {'source': self.descriptor, 'source_sha256_verified': True,
                        'sha256': output_hash.hexdigest(), 'size': output_count,
                        'quant': self.quant, 'tensors': proofs, 'tensor_count': len(proofs),
                        'native': getattr(self.quantizer, 'identity', {}),
                        'source_bytes_hashed': cursor,
                        'max_float_workspace_bytes': ROW_WORK_BYTES,
                        'source_materialized': False, 'output_materialized': False,
                        'full_inference_test_performed': False, 'stream_complete': True}
        if source_checks is not None:
            self.receipt['source_verification'] = source_checks
            self.receipt['concatenated_source_sha256'] = source_hash.hexdigest()


def stream_quantize(remote, descriptor, quant, quantizer, sink, should_stop=None):
    """Write bounded output chunks to a pipe/sink, return only a verified receipt."""
    stream = QuantizedStream(remote, descriptor, quant, quantizer, should_stop)
    for raw in stream:
        written = sink.write(raw)
        if written is not None and written != len(raw):
            raise RuntimeError('output sink accepted a short write')
    if hasattr(sink, 'flush'):
        sink.flush()
    if not stream.receipt:
        raise RuntimeError('stream ended without a verified receipt')
    return stream.receipt


class ShardedQuantizedStream:
    """One global native plan and source pass; tensor-boundary output shards.

    Iteration yields ``(zero_based_shard_index, bounded_bytes)``. Each file is
    valid split GGUF; tensors never span files. No receipt exists until the full
    source hashes have passed. Earlier shard ingestion is only unreferenced CAS.
    ``target_shard_bytes`` includes the header and is a hard per-file cap.
    """
    def __init__(self, remote, descriptor, quant, quantizer, should_stop=None,
                 target_shard_bytes=DEFAULT_SHARD_BYTES):
        if type(target_shard_bytes) is not int or not 1024 <= target_shard_bytes <= MAX_SHARD_BYTES:
            raise ValueError('output shard byte limit must be between 1024 and 50000000000')
        self.base = QuantizedStream(remote, descriptor, quant, quantizer, should_stop)
        self.used, self.receipt = False, None
        self.quant, self.target_shard_bytes = quant, target_shard_bytes
        # Every shard repeats the bounded metadata so headers remain self
        # describing. UINT16 split ids/count and INT32 total match gguf-split.
        entries = [entry for entry in remote.metadata_entries if not entry[0].startswith('split.')]
        entries += [('split.no', 2, struct.pack('<H', 0)),
                    ('split.count', 2, struct.pack('<H', 0)),
                    ('split.tensors.count', 5, struct.pack('<i', len(self.base.tensors)))]
        self._remote = SimpleNamespace(version=remote.version, alignment=remote.alignment,
                                       metadata_entries=entries)
        self._entries = entries
        # Header length is linear in per-tensor entries. Partition with exact
        # per-tensor sizes; split ids/count have fixed-width values.
        # Use unpadded header length to avoid off-by-one alignment calculations.
        replaced = {'general.file_type', 'general.quantization_version'}
        effective = [e for e in entries if e[0] not in replaced]
        effective += [('general.file_type', 4, struct.pack('<I', QUANTS[quant])),
                      ('general.quantization_version', 4, struct.pack('<I', 2))]
        fixed = 24 + sum(8 + len(key.encode('utf-8')) + 4 + len(raw) for key, _, raw in effective)
        groups, group, raw_header, data_size = [], [], fixed, 0
        for tensor in self.base.tensors:
            entry_size = 8 + len(tensor['name'].encode('utf-8')) + 4 + 8 * len(tensor['shape']) + 12
            payload_size = tensor['nbytes'] + (-tensor['nbytes'] % remote.alignment)
            prospective = raw_header + entry_size
            padded_header = prospective + (-prospective % remote.alignment)
            if group and padded_header + data_size + payload_size > target_shard_bytes:
                groups.append(group)
                group, raw_header, data_size = [], fixed, 0
                prospective = raw_header + entry_size
                padded_header = prospective + (-prospective % remote.alignment)
            if padded_header + data_size + payload_size > target_shard_bytes:
                raise ValueError('a single complete tensor plus GGUF metadata exceeds the output shard limit')
            group.append(tensor)
            raw_header, data_size = prospective, data_size + payload_size
        if group:
            groups.append(group)
        if len(groups) > 1024:
            raise ValueError('output shard count exceeds the qualified budget')
        self.shards = []
        for index, group in enumerate(groups):
            tensors, offset = [], 0
            for source in group:
                tensors.append(dict(source, offset=offset))
                offset += source['nbytes'] + (-source['nbytes'] % remote.alignment)
            header_size = fixed + sum(8 + len(tensor['name'].encode('utf-8')) + 4 + 8 * len(tensor['shape']) + 12 for tensor in tensors)
            header_size += -header_size % remote.alignment
            size = header_size + offset
            if size > target_shard_bytes:
                raise RuntimeError('output shard plan exceeded its exact byte limit')
            self.shards.append({'index': index, 'header_size': header_size, 'size': size, 'tensors': tensors})
        self.size = sum(item['size'] for item in self.shards)

    def _header_for(self, shard):
        entries = [e for e in self._entries if e[0] not in {'split.no', 'split.count'}]
        entries += [('split.no', 2, struct.pack('<H', shard['index'])),
                    ('split.count', 2, struct.pack('<H', len(self.shards)))]
        remote = SimpleNamespace(version=self._remote.version, alignment=self._remote.alignment,
                                 metadata_entries=entries)
        result = _output_header(remote, self.quant, shard['tensors'])
        if len(result) != shard['header_size']:
            raise RuntimeError('split metadata header differs from planned size')
        return result

    def __iter__(self):
        if self.used:
            raise RuntimeError('a sharded quantized stream cannot be reused')
        self.used = True
        iterator = iter(self.base)
        next(iterator)  # Replace the logical unsplit header with split headers.
        proofs = []
        for shard in self.shards:
            digest, count = hashlib.sha256(), 0
            raw = self._header_for(shard)
            digest.update(raw)
            count += len(raw)
            yield shard['index'], raw
            for tensor in shard['tensors']:
                remaining = tensor['nbytes'] + (-tensor['nbytes'] % self.base.remote.alignment)
                while remaining:
                    raw = next(iterator)
                    if not raw or len(raw) > remaining:
                        raise RuntimeError('native stream crossed a planned tensor boundary')
                    remaining -= len(raw)
                    digest.update(raw)
                    count += len(raw)
                    yield shard['index'], raw
            if count != shard['size']:
                raise RuntimeError('output shard byte count differs from its exact plan')
            proofs.append({'index': shard['index'], 'size': count, 'sha256': digest.hexdigest(),
                           'tensor_names': [tensor['name'] for tensor in shard['tensors']]})
        try:
            next(iterator)  # Exhaust source trailer and verify every source SHA.
        except StopIteration:
            pass
        else:
            raise RuntimeError('native source yielded unplanned tensor data')
        if not self.base.receipt or self.base.receipt.get('source_sha256_verified') is not True:
            raise RuntimeError('sharded stream ended without complete source verification')
        self.receipt = {key: value for key, value in self.base.receipt.items() if key not in {'sha256', 'size'}}
        self.receipt.update({'shards': proofs, 'size': sum(item['size'] for item in proofs),
                             'shard_count': len(proofs), 'target_shard_bytes': self.target_shard_bytes,
                             'logical_unsplit_sha256': self.base.receipt['sha256']})


def shard_filename(stem, quant, index, count):
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*', stem) or quant not in QUANTS or not 0 <= index < count <= 1024:
        raise ValueError('invalid streamed shard filename components')
    if count == 1:
        return f'{stem}-{quant}.gguf'  # one file, the name people expect: Model-Q4_K_M.gguf
    return f'{stem}-{quant}-{index + 1:05d}-of-{count:05d}.gguf'


def _xet_environment(api, repo_id, revision):
    import hf_xet
    from huggingface_hub.utils._xet import XetTokenType, xet_connection_info_refresh_url, xet_headers_without_auth
    from hfqa.target_policy import authorize_target
    authorize_target(repo_id,api.whoami())
    config = hf_xet.XetConfig().with_config({'data.max_concurrent_file_ingestion': 1,
        'client.ac_max_upload_concurrency': 4, 'client.ac_initial_upload_concurrency': 2,
        'shard.cache_size_limit': '256mb', 'shard.chunk_index_table_max_size': 1048576,
        'telemetry.enabled': False})
    session = hf_xet.XetSession(config=config)
    headers = api._build_hf_headers()
    connection = {'token_refresh_url': xet_connection_info_refresh_url(token_type=XetTokenType.WRITE,
        repo_id=repo_id, repo_type='model', revision=revision, endpoint=api.endpoint),
        'token_refresh_headers': headers, 'custom_headers': xet_headers_without_auth(headers)}
    return hf_xet, session, connection


def upload_xet_sharded_stream(api, repo_id, revision, stem, stream):
    """Stage all split files; finalize CAS only after full-source proof.

    No Hub commit is created here. A separate publisher must require real
    output inference receipts before adding any staged LFS pointers to a repo.
    """
    if not isinstance(stream, ShardedQuantizedStream) or stream.used:
        raise ValueError('Xet split ingestion requires a fresh sharded quantized stream')
    names = [shard_filename(stem, stream.quant, i, len(stream.shards)) for i in range(len(stream.shards))]
    hf_xet, session, connection = _xet_environment(api, repo_id, revision)
    files, active, index, sample = [], None, -1, b''
    def finish():
        nonlocal active
        native = active.finish().xet_info
        active = None
        planned = stream.shards[index]
        if native.file_size != planned['size'] or not re.fullmatch('[a-fA-F0-9]{64}', native.hash):
            raise RuntimeError('native Xet shard size/hash disagrees with the split plan')
        files.append({'path': names[index], 'size': native.file_size, 'sha256': native.sha256,
                      'xet_hash': native.hash, 'sample_base64': base64.b64encode(sample).decode('ascii')})
    with session.new_upload_commit(**connection) as commit:
        try:
            for next_index, raw in stream:
                if next_index != index:
                    if active is not None:
                        finish()
                    if next_index != index + 1:
                        raise RuntimeError('unexpected split output stream order')
                    index, sample = next_index, b''
                    active = commit.start_upload_stream(name=names[index], sha256=hf_xet.COMPUTE_SHA256)
                if len(sample) < 512:
                    sample = (sample + raw)[:512]
                written = active.write(raw)
                if written is not None and written != len(raw):
                    raise RuntimeError('native Xet shard accepted a short write')
            if not stream.receipt or stream.receipt.get('source_sha256_verified') is not True:
                raise RuntimeError('split source receipts are incomplete')
            if active is not None:
                finish()
            proofs = stream.receipt['shards']
            if len(files) != len(proofs) or any(item['sha256'] != proof['sha256'] or item['size'] != proof['size'] for item, proof in zip(files, proofs)):
                raise RuntimeError('native Xet shard SHA-256 differs from independent producer')
        except BaseException:
            if active is not None:
                active.abort()
            raise
    return {'files': files, 'receipt': stream.receipt, 'native_sha256_verified': True,
            'hub_commit_created': False, 'published': False}


def upload_xet_stream(api, repo_id, revision, filename, stream):
    """Native Xet ingestion only; caller commits the pointer AFTER receipt gates.

    This helper never publishes a Hub model card or creates a Hub commit.
    """
    if not isinstance(stream, QuantizedStream) or stream.used:
        raise ValueError('Xet ingestion requires a fresh verified quantized stream')
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*[.]gguf', filename):
        raise ValueError('invalid streamed GGUF output filename')
    if stream.size > MAX_SHARD_BYTES:
        raise ValueError('single streamed file exceeds the qualified Hub file-size budget; split output is required')
    hf_xet, session, connection = _xet_environment(api, repo_id, revision)
    sample = b''
    with session.new_upload_commit(**connection) as commit:
        native_stream = commit.start_upload_stream(name=filename, sha256=hf_xet.COMPUTE_SHA256)
        try:
            for raw in stream:
                if len(sample) < 512:
                    sample = (sample + raw)[:512]
                written = native_stream.write(raw)
                if written is not None and written != len(raw):
                    raise RuntimeError('native Xet stream accepted a short write')
            if not stream.receipt or not stream.receipt['source_sha256_verified']:
                raise RuntimeError('source/output receipts are incomplete')
            native = native_stream.finish().xet_info
            if (native.file_size != stream.receipt['size'] or native.sha256 != stream.receipt['sha256']
                    or not re.fullmatch('[a-fA-F0-9]{64}', native.hash)):
                raise RuntimeError('native Xet hash/size differs from independent producer')
        except BaseException:
            native_stream.abort()
            raise
    return {'path': filename, 'size': native.file_size, 'sha256': native.sha256,
            'xet_hash': native.hash, 'sample_base64': base64.b64encode(sample).decode('ascii'),
            'receipt': stream.receipt, 'native_sha256_verified': True,
            'hub_commit_created': False, 'published': False}
