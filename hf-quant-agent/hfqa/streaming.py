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

QUANTS = {'Q2_K': 10, 'Q3_K_M': 12, 'Q4_K_M': 15, 'Q5_K_M': 17}
# GGML type -> (block elements, block bytes)
LAYOUTS = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20),
           6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 40),
           10: (256, 84), 11: (256, 110), 12: (256, 144),
           13: (256, 176), 14: (256, 210), 15: (256, 292), 30: (1, 2)}
SOURCE_TYPES = {0, 1, 8, 30}
ROW_WORK_BYTES = 8 * 1024**2
DEFAULT_SHARD_BYTES = 4 * 1024**3
MAX_SHARD_BYTES = 50_000_000_000


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


def model_hparams(metadata):
    arch = metadata.get('general.architecture')
    if not isinstance(arch, str) or not re.fullmatch('[a-z0-9_-]+', arch):
        raise ValueError('missing or invalid stock GGUF architecture')
    prefix = arch + '.'
    embd = _uniform_int(metadata, prefix + 'embedding_length')
    head = _uniform_int(metadata, prefix + 'attention.head_count')
    if not head or not embd or embd % head:
        raise ValueError('invalid native attention dimensions')
    params = [embd, _uniform_int(metadata, prefix + 'feed_forward_length'),
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
        hparams = (ctypes.c_uint32 * 8)(*values)
        if self.bridge.hfqa_plan(QUANTS[quant], arch.encode(), hparams, count, names, shapes, dims, inputs, outputs):
            raise RuntimeError((self.bridge.hfqa_error() or b'native planner failed').decode('utf-8', 'replace'))
        return {_field(t, 'name'): int(outputs[i]) for i, t in enumerate(tensors)}

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
        …12941 tokens truncated…b/hf-quant-agent/data", "/var/lib/hf-quant-agent/work"):
        shutil.chown(path, user="hfquant", group="hfquant")
        os.chmod(path, 0o750)
    if not os.path.exists(ROOT + "/.venv/bin/pip"):
        run(["python3", "-m", "venv", ROOT + "/.venv"])
        run([ROOT + "/.venv/bin/pip", "install", "-r", ROOT + "/requirements.txt"])
    run(["systemctl", "daemon-reload"])
    run(["systemctl", "enable", "--now", "hf-quant-agent"])
    time.sleep(2)
    print(subprocess.check_output(["systemctl", "is-active", "hf-quant-agent", "AdGuardHome", "unbound", "tailscaled"], text=True))


if __name__ == "__main__":
    main()
