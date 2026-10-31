"""Experimental helpers for legacy llama.cpp importance matrices.

Not connected to a release route.  A complete route also needs a matrix-aware
global type planner, verified provenance, and stock-quantizer parity tests.
"""
from __future__ import annotations

import hashlib
import re
import struct
from pathlib import Path

MAX_MATRIX_BYTES = 256 * 1024 ** 2


def load_legacy_matrix(path, expected_sha256):
    """Load a pinned .dat matrix and normalize it as the stock quantizer does.

    Reject malformed data before any native code sees a pointer. Newer GGUF
    matrix formats intentionally are not accepted by this experimental helper.
    """
    import numpy as np

    path = Path(path)
    if not 16 <= path.stat().st_size <= MAX_MATRIX_BYTES:
        raise ValueError('invalid importance matrix size')
    raw = path.read_bytes()
    if not 16 <= len(raw) <= MAX_MATRIX_BYTES:
        raise ValueError('invalid importance matrix size')
    if not isinstance(expected_sha256, str) or hashlib.sha256(raw).hexdigest() != expected_sha256.lower():
        raise ValueError('importance matrix checksum mismatch')
    pos = 0

    def take(n):
        nonlocal pos
        if n < 0 or pos + n > len(raw):
            raise ValueError('truncated importance matrix')
        data = raw[pos:pos + n]
        pos += n
        return data

    def integer():
        return struct.unpack('<i', take(4))[0]

    count = integer()
    if not 1 <= count <= 100000:
        raise ValueError('invalid importance matrix entry count')
    entries = {}
    for _ in range(count):
        length = integer()
        if not 1 <= length <= 1024:
            raise ValueError('invalid importance matrix name length')
        name = take(length).decode('utf-8', 'strict')
        if not name or '\0' in name or name in entries:
            raise ValueError('invalid or duplicate importance matrix name')
        calls, values = integer(), integer()
        if calls < 1 or not 1 <= values <= MAX_MATRIX_BYTES // 4:
            raise ValueError('invalid importance matrix counts')
        sums = np.frombuffer(take(values * 4), dtype='<f4')
        if not np.isfinite(sums).all() or (sums < 0).any():
            raise ValueError('invalid importance matrix values')
        # np.float32 division matches the stock loader's float arithmetic.
        weights = np.ascontiguousarray(sums / np.float32(calls), dtype=np.float32)
        if not np.isfinite(weights).all():
            raise ValueError('invalid normalized importance matrix values')
        entries[name] = weights
    chunks, length = integer(), integer()
    if chunks < 1 or not 1 <= length <= 8192:
        raise ValueError('invalid calibration metadata')
    dataset = take(length).decode('utf-8', 'strict')
    if pos != len(raw) or not any((v > 0).any() for v in entries.values()):
        raise ValueError('trailing data or empty importance matrix')
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'entries': entries,
            'chunks': chunks, 'dataset': dataset}


def expert_weight_slices(shape, weights, first_row, nrows):
    """Yield (offset, row count, weights) without crossing expert boundaries.

    llama.cpp expects ne[0] weights per expert, ne[1] rows per expert,
    and ne[2]*ne[3] experts. Shape uses GGML dimension order.
    """
    if not 1 <= len(shape) <= 4 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError('invalid tensor shape')
    dims = list(shape) + [1] * (4 - len(shape))
    cols, rows_per_expert, experts = dims[0], dims[1], dims[2] * dims[3]
    if len(weights) != cols * experts:
        raise ValueError('importance matrix does not match tensor shape')
    if type(first_row) is not int or type(nrows) is not int or first_row < 0 or nrows < 1 or first_row + nrows > rows_per_expert * experts:
        raise ValueError('invalid tensor row interval')
    offset = 0
    while offset < nrows:
        row = first_row + offset
        expert = row // rows_per_expert
        take = min(nrows - offset, rows_per_expert - row % rows_per_expert)
        yield offset, take, weights[expert * cols:(expert + 1) * cols]
        offset += take


# Inspected upstream source: the has_imatrix branches change IQ types and Q4_0/
# Q5_0, but not these two K presets. This narrow equivalence is revision-bound;
# it is NOT proof that the whole stream matches llama-quantize.
EXPERIMENTAL_LLAMA_REVISION = '19ff817fb25e9a1b838b1d7f9f4325f7bf9779c4'
EXPERIMENTAL_PRESETS = frozenset({'Q4_K_M', 'Q5_K_M'})


from hfqa.streaming import DEFAULT_SHARD_BYTES, LAYOUTS, QuantizedStream, ShardedQuantizedStream


class ExperimentalImatrixStream(QuantizedStream):
    """Revision- and preset-gated prototype, never wired to publishing.

    No claim is made that the matrix was made from this source; callers must
    independently establish that provenance and stock quantizer parity.
    """
    def __init__(self, remote, descriptor, quant, quantizer, matrix, should_stop=None):
        import numpy as np

        if quant not in EXPERIMENTAL_PRESETS or getattr(quantizer, 'identity', {}).get('runtime_revision') != EXPERIMENTAL_LLAMA_REVISION:
            raise ValueError('experimental matrix planning is not qualified for this revision or preset')
        if (not isinstance(matrix, dict) or not isinstance(matrix.get('entries'), dict) or not matrix['entries']
                or not re.fullmatch(r'[0-9a-f]{64}', str(matrix.get('sha256', '')))):
            raise ValueError('a checked importance matrix is required')
        super().__init__(remote, descriptor, quant, quantizer, should_stop)
        self._matrix_sha256 = matrix['sha256']
        self._weights = {}
        for tensor in self.tensors:
            weights = matrix['entries'].get(tensor['name'])
            if weights is None or tensor['input_type'] == tensor['type']:
                continue
            shape = tensor['shape'] + (1,) * (4 - len(tensor['shape']))
            arr = np.array(weights, dtype=np.float32, copy=True, order='C')
            if arr.ndim != 1 or len(arr) != shape[0] * shape[2] * shape[3] or not np.isfinite(arr).all() or (arr < 0).any():
                raise ValueError('importance matrix dimensions or values do not match tensor')
            arr.flags.writeable = False
            self._weights[tensor['name']] = arr
        if not self._weights or not any((value > 0).any() for value in self._weights.values()):
            raise ValueError('importance matrix has no positive weights for quantized tensors')

    def _quantize_chunk(self, tensor, raw, first_row):
        weights = self._weights.get(tensor['name'])
        if weights is None:
            return super()._quantize_chunk(tensor, raw, first_row)
        cols = tensor['shape'][0]
        block, width = LAYOUTS[tensor['input_type']]
        row_bytes = cols // block * width
        if len(raw) % row_bytes:
            raise ValueError('source row boundary mismatch')
        results = []
        for offset, nrows, expert_weights in expert_weight_slices(tensor['shape'], weights, first_row, len(raw) // row_bytes):
            chunk = raw[offset * row_bytes:(offset + nrows) * row_bytes]
            results.append(self.quantizer.quantize_rows(chunk, tensor['input_type'], tensor['type'], cols, weights=expert_weights))
        return b''.join(results)

    def __iter__(self):
        yield from super().__iter__()
        if self.receipt is not None:
            self.receipt.update(experimental=True, matrix_sha256=self._matrix_sha256,
                                matrix_tensor_count=len(self._weights), published=False)


class ProvenanceCheckedImatrixStream(ExperimentalImatrixStream):
    """Experimental stream with pinned matrix evidence; not a release gate.

    The caller must authenticate the worker and tool build independently.
    This wrapper does not establish inference or publication eligibility.
    """
    def __init__(self, remote, descriptor, quant, quantizer, matrix_path, evidence, calibration, should_stop=None):
        from hfqa.imatrix_provenance import validate_matrix_provenance
        revision = getattr(quantizer, 'identity', {}).get('runtime_revision')
        from hfqa.source_identity import identity
        checked = validate_matrix_provenance(evidence, source=identity(descriptor), calibration=calibration, runtime_revision=revision)
        # Load and hash the bytes here; an arbitrary caller-supplied matrix dict
        # could otherwise claim the correct hash while containing other weights.
        matrix = load_legacy_matrix(matrix_path, checked['matrix_sha256'])
        self._provenance = checked
        super().__init__(remote, descriptor, quant, quantizer, matrix, should_stop)

    def __iter__(self):
        yield from super().__iter__()
        if self.receipt is not None:
            self.receipt['imatrix_provenance'] = dict(self._provenance)


class ProvenanceCheckedShardedImatrixStream(ShardedQuantizedStream):
    """Experimental shard writer using the verified weighted stream.

    Publication and inference remain outside this class. A split source is bound
    by the identity of its ordered file set (hfqa.source_identity).
    """
    def __init__(self, remote, descriptor, quant, quantizer, matrix_path, evidence, calibration,
                 should_stop=None, target_shard_bytes=DEFAULT_SHARD_BYTES):
        self._matrix_args = (matrix_path, evidence, calibration)
        super().__init__(remote, descriptor, quant, quantizer, should_stop, target_shard_bytes)

    def _make_base(self, remote, descriptor, quant, quantizer, should_stop):
        return ProvenanceCheckedImatrixStream(remote, descriptor, quant, quantizer,
                                              *self._matrix_args, should_stop=should_stop)
