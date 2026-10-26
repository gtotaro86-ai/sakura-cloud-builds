"""Runner side of a hybrid release (portable, public worker): build the replacement table file, check it, stage it on a branch.

The backbone files of the other provider are NOT touched here: Oracle copies them server-side at release time, byte for byte.
This step only produces the one new table file. Order is the contract: re-check that the source table is the backbone's own
table -> stream the conversion (every source file hashed completely) -> write the file -> verify bytes and structure -> only
then upload to a staging branch. `main` is never written. No inference is run (these models are far larger than a runner can
load), and the receipt says so.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path

from hfqa import hybrid
from hfqa.remote_gguf import BufferedRangeReader, RangeReader, inspect_remote_gguf
from hfqa.runner_release import DISK_RESERVE, runner_facts, sha256_file, stage_on_branch

SCHEMA = 'hfqa.hybrid-receipt/1'
SELF_CHECK_TOLERANCE = 0.30


def open_remote(descriptor: dict):
    reader = BufferedRangeReader(RangeReader(descriptor))
    return inspect_remote_gguf(descriptor, reader=reader, allow_split=True, extra_layouts=hybrid.NUMPY_LAYOUTS)


class _LocalReader:
    def __init__(self, path):
        self.path, self.size = Path(path), Path(path).stat().st_size

    def read_range(self, start, count):
        with self.path.open('rb') as handle:
            handle.seek(start)
            return handle.read(count)


def materialize(stream, path: Path, *, log=print, every: float = 60.0) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path.parent).free
    if free < stream.size + DISK_RESERVE:
        raise OSError(f'not enough disk for the table file ({stream.size} bytes) plus reserve; {free} bytes free')
    if path.exists():
        raise FileExistsError('output file already exists; use a fresh work directory')
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.part')
    written, last, started = 0, time.monotonic(), time.monotonic()
    try:
        with temp.open('xb') as handle:
            for raw in stream:
                if handle.write(raw) != len(raw):
                    raise OSError('short write while materializing the table file')
                written += len(raw)
                if time.monotonic() - last >= every:
                    last = time.monotonic()
                    log(json.dumps({'event': 'progress', 'output_bytes': written, 'planned_bytes': stream.size, 'seconds': round(last - started)}), flush=True)
            handle.flush()
            os.fsync(handle.fileno())
        receipt = stream.receipt
        if not receipt or receipt.get('stream_complete') is not True:
            raise RuntimeError('the table stream ended without complete source verification')
        if temp.stat().st_size != receipt['size'] or sha256_file(temp) != receipt['sha256']:
            raise RuntimeError('the written table file differs from the streamed bytes')
        os.replace(temp, path)
        return receipt
    finally:
        temp.unlink(missing_ok=True)


def verify_output(path: Path, template, plan: dict, ggml: hybrid.Ggml, sources: dict) -> dict:
    """The file is a GGUF with the template's key/value block and the planned tensors, and its rows decode like the sources."""
    import numpy as np
    out = inspect_remote_gguf({'repo_id': 'local/file', 'revision': '0' * 40, 'filename': path.name, 'size': path.stat().st_size, 'sha256': '0' * 64},
                              reader=_LocalReader(path), allow_split=True)
    if [tuple(e) for e in out.metadata_entries] != [tuple(e) for e in template.metadata_entries]:
        raise RuntimeError('the output key/value block differs from the backbone template')
    expected = {t['name']: (tuple(t['shape']), t['type']) for t in plan['tensors']}
    got = {t.name: (tuple(t.shape), t.type) for t in out.tensors}
    if got != expected:
        raise RuntimeError('the output tensor catalogue differs from the plan')
    checked = []
    for tensor in plan['tensors']:
        block, width = ggml.layout(tensor['type'])
        row_bytes = tensor['shape'][0] // block * width
        base = out.data_offset + next(t.offset for t in out.tensors if t.name == tensor['name'])
        first = 0
        for part in tensor['parts']:
            source = sources[part['source']]['remote']
            src_block, src_width = ggml.layout(part['in_type'])
            src_row = tensor['shape'][0] // src_block * src_width
            for local in sorted({0, part['rows'] // 2, part['rows'] - 1}):
                produced = ggml.decode(out.reader.read_range(base + (first + local) * row_bytes, row_bytes), tensor['type'], tensor['shape'][0])[0]
                original = ggml.decode(source.reader.read_range(part['start'] + local * src_row, src_row), part['in_type'], tensor['shape'][0])[0]
                error = float(np.sqrt(np.mean((produced - original) ** 2)) / (np.sqrt(np.mean(original ** 2)) + 1e-9))
                if error > SELF_CHECK_TOLERANCE:
                    raise RuntimeError(f"row {first + local} of {tensor['name']} decodes {error:.3f} away from its source")
                checked.append(round(error, 5))
            first += part['rows']
    return {'structure_valid': True, 'kv_block_identical_to_backbone_template': True, 'tensors': sorted(got), 'rows_decoded_against_source': len(checked),
            'max_relative_rms': max(checked) if checked else None}


def run_hybrid(manifest: dict, *, ggml_path, work: Path, job_id: str, api=None, branch: str | None = None, env: dict | None = None, log=print) -> dict:
    spec = manifest['hybrid']
    t0 = time.monotonic()
    facts = runner_facts(work)
    ggml = hybrid.Ggml(ggml_path, manifest['runtime_revision'])
    template = open_remote(spec['template'])
    sources = {key: {'remote': open_remote(descriptor), 'descriptor': descriptor} for key, descriptor in spec['sources'].items()}
    own_name = spec.get('compat', {}).get('own_tensor')
    compat = None
    if own_name:
        parts = [(sources[p['source']]['remote'], p['tensor']) for output in spec['outputs'] for p in output['parts']]
        compat = hybrid.sample_compatibility(ggml, (template, own_name), parts)
        if not compat['passes']:
            raise RuntimeError('the replacement table does not match the backbone table: ' + json.dumps(compat))
    plan = hybrid.build_plan(template, spec['outputs'], sources, int(spec['out_type']), ggml, kv_overrides=None)
    stream = hybrid.HybridTableStream(template, plan, sources, ggml)
    log(json.dumps({'event': 'plan', 'planned_bytes': stream.size, 'runner': facts, 'compat': compat}), flush=True)
    t1 = time.monotonic()
    name = spec['output_filename']
    path = work / name
    receipt = materialize(stream, path, log=log)
    t2 = time.monotonic()
    verification = verify_output(path, template, plan, ggml, sources)
    t3 = time.monotonic()
    staging = None
    if api is not None:
        staging = stage_on_branch(api, manifest['target_repo_id'], branch or f'hfqa-staging-{job_id}',
                                  [{'name': name, 'path': str(path), 'size': receipt['size'], 'sha256': receipt['sha256']}],
                                  private=bool(manifest.get('private')), message='HFQA hybrid table file (staging, not a release)')
    t4 = time.monotonic()
    path.unlink(missing_ok=True)
    return {'schema': SCHEMA, 'job_id': job_id, 'state': 'HYBRID_BUILT' + ('_STAGED' if staging else ''), 'runner': {**facts, **(env or {})},
            'native_identity': ggml.identity, 'table': receipt, 'compatibility': compat, 'verification': verification,
            'output': {'name': name, 'size': receipt['size'], 'sha256': receipt['sha256']}, 'staging': staging,
            'inference_tested': False, 'full_inference_test_performed': False, 'integrity_only': True, 'published': False, 'hub_main_changed': False,
            'timings_seconds': {'plan': round(t1 - t0, 1), 'convert_and_write': round(t2 - t1, 1), 'verify': round(t3 - t2, 1), 'upload': round(t4 - t3, 1)},
            'note': 'Integrity checks only: source hashes, byte identity of the written file, structure and sampled rows. No model was loaded or run.'}
