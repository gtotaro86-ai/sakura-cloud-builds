"""Runner-side release with an importance matrix, made entirely by the STOCK llama.cpp tools (public worker).

For models whose source is one file small enough for a hosted runner (a few GB): download the pinned source and check its hash, compute ONE importance matrix
with `llama-imatrix` on the fixed calibration text, then for every requested quant run `llama-quantize --imatrix`, verify the output file (valid GGUF, the
same tensors as the source), run a real inference with the stock runtime and stage the file on a branch of the target repository. `main` is never written here.
Because the output comes from the stock quantizer there is nothing to prove against it; what is recorded is the evidence: hashes of the source, the matrix, the
calibration text and the tools, plus the commands.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from hfqa.gguf import validate_gguf
from hfqa.imatrix_data import DATASET, DATASET_ID, DATASET_LICENSE, DATASET_SHA256
from hfqa.runner_release import SCHEMA, remaining_job_seconds, run_inference, runner_facts, sha256_file, stage_on_branch
from hfqa.streaming import shard_filename

MAX_SOURCE_BYTES = 8 * 10 ** 9          # the model must run on the runner's CPU and memory for the calibration
CONTEXT = 128
IMATRIX_TIMEOUT_S = 40 * 60
QUANTIZE_TIMEOUT_S = 40 * 60
METHOD = 'stock_imatrix'


def fetch_source(descriptor: dict, directory: Path) -> Path:
    """Download the one pinned source file and prove it is the file Oracle pinned (size and SHA-256)."""
    from huggingface_hub import hf_hub_download
    directory.mkdir(parents=True, exist_ok=True)
    path = Path(hf_hub_download(repo_id=descriptor['repo_id'], filename=descriptor['filename'], revision=descriptor['revision'], local_dir=str(directory)))
    if path.stat().st_size != descriptor['size'] or sha256_file(path) != descriptor['sha256']:
        raise RuntimeError('the downloaded source differs from the pinned descriptor')
    return path


def _tool_env(tool: Path) -> dict:
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(tool.parent) + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    return env


def run_tool(command: list[str], logs: Path, name: str, timeout: int) -> float:
    """Run one stock tool; stdout and stderr go to files (bounded memory, kept as evidence). Returns seconds."""
    logs.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (logs / f'{name}.stdout.log').open('wb') as out, (logs / f'{name}.stderr.log').open('wb') as err:
        completed = subprocess.run(command, check=False, stdin=subprocess.DEVNULL, stdout=out, stderr=err, timeout=timeout, env=_tool_env(Path(command[0])))
    if completed.returncode != 0:
        raise RuntimeError(f'{Path(command[0]).name} failed (exit {completed.returncode}); logs: {logs / (name + ".stderr.log")}')
    return round(time.monotonic() - started, 1)


def validate_matrix(path: Path) -> dict:
    """Legacy .dat structure: refuse an empty, truncated, non-finite or negative matrix before any quantization uses it."""
    import math
    import struct
    if not 16 <= path.stat().st_size <= 256 * 1024 ** 2:
        raise ValueError('invalid importance matrix size')
    with path.open('rb') as handle:
        def read(n):
            data = handle.read(n)
            if len(data) != n:
                raise ValueError('truncated importance matrix')
            return data
        entries = struct.unpack('<i', read(4))[0]
        if not 1 <= entries <= 100000:
            raise ValueError('invalid importance matrix entry count')
        names, positive = set(), 0
        for _ in range(entries):
            length = struct.unpack('<i', read(4))[0]
            if not 1 <= length <= 1024:
                raise ValueError('invalid importance matrix tensor name')
            name = read(length).decode('utf-8', 'strict')
            if name in names:
                raise ValueError('duplicate importance matrix tensor name')
            names.add(name)
            calls, count = struct.unpack('<ii', read(8))
            if calls < 1 or not 1 <= count <= 64_000_000:
                raise ValueError('invalid importance matrix counts')
            values = struct.unpack(f'<{count}f', read(count * 4))
            if any(not math.isfinite(v) or v < 0 for v in values):
                raise ValueError('non-finite or negative importance matrix value')
            positive += any(v > 0 for v in values)
        chunks = struct.unpack('<i', read(4))[0]
        length = struct.unpack('<i', read(4))[0]
        if chunks < 1 or not 1 <= length <= 8192:
            raise ValueError('importance matrix without calibration chunk count')
        read(length)
        if handle.read(1):
            raise ValueError('unexpected trailing data in the importance matrix')
    if not positive:
        raise ValueError('importance matrix holds no data')
    return {'entries': entries, 'chunks_processed': chunks}


def make_matrix(source: Path, tools: dict, work: Path, cpus: int, *, log=print) -> dict:
    dataset = work / f'{DATASET_ID}.txt'
    dataset.write_bytes(DATASET)
    matrix = work / 'importance.dat'
    command = [str(tools['imatrix']), '-m', str(source), '-f', str(dataset), '-o', str(matrix), '--output-format', 'dat', '--no-ppl',
               '-t', str(cpus), '-tb', str(cpus), '-c', str(CONTEXT), '-b', str(CONTEXT), '-ub', str(CONTEXT), '-ngl', '0']
    seconds = run_tool(command, work / 'logs', 'imatrix', IMATRIX_TIMEOUT_S)
    parsed = validate_matrix(matrix)
    evidence = {'matrix_sha256': sha256_file(matrix), 'dataset_id': DATASET_ID, 'dataset_license': DATASET_LICENSE, 'dataset_sha256': DATASET_SHA256,
                'dataset_bytes': len(DATASET), 'context': CONTEXT, 'tool_sha256': sha256_file(tools['imatrix']), 'seconds': seconds,
                'command': [Path(command[0]).name] + [Path(a).name if a.startswith('/') else a for a in command[1:]], **parsed}
    log(json.dumps({'event': 'imatrix', **{k: evidence[k] for k in ('matrix_sha256', 'entries', 'chunks_processed', 'seconds')}}), flush=True)
    return {'path': matrix, 'evidence': evidence}


def release_quants(manifest: dict, source: dict, quants: list[str], *, runtime_dir: Path, work: Path, job_id: str, api=None, branch: str | None = None,
                   env: dict | None = None, log=print, fetch=fetch_source, tools=None) -> dict:
    """All requested quants in this job (cheap for models of this size). Returns {quant: receipt}; nothing is written to `main`."""
    if source.get('sources') and len(source['sources']) != 1:
        raise ValueError('the importance-matrix path takes exactly one source file')
    descriptor = (source.get('sources') or [source])[0]
    if descriptor['size'] > MAX_SOURCE_BYTES:
        raise ValueError('the source is too large for the importance-matrix path')
    runtime_dir = Path(runtime_dir)
    tools = tools or {'imatrix': runtime_dir / 'llama-imatrix', 'quantize': runtime_dir / 'llama-quantize', 'cli': runtime_dir / 'llama-cli'}
    for name, tool in tools.items():
        if not Path(tool).is_file():
            raise FileNotFoundError(f'the stock runtime lacks {Path(tool).name}')
    t0 = time.monotonic()
    facts = runner_facts(work)
    cpus = facts['cpus'] or 1
    source_path = fetch(descriptor, work / 'source')
    report = validate_gguf(source_path)
    if report.get('structure_valid') is not True:
        raise RuntimeError('the source is not a valid GGUF')
    structure = {'architecture': report['architecture'], 'tensor_count': report['tensor_count'], 'tensor_name_set_sha256': report['tensor_name_set_sha256'],
                 'tensor_type_histogram': report.get('tensor_type_histogram')}
    matrix = make_matrix(source_path, tools, work, cpus, log=log)
    t_matrix = time.monotonic()
    stem = manifest.get('output_stem') or Path(descriptor['filename']).stem
    requantize = str(manifest.get('source_precision') or '').lower() == 'q8_0'
    receipts = {}
    for quant in quants:
        t1 = time.monotonic()
        out = work / quant / shard_filename(stem, quant, 0, 1)
        out.parent.mkdir(parents=True, exist_ok=True)
        command = [str(tools['quantize']), '--imatrix', str(matrix['path'])] + (['--allow-requantize'] if requantize else []) + [str(source_path), str(out), quant, str(cpus)]
        q_seconds = run_tool(command, work / quant / 'logs', 'quantize', QUANTIZE_TIMEOUT_S)
        check = validate_gguf(out, expected_architecture=structure['architecture'])
        if check.get('structure_valid') is not True or check['tensor_count'] != structure['tensor_count'] or check['tensor_name_set_sha256'] != structure['tensor_name_set_sha256']:
            raise RuntimeError(f'{quant}: the quantized file does not hold exactly the source tensors')
        item = {'path': str(out), 'name': out.name, 'size': out.stat().st_size, 'sha256': sha256_file(out)}
        inference = run_inference(Path(tools['cli']), [item], work / quant / 'logs', cpus, facts.get('memory_total_bytes'), remaining_job_seconds(t0))
        t2 = time.monotonic()
        staging = None
        if api is not None:
            staging = stage_on_branch(api, manifest['target_repo_id'], branch or f'hfqa-staging-{job_id}', [item], private=bool(manifest.get('private')),
                                      message=f'HFQA validated {quant} (staging, not a release)')
        t3 = time.monotonic()
        receipts[quant] = {
            'schema': SCHEMA, 'method': METHOD, 'job_id': job_id, 'quant': quant, 'state': 'RUNNER_VALIDATED' + ('_STAGED' if staging else ''),
            'runner': {**facts, **(env or {})},
            'native_identity': {'runtime_revision': manifest['runtime_revision'], 'quantize_sha256': sha256_file(tools['quantize']),
                                'imatrix_tool_sha256': matrix['evidence']['tool_sha256'], 'llama_sha256': sha256_file(tools['cli'])},
            'source': {'descriptors': [descriptor], 'structure': structure, 'source_sha256_verified': True, 'concatenated_source_sha256': descriptor['sha256'],
                       'source_materialized': True},
            'imatrix': matrix['evidence'],
            'quantize': {'command': [Path(command[0]).name] + [Path(a).name if a.startswith('/') else a for a in command[1:]], 'seconds': q_seconds},
            'output': {'shard_count': 1, 'total_bytes': item['size'], 'tensor_count': structure['tensor_count'],
                       'shards': [{'name': item['name'], 'size': item['size'], 'sha256': item['sha256']}],
                       'verification': [{'name': item['name'], 'tensor_count': check['tensor_count'], 'architecture': check.get('architecture'),
                                         'tensor_type_histogram': check.get('tensor_type_histogram'), 'tensor_name_set_sha256': check['tensor_name_set_sha256']}]},
            'inference': {k: inference[k] for k in ('runtime_sha256', 'runtime_build', 'model_sha256', 'returncode', 'prompt', 'completion',
                                                     'context', 'predict', 'threads', 'command', 'seconds', 'model_bytes', 'shard_count')},
            'staging': staging, 'inference_tested': True, 'full_inference_test_performed': True, 'runtime_compatible': True,
            'published': False, 'hub_main_changed': False,
            'timings_seconds': {'matrix': round(t_matrix - t0, 1), 'quantize': round(q_seconds, 1), 'verify_and_infer': round(t2 - t1 - q_seconds, 1), 'upload': round(t3 - t2, 1)},
            'note': 'Stock llama-quantize with an importance matrix from fixed calibration text; one smoke prompt; no quality benchmark is claimed.',
        }
        if not manifest.get('keep_outputs'):
            out.unlink(missing_ok=True)  # disk is the scarce resource on a runner
        log(json.dumps({'event': 'quant_done', 'quant': quant, 'bytes': item['size']}), flush=True)
    return receipts
