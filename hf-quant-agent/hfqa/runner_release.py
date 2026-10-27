"""Runner-side release step (portable, public worker): quantize one quant, prove it, then stage it.

Order is the contract: stream-quantize from the pinned source (never stored) -> write the shards -> verify every
shard (structure, tensor coverage, hashes) -> run a real inference with the unchanged stock runtime on the exact
shards -> only then upload them to a staging branch. `main` of the target repository is never written here.
The receipt it returns is what the controller on Oracle checks before it makes the final commit.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import struct
import time
import uuid
from pathlib import Path

from hfqa.gguf import TENSOR_TYPES, validate_gguf
from hfqa.quantize import inference_budget, test_inference
from hfqa.remote_gguf import BufferedRangeReader, RangeReader, inspect_remote_gguf, inspect_remote_source_set
from hfqa.streaming import DEFAULT_SHARD_BYTES, NativeQuantizer, ShardedQuantizedStream, shard_filename

SCHEMA = 'hfqa.runner-release-receipt/1'
DISK_RESERVE = 2 * 1024 ** 3


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def names_hash(names) -> str:
    digest = hashlib.sha256()
    for name in sorted(names):
        raw = name.encode('utf-8')
        digest.update(struct.pack('<Q', len(raw)) + raw)
    return digest.hexdigest()


def source_structure(remote) -> dict:
    counts: dict = {}
    for tensor in remote.tensors:
        kind = TENSOR_TYPES[tensor.type]
        counts[kind] = counts.get(kind, 0) + 1
    return {'architecture': remote.metadata['general.architecture'], 'tensor_count': len(remote.tensors),
            'tensor_name_set_sha256': names_hash([tensor.name for tensor in remote.tensors]), 'tensor_type_histogram': counts}


def runner_facts(work: Path) -> dict:
    """Measured, not assumed: what machine actually ran the job."""
    memory = None
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemTotal:'):
                memory = int(line.split()[1]) * 1024
    except OSError:
        pass
    work.mkdir(parents=True, exist_ok=True)
    return {'cpus': os.cpu_count(), 'memory_total_bytes': memory, 'disk_free_bytes': shutil.disk_usage(work).free,
            'platform': platform.platform(), 'machine': platform.machine()}


def open_source(source: dict):
    if source.get('sources'):
        readers = [BufferedRangeReader(RangeReader(item)) for item in source['sources']]
        return inspect_remote_source_set(source['sources'], readers=readers)
    return inspect_remote_gguf(source, reader=BufferedRangeReader(RangeReader(source)))


def materialize(stream: ShardedQuantizedStream, directory: Path, stem: str, *, log=print, every: float = 60.0) -> list[dict]:
    """Write every output shard. All bytes are checked against the stream's own receipt before the files appear."""
    directory.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(directory).free
    if free < stream.size + DISK_RESERVE:
        raise OSError(f'not enough disk for the planned output ({stream.size} bytes) plus reserve; {free} bytes free')
    count = len(stream.shards)
    names = [shard_filename(stem, stream.quant, i, count) for i in range(count)]
    finals = [directory / name for name in names]
    if any(path.exists() for path in finals):
        raise FileExistsError('output shards already exist; use a fresh work directory')
    temps, handle, current, written, last, started = [], None, -1, 0, time.monotonic(), time.monotonic()
    try:
        for index, raw in stream:
            if index != current:
                if handle:
                    handle.flush(); os.fsync(handle.fileno()); handle.close()
                temp = finals[index].with_name(finals[index].name + '.' + uuid.uuid4().hex + '.part')
                handle = temp.open('xb'); temps.append(temp); current = index
            if handle.write(raw) != len(raw):
                raise OSError('short write while materializing a shard')
            written += len(raw)
            if time.monotonic() - last >= every:
                last = time.monotonic()
                log(json.dumps({'event': 'progress', 'quant': stream.quant, 'output_bytes': written, 'planned_bytes': stream.size,
                                'shard': current + 1, 'shards': count, 'seconds': round(last - started)}), flush=True)
        if handle:
            handle.flush(); os.fsync(handle.fileno()); handle.close(); handle = None
        receipt = stream.receipt
        if not receipt or receipt.get('source_sha256_verified') is not True:
            raise RuntimeError('quantization finished without complete source verification')
        files = []
        for temp, final, proof in zip(temps, finals, receipt['shards']):
            if temp.stat().st_size != proof['size'] or sha256_file(temp) != proof['sha256']:
                raise RuntimeError('written shard differs from the streamed bytes')
            os.replace(temp, final)
            files.append({'path': str(final), 'name': final.name, 'size': proof['size'], 'sha256': proof['sha256'],
                          'tensor_names': proof['tensor_names']})
        return files
    finally:
        if handle:
            handle.close()
        for temp in temps:
            temp.unlink(missing_ok=True)


def verify_shards(files: list[dict], structure: dict, tensor_count: int) -> list[dict]:
    """Every shard is a valid GGUF of the right architecture; together they cover exactly the source tensors."""
    reports, names = [], []
    for item in files:
        report = validate_gguf(item['path'], expected_architecture=structure['architecture'])
        if report.get('structure_valid') is not True:
            raise RuntimeError(f"shard {item['name']} is not a valid GGUF: {report.get('note')}")
        if report['tensor_count'] != len(item['tensor_names']) or report['tensor_name_set_sha256'] != names_hash(item['tensor_names']):
            raise RuntimeError(f"shard {item['name']} does not hold the tensors the plan assigned to it")
        names.extend(item['tensor_names'])
        reports.append({'name': item['name'], 'tensor_count': report['tensor_count'], 'architecture': report.get('architecture'),
                        'tensor_type_histogram': report.get('tensor_type_histogram'), 'tensor_name_set_sha256': report['tensor_name_set_sha256']})
    if len(set(names)) != len(names) or len(names) != tensor_count or names_hash(names) != structure['tensor_name_set_sha256']:
        raise RuntimeError('the shard set does not cover exactly the source tensor catalogue')
    return reports


JOB_SECONDS = 350 * 60      # timeout-minutes of one release job in the workflow
SETUP_SECONDS = 10 * 60     # checkout, runtime download and bridge build before the step starts
UPLOAD_RESERVE_SECONDS = 15 * 60
REPACK_OFF_MEMORY_MULTIPLE = 2.0  # a model more than twice the machine's memory cannot be repacked at all: do not waste the first attempt
ALLOCATION_FAILURES = ('CPU_REPACK', 'insufficient memory', 'failed to allocate')


def remaining_job_seconds(started_monotonic: float) -> int:
    """Time left in the job for the real inference, so a long timeout cannot outlive the job and throw away the quantization."""
    return int(JOB_SECONDS - SETUP_SECONDS - UPLOAD_RESERVE_SECONDS - (time.monotonic() - started_monotonic))


def run_inference(llama_cli: Path, files: list[dict], logs: Path, cpus: int, memory_bytes: int | None = None, remaining_seconds: int | None = None) -> dict:
    """One real generation by the stock runtime on the exact shards (the first shard loads the rest).

    The proven default (weight repacking on) is tried first. If the runtime then cannot allocate its repacked copy of the weights (a model larger than
    the machine's memory), the same test is repeated once with the stock option --no-repack; any other failure is reported as it is.
    """
    total = sum(item['size'] for item in files)
    budget = inference_budget(total, cpus)
    repack = not (memory_bytes and total > REPACK_OFF_MEMORY_MULTIPLE * memory_bytes)
    started = time.monotonic()
    retried = False
    while True:
        timeout = budget['timeout_seconds']
        if remaining_seconds is not None:
            timeout = int(max(300, min(timeout, remaining_seconds - (time.monotonic() - started))))
        try:
            result = test_inference(llama_cli, files[0]['path'], logs, timeout, threads=budget['threads'], predict=budget['predict'], repack=repack)
            break
        except RuntimeError as error:
            log_path = str(error).rpartition('logs: ')[2].strip()
            try:
                tail = Path(log_path).read_bytes()[-65536:].decode('utf-8', 'replace') if log_path else ''
            except OSError:
                tail = ''
            if repack and not retried and any(marker in tail for marker in ALLOCATION_FAILURES):
                repack, retried = False, True
                continue
            raise
    result['repack_retried_without_repacking'] = retried
    result.update(seconds=round(time.monotonic() - started, 1), model_bytes=total, shard_count=len(files),
                  shard_sha256_after_inference={item['name']: sha256_file(item['path']) for item in files})
    if any(result['shard_sha256_after_inference'][item['name']] != item['sha256'] for item in files):
        raise RuntimeError('a shard changed during inference')
    return result


def stage_on_branch(api, repo_id: str, branch: str, files: list[dict], *, private: bool, message: str) -> dict:
    """Upload validated shards to a staging branch and prove the Hub stores exactly these bytes."""
    from huggingface_hub import CommitOperationAdd
    from hfqa.target_policy import ensure_repository
    info = ensure_repository(api, repo_id, private)
    base = getattr(info, 'sha', None)
    api.create_branch(repo_id=repo_id, branch=branch, revision=base, exist_ok=True)
    commit = api.create_commit(repo_id=repo_id, revision=branch, commit_message=message,
                               operations=[CommitOperationAdd(path_in_repo=item['name'], path_or_fileobj=item['path']) for item in files])
    remote = {entry.path: entry for entry in api.get_paths_info(repo_id=repo_id, revision=commit.oid, paths=[item['name'] for item in files])}
    for item in files:
        entry = remote.get(item['name'])
        lfs = getattr(entry, 'lfs', None)
        digest = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
        if entry is None or entry.size != item['size'] or digest != item['sha256']:
            raise RuntimeError(f"the Hub does not hold the validated bytes of {item['name']}")
    return {'repo_id': repo_id, 'branch': branch, 'commit_oid': commit.oid, 'base_main_sha': base, 'lfs_sha256_verified': True,
            'files': [{'name': item['name'], 'size': item['size'], 'sha256': item['sha256']} for item in files]}


def release_quant(manifest: dict, source: dict, quant: str, *, bridge, ggml, llama_cli, work: Path, job_id: str,
                  api=None, branch: str | None = None, env: dict | None = None, log=print) -> dict:
    """Run the whole runner step for one quant and return the receipt (nothing is written to `main`)."""
    t0 = time.monotonic()
    facts = runner_facts(work)
    native = NativeQuantizer(bridge, ggml, manifest['runtime_revision'])
    remote = open_source(source)
    structure = source_structure(remote)
    stem = manifest.get('output_stem') or Path(source['filename']).stem
    stream = ShardedQuantizedStream(remote, source, quant, native, target_shard_bytes=int(manifest.get('target_shard_bytes') or DEFAULT_SHARD_BYTES))
    log(json.dumps({'event': 'plan', 'quant': quant, 'planned_bytes': stream.size, 'shards': len(stream.shards), 'runner': facts}), flush=True)
    t1 = time.monotonic()
    files = materialize(stream, work / quant, stem, log=log)
    t2 = time.monotonic()
    shard_reports = verify_shards(files, structure, structure['tensor_count'])
    inference = run_inference(Path(llama_cli), files, work / quant / 'logs', facts['cpus'] or 1, facts.get('memory_total_bytes'), remaining_job_seconds(t0))
    t3 = time.monotonic()
    staging = None
    if api is not None:
        staging = stage_on_branch(api, manifest['target_repo_id'], branch or f'hfqa-staging-{job_id}', files,
                                  private=bool(manifest.get('private')), message=f'HFQA validated {quant} shards (staging, not a release)')
    t4 = time.monotonic()
    receipt = {
        'schema': SCHEMA, 'job_id': job_id, 'quant': quant, 'state': 'RUNNER_VALIDATED' + ('_STAGED' if staging else ''),
        'runner': {**facts, **(env or {})},
        'native_identity': native.identity,
        'source': {'descriptors': source.get('sources') or [source], 'structure': structure,
                   'source_sha256_verified': True, 'concatenated_source_sha256': (stream.receipt or {}).get('concatenated_source_sha256'),
                   'source_materialized': False},
        'output': {'shard_count': len(files), 'total_bytes': sum(f['size'] for f in files), 'tensor_count': structure['tensor_count'],
                   'shards': [{'name': f['name'], 'size': f['size'], 'sha256': f['sha256']} for f in files], 'verification': shard_reports},
        'inference': {k: inference[k] for k in ('runtime_sha256', 'runtime_build', 'model_sha256', 'returncode', 'prompt', 'completion',
                                                 'context', 'predict', 'threads', 'command', 'seconds', 'model_bytes', 'shard_count')},
        'staging': staging,
        'inference_tested': True, 'full_inference_test_performed': True, 'runtime_compatible': True,
        'published': False, 'hub_main_changed': False,
        'timings_seconds': {'plan': round(t1 - t0, 1), 'quantize_and_write': round(t2 - t1, 1), 'verify_and_infer': round(t3 - t2, 1), 'upload': round(t4 - t3, 1)},
        'note': 'One smoke prompt on the exact shards; no quality benchmark is claimed.',
    }
    return receipt
