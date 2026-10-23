"""Durable Oracle HTTP-to-native-to-Xet staging worker; never publishes a release.

Queue tickets contain job_id, execution_backend=oracle and an inline manifest
(or the manifest fields directly). Only metadata/receipts reach local disk.
Successful quant receipts survive pause, restart, and later operator retries.
An independently validated publisher must finish the release; staging leaves
the job BLOCKED with stage_state=AWAITING_RUNTIME_VALIDATION.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import sys
import threading
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hfqa.jobs import JobStore, TransitionError
from hfqa.remote_gguf import BufferedRangeReader, RangeReader, inspect_remote_gguf, inspect_remote_source_set
from hfqa.security import redact
from hfqa.streaming import DEFAULT_SHARD_BYTES, NativeQuantizer, ShardedQuantizedStream, shard_filename, upload_xet_sharded_stream
from scripts.run_stream_worker import read_manifest
from scripts.run_worker_ticket import _read_token

RUNTIME_REVISION = '620d2225cc436ab0b001e3200e590e3b286ef9ab'
RUNTIME_DIR = Path('/opt/hf-quant-agent/runtime/llama-b11669')
QUEUE = Path(os.environ.get('HFQA_STREAM_QUEUE', '/var/lib/hf-quant-agent/stream-queue'))
DB = Path(os.environ.get('HFQA_STREAM_DB', '/var/lib/hf-quant-agent/data/jobs.sqlite'))
RECEIPTS = Path(os.environ.get('HFQA_STREAM_RECEIPTS', '/var/lib/hf-quant-agent/stream-receipts'))
ACTIVE = {'WAITING_FOR_WORKER', 'QUANTIZING', 'VALIDATING'}
SHUTDOWN = threading.Event()


class WorkerStopped(RuntimeError):
    pass


class WorkerBusy(RuntimeError):
    pass


class LeaseGuard:
    """Renew independently of slow Xet finalization; cancellation checked per row."""
    def __init__(self, store, job_id, owner):
        self.store, self.job_id, self.owner = store, job_id, owner
        self.stop, self.lost = threading.Event(), threading.Event()
        self.thread = threading.Thread(target=self._heartbeat, name='stream-lease', daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=2)

    def _heartbeat(self):
        while not self.stop.wait(30):
            try:
                job = self.store.get(self.job_id)
                if not job or job['state'] not in ACTIVE or not self.store.renew_claim(self.job_id, self.owner, ttl_seconds=180):
                    self.lost.set()
                    return
            except Exception:
                self.lost.set()
                return

    def check(self):
        job = self.store.get(self.job_id)
        if SHUTDOWN.is_set() or self.lost.is_set() or not job or job['state'] not in ACTIVE:
            raise WorkerStopped('stream worker stopped, paused, cancelled or lost its lease')
        return job

    def should_stop(self):
        try:
            self.check()
            return False
        except WorkerStopped:
            return True


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _durable_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.part')
    try:
        with temporary.open('x', encoding='utf-8') as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != 'nt':
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _policy(ticket, manifest, source, job):
    plan, analysis = job.get('plan') or {}, job.get('analysis') or {}
    if ticket.get('execution_backend') != 'oracle' or plan.get('execution_backend') != 'oracle' or plan.get('route') != 'native_gguf_stream' or job.get('execution_backend', 'auto') not in {'auto', 'oracle'}:
        raise ValueError('ticket and plan must explicitly select the qualified Oracle stream backend')
    if plan.get('allow_new_patch') or job.get('variants', ['normal']) != ['normal']:
        raise ValueError('Oracle streaming requires the stock runtime and normal variant')
    if manifest.get('runtime_revision') != RUNTIME_REVISION:
        raise ValueError('Oracle streaming runtime revision differs from the qualified stock build')
    if manifest.get('quants') != job.get('quants'):
        raise ValueError('stream manifest quantization set differs from the durable job')
    if not job.get('revision') or analysis.get('revision') != job['revision'] or analysis.get('trust_remote_code_executed'):
        raise ValueError('stream job requires pinned inspected metadata and no remote-code execution')
    expected = analysis.get('native_gguf_repo') or analysis
    if any(source.get(key) != expected.get(key) for key in ('repo_id', 'revision')):
        raise ValueError('stream source differs from the inspected immutable GGUF repository')
    if job.get('publish') is not True:
        raise ValueError('Xet staging is an external write and requires publication requested by this job')
    if ((manifest.get('source_license') or {}).get('redistributable') is not True
            or (analysis.get('license_decision') or {}).get('redistributable') is not True):
        raise ValueError('both inspected source license and stream manifest must permit redistribution')
    target = manifest.get('target_repo_id', '')
    if not re.fullmatch('webmp3/[A-Za-z0-9][A-Za-z0-9._-]*', target) or target != plan.get('release_repo_id'):
        raise ValueError('stream target differs from the qualified publisher repository')
    if not isinstance(manifest.get('target_revision', 'main'), str):
        raise ValueError('invalid target staging revision')
    cap = manifest.get('target_shard_bytes', DEFAULT_SHARD_BYTES)
    if type(cap) is not int or not 1024 <= cap <= DEFAULT_SHARD_BYTES:
        raise ValueError('Oracle stream shard cap must remain at most 4 GiB')


def _validate_stage(stage, manifest_sha, source, native, quant, stem, target_repo, target_revision):
    receipt, files = stage.get('receipt') or {}, stage.get('files') or []
    proofs = receipt.get('shards') or []
    if (stage.get('manifest_sha256') != manifest_sha or stage.get('quant') != quant
            or stage.get('target_repo_id') != target_repo or stage.get('target_revision') != target_revision
            or stage.get('native_sha256_verified') is not True or stage.get('hub_commit_created') is not False
            or stage.get('published') is not False or receipt.get('source') != source
            or receipt.get('native') != native.identity or receipt.get('quant') != quant
            or receipt.get('source_sha256_verified') is not True or receipt.get('stream_complete') is not True
            or receipt.get('output_materialized') is not False or receipt.get('source_materialized') is not False
            or receipt.get('source_bytes_hashed') != source['size'] or not files or len(files) != len(proofs)
            or receipt.get('shard_count') != len(files)):
        raise RuntimeError('durable stream receipt does not match source, native identity or complete staging proof')
    for index, (item, proof) in enumerate(zip(files, proofs)):
        if (item.get('path') != shard_filename(stem, quant, index, len(files))
                or not re.fullmatch('[a-fA-F0-9]{64}', item.get('sha256', ''))
                or not re.fullmatch('[a-fA-F0-9]{64}', item.get('xet_hash', ''))
                or type(item.get('size')) is not int or item['size'] <= 0
                or item['size'] > DEFAULT_SHARD_BYTES or proof.get('index') != index
                or any(item[key] != proof.get(key) for key in ('size', 'sha256'))):
            raise RuntimeError('durable stream shard receipt is incomplete or inconsistent')


def _checkpoint(store, job_id, quant, stage, receipt_path):
    for index, item in enumerate(stage['files']):
        store.upsert_shard(job_id, quant, index, 'staged_awaiting_validation',
            checksum=item['sha256'], nbytes=item['size'], path=f"xet:{stage['target_repo_id']}/{item['path']}",
            detail={'receipt_path': str(receipt_path), 'manifest_sha256': stage['manifest_sha256'],
                    'xet_hash': item['xet_hash'], 'native': stage['receipt']['native'],
                    'source_sha256_verified': True, 'hub_commit_created': False, 'published': False})


def _inspect(source):
    if source.get('sources'):
        readers = [BufferedRangeReader(RangeReader(item)) for item in source['sources']]
        return inspect_remote_source_set(source['sources'], readers=readers)
    return inspect_remote_gguf(source, reader=BufferedRangeReader(RangeReader(source)))


def _process(ticket, ticket_path, store, guard, bridge, ggml, receipts_root):
    job = guard.check()
    job_id = job['id']
    manifest = ticket.get('manifest')
    if manifest is None:
        manifest = {key: value for key, value in ticket.items() if key not in {'job_id', 'execution_backend'}}
    if not isinstance(manifest, dict):
        raise ValueError('stream ticket manifest must be an inline object')
    manifest_sha = _canonical_sha(manifest)
    receipt_dir = Path(receipts_root) / job_id
    receipt_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = receipt_dir / 'manifest.json'
    if manifest_path.exists():
        if _canonical_sha(json.loads(manifest_path.read_text(encoding='utf-8'))) != manifest_sha:
            raise RuntimeError('stream manifest changed since the durable checkpoint')
    else:
        _durable_json(manifest_path, manifest)
    manifest, source = read_manifest(manifest_path)
    _policy(ticket, manifest, source, job)
    native = NativeQuantizer(bridge, ggml, manifest['runtime_revision'])
    if not all(re.fullmatch('[a-fA-F0-9]{64}', native.identity.get(key, '')) for key in ('bridge_sha256', 'ggml_sha256', 'llama_sha256')):
        raise RuntimeError('stock native identity requires bridge, ggml and loaded llama hashes')
    stem = Path(source['filename']).stem
    results, remote, api = [], None, None
    def save(**fields):
        store.update_fields(job_id, expected_state=guard.check()['state'], **fields)
    for quant in manifest['quants']:
        job = guard.check()
        receipt_path = receipt_dir / (quant + '.json')
        if receipt_path.exists():
            if receipt_path.stat().st_size > 64 * 1024**2:
                raise RuntimeError('durable stage receipt exceeds metadata budget')
            stage = json.loads(receipt_path.read_text(encoding='utf-8'))
            _validate_stage(stage, manifest_sha, source, native, quant, stem, manifest['target_repo_id'], manifest.get('target_revision', 'main'))
            guard.check()
            _checkpoint(store, job_id, quant, stage, receipt_path)
            results.append(stage)
            store.event(job_id, job['state'], job['state'], 'reused complete native Xet stream staging receipt', {'quant': quant})
            continue
        if job['state'] != 'QUANTIZING':
            store.transition(job_id, 'QUANTIZING', 'Oracle native HTTP-to-Xet stream started', {'quant': quant, 'source_materialized': False}, expected_state=job['state'])
        if remote is None:
            remote = _inspect(source)
        guard.check()
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi(token=_read_token())
            if api.whoami().get('name') != 'webmp3':
                raise RuntimeError('HF token does not belong to the qualified publisher')
            api.create_repo(manifest['target_repo_id'], repo_type='model', exist_ok=True, private=False)
        stream = ShardedQuantizedStream(remote, source, quant, native, should_stop=guard.should_stop,
            target_shard_bytes=manifest.get('target_shard_bytes', DEFAULT_SHARD_BYTES))
        stage = upload_xet_sharded_stream(api, manifest['target_repo_id'], manifest.get('target_revision', 'main'), stem, stream)
        stage.update({'manifest_sha256': manifest_sha, 'quant': quant,
                      'target_repo_id': manifest['target_repo_id'],
                      'target_revision': manifest.get('target_revision', 'main'),
                      'state': 'STAGED_AWAITING_RUNTIME_VALIDATION'})
        _validate_stage(stage, manifest_sha, source, native, quant, stem, manifest['target_repo_id'], manifest.get('target_revision', 'main'))
        # CAS is complete and all source/native hashes passed. Persist before
        # observing pause so completed CPU work remains reusable after restart.
        _durable_json(receipt_path, stage)
        guard.check()
        _checkpoint(store, job_id, quant, stage, receipt_path)
        results.append(stage)
        previous = dict((guard.check().get('result') or {}))
        previous.update({'execution_backend': 'oracle', 'stage_state': 'STAGING',
            'stream_receipt_dir': str(receipt_dir), 'staged_quants': [item['quant'] for item in results],
            'source_materialized': False, 'output_materialized': False,
            'published': False, 'hub_commit_created': False})
        save(result_json=previous, error=None)
    job = guard.check()
    if job['state'] != 'VALIDATING':
        store.transition(job_id, 'VALIDATING', 'all requested quants staged; independent runtime validation required', expected_state=job['state'])
    report = {'execution_backend': 'oracle', 'route': 'native_gguf_stream',
        'stage_state': 'AWAITING_RUNTIME_VALIDATION', 'source': source,
        'source_materialized': False, 'output_materialized': False, 'published': False,
        'hub_commit_created': False, 'inference_tested': False,
        'runtime_compatible': False, 'staged_quants': [item['quant'] for item in results],
        'outputs': results, 'stream_receipt_dir': str(receipt_dir),
        'summary': 'All requested GGUF shards staged in unreferenced Xet CAS. No Hub commit or release exists. Independent per-output stock runtime inference is still required.'}
    _durable_json(receipt_dir / 'stream-result.json', report)
    save(result_json=report, error='AWAITING_RUNTIME_VALIDATION: staged CAS blobs have no independently verified release')
    store.transition(job_id, 'BLOCKED', 'awaiting independent runtime validation and verified publication; staged quant receipts retained', expected_state=guard.check()['state'])
    return store.get(job_id)


def process_ticket(ticket_path, store, bridge=RUNTIME_DIR / 'libhfqa_plan.so', ggml=RUNTIME_DIR / 'libggml-base.so', receipts_root=RECEIPTS):
    ticket_path = Path(ticket_path)
    if ticket_path.stat().st_size > 1024**2:
        raise ValueError('stream ticket exceeds metadata budget')
    ticket = json.loads(ticket_path.read_text(encoding='utf-8'))
    job_id = ticket.get('job_id', '')
    if not re.fullmatch('[a-f0-9]{16}', job_id) or ticket_path.stem != job_id:
        raise ValueError('invalid stream ticket job identity')
    job = store.get(job_id)
    if not job:
        raise ValueError('stream ticket references an unknown job')
    if job['state'] in {'COMPLETED', 'CANCELLED'}:
        ticket_path.unlink(missing_ok=True)
        return job
    if job['state'] not in ACTIVE:
        return job
    owner = uuid.uuid4().hex
    if not store.claim(job_id, owner, ttl_seconds=180):
        raise WorkerBusy('stream job already has a live worker lease')
    try:
        with LeaseGuard(store, job_id, owner) as guard:
            return _process(ticket, ticket_path, store, guard, bridge, ggml, receipts_root)
    except (WorkerStopped, InterruptedError, TransitionError):
        return store.get(job_id)
    except Exception as error:
        current = store.get(job_id)
        if current and current['state'] in ACTIVE:
            text = redact(f'{type(error).__name__}: {error}')[:2000]
            try:
                store.update_fields(job_id, error=text, expected_state=current['state'])
                store.transition(job_id, 'BLOCKED', 'Oracle stream stopped; completed quant receipts retained', {'error': text}, expected_state=current['state'])
            except TransitionError:
                pass
        return store.get(job_id)
    finally:
        store.release_claim(job_id, owner)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, default=QUEUE)
    parser.add_argument('--db', type=Path, default=DB)
    parser.add_argument('--receipts', type=Path, default=RECEIPTS)
    parser.add_argument('--bridge', type=Path, default=RUNTIME_DIR / 'libhfqa_plan.so')
    parser.add_argument('--ggml', type=Path, default=RUNTIME_DIR / 'libggml-base.so')
    args = parser.parse_args()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: SHUTDOWN.set())
    if hasattr(os, 'nice'):
        os.nice(15)
    with JobStore(args.db) as store:
        for path in sorted(args.queue.glob('*.json')) if args.queue.is_dir() else []:
            try:
                if path.stat().st_size > 1024**2:
                    raise ValueError('stream ticket exceeds metadata budget')
                ticket = json.loads(path.read_text(encoding='utf-8'))
                before = store.get(ticket.get('job_id', ''))
                if not before or before['state'] not in ACTIVE:
                    continue
                result = process_ticket(path, store, args.bridge, args.ggml, args.receipts)
                print(json.dumps({'job_id': result['id'], 'state': result['state'],
                    'stage_state': (result.get('result') or {}).get('stage_state'), 'published': False}), flush=True)
                break  # One active ticket per timer invocation.
            except WorkerBusy:
                continue
            except Exception as error:
                print(redact(f'invalid Oracle stream ticket {path.name}: {type(error).__name__}: {error}')[:2000], file=sys.stderr)


if __name__ == '__main__':
    main()
