"""Bounded remote GGUF worker: small local proof or unreferenced Xet staging.

This command never creates a Hub commit. Xet staging is not a validated release;
per-output inference evidence is still required before a separate publisher may
make the files visible. A manifest is pinned to one source and one runtime SHA.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hfqa.remote_gguf import RangeReader, BufferedRangeReader, inspect_remote_gguf, inspect_remote_source_set
from hfqa.security import redact
from hfqa.streaming import (NativeQuantizer, QuantizedStream, ShardedQuantizedStream,
    DEFAULT_SHARD_BYTES, QUANTS, stream_quantize, upload_xet_sharded_stream, shard_filename)

LOCAL_PROOF_LIMIT = 2 * 1024**3


def read_manifest(path):
    path = Path(path)
    if path.stat().st_size > 1024**2:
        raise ValueError('stream manifest exceeds metadata budget')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    sources = manifest.get('sources')
    if sources is None:
        sources = [manifest.get('source')]
    if not isinstance(sources, list) or not sources or len(sources) > 1024 or any(not isinstance(item, dict) for item in sources):
        raise ValueError('stream manifest requires a bounded nonempty list of complete GGUF sources')
    for source in sources:
        if (not isinstance(source.get('size'), int) or isinstance(source['size'], bool) or source['size'] < 24
                or not re.fullmatch('[a-fA-F0-9]{64}', source.get('sha256', ''))
                or not re.fullmatch('[a-f0-9]{40}', source.get('revision', ''))):
            raise ValueError('source descriptor requires exact byte size, full SHA-256 and immutable revision')
    sources = sorted(sources, key=lambda item: item.get('split_index', 1))
    if len(sources) > 1:
        if (any(item.get('split_count') != len(sources) for item in sources)
                or [item.get('split_index') for item in sources] != list(range(1, len(sources) + 1))
                or len({(item['repo_id'], item['revision']) for item in sources}) != 1):
            raise ValueError('split source descriptors require every file from one immutable revision')
        source = {'sources': sources, 'size': sum(item['size'] for item in sources),
                  'repo_id': sources[0]['repo_id'], 'revision': sources[0]['revision'],
                  'filename': re.sub(r'-\d{5}-of-\d{5}(?=\.gguf$)', '', sources[0]['filename'])}
    else:
        source = sources[0]
    if not re.fullmatch('[a-fA-F0-9]{40}', manifest.get('runtime_revision', '')):
        raise ValueError('manifest requires the full pinned stock runtime revision')
    quants = manifest.get('quants')
    if not isinstance(quants, list) or not quants or len(set(quants)) != len(quants) or any(q not in QUANTS for q in quants):
        raise ValueError('manifest requires a unique supported quantization set')
    return manifest, source


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.part')
    created = False
    try:
        with temporary.open('x', encoding='utf-8') as handle:
            created = True
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if created:
            temporary.unlink(missing_ok=True)


class _ProofSink:
    def __init__(self, handle):
        self.handle, self.count = handle, 0

    def write(self, raw):
        if self.count + len(raw) > LOCAL_PROOF_LIMIT:
            raise ValueError('local output exceeds the small proof budget')
        written = self.handle.write(raw)
        self.count += written
        return written

    def flush(self):
        self.handle.flush()


def _local_shards(stream, directory, stem):
    if stream.size > LOCAL_PROOF_LIMIT:
        raise ValueError('combined output shards exceed the small local proof budget')
    outputs = [Path(directory) / shard_filename(stem, stream.quant, index, len(stream.shards)) for index in range(len(stream.shards))]
    temporaries = [path.with_suffix('.gguf.part') for path in outputs]
    if any(path.exists() for path in outputs + temporaries):
        raise FileExistsError('proof shard output already exists; choose a fresh output directory')
    active, current, created = None, -1, []
    try:
        for index, raw in stream:
            if index != current:
                if active is not None:
                    active.flush()
                    os.fsync(active.fileno())
                    active.close()
                active = temporaries[index].open('xb')
                created.append(temporaries[index])
                current = index
            if active.write(raw) != len(raw):
                raise RuntimeError('local proof shard accepted a short write')
        if not stream.receipt:
            raise RuntimeError('split proof has no full-source verification receipt')
        if active is not None:
            active.flush()
            os.fsync(active.fileno())
            active.close()
            active = None
        for temporary, output in zip(temporaries, outputs):
            os.replace(temporary, output)
        stream.receipt['output_materialized'] = True
        return {'files': [{'path': str(path.resolve()), **proof} for path, proof in zip(outputs, stream.receipt['shards'])],
                'receipt': stream.receipt, 'state': 'LOCAL_PROOF_AWAITING_RUNTIME_VALIDATION',
                'published': False, 'hub_commit_created': False}
    finally:
        if active is not None:
            active.close()
        for temporary in created:
            temporary.unlink(missing_ok=True)


def execute(args):
    manifest, source = read_manifest(args.manifest)
    if args.xet_repo:
        if args.xet_repo != manifest.get('target_repo_id'):
            raise ValueError('explicit Xet target differs from pinned manifest target')
        if (manifest.get('source_license') or {}).get('redistributable') is not True:
            raise ValueError('Xet staging requires an affirmative source redistribution decision')
        from huggingface_hub import HfApi
        from scripts.run_worker_ticket import _read_token
        api = HfApi(token=_read_token())
        target_revision = manifest.get('target_revision', 'main')
    else:
        if source['size'] > LOCAL_PROOF_LIMIT:
            raise ValueError('local output is restricted to small proof sources; use Xet streaming for large models')
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    receipt_dir = Path(args.receipt_dir)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    native = NativeQuantizer(args.bridge, args.ggml, manifest['runtime_revision'])
    if source.get('sources'):
        readers = [BufferedRangeReader(RangeReader(item)) for item in source['sources']]
        remote = inspect_remote_source_set(source['sources'], readers=readers)
    else:
        remote = inspect_remote_gguf(source, reader=BufferedRangeReader(RangeReader(source)))
    stem = Path(source['filename']).stem
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9._-]*', stem):
        raise ValueError('source basename cannot be used as a safe output name')
    results = []
    for quant in manifest['quants']:
        filename = f'{stem}-{quant}-requant.gguf'
        if args.xet_repo:
            stream = ShardedQuantizedStream(remote, source, quant, native,
                target_shard_bytes=getattr(args, 'shard_bytes', None) or DEFAULT_SHARD_BYTES)
            result = upload_xet_sharded_stream(api, args.xet_repo, target_revision, stem, stream)
            result['state'] = 'STAGED_AWAITING_RUNTIME_VALIDATION'
            result['target_repo_id'] = args.xet_repo
        elif getattr(args, 'shard_bytes', None):
            stream = ShardedQuantizedStream(remote, source, quant, native, target_shard_bytes=args.shard_bytes)
            result = _local_shards(stream, args.output_dir, stem)
        else:
            output = Path(args.output_dir) / filename
            temporary = output.with_suffix('.gguf.part')
            if output.exists() or temporary.exists():
                raise FileExistsError('proof output already exists; choose a fresh output directory')
            try:
                with temporary.open('xb') as handle:
                    receipt = stream_quantize(remote, source, quant, native, _ProofSink(handle))
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, output)
            finally:
                temporary.unlink(missing_ok=True)
            receipt['output_materialized'] = True
            result = {'path': str(output.resolve()), 'receipt': receipt,
                      'state': 'LOCAL_PROOF_AWAITING_RUNTIME_VALIDATION',
                      'published': False, 'hub_commit_created': False}
        atomic_json(receipt_dir / (quant + '.json'), result)
        results.append(result)
        print(json.dumps({'quant': quant, 'state': result['state'], 'published': False}), flush=True)
    report = {'route': 'native_gguf_stream', 'source': source,
              'source_materialized': False, 'published': False, 'hub_commit_created': False,
              'full_inference_test_performed': False, 'outputs': results,
              'state': 'AWAITING_RUNTIME_VALIDATION'}
    atomic_json(receipt_dir / 'stream-result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--bridge', type=Path, required=True)
    parser.add_argument('--ggml', type=Path, required=True)
    parser.add_argument('--receipt-dir', type=Path, required=True)
    parser.add_argument('--shard-bytes', type=int, help='Hard per-file tensor-boundary GGUF cap; Xet default is 4 GiB')
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument('--output-dir', type=Path, help='Small source proof only; at most 2 GiB')
    destination.add_argument('--xet-repo', help='Existing publisher repo; stage CAS without a Hub commit')
    args = parser.parse_args()
    if hasattr(os, 'nice'):
        os.nice(15)
    try:
        execute(args)
    except Exception as error:
        print(redact(f'{type(error).__name__}: {error}')[:2000], file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
