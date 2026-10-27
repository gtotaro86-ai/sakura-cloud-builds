"""Runner-side release worker for the importance-matrix path: all requested quants in one job (small and medium models).

Download and verify the pinned source, compute one matrix with the stock llama-imatrix, quantize every requested preset with the stock llama-quantize
--imatrix, verify each file, run a real inference on it and stage it on a Hub branch. It never writes the target's `main`; the receipts are judged on Oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hfqa.security import redact
from hfqa.runner_imatrix import release_quants
from scripts.run_stream_worker import atomic_json, read_manifest


def source_build_evidence(manifest: dict, runtime_dir: Path, receipt_path: Path | None) -> dict | None:
    """For a runtime built from source: what the build job recorded, checked against the binaries this job really uses."""
    from hfqa import source_builds
    if not source_builds.is_source(manifest.get('runtime')):
        return None
    if receipt_path is None:
        raise ValueError('the manifest names a source build but no build receipt was given')
    chosen = source_builds.validate(manifest['runtime'])
    recorded = json.loads(Path(receipt_path).read_text(encoding='utf-8'))
    if recorded.get('head') != chosen['revision'] or recorded.get('repo') != source_builds.REPO or recorded.get('static') is not True:
        raise ValueError('the build receipt does not belong to the commit named in the manifest')
    for name in source_builds.TARGETS:
        digest = hashlib.sha256(Path(runtime_dir, name).read_bytes()).hexdigest()
        if recorded['binaries'].get(name) != digest:
            raise ValueError(f'{name} differs from the binary the build job hashed')
    return {key: recorded[key] for key in ('repo', 'head', 'binaries', 'static')} | {'build_run_id': int(recorded.get('build_run_id') or 0)}


def run(args) -> dict:
    manifest, source = read_manifest(args.manifest)
    if not isinstance(manifest.get('imatrix'), dict):
        raise ValueError('the manifest does not request the importance-matrix path')
    api = None
    if args.upload:
        if (manifest.get('source_license') or {}).get('redistributable') is not True:
            raise ValueError('staging requires an affirmative source redistribution decision in the manifest')
        if not manifest.get('target_repo_id'):
            raise ValueError('staging target must be explicitly pinned')
        from hfqa.inspect import inspect_repo
        descriptors = source.get('sources') or [source]
        checked = inspect_repo(f"https://huggingface.co/{descriptors[0]['repo_id']}/tree/{descriptors[0]['revision']}")
        from hfqa.license_pin import runner_accepts
        if not runner_accepts(manifest.get('source_license'), checked):
            raise ValueError('the live pinned source license blocks automatic staging')
        token = os.environ.get('HF_TOKEN', '').strip()
        if not token:
            raise RuntimeError('HF_TOKEN is required to stage')
        from huggingface_hub import HfApi
        api = HfApi(token=token)
    work = Path(args.work)
    env = {k: os.environ[k] for k in ('GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT', 'GITHUB_SHA', 'GITHUB_REPOSITORY', 'RUNNER_NAME', 'RUNNER_OS', 'RUNNER_ARCH') if k in os.environ}
    build = source_build_evidence(manifest, args.runtime_dir, args.build_receipt)
    receipts = release_quants(manifest, source, list(manifest['quants']), runtime_dir=args.runtime_dir, work=work, job_id=args.job_id, api=api, branch=args.branch, env=env)
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    folder = work / 'receipts'
    folder.mkdir(parents=True, exist_ok=True)
    for quant, receipt in receipts.items():
        receipt['manifest_sha256'] = digest
        if build:
            receipt['native_identity']['source_build'] = build
        atomic_json(folder / f'{quant}.json', receipt)
    return receipts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--runtime-dir', type=Path, required=True, help='directory of the stock llama.cpp release (llama-imatrix, llama-quantize, llama-cli)')
    parser.add_argument('--build-receipt', type=Path, help='receipt of the build job, for a runtime built from source')
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--branch')
    parser.add_argument('--upload', action='store_true')
    args = parser.parse_args()
    try:
        receipts = run(args)
    except Exception as error:
        print(redact(f'{type(error).__name__}: {error}')[:2000], file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps({'event': 'done', 'quants': sorted(receipts), 'hub_main_changed': False}), flush=True)


if __name__ == '__main__':
    main()
