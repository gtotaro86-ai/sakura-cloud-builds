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
    if recorded.get('head') != chosen['revision'] or recorded.get('repo') != chosen['repo'] or recorded.get('static') is not True:
        raise ValueError('the build receipt does not belong to the commit named in the manifest')
    for name in source_builds.TARGETS:
        digest = hashlib.sha256(Path(runtime_dir, name).read_bytes()).hexdigest()
        if recorded['binaries'].get(name) != digest:
            raise ValueError(f'{name} differs from the binary the build job hashed')
    return {key: recorded[key] for key in ('repo', 'head', 'binaries', 'static')} | {'build_run_id': int(recorded.get('build_run_id') or 0)}


FORBIDDEN_IN_PHASE_QUANTIZE = ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_HUB_TOKEN', 'GITHUB_TOKEN', 'GH_TOKEN')
PHASE_FILE = 'phase-a.json'


def stage_api(manifest: dict, source: dict):
    """The checks that stand before every upload, and the Hub client (this is where the Hugging Face key is first used)."""
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
    return HfApi(token=token)


def tools_for(manifest: dict, runtime_dir: Path, build_receipt):
    build = source_build_evidence(manifest, runtime_dir, build_receipt)
    tools = None
    if build:  # built from source: the plain text generation tool is llama-completion (llama-cli exists only with the server build)
        from hfqa import source_builds
        folder = Path(runtime_dir)
        tools = {'imatrix': folder / 'llama-imatrix', 'quantize': folder / 'llama-quantize', 'cli': folder / source_builds.INFERENCE_TOOL, 'perplexity': folder / 'llama-perplexity'}
    return build, tools


def write_receipts(receipts: dict, manifest: dict, build, out: Path) -> None:
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    folder = Path(out) / 'receipts'
    folder.mkdir(parents=True, exist_ok=True)
    for quant, receipt in receipts.items():
        receipt['manifest_sha256'] = digest
        if build:
            receipt['native_identity']['source_build'] = build
        atomic_json(folder / f'{quant}.json', receipt)


def run(args) -> dict:
    """Three ways to run: `all` (one process, the original path for official builds and master commits), and for a llama.cpp FORK the two phases `quantize` (no key, third-party code) and
    `stage` (the key, only our own code, nothing of phase A is trusted but the files)."""
    manifest, source = read_manifest(args.manifest)
    if isinstance(manifest.get('imatrix'), dict) == isinstance(manifest.get('plain'), dict):
        raise ValueError('the manifest must request exactly one of the importance-matrix path and a plain quantization with the stock llama-quantize')
    phase = getattr(args, 'phase', 'all') or 'all'
    work = Path(args.work)
    out = Path(getattr(args, 'out', None) or work)
    from hfqa import source_builds
    fork = source_builds.is_fork(manifest.get('runtime'))
    if fork and phase == 'all':
        raise ValueError('a build from a fork runs only in the two phases (quantize without the key, then stage)')
    if phase == 'stage':
        if not args.upload:
            raise ValueError('phase stage stages: it needs --upload')
        build, _ = tools_for(manifest, args.runtime_dir, args.build_receipt)   # the binaries are hashed again here: phase A cannot have swapped them unnoticed
        phase_file = work / PHASE_FILE
        if phase_file.stat().st_size > 8 * 1024 ** 2:
            raise ValueError('the phase A file is unreasonably large')
        recorded = json.loads(phase_file.read_text(encoding='utf-8'))
        if recorded.get('job_id') != args.job_id or not isinstance(recorded.get('receipts'), dict) or sorted(recorded['receipts']) != sorted(manifest['quants']):
            raise ValueError('the phase A file does not belong to this job and its quants')
        api = stage_api(manifest, source)
        from hfqa import runner_imatrix
        descriptor = (source.get('sources') or [source])[0]
        if not (work / 'source' / descriptor['filename']).is_file():   # phase A ran elsewhere (Modal): the source is fetched again here, by our own code, and proven against the pinned hash
            runner_imatrix.fetch_source(descriptor, work / 'source')
        receipts = runner_imatrix.stage_deferred(manifest, source, recorded['receipts'], work=work, api=api, branch=args.branch, job_id=args.job_id)
        write_receipts(receipts, manifest, build, out)
        _copy_logs(work, out)
        return receipts
    if phase == 'quantize':
        if args.upload:
            raise ValueError('phase quantize never stages')
        present = [name for name in FORBIDDEN_IN_PHASE_QUANTIZE if os.environ.get(name)]
        if present:
            raise RuntimeError('phase quantize runs third-party code and must run without any key in its environment: ' + ', '.join(present))
    api = stage_api(manifest, source) if (phase == 'all' and args.upload) else None
    env = {k: os.environ[k] for k in ('GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT', 'GITHUB_SHA', 'GITHUB_REPOSITORY', 'RUNNER_NAME', 'RUNNER_OS', 'RUNNER_ARCH') if k in os.environ}
    build, tools = tools_for(manifest, args.runtime_dir, args.build_receipt)
    receipts = release_quants(manifest, source, list(manifest['quants']), runtime_dir=args.runtime_dir, work=work, job_id=args.job_id, api=api, branch=args.branch, env=env, tools=tools,
                              defer_staging=(phase == 'quantize'))
    if phase == 'quantize':
        work.mkdir(parents=True, exist_ok=True)
        atomic_json(work / PHASE_FILE, {'job_id': args.job_id, 'receipts': receipts})
        return receipts
    write_receipts(receipts, manifest, build, out)
    return receipts


def _copy_logs(work: Path, out: Path) -> None:
    """The tool logs of phase A go to the artifact folder next to the final receipts."""
    import shutil
    target = Path(out) / 'logs'
    target.mkdir(parents=True, exist_ok=True)
    for log in sorted(Path(work).glob('*/logs/*.log'))[:200]:
        if log.stat().st_size <= 8 * 1024 ** 2:
            shutil.copy2(log, target / f'{log.parent.parent.name}-{log.name}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--runtime-dir', type=Path, required=True, help='directory of the stock llama.cpp release (llama-imatrix, llama-quantize, llama-cli)')
    parser.add_argument('--build-receipt', type=Path, help='receipt of the build job, for a runtime built from source')
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--branch')
    parser.add_argument('--upload', action='store_true')
    parser.add_argument('--phase', choices=('all', 'quantize', 'stage'), default='all', help='all: one process; quantize / stage: the two phases of a fork build (docs/FORK-BUILDS.md)')
    parser.add_argument('--out', type=Path, help='where receipts/ and logs/ go (default: --work)')
    args = parser.parse_args()
    try:
        receipts = run(args)
    except Exception as error:
        print(redact(f'{type(error).__name__}: {error}')[:2000], file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps({'event': 'done', 'quants': sorted(receipts), 'hub_main_changed': False}), flush=True)


if __name__ == '__main__':
    main()
