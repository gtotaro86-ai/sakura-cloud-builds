"""Runner-side release worker: one quant per invocation (GitHub matrix job, or Oracle for a proof without upload).

Stream-quantize from the pinned source, verify the shards, run a real inference on them, then optionally stage the
validated bytes on a Hub branch. It never writes the target's `main`; the receipt it prints is judged on Oracle.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hfqa.security import redact
from hfqa.runner_release import release_quant
from scripts.run_stream_worker import atomic_json, read_manifest


def run(args) -> dict:
    manifest, source = read_manifest(args.manifest)
    if args.quant not in manifest['quants']:
        raise ValueError('requested quant is not part of the pinned manifest')
    api = None
    if args.upload:
        if (manifest.get('source_license') or {}).get('redistributable') is not True:
            raise ValueError('staging requires an affirmative source redistribution decision in the manifest')
        if not manifest.get('target_repo_id'):
            raise ValueError('staging target must be explicitly pinned')
        from hfqa.inspect import inspect_repo
        descriptors = source.get('sources') or [source]
        checked = inspect_repo(f"https://huggingface.co/{descriptors[0]['repo_id']}/tree/{descriptors[0]['revision']}")
        if (checked.get('license_decision') or {}).get('redistributable') is not True:
            raise ValueError('the live pinned source license blocks automatic staging')
        token = os.environ.get('HF_TOKEN', '').strip()
        if not token:
            raise RuntimeError('HF_TOKEN is required to stage')
        from huggingface_hub import HfApi
        api = HfApi(token=token)
    work = Path(args.work)
    env = {k: os.environ[k] for k in ('GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT', 'GITHUB_SHA', 'GITHUB_REPOSITORY', 'RUNNER_NAME', 'RUNNER_OS', 'RUNNER_ARCH') if k in os.environ}
    receipt = release_quant(manifest, source, args.quant, bridge=args.bridge, ggml=args.ggml, llama_cli=args.llama_cli, work=work,
                            job_id=args.job_id, api=api, branch=args.branch, env=env)
    receipt['manifest_sha256'] = __import__('hashlib').sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    receipts = work / 'receipts'
    receipts.mkdir(parents=True, exist_ok=True)
    atomic_json(receipts / f'{args.quant}.json', receipt)
    if not args.keep_shards:
        for path in (work / args.quant).glob('*.gguf'):
            path.unlink(missing_ok=True)  # disk is the scarce resource on a runner
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--quant', required=True)
    parser.add_argument('--bridge', type=Path, required=True)
    parser.add_argument('--ggml', type=Path, required=True)
    parser.add_argument('--llama-cli', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--branch', help='staging branch name; default hfqa-staging-<job>')
    parser.add_argument('--upload', action='store_true', help='stage the validated shards on the Hub (needs HF_TOKEN)')
    parser.add_argument('--keep-shards', action='store_true')
    args = parser.parse_args()
    try:
        receipt = run(args)
    except Exception as error:
        print(redact(f'{type(error).__name__}: {error}')[:2000], file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps({'event': 'done', 'quant': args.quant, 'state': receipt['state'], 'hub_main_changed': False,
                      'inference_tested': receipt['inference_tested'], 'timings_seconds': receipt['timings_seconds']}), flush=True)


if __name__ == '__main__':
    main()
