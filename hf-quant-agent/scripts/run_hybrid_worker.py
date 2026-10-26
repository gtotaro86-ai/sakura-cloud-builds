"""Public runner entry point for a hybrid release (see hfqa/hybrid_release.py). Never writes `main`; publishes nothing."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hfqa import runtimes
from hfqa.hybrid_release import run_hybrid
from scripts.run_stream_worker import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--ggml', required=True)
    parser.add_argument('--work', required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--branch', required=True)
    parser.add_argument('--upload', action='store_true')
    args = parser.parse_args()
    raw = Path(args.manifest)
    if raw.stat().st_size > 1024 ** 2:
        raise ValueError('manifest exceeds the metadata budget')
    manifest = json.loads(raw.read_text(encoding='utf-8-sig'))
    runtimes.descriptor_for_manifest(manifest)  # the runtime must be a fully pinned official build
    api = None
    if args.upload:
        if (manifest.get('source_license') or {}).get('redistributable') is not True:
            raise ValueError('staging requires an affirmative redistribution decision in the manifest')
        token = os.environ.get('HF_TOKEN', '').strip()
        if not token or not manifest.get('target_repo_id'):
            raise RuntimeError('HF_TOKEN and a pinned target repository are required to stage')
        from huggingface_hub import HfApi
        api = HfApi(token=token)
    env = {k: os.environ[k] for k in ('GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT', 'GITHUB_SHA', 'GITHUB_REPOSITORY', 'RUNNER_NAME', 'RUNNER_OS', 'RUNNER_ARCH') if k in os.environ}
    receipt = run_hybrid(manifest, ggml_path=args.ggml, work=Path(args.work), job_id=args.job_id, api=api, branch=args.branch, env=env)
    import hashlib
    receipt['manifest_sha256'] = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    receipts = Path(args.work) / 'receipts'
    receipts.mkdir(parents=True, exist_ok=True)
    atomic_json(receipts / 'hybrid.json', receipt)
    print(json.dumps({'state': receipt['state'], 'size': receipt['output']['size'], 'sha256': receipt['output']['sha256']}))


if __name__ == '__main__':
    main()
