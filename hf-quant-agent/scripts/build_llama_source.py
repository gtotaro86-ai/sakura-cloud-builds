"""Build llama.cpp from the exact master commit Oracle chose, for the importance-matrix path (runs in a job that has no secrets).

The checkout must be that commit (`git rev-parse HEAD`), the build is static (no library path surprises) and limited to the four tools the
imatrix path uses (llama-imatrix, llama-quantize, llama-completion for the inference test, llama-perplexity). The receipt carries the SHA-256 of every binary; the imatrix job re-hashes what it downloads and Oracle compares the same
hashes with the ones in the job receipt. Without a secret in this job's environment the CMake scripts of an unreleased commit can read nothing
worth stealing; the build also runs with a minimal environment of its own.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hfqa import source_builds

FLAGS = ['-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=OFF', '-DGGML_NATIVE=ON', '-DLLAMA_CURL=OFF', '-DLLAMA_OPENSSL=OFF',
         '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_SERVER=OFF', '-DLLAMA_BUILD_EXAMPLES=ON', '-DLLAMA_BUILD_TOOLS=ON']


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(1024 ** 2):
            digest.update(chunk)
    return digest.hexdigest()


def clean_env() -> dict:
    return {key: os.environ[key] for key in ('PATH', 'HOME', 'LANG', 'TMPDIR') if key in os.environ}


def run(command: list[str], cwd: Path, log: Path) -> None:
    with log.open('ab') as out:
        completed = subprocess.run(command, cwd=cwd, env=clean_env(), stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        tail = log.read_bytes()[-3000:].decode('utf-8', 'replace')
        raise RuntimeError(f"{' '.join(command[:3])} failed with exit code {completed.returncode}:\n{tail}")


def build(manifest: dict, llama: Path, destination: Path, env: dict | None = None) -> dict:
    chosen = source_builds.validate(manifest.get('runtime'))
    if manifest.get('runtime_revision') != chosen['revision']:
        raise ValueError('the manifest names another runtime revision than its source build')
    llama, destination = Path(llama), Path(destination)
    head = subprocess.run(['git', '-C', str(llama), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True, env=clean_env()).stdout.strip()
    if head != chosen['revision']:
        raise ValueError(f'the checkout is at {head}, not at the commit Oracle chose')
    destination.mkdir(parents=True, exist_ok=True)
    log = destination / 'build.log'
    started = time.monotonic()
    folder = llama / 'build'
    cpus = os.cpu_count() or 2
    # The commit goes into the binary exactly, so that the tools report it (the shallow checkout would otherwise yield build 1 and a short hash).
    run(['cmake', '-S', str(llama), '-B', str(folder), *FLAGS, f"-DLLAMA_BUILD_COMMIT={chosen['revision']}", '-DLLAMA_BUILD_NUMBER=0'], llama, log)
    run(['cmake', '--build', str(folder), '--config', 'Release', '-j', str(cpus), '--target', *source_builds.TARGETS], llama, log)
    binaries = {}
    for name in source_builds.TARGETS:
        found = [path for path in folder.rglob(name) if path.is_file() and os.access(path, os.X_OK)]
        if len(found) != 1:
            raise FileNotFoundError(f'{name} was not produced exactly once')
        shutil.copy2(found[0], destination / name)
        binaries[name] = sha256(destination / name)
    probe = subprocess.run([str(destination / source_builds.INFERENCE_TOOL), '--version'], capture_output=True, text=True, env=clean_env(), timeout=60, check=False)
    shown = probe.stdout + probe.stderr
    if probe.returncode != 0 or chosen['revision'][:9] not in shown:
        raise RuntimeError(f'{source_builds.INFERENCE_TOOL} does not report the commit it was built from (exit {probe.returncode}): {shown[-400:]}')
    receipt = {'repo': source_builds.REPO, 'head': head, 'binaries': binaries, 'static': True, 'flags': FLAGS, 'version_output': shown.strip()[-400:],
               'seconds': round(time.monotonic() - started, 1), 'cpus': cpus, 'build_run_id': int((env or os.environ).get('GITHUB_RUN_ID') or 0)}
    (destination / 'build-receipt.json').write_text(json.dumps(receipt, indent=1, sort_keys=True), encoding='utf-8')
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--llama', type=Path, required=True, help='checkout of ggml-org/llama.cpp at the commit named in the manifest')
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    raw = args.manifest
    if raw.stat().st_size > 1024 ** 2:
        raise ValueError('manifest exceeds metadata budget')
    receipt = build(json.loads(raw.read_text(encoding='utf-8-sig')), args.llama, args.destination)
    print(json.dumps({'event': 'built', 'head': receipt['head'], 'seconds': receipt['seconds'], 'binaries': sorted(receipt['binaries'])}), flush=True)


if __name__ == '__main__':
    main()
