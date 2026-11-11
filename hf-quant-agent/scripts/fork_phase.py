"""What phase A of a llama.cpp FORK build does inside its container (scripts/modal_fork_phase.py), without any dependency on Modal so that it can be tested anywhere.

Clone the pinned commit of the fork, build the four tools, run the quantizations and the inference test with `scripts/run_imatrix_worker.py --phase quantize` (which refuses to start
when a key is in its environment), and hand the files over. Every subprocess gets a minimal environment of its own; the container has no secret to begin with.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

MOUNT = '/vol'
APP_DIR = '/app'
JOB_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}')
GITHUB_ENV = ('GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT', 'GITHUB_SHA', 'GITHUB_REPOSITORY', 'RUNNER_NAME', 'RUNNER_OS', 'RUNNER_ARCH')
LOG_TAIL = 4000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(8 * 1024 ** 2):
            digest.update(chunk)
    return digest.hexdigest()


def minimal_env(github_env: dict, app_dir: str = APP_DIR, home: str = '/tmp/home') -> dict:
    """Nothing of the container's own variables, only what the receipts name about the GitHub run."""
    env = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': home, 'LANG': 'C.UTF-8', 'TMPDIR': '/tmp', 'OMP_NUM_THREADS': str(os.cpu_count() or 4),
           'HF_HOME': home + '/hf', 'HF_HUB_DISABLE_TELEMETRY': '1', 'HF_HUB_DISABLE_PROGRESS_BARS': '1', 'PYTHONPATH': app_dir}
    env.update({key: str(github_env[key])[:200] for key in GITHUB_ENV if key in (github_env or {})})
    return env


def run_command(command: list[str], env: dict, log: Path, cwd: str | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('ab') as out:
        completed = subprocess.run(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        tail = log.read_bytes()[-LOG_TAIL:].decode('utf-8', 'replace')
        raise RuntimeError(f"{' '.join(command[:3])} failed with exit code {completed.returncode}:\n{tail}")


def build_and_quantize(manifest: dict, job_id: str, github_env: dict, root: Path, *, run=run_command, app_dir: str = APP_DIR) -> dict:
    """Everything phase A does, in `root` (a scratch directory). Returns the files to hand over: {'runtime/<name>' | 'work/<path>': [size, sha256]}."""
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    from hfqa import source_builds
    chosen = source_builds.validate(manifest.get('runtime'))
    if not source_builds.is_fork(chosen) or manifest.get('runtime_revision') != chosen['revision']:
        raise ValueError('this builds a fork at the commit named in the manifest, nothing else')
    if not JOB_ID.fullmatch(job_id or ''):
        raise ValueError('not a job id')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    env = minimal_env(github_env, app_dir, str(root / 'home'))
    llama, build, work, logs = root / 'llama', root / 'out' / 'llama-build', root / 'out' / 'work', root / 'logs'
    request = root / 'request.json'
    request.write_text(json.dumps(manifest), encoding='utf-8')
    Path(env['HOME']).mkdir(parents=True, exist_ok=True)
    llama.mkdir(parents=True)
    git = ['git', '-c', 'protocol.file.allow=never', '-c', 'advice.detachedHead=false']
    run(git + ['init', '-q', str(llama)], env, logs / 'git.log')
    run(git + ['-C', str(llama), 'remote', 'add', 'origin', f"https://github.com/{chosen['repo']}.git"], env, logs / 'git.log')
    run(git + ['-C', str(llama), 'fetch', '-q', '--depth', '1', 'origin', chosen['revision']], env, logs / 'git.log')
    run(git + ['-C', str(llama), 'checkout', '-q', 'FETCH_HEAD'], env, logs / 'git.log')
    py = sys.executable
    run([py, f'{app_dir}/scripts/build_llama_source.py', '--manifest', str(request), '--llama', str(llama), '--destination', str(build)], env, logs / 'build.log')
    run([py, f'{app_dir}/scripts/run_imatrix_worker.py', '--manifest', str(request), '--runtime-dir', str(build), '--build-receipt', str(build / 'build-receipt.json'),
         '--work', str(work), '--job-id', job_id, '--phase', 'quantize'], env, logs / 'quantize.log')
    handed = {}
    for base, prefix in ((build, 'runtime'), (work, 'work')):
        for path in sorted(base.rglob('*')):
            relative = path.relative_to(base)
            if path.is_file() and not path.is_symlink() and not (prefix == 'work' and relative.parts[0] == 'source'):   # the source is fetched again by the GitHub job, never taken from here
                handed[f'{prefix}/{relative.as_posix()}'] = [path.stat().st_size, sha256_file(path)]
    return {'handed': handed, 'build': str(build), 'work': str(work)}


def run_and_report(manifest_json: str, job_id: str, github_env_json: str, scratch: Path, volume_root: Path = Path(MOUNT), *, build=build_and_quantize) -> dict:
    """The whole container job. Never raises: the outcome is written to <volume>/<job id>/status.json (and returned)."""
    started = time.monotonic()
    target = Path(volume_root) / job_id if JOB_ID.fullmatch(job_id or '') else None
    status = {'ok': False}
    try:
        if target is None:
            raise ValueError('not a job id')
        manifest = json.loads(manifest_json)
        github_env = json.loads(github_env_json or '{}')
        shutil.rmtree(scratch, ignore_errors=True)
        Path(scratch).mkdir(parents=True)
        result = build(manifest, job_id, github_env, Path(scratch))
        shutil.rmtree(target, ignore_errors=True)
        for name in result['handed']:
            kind, _, rest = name.partition('/')
            destination = target / kind / rest
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(result['build'] if kind == 'runtime' else result['work']) / rest, destination)
        status = {'ok': True, 'files': result['handed']}
    except BaseException as error:   # noqa: BLE001 - the caller must not receive an exception object from this container
        status = {'ok': False, 'error': f'{type(error).__name__}: {error}'[:LOG_TAIL * 2], 'trace': traceback.format_exc()[-LOG_TAIL:]}
    status['seconds'] = round(time.monotonic() - started, 1)
    try:
        if target is not None:
            target.mkdir(parents=True, exist_ok=True)
            partial = target / 'status.json.partial'
            partial.write_text(json.dumps(status), encoding='utf-8')
            partial.replace(target / 'status.json')
    except OSError:
        pass
    return status
