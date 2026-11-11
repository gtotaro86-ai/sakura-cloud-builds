"""The GitHub job's side of a fork build: start phase A on Modal, wait, download what it left and check it is what it claims to be (docs/FORK-BUILDS.md).

This process holds a Modal token and no Hugging Face key, and it executes nothing that comes from Modal: it only writes files. Every path is checked, the total size is bounded, and the
hash of each file is compared with what the container reported (a truncated download, nothing more: phase B hashes everything again and does not trust the report).

    python scripts/run_fork_on_modal.py --manifest request.json --job-id <id> --work <dir> --runtime-dir <dir>

Needs MODAL_TOKEN_ID and MODAL_TOKEN_SECRET (GitHub secrets of the runner repository) and the `modal` package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MAX_FILES = 400
MAX_TOTAL_BYTES = 110 * 1024 ** 3     # the runner has about this much free after cleaning
STATUS_WAIT_S = 180                   # the volume is committed when the container ends: the status may show up a little after the call returned
ALLOWED = re.compile(r'(runtime|work)/[A-Za-z0-9._+=@-]+(/[A-Za-z0-9._+=@-]+){0,3}')


def safe_destination(name: str, work: Path, runtime: Path) -> Path:
    """Where a reported file goes, or ValueError. Only plain relative names below `runtime/` or `work/`."""
    if not isinstance(name, str) or not ALLOWED.fullmatch(name) or '..' in name.split('/') or any(part in ('.', '') for part in name.split('/')):
        raise ValueError('a file name from the container is not acceptable')
    kind, _, rest = name.partition('/')
    base = (runtime if kind == 'runtime' else work).resolve()
    destination = (base / rest).resolve()
    if base not in destination.parents:
        raise ValueError('a file would leave its directory')
    return destination


def read_status(volume, job_id: str, wait: float = STATUS_WAIT_S, sleep=time.sleep) -> dict:
    deadline = time.monotonic() + wait
    last = 'nothing read yet'
    while True:
        try:   # no volume.reload() here: it works only inside a container, and a client always reads the committed state
            raw = b''.join(volume.read_file(f'{job_id}/status.json'))
            if len(raw) > 4 * 1024 ** 2:
                raise ValueError('the status file is unreasonably large')
            value = json.loads(raw.decode('utf-8'))
            if isinstance(value, dict):
                return value
            last = 'the status file is not a JSON object'
        except Exception as error:   # noqa: BLE001 - not there yet (Modal's own not-found error), or not valid JSON yet: wait, and remember why
            if isinstance(error, ValueError) and 'unreasonably' in str(error):
                raise
            last = f'{type(error).__name__}: {str(error)[:200]}'
        if time.monotonic() > deadline:
            raise RuntimeError('the container left no status file (' + last + ')')
        sleep(5)


def download(volume, job_id: str, status: dict, work: Path, runtime: Path) -> dict:
    files = status.get('files')
    if not isinstance(files, dict) or not 1 <= len(files) <= MAX_FILES:
        raise RuntimeError('the container reported no usable file list')
    total = 0
    for name, entry in files.items():
        if not (isinstance(entry, list) and len(entry) == 2 and isinstance(entry[0], int) and entry[0] >= 0 and isinstance(entry[1], str) and re.fullmatch('[a-f0-9]{64}', entry[1])):
            raise RuntimeError('a file entry from the container is malformed')
        total += entry[0]
        safe_destination(name, work, runtime)
    if total > MAX_TOTAL_BYTES:
        raise RuntimeError('the container hands over more than the runner can hold')
    for name, (size, digest) in files.items():
        destination = safe_destination(name, work, runtime)
        destination.parent.mkdir(parents=True, exist_ok=True)
        hasher, written = hashlib.sha256(), 0
        with destination.open('wb') as out:
            for chunk in volume.read_file(f'{job_id}/{name}'):
                written += len(chunk)
                if written > size:
                    raise RuntimeError(f'{name} is larger than reported')
                hasher.update(chunk)
                out.write(chunk)
        if written != size or hasher.hexdigest() != digest:
            raise RuntimeError(f'{name} was damaged on the way')
        if name.startswith('runtime/llama-') and '.' not in name.rsplit('/', 1)[-1]:
            destination.chmod(0o755)
    return {'files': len(files), 'bytes': total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--runtime-dir', type=Path, required=True)
    args = parser.parse_args()
    if not (os.environ.get('MODAL_TOKEN_ID') and os.environ.get('MODAL_TOKEN_SECRET')):
        raise SystemExit('a fork build runs on Modal and needs the Modal token (MODAL_TOKEN_ID, MODAL_TOKEN_SECRET) as secrets of the runner repository')
    if os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN'):
        raise SystemExit('this step must run without a Hugging Face key')
    if args.manifest.stat().st_size > 1024 ** 2:
        raise ValueError('manifest exceeds metadata budget')
    from hfqa import source_builds
    manifest = json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    if not source_builds.is_fork(source_builds.validate(manifest.get('runtime'))):
        raise ValueError('this step is for builds from a fork only')
    from scripts import fork_phase
    if not fork_phase.JOB_ID.fullmatch(args.job_id):
        raise ValueError('not a job id')
    import modal
    from scripts import modal_fork_phase as phase
    github_env = {key: os.environ[key] for key in fork_phase.GITHUB_ENV if key in os.environ}
    started = time.monotonic()
    outcome = 'failed'
    try:
        with modal.enable_output(), phase.app.run():
            outcome = phase.phase_a.remote(args.manifest.read_text(encoding='utf-8-sig'), args.job_id, json.dumps(github_env))
        volume = phase.volume
        status = read_status(volume, args.job_id)
        if outcome != 'done' or status.get('ok') is not True:
            raise RuntimeError('phase A on Modal failed: ' + str(status.get('error') or 'no reason given')[:3000] + '\n' + str(status.get('trace') or '')[-1500:])
        for directory in (args.work, args.runtime_dir):
            Path(directory).mkdir(parents=True, exist_ok=True)
        summary = download(volume, args.job_id, status, Path(args.work), Path(args.runtime_dir))
        print(json.dumps({'event': 'fork_phase_a_downloaded', **summary, 'container_seconds': status.get('seconds'), 'seconds': round(time.monotonic() - started, 1)}), flush=True)
    finally:
        try:   # the hand-over directory is not kept
            phase.volume.remove_file(args.job_id, recursive=True)
        except Exception:   # noqa: BLE001
            pass
        if outcome != 'done':
            shutil.rmtree(args.work, ignore_errors=True)


if __name__ == '__main__':
    main()
