"""llama.cpp built from an exact master commit, for architectures no official release knows yet.

An official release is always preferred (`hfqa/runtimes.py`: pinned asset, parity, qualification). A source build exists only for
the importance-matrix path, whose tools are llama.cpp's own (llama-imatrix, llama-quantize --imatrix, llama-cli, llama-perplexity); no
bridge and no parity run are involved. Oracle chooses the commit (never a moving name), the runner builds it without any secret in the
build job, and the evidence is the job's own receipt: the binaries' SHA-256, a real inference, the measured deviation from the source.
A source build never enters `runtimes.approved()` / `for_architecture()`, so no other path can pick it up.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = 'ggml-org/llama.cpp'
KIND = 'source'
HEX40 = re.compile(r'[a-f0-9]{40}')
TARGETS = ('llama-imatrix', 'llama-quantize', 'llama-cli', 'llama-perplexity')
FILE = 'source-builds.json'


def _data() -> Path:
    return Path(os.environ.get('HFQA_DATA', 'data'))


def descriptor(revision: str) -> dict:
    return validate({'kind': KIND, 'repo': REPO, 'revision': revision, 'tag': 'src-' + str(revision)[:9]})


def is_source(value) -> bool:
    return isinstance(value, dict) and value.get('kind') == KIND


def validate(value) -> dict:
    """The only fields a runner may act on: this repository, one 40-hex commit and the tag derived from it."""
    if not isinstance(value, dict):
        raise ValueError('source build descriptor must be an object')
    revision = value.get('revision')
    if not (value.get('kind') == KIND and value.get('repo') == REPO and isinstance(revision, str) and HEX40.fullmatch(revision)
            and value.get('tag') == 'src-' + revision[:9] and set(value) <= {'kind', 'repo', 'revision', 'tag'}):
        raise ValueError('not an exact ggml-org/llama.cpp source commit')
    return {'kind': KIND, 'repo': REPO, 'revision': revision, 'tag': value['tag']}


def candidate(architecture, cache_dir: Path | None = None) -> dict | None:
    """The descriptor of the cached master commit when that commit knows the architecture (no network access)."""
    if not architecture:
        return None
    from hfqa import llama_arch
    cached = llama_arch.cached_master(cache_dir or _data())
    if not cached or not llama_arch.knows(cached.get('gguf_architectures'), architecture):
        return None
    try:
        return descriptor(cached['revision'])
    except (KeyError, ValueError):
        return None


def qualified_quants() -> set:
    """Every preset: llama-quantize of the very commit is the reference, there is no second implementation to compare with."""
    from hfqa import quants
    return set(quants.CORE) | set(quants.EXTRA)


def _load() -> dict:
    try:
        value = json.loads((_data() / FILE).read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def register(value: dict, architecture: str, job_id: str) -> dict:
    """Oracle's own record that it chose this commit for this job (the receipt check accepts nothing else)."""
    value = validate(value)
    registry = _load()
    entry = registry.get(value['revision']) or {'registered_at': datetime.now(timezone.utc).isoformat(), 'jobs': []}
    entry.update(value, architecture=str(architecture))
    entry['jobs'] = sorted(set(entry.get('jobs') or []) | {str(job_id)})
    registry[value['revision']] = entry
    path = _data() / FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix='.source-builds-', dir=path.parent)
    with os.fdopen(handle, 'w', encoding='utf-8') as out:
        json.dump(registry, out, indent=1, sort_keys=True)
    os.replace(name, path)
    return entry


def is_registered(revision, job_id: str | None = None) -> bool:
    entry = _load().get(revision) if isinstance(revision, str) else None
    return bool(entry) and (job_id is None or str(job_id) in (entry.get('jobs') or []))


def accepts(manifest: dict, job_id: str) -> bool:
    """A manifest that names a source build is valid for the importance-matrix path only, and only for a commit Oracle registered for this job."""
    try:
        chosen = validate(manifest.get('runtime'))
    except ValueError:
        return False
    return (manifest.get('runtime_revision') == chosen['revision'] and isinstance(manifest.get('imatrix'), dict)
            and is_registered(chosen['revision'], job_id))
