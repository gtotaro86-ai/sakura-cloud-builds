"""Approved llama.cpp runtimes for the GitHub runner path.

The baseline (b11669) ships with the source. Newer builds become usable only after a qualification run on the runner
(`scripts/qualify_runtime.py`: pinned asset hash, bridge built against that exact revision, stock-quantize parity,
real inference) was judged by Oracle (`hfqa/runtime_channel.py`). Approvals live in the data directory, because a
deployment replaces the source tree. A runtime is always identified by its 40-hex commit; the manifest of a job carries
the exact asset, so the runner never looks anything up at release time.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

BASELINE_TAG = 'b11669'
BASELINE_REVISION = '620d2225cc436ab0b001e3200e590e3b286ef9ab'
BASELINE_SHA256 = 'cf07c1c112d709bd67e522017471de57ed0ec8bea031d732c5422351c07f65ac'
MAX_ASSET_BYTES = 64 * 1024 ** 2
TAG = re.compile(r'b\d{3,6}')
HEX40 = re.compile(r'[a-f0-9]{40}')
HEX64 = re.compile(r'[a-f0-9]{64}')


def asset_name(tag: str) -> str:
    return f'llama-{tag}-bin-ubuntu-x64.tar.gz'


def asset_url(tag: str) -> str:
    return f'https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{asset_name(tag)}'


def baseline() -> dict:
    return {'tag': BASELINE_TAG, 'revision': BASELINE_REVISION, 'url': asset_url(BASELINE_TAG), 'sha256': BASELINE_SHA256}


def validate_descriptor(value) -> dict:
    """The only fields a runner may act on: tag, commit, the official asset URL of that tag, and its SHA-256."""
    if not isinstance(value, dict):
        raise ValueError('runtime descriptor must be an object')
    tag, revision, url, digest = value.get('tag'), value.get('revision'), value.get('url'), value.get('sha256')
    if not (isinstance(tag, str) and TAG.fullmatch(tag) and isinstance(revision, str) and HEX40.fullmatch(revision)
            and isinstance(digest, str) and HEX64.fullmatch(digest) and url == asset_url(tag)):
        raise ValueError('runtime descriptor is not an official, fully pinned llama.cpp build')
    return {'tag': tag, 'revision': revision, 'url': url, 'sha256': digest}


def descriptor_for_manifest(manifest: dict) -> dict:
    """What a runner must fetch for this manifest: the pinned descriptor, or the baseline when none is named."""
    revision = manifest.get('runtime_revision')
    named = manifest.get('runtime')
    if named is None:
        if revision != BASELINE_REVISION:
            raise ValueError('manifest names a runtime revision without its asset descriptor')
        return baseline()
    descriptor = validate_descriptor(named)
    if descriptor['revision'] != revision:
        raise ValueError('runtime descriptor differs from the manifest revision')
    return descriptor


def _path() -> Path:
    return Path(os.environ.get('HFQA_DATA', 'data')) / 'approved-runtimes.json'


def _overlay() -> dict:
    try:
        value = json.loads(_path().read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def approved() -> list[dict]:
    """Baseline first, then approved newer builds, newest tag first. Each entry also carries its architecture list."""
    from hfqa import llama_arch
    entries = [{**baseline(), 'gguf_architectures': list(llama_arch.stock()['gguf_architectures']), 'baseline': True}]
    extra = []
    for item in _overlay().values():
        if item.get('revision') == BASELINE_REVISION:
            continue  # extra presets proven on the baseline are recorded in the overlay but do not make it a second runtime
        try:
            extra.append({**validate_descriptor(item), 'gguf_architectures': [str(a) for a in item.get('gguf_architectures') or []],
                          'approved_at': item.get('approved_at'), 'qualification': item.get('qualification')})
        except ValueError:
            continue
    return entries + sorted(extra, key=lambda e: int(e['tag'][1:]), reverse=True)


def is_approved(revision) -> bool:
    return isinstance(revision, str) and any(item['revision'] == revision for item in approved())


def qualified_quants(revision: str) -> set:
    """Presets this build has proven (tensor-for-tensor parity with its own llama-quantize plus a real inference)."""
    from hfqa import quants
    found = set(quants.CORE) if revision == BASELINE_REVISION else set()
    entry = _overlay().get(revision) or {}
    return found | set((entry.get('qualification') or {}).get('quants') or [])


def for_architecture(architecture: str | None) -> dict | None:
    """The baseline when it knows the architecture, otherwise the newest approved build that does."""
    if not architecture:
        return None
    for item in approved():
        if architecture in item['gguf_architectures']:
            return {key: item[key] for key in ('tag', 'revision', 'url', 'sha256')}
    return None


def approve(descriptor: dict, gguf_architectures: list[str], qualification: dict) -> dict:
    descriptor = validate_descriptor(descriptor)
    if not gguf_architectures:
        raise ValueError('a runtime needs its architecture list before approval')
    overlay = _overlay()
    old = set(((overlay.get(descriptor['revision']) or {}).get('qualification') or {}).get('quants') or [])
    qualification = {**qualification, 'quants': sorted(old | set(qualification.get('quants') or []))}
    overlay[descriptor['revision']] = {**descriptor, 'gguf_architectures': sorted(set(gguf_architectures)),
                                       'approved_at': datetime.now(timezone.utc).isoformat(), 'qualification': qualification}
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix='.approved-runtimes-', dir=path.parent)
    with os.fdopen(handle, 'w', encoding='utf-8') as out:
        json.dump(overlay, out, indent=1, sort_keys=True)
    os.replace(name, path)
    return overlay[descriptor['revision']]


def revoke(revision: str) -> bool:
    overlay = _overlay()
    if overlay.pop(revision, None) is None:
        return False
    path = _path()
    handle, name = tempfile.mkstemp(prefix='.approved-runtimes-', dir=path.parent)
    with os.fdopen(handle, 'w', encoding='utf-8') as out:
        json.dump(overlay, out, indent=1, sort_keys=True)
    os.replace(name, path)
    return True
