"""Licenses the operator has read and approved for redistribution of quantizations.

The automatic allow-list only knows plain licenses (Apache-2.0, MIT, ...). Many vendor licenses show up on Hugging Face as
`license: other` plus a `license_name` and a link. Whether such a license permits republishing quantized derivatives is a
legal decision that stays with the operator: it is recorded here, per license name, with the link that was read.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{1,99}')


def _path() -> Path:
    return Path(os.environ.get('HFQA_DATA', 'data')) / 'license-approvals.json'


def load() -> dict:
    try:
        value = json.loads(_path().read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def is_approved(license_name) -> dict | None:
    key = str(license_name or '').strip().lower()
    return load().get(key) if key else None


def _save(value: dict) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix='.license-approvals-', dir=path.parent)
    with os.fdopen(handle, 'w', encoding='utf-8') as out:
        json.dump(value, out, indent=1, sort_keys=True)
    os.replace(name, path)


def approve(license_name: str, license_link: str, *, auto: bool = False, evidence: dict | None = None) -> dict:
    if not NAME.fullmatch(license_name or ''):
        raise ValueError('invalid license name')
    if not re.fullmatch(r'https://[A-Za-z0-9._~:/?#\[\]@!$&()*+,;=%-]{4,400}', license_link or ''):
        raise ValueError('the license link must be an https address that was read before approving')
    approvals = load()
    approvals[license_name.lower()] = {'license_name': license_name, 'license_link': license_link,
                                       'approved_at': datetime.now(timezone.utc).isoformat(), 'automatic': bool(auto),
                                       **({'evidence': evidence} if evidence else {})}
    _save(approvals)
    return approvals[license_name.lower()]


def revoke(license_name: str) -> bool:
    approvals = load()
    if approvals.pop(str(license_name or '').lower(), None) is None:
        return False
    _save(approvals)
    return True
