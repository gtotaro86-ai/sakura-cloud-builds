"""Automatic license check for vendor licenses published as `license: other`.

The operator asked for this to run by itself. It is a text check, not legal advice, so it is deliberately strict:
the license text must be fetched from a public https address, show no clear restriction on sharing or on derivatives
(non-commercial, research only, no redistribution, no derivatives, internal use only) and must grant distribution of
derivative works. Anything else stays blocked and the plan offers the manual decision instead. Every automatic approval
is stored with its evidence (link, SHA-256 of the text, matched phrases) and can be revoked at any time.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import socket
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

MAX_BYTES = 1024 ** 2
TIMEOUT = 8
RECHECK_SECONDS = 6 * 3600
HEADERS = {'User-Agent': 'hf-quant-agent license check', 'Accept': 'text/plain, text/html, text/markdown;q=0.9'}

RESTRICTIONS = [
    ('non-commercial', r'non[- ]?commercial|not for commercial'),
    ('research or evaluation only', r'(for )?(research|academic|evaluation|educational) (and (research|evaluation) )?(purposes |use )?only|research use only'),
    ('redistribution forbidden', r'(may|shall|will|must|can) not (\w+ ){0,3}(re)?distribut|no (re)?distribution|prohibited from (re)?distribut|(do|does) not (\w+ ){0,3}(re)?distribut'),
    ('no derivatives', r'no[- ]?derivative|may not (create|make|prepare) (\w+ ){0,3}derivative'),
    ('internal or personal use only', r'internal use only|personal use only|for internal purposes only'),
]
GRANTS = [
    ('grants distribution', r'\b(re)?distribut(e|ion|ing|ed)\b'),
    ('covers derivative works', r'derivative (works|models|model|work)'),
    ('contains a grant', r'\b(grant|grants|granted|license|licenses|permit|permits|permitted)\b'),
]


def _data() -> Path:
    return Path(os.environ.get('HFQA_DATA', 'data'))


def _read(name: str) -> dict:
    try:
        value = json.loads((_data() / name).read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(name: str, value: dict) -> None:
    path = _data() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix='.' + name + '-', dir=path.parent)
    with os.fdopen(handle, 'w', encoding='utf-8') as out:
        json.dump(value, out, indent=1, sort_keys=True)
    os.replace(temporary, path)


def auto_enabled() -> bool:
    """On by default, as the operator asked; switchable from the app."""
    return _read('license-policy.json').get('auto', True) is not False


def set_auto(enabled: bool) -> bool:
    _write('license-policy.json', {'auto': bool(enabled)})
    return auto_enabled()


def _public_host(host: str) -> None:
    for family, _, _, _, address in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM):
        if not ipaddress.ip_address(address[0]).is_global:
            raise ValueError('license link points to a non-public address')


class _CheckedRedirect(urllib.request.HTTPRedirectHandler):
    max_redirections = 3

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_url(urllib.parse.urljoin(req.full_url, newurl))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _check_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.port not in (None, 443):
        raise ValueError('license link must be a plain https address')
    _public_host(parsed.hostname)


def fetch_text(url: str) -> str:
    """The link comes from an untrusted model card: https only, public addresses only, bounded size, no scripts executed."""
    _check_url(url)
    opener = urllib.request.build_opener(_CheckedRedirect())
    with opener.open(urllib.request.Request(url, headers=HEADERS), timeout=TIMEOUT) as response:
        kind = response.headers.get('Content-Type', '').split(';')[0].strip().lower()
        if kind and not (kind.startswith('text/') or kind in {'application/xhtml+xml', 'application/json'}):
            raise ValueError('license text is not a text document')
        raw = response.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError('license text is too large')
    text = raw.decode('utf-8', 'replace')
    if '<' in text and re.search(r'<(html|body|div|p)\b', text, re.I):
        text = re.sub(r'<(script|style)\b.*?</\1>', ' ', text, flags=re.I | re.S)
        text = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def review_text(text: str) -> dict:
    """Verdict on a license text: 'approve' only without any restriction hint and with all grant hints."""
    flat = re.sub(r'\s+', ' ', text or '').lower()
    if len(flat) < 400:
        return {'verdict': 'unclear', 'reasons': ['the license text is too short to judge'], 'matched': []}
    restrictions = [label for label, pattern in RESTRICTIONS if re.search(pattern, flat)]
    if restrictions:
        return {'verdict': 'block', 'reasons': ['the text mentions: ' + ', '.join(restrictions)], 'matched': restrictions}
    grants = [label for label, pattern in GRANTS if re.search(pattern, flat)]
    if len(grants) < len(GRANTS):
        return {'verdict': 'unclear', 'reasons': ['no clear grant to distribute derivative works found'], 'matched': grants}
    return {'verdict': 'approve', 'reasons': ['no restriction on sharing or derivatives found; distribution of derivative works is granted'], 'matched': grants}


def auto_decide(license_name, license_link, *, fetch=None, now=None) -> dict | None:
    """The stored approval when the license passes the check, otherwise None. The review itself is kept for the plan view."""
    from hfqa import license_approvals
    name, link = str(license_name or '').strip(), str(license_link or '').strip()
    if not name or not link.startswith('https://') or not auto_enabled():
        return None
    existing = license_approvals.is_approved(name)
    if existing:
        return existing
    reviews = _read('license-reviews.json')
    key = name.lower()
    seen = reviews.get(key)
    stamp = (now or datetime.now(timezone.utc))
    if seen and seen.get('link') == link and (stamp.timestamp() - float(seen.get('checked_epoch', 0))) < RECHECK_SECONDS:
        return None
    try:
        text = (fetch or fetch_text)(link)
        result = review_text(text)
        evidence = {'sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(), 'text_length': len(text)}
    except (OSError, ValueError, urllib.error.URLError) as exc:
        result, evidence = {'verdict': 'unclear', 'reasons': [f'license text could not be fetched ({type(exc).__name__})'], 'matched': []}, {}
    reviews[key] = {**result, **evidence, 'link': link, 'license_name': name, 'checked_at': stamp.isoformat(), 'checked_epoch': stamp.timestamp()}
    _write('license-reviews.json', reviews)
    if result['verdict'] != 'approve':
        return None
    return license_approvals.approve(name, link, auto=True, evidence={**evidence, 'matched': result['matched']})


def review_for(license_name) -> dict | None:
    value = _read('license-reviews.json').get(str(license_name or '').strip().lower())
    return {k: v for k, v in value.items() if k != 'checked_epoch'} if value else None
