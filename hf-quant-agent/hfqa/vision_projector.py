"""Discover vision projector candidates without authorising publication.

Same-repository membership is provenance, not proof that a projector is
compatible with a quant or that image inference works. No download or copy is
performed here. The release path remains text-only until separately qualified.
"""
from __future__ import annotations

import re
from hfqa.remote_gguf import descriptor_url

MAX_CANDIDATES = 32


def discover(repo: str, revision: str, info: dict) -> dict:
    result = {'status': 'unavailable', 'candidates': [], 'automatic_copy': False,
              'compatibility_tested': False, 'image_inference_tested': False}
    if not isinstance(info, dict) or not re.fullmatch(r'[a-fA-F0-9]{40}', revision or ''):
        result['reason'] = 'source revision not pinned'
        return result
    if str(info.get('sha') or '').lower() != revision.lower():
        result['reason'] = 'Hub response did not confirm the pinned revision'
        return result
    if info.get('private') is not False or info.get('gated') is not False:
        result['reason'] = 'source is private, gated, or its access status is unknown'
        return result
    matches = []
    for item in info.get('siblings') or []:
        if not isinstance(item, dict):
            continue
        name = item.get('rfilename') or ''
        if not isinstance(name, str) or not re.search(r'mmproj', name, re.I) or not name.lower().endswith('.gguf'):
            continue
        lfs = item.get('lfs') or {}
        if not isinstance(lfs, dict):
            continue
        digest = str(lfs.get('sha256') or lfs.get('oid') or '').removeprefix('sha256:')
        size = item.get('size') or lfs.get('size')
        if type(size) is not int or size < 24 or not re.fullmatch(r'[a-fA-F0-9]{64}', digest):
            continue
        desc = {'repo_id': repo, 'revision': revision.lower(), 'filename': name, 'size': size, 'sha256': digest.lower()}
        try:
            descriptor_url(desc)  # also rejects unsafe paths
        except (KeyError, TypeError, ValueError):
            continue
        matches.append(desc)
    if len(matches) > MAX_CANDIDATES:
        result['reason'] = 'too many projector candidates'
        return result
    result.update(status='candidates_found' if matches else 'none_found', candidates=matches,
                  reason='candidates require license, compatibility and image-inference checks before copying')
    return result
