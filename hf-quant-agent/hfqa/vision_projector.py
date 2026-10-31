"""Discover vision projector candidates without authorising publication.

Same-repository membership is provenance, not proof that a projector is
compatible with a quant or that image inference works. No download or copy is
performed here. The release path remains text-only until separately qualified.
"""
from __future__ import annotations

import re
from hfqa.remote_gguf import descriptor_url

MAX_CANDIDATES = 32


PRECISION_ORDER = ('f16', 'bf16', 'f32', 'q8_0')  # f16 is read by every llama.cpp backend; the others follow by how little they lose


def choose(discovery: dict | None) -> dict | None:
    """The one projector a release carries over: the best-supported precision of the candidates found next to the pinned source file, or None."""
    candidates = (discovery or {}).get('candidates') if (discovery or {}).get('status') == 'candidates_found' else None
    if not candidates:
        return None

    def rank(item):
        name = item['filename'].lower()
        order = next((i for i, tag in enumerate(PRECISION_ORDER) if re.search(rf'(?<![a-z0-9]){tag}(?![a-z0-9])', name)), len(PRECISION_ORDER))
        return order, item['size'], name
    return dict(min(candidates, key=rank))


def target_name(projector: dict) -> str:
    """File name in the release: the one of the original repository (flat), which always names the projector."""
    name = projector['filename'].rsplit('/', 1)[-1]
    if not re.fullmatch(r'[A-Za-z0-9._+-]{1,150}\.gguf', name) or 'mmproj' not in name.lower():
        raise ValueError('projector file name is not acceptable')
    return name


def _lfs_sha(entry):
    lfs = getattr(entry, 'lfs', None)
    return lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)


def projector_operation(api, projector: dict, remote: dict):
    """Hub-side copy of the vision projector of the original GGUF repository into the release (nothing is downloaded or changed).

    The source file is looked up again at its pinned revision and must have exactly the size and SHA-256 that were pinned when the job was planned.
    Returns (operation or None when the release already holds the identical file, target name, expected facts)."""
    from huggingface_hub import CommitOperationCopy
    name = target_name(projector)
    entries = {entry.path: entry for entry in api.get_paths_info(repo_id=projector['repo_id'], revision=projector['revision'], paths=[projector['filename']])}
    entry = entries.get(projector['filename'])
    if entry is None or getattr(entry, 'size', None) != projector['size'] or _lfs_sha(entry) != projector['sha256']:
        raise RuntimeError('the vision projector of the original repository differs from the one pinned for this release')
    expected = {'size': projector['size'], 'sha256': projector['sha256']}
    current = remote.get(name)
    if current is not None:
        if getattr(current, 'size', None) == expected['size'] and _lfs_sha(current) == expected['sha256']:
            return None, name, expected
        raise RuntimeError('a different file named like the vision projector already exists in the release repository')
    return CommitOperationCopy(src_repo_id=projector['repo_id'], src_repo_type='model', src_revision=projector['revision'],
                               src_path_in_repo=projector['filename'], path_in_repo=name), name, expected


def for_source(repo: str, revision: str, read_json) -> dict | None:
    """The projector next to a pinned source file, read from the Hub (`read_json(url)` returns the parsed revision listing), or None."""
    import urllib.parse
    info = read_json(f"https://huggingface.co/api/models/{urllib.parse.quote(repo, safe='/')}/revision/{revision}?blobs=true")
    return choose(discover(repo, revision, info))


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
