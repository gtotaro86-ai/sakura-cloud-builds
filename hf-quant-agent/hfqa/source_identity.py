"""What a matrix is bound to when the source is a set of files: one identity for the ordered set, made only from the pinned per-file facts.

A single source file keeps its own descriptor (nothing changes for it). A split source (every model above ~50 GB, because the Hub caps a file) is bound by the
SHA-256 of the ordered list of [index, size, per-file SHA-256]; the matrix job verifies every file against its pinned hash first, so the set hash vouches for
exactly those bytes. Oracle computes the same value from the manifest it pinned, with no weights involved.
"""
from __future__ import annotations

import hashlib
import json


def set_sha256(descriptors: list[dict]) -> str:
    ordered = sorted(descriptors, key=lambda item: item.get('split_index', 1))
    rows = [[item.get('split_index', 1), item['size'], str(item['sha256']).lower()] for item in ordered]
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode('utf-8')).hexdigest()


def identity_of(descriptors: list[dict]) -> dict:
    """The descriptor the matrix provenance is checked against."""
    if len(descriptors) == 1:
        return descriptors[0]
    first = sorted(descriptors, key=lambda item: item.get('split_index', 1))[0]
    return {'repo_id': first['repo_id'], 'revision': first['revision'], 'filename': first['filename'], 'size': sum(item['size'] for item in descriptors),
            'sha256': set_sha256(descriptors), 'split_count': len(descriptors)}


def identity(source: dict) -> dict:
    """Same, for the source structure the runner works with (`{'sources': [...]}` for a set, the descriptor itself for one file)."""
    return identity_of(source['sources']) if source.get('sources') else source
