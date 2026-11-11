"""Environment for processes that run third-party binaries (llama.cpp tools, possibly built from a fork): no secret goes along.

This is defence in depth only. A child of the same user can still read the environment of its siblings through /proc, so the real protection is that no process holds the key while
fork code runs (scripts/run_imatrix_worker.py --phase quantize runs as a separate user without the key). Nothing the llama.cpp tools need is removed: they read local files.
"""
from __future__ import annotations

import os
import re

SECRET = re.compile(r'(TOKEN|SECRET|PASSW|CREDENTIAL|PRIVATE|APIKEY|API_KEY|(^|_)KEY$|AUTHORIZATION|^ACTIONS_)', re.I)


def clean(env=None) -> dict:
    """A copy of `env` (default: the current environment) without anything that looks like a secret."""
    source = os.environ if env is None else env
    return {key: value for key, value in source.items() if not SECRET.search(key)}
