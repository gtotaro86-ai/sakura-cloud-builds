"""Secret redaction, untrusted-text handling, and path confinement."""

from __future__ import annotations

import re
from pathlib import Path

SECRET_RE = re.compile(
    r"(?i)(hf_[A-Za-z0-9]{8,}|sk-[A-Za-z0-9]{8,}|github_pat_[A-Za-z0-9_]{8,}|ghp_[A-Za-z0-9]{8,}|Bearer\s+[A-Za-z0-9._\-]{8,})"
)
INSTRUCTION_MARKERS = (
    "ignore previous",
    "ignore all previous",
    "system prompt",
    "you are now",
    "disregard",
    "</tool",
    "<tool_call>",
)


def redact(text: str) -> str:
    return SECRET_RE.sub("[REDACTED]", text or "")


def as_untrusted_data(text: str, limit: int = 4000) -> str:
    """HF cards and repo files are data, never instructions."""
    cleaned = redact(text or "")
    cleaned = cleaned.replace("\x00", "")
    lowered = cleaned.lower()
    flagged = [marker for marker in INSTRUCTION_MARKERS if marker in lowered]
    body = cleaned[:limit]
    note = ""
    if flagged:
        note = " [untrusted-data: instruction-like markers ignored: " + ", ".join(flagged) + "]"
    return "UNTRUSTED_REPOSITORY_DATA:" + body + note


def confine(root: Path, candidate: Path) -> Path:
    root = root.resolve()
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError(f"path escapes workspace: {candidate}")
    return resolved
