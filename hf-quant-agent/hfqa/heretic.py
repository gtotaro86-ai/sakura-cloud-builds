"""Optional Heretic release variant. The tool is external; this module does not vendor it.

Heretic (https://github.com/p-e-w/heretic, AGPL-3.0-or-later) ablates a
transformer checkpoint. GGUF export afterwards must still use stock llama.cpp.
Heretic is not run on the Oracle DNS host.
"""

from __future__ import annotations

import re

TOOL_URL = "https://github.com/p-e-w/heretic"
TOOL_PACKAGE = "heretic-llm"
TOOL_LICENSE = "AGPL-3.0-or-later"
VARIANTS = ("normal", "heretic")


def normalize_variants(requested: list[str] | None) -> list[str]:
    if not requested:
        return ["normal"]
    alias = {
        "normal": "normal",
        "original": "normal",
        "heretic": "heretic",
        "both": "both",
        "beide": "both",
    }
    chosen: list[str] = []
    for item in requested:
        key = alias.get(item.strip().lower())
        if key is None:
            raise ValueError("variant must be normal, heretic, or both")
        if key == "both":
            chosen.extend(item for item in VARIANTS if item not in chosen)
            continue
        if key not in chosen:
            chosen.append(key)
    return [item for item in VARIANTS if item in chosen] or ["normal"]


def release_plan(repo_id: str, variants: list[str], publisher: str = "webmp3") -> list[dict]:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", repo_id.split("/", 1)[-1]).strip("-") or "model"
    releases = []
    if "normal" in variants:
        releases.append({
            "variant": "normal",
            "repo_id": f"{publisher}/{slug}-GGUF",
            "heretic": False,
        })
    if "heretic" in variants:
        releases.append({
            "variant": "heretic",
            "repo_id": f"{publisher}/{slug}-Heretic-GGUF",
            "heretic": True,
            "tool": TOOL_URL,
            "tool_license": TOOL_LICENSE,
        })
    return releases


def assess(analysis: dict, variants: list[str], host: str = "oracle", worker_configured: bool = False) -> dict:
    selected = normalize_variants(variants)
    unsupported = str(analysis.get("model_type") or "").lower() in {"mamba", "mamba2", "rwkv", "rwkv6", "rwkv7"}
    oracle = host.lower() == "oracle"
    heretic_selected = "heretic" in selected
    runnable = heretic_selected and not unsupported and not oracle and worker_configured
    blockers = []
    if heretic_selected and unsupported:
        blockers.append("Heretic does not support this architecture out of the box; no custom patch is started")
    if heretic_selected and oracle:
        blockers.append("Heretic is not executed on the Oracle DNS host; it needs a separate PyTorch machine")
    if heretic_selected:
        if not worker_configured:
            blockers.append("Heretic execution worker is not configured; the selection only records a release plan")
        blockers.append("Heretic refusal and KL numbers are untested until a real run records them")
    return {
        "variants": selected,
        "tool": TOOL_URL if heretic_selected else None,
        "package": TOOL_PACKAGE if heretic_selected else None,
        "license": TOOL_LICENSE if heretic_selected else None,
        "releases": release_plan(analysis.get("repo_id") or "unknown/model", selected),
        "runnable_here": runnable,
        "executed": False,
        "allow_custom_patch": False,
        "gguf_runtime": "stock-llama.cpp",
        "blockers": blockers,
    }
