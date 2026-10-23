"""Choose a route from inspection, registry and resource limits. Never invents capacity."""

from __future__ import annotations

import json
from pathlib import Path

from hfqa.heretic import assess as assess_heretic
from hfqa.native import classify as classify_native

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "converters" / "architecture_registry" / "registry.json"
QUANT_ORDER = ("Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M")
QUANT_RATIO = {"Q2_K": 0.28, "Q3_K_M": 0.36, "Q4_K_M": 0.47, "Q5_K_M": 0.56}


def load_registry(path: Path | None = None) -> dict:
    return json.loads((path or REGISTRY_PATH).read_text(encoding="utf-8"))


def normalize_quants(requested: list[str] | None) -> list[str]:
    if not requested:
        return list(QUANT_ORDER)
    alias = {
        "q2": "Q2_K", "q2_k": "Q2_K",
        "q3": "Q3_K_M", "q3_k_m": "Q3_K_M",
        "q4": "Q4_K_M", "q4_k_m": "Q4_K_M",
        "q5": "Q5_K_M", "q5_k_m": "Q5_K_M",
    }
    out = []
    for item in requested:
        key = alias.get(item.strip().lower(), item.strip())
        if key not in QUANT_ORDER:
            raise ValueError(f"unsupported quant {item}")
        if key not in out:
            out.append(key)
    return out


def match_adapter(analysis: dict, registry: dict | None = None) -> dict | None:
    registry = registry or load_registry()
    repo = (analysis.get("repo_id") or "").lower()
    model_type = (analysis.get("model_type") or "").lower()
    architectures = [str(item).lower() for item in analysis.get("architectures") or []]
    for adapter in registry.get("adapters") or []:
        match = adapter.get("match") or {}
        if any(item.lower() == repo for item in match.get("repo_ids") or []):
            return adapter
        if model_type and model_type in [item.lower() for item in match.get("model_types") or []]:
            return adapter
        if any(arch in architectures for arch in [item.lower() for item in match.get("architectures") or []]):
            return adapter
        if any(token.lower() in repo for token in match.get("name_contains") or []):
            return adapter
    return None


def estimate_bytes(analysis: dict, quant: str) -> int | None:
    parameters = analysis.get("parameters")
    if not parameters:
        storage = analysis.get("used_storage")
        if storage and "gguf" in (analysis.get("formats") or []):
            return int(storage * QUANT_RATIO.get(quant, 0.5))
        return None
    # BF16 source is 2 bytes/parameter. K-quant ratios are planning estimates, not measurements.
    return int(int(parameters) * 2 * QUANT_RATIO[quant])


def plan(
    analysis: dict,
    quants: list[str],
    resources: dict | None = None,
    active_workflows: list[dict] | None = None,
    variants: list[str] | None = None,
) -> dict:
    resources = resources or {}
    registry = load_registry()
    adapter = match_adapter(analysis, registry)
    free_disk = int(resources.get("free_disk_bytes") or 0)
    available_ram = int(resources.get("available_ram_bytes") or 0)
    estimates = {quant: estimate_bytes(analysis, quant) for quant in quants}
    known = [item for item in estimates.values() if item]
    largest = max(known) if known else None
    route = "blocked"
    blockers = []
    actions = []
    native = classify_native(analysis, adapter)
    heretic = assess_heretic(analysis, variants or ["normal"], host=str((resources or {}).get("host") or "oracle"))
    actions.append(native["reason"])
    actions.extend(heretic["blockers"])
    blockers.extend(heretic["blockers"])
    if analysis.get("gated") or analysis.get("private"):
        blockers.append("private or gated repository cannot be processed without an explicit authorized token and a separate approval")
        route = "blocked"
    elif adapter and adapter.get("route") == "agent_runtime":
        route = "agent_runtime"
        actions.append("Do not quantize the Spark agent model as a release target unless explicitly requested.")
    elif adapter and adapter.get("route") == "existing_pipeline":
        route = "existing_pipeline"
        running = [
            run for run in (active_workflows or [])
            if run.get("path", "").endswith(adapter.get("workflow", "___")) and run.get("status") in {"queued", "in_progress", "waiting"}
        ]
        actions.append(f"Use existing workflow {adapter.get('workflow')} in {registry.get('github_repo')}.")
        actions.append("Do not add a new llama.cpp patch. If stock llama.cpp already contains the architecture, prefer that runtime.")
        actions.append(adapter.get("notes") or "")
        if running:
            actions.append(f"Active run already exists ({running[0].get('html_url')}). Do not dispatch a duplicate.")
        else:
            actions.append("No active public run observed. Dispatch stays disabled until a GitHub token and an explicit recompute flag are configured.")
        if adapter.get("recompute") is False:
            actions.append("Recompute is disabled for this adapter.")
    elif (adapter and adapter.get("route") == "standard") or native["level"] == "native":
        local_max = int((adapter or {}).get("local_max_bytes") or 1_610_612_736)
        source_is_gguf = "gguf" in (analysis.get("formats") or [])
        fits_disk = free_disk == 0 or (largest or 0) * 3 < free_disk
        fits_ram = available_ram == 0 or (largest or 0) < min(local_max, max(available_ram // 2, 1))
        if not native["in_stock_binary"]:
            route = "blocked"
            blockers.append("registered family has no confirmed architecture in the pinned stock runtime")
        elif (largest is None or largest <= local_max) and fits_disk and fits_ram and source_is_gguf:
            route = "oracle_local_requant"
            actions.append("Source is already GGUF and the architecture is native. Requantize with stock llama-quantize only. No patch.")
        elif not source_is_gguf and analysis.get("native_gguf_repo"):
            route = "native_gguf_requant"
            actions.append(
                "Safatensors source has an existing public GGUF repo "
                f"{analysis['native_gguf_repo'].get('repo_id')}. Requantize that file with stock llama-quantize. Do not convert or patch."
            )
        elif largest is not None and largest <= local_max and not source_is_gguf:
            route = "needs_converter"
            actions.append("Native architecture, but weights are not GGUF and no public GGUF repo was found. Stock convert_hf_to_gguf is not qualified. No patch.")
            blockers.append("stock convert_hf_to_gguf is not qualified and no public GGUF sibling was found")
        else:
            if source_is_gguf or analysis.get("native_gguf_repo"):
                route = "native_gguf_stream"
                actions.append("Use bounded HTTP Range streaming on Oracle or a free standard GitHub worker; never download the whole large source to Oracle.")
            else:
                route = "github_standard_unqualified"
                blockers.append("no qualified streaming converter or native GGUF source is registered")
    else:
        route = "blocked"
        blockers.append("no native GGUF support in the pinned stock llama.cpp build; custom patch not started")
        actions.append("Search upstream llama.cpp before any local change. Do not fork or patch the runtime automatically.")
        if analysis.get("custom_code"):
            blockers.append("repository ships custom code; trust_remote_code will not be executed")
    license_ok = (analysis.get("license_decision") or {}).get("redistributable") is True
    if not license_ok:
        blockers.append((analysis.get("license_decision") or {}).get("reason") or "license blocks publication")
    sibling_license_ok = True
    if route == "native_gguf_requant":
        sibling_license_ok = ((analysis.get("native_gguf_repo") or {}).get("license_decision") or {}).get("redistributable") is True
        if not sibling_license_ok:
            blockers.append("GGUF source repository license has not been cleared for redistribution")
    return {
        "route": route,
        "adapter_id": (adapter or {}).get("id"),
        "workflow": (adapter or {}).get("workflow"),
        "quants": quants,
        "estimated_bytes": estimates,
        "largest_estimated_bytes": largest,
        "license_allows_publish": license_ok,
        "blockers": [item for item in blockers if item],
        "actions": [item for item in actions if item],
        "native": native,
        "heretic": heretic,
        "variants": heretic["variants"],
        "allow_new_patch": False,
        "publish_allowed": license_ok and sibling_license_ok and route in {"oracle_local_requant", "native_gguf_requant"} and native["publish_as_native"] and not blockers,
        "estimates_are_measurements": False,
        "oracle_budget": {
            "free_disk_bytes": free_disk or None,
            "available_ram_bytes": available_ram or None,
            "dns_priority": True,
        },
    }
