"""Read-only Hugging Face inspection. Never executes trust_remote_code."""

from __future__ import annotations

import json
import re
import struct
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

USER_AGENT = "hf-quant-agent/0.1"
MAX_CONFIG_BYTES = 2_000_000
MAX_INDEX_BYTES = 8_000_000
MAX_SAFETENSORS_HEADER = 8_000_000
REDISTRIBUTABLE = {
    "apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "isc", "unlicense", "zlib",
}
RESTRICTED_HINTS = ("non-commercial", "research only", "no redistribution", "cc-by-nc", "llama2", "llama3")


class InspectError(RuntimeError):
    pass


def find_own_bf16_repo(repo_id: str, revision: str | None, timeout: float = 20) -> dict | None:
    """Our own earlier BF16 conversion of exactly this revision (the converter writes the revision into the card), reused instead of converting again."""
    if not revision:
        return None
    from hfqa.stream_controller import output_stem
    stem = output_stem({'repo_id': repo_id})
    legacy = stem[len('Sakura-'):] if stem.startswith('Sakura-') else stem  # conversions made before the brand name led the repository names
    info = candidate = None
    for name in dict.fromkeys([stem, legacy]):
        candidate = f"webmp3/{name}-BF16-GGUF"
        try:
            info = _json(f"https://huggingface.co/api/models/{urllib.parse.quote(candidate)}", timeout)
            break
        except InspectError:
            continue
    if info is None:
        return None
    card = info.get("cardData") or {}
    files = [item.get("rfilename") for item in info.get("siblings") or [] if str(item.get("rfilename")).endswith(".gguf")]
    if not files or card.get("hfqa_source_revision") != revision or card.get("base_model") != repo_id or info.get("private"):
        return None
    return {"repo_id": info.get("id") or candidate, "revision": info.get("sha"), "architecture": (info.get("gguf") or {}).get("architecture"), "gguf_files": files[:40],
            "license": card.get("license"), "license_decision": classify_license(card.get("license"), _absolute_license_link(card, info.get("id") or candidate, info.get("sha"))),
            "weights_downloaded": False, "source_precision": "bf16", "converted_by_agent": True, "converter_revision": card.get("hfqa_converter_revision")}


def find_native_gguf_repo(repo_id: str, timeout: float = 20, revision: str | None = None) -> dict | None:
    """Look for an existing public GGUF repo. Does not download weights or write a patch."""
    owner, _, name = repo_id.partition("/")
    candidates = []
    if name:
        candidates.extend([f"{owner}/{name}-GGUF", f"{owner}/{name}-gguf"])
    else:
        candidates.append(f"{repo_id}-GGUF")
    for candidate in candidates:
        try:
            info = _json(f"https://huggingface.co/api/models/{urllib.parse.quote(candidate)}", timeout)
        except InspectError:
            continue
        files = [item.get("rfilename") for item in info.get("siblings") or [] if item.get("rfilename")]
        gguf_files = [item for item in files if str(item).endswith(".gguf")]
        if not gguf_files and "gguf" not in (info.get("tags") or []):
            continue
        return {
            "repo_id": info.get("id") or candidate,
            "revision": info.get("sha"),
            "architecture": (info.get("gguf") or {}).get("architecture"),
            "gguf_files": gguf_files[:40],
            "license": (info.get("cardData") or {}).get("license") or _tag_license(info.get("tags") or []),
            "license_decision": classify_license((info.get("cardData") or {}).get("license") or _tag_license(info.get("tags") or []),
                                                 _absolute_license_link(info.get("cardData") or {}, info.get("id") or candidate, info.get("sha"))),
            "weights_downloaded": False,
        }
    try:
        return find_third_party_gguf_repo(repo_id, timeout) or find_own_bf16_repo(repo_id, revision, timeout)
    except InspectError:
        return find_own_bf16_repo(repo_id, revision, timeout)


# Only established quantizers are trusted to have converted the ORIGINAL weights unchanged. A repository of anyone else,
# or one whose name signals changed weights, is never picked automatically: it could pass off a modified model as the original.
TRUSTED_QUANTIZERS = frozenset(x.lower() for x in (
    "bartowski", "unsloth", "lmstudio-community", "mradermacher", "QuantFactory", "ggml-org", "second-state", "TheBloke", "MaziyarPanahi"))
MODIFIED_WEIGHTS_HINT = re.compile(r"abliterat|uncensor|heretic|merge|merged|finetune|fine-tune|lora|dpo|sft|rp[-_]|roleplay|distill", re.I)


def find_third_party_gguf_repo(repo_id: str, timeout: float = 20, limit: int = 8) -> dict | None:
    """A GGUF repository of ANOTHER author that Hugging Face lists as a quantization of this model.

    Used only as the INPUT of our own quantization (never copied). The best candidate has a complete, hash-pinned
    BF16 set, else F16, else Q8_0; ties go to the more downloaded repository. Metadata only, no weights.
    """
    from hfqa.stream_sources import select_stream_sources
    query = ("https://huggingface.co/api/models?filter=gguf&filter=" + urllib.parse.quote("base_model:quantized:" + repo_id, safe="")
             + f"&sort=downloads&direction=-1&limit={limit}")
    ranks = {"bf16": 0, "f16": 1, "q8_0": 2}
    best = None
    for position, item in enumerate(_json(query, timeout) or []):
        candidate = item.get("id") or item.get("modelId")
        if not candidate or candidate == repo_id:
            continue
        owner = candidate.split("/", 1)[0].lower()
        if owner not in TRUSTED_QUANTIZERS or MODIFIED_WEIGHTS_HINT.search(candidate.split("/", 1)[-1]):
            continue
        try:
            info = _json(f"https://huggingface.co/api/models/{urllib.parse.quote(candidate)}", timeout)
            revision = info.get("sha")
            selected = select_stream_sources({"repo_id": candidate, "revision": revision}, timeout)
        except (InspectError, ValueError):
            continue
        card = info.get("cardData") or {}
        if str(card.get("base_model_relation") or "quantized").lower() != "quantized" or MODIFIED_WEIGHTS_HINT.search(" ".join(info.get("tags") or [])):
            continue
        licence = card.get("license") or _tag_license(info.get("tags") or [])
        rank = ranks.get(selected.get("source_precision"), 3)
        if best is None or rank < best[0]:
            best = (rank, position, {
                "repo_id": info.get("id") or candidate, "revision": revision, "architecture": (info.get("gguf") or {}).get("architecture"),
                "gguf_files": [s["filename"] for s in selected["sources"]][:40], "license": licence,
                "license_decision": classify_license(licence, _absolute_license_link(card, candidate, revision)), "weights_downloaded": False,
                "third_party": True, "source_set": selected["source_set"], "source_precision": selected["source_precision"],
                "source_bytes": selected["total_source_bytes"], "downloads": item.get("downloads"),
            })
        if rank == 0:
            break
    return best[2] if best else None


def list_gguf_candidates(repo_id: str, timeout: float = 20, limit: int = 8) -> list[dict]:
    """GGUF repositories Hugging Face lists as quantizations of this model; shown to the user, never used automatically."""
    query = ("https://huggingface.co/api/models?filter=gguf&filter=" + urllib.parse.quote("base_model:quantized:" + repo_id, safe="")
             + f"&sort=downloads&direction=-1&limit={limit}")
    try:
        items = _json(query, timeout) or []
    except InspectError:
        return []
    found = []
    for item in items:
        name = item.get("id") or item.get("modelId")
        if name and name != repo_id:
            found.append({"repo_id": name, "downloads": item.get("downloads"), "trusted": name.split("/", 1)[0].lower() in TRUSTED_QUANTIZERS})
    return found


def parse_hf_url(value: str) -> tuple[str, str | None]:
    text = (value or "").strip()
    if not text:
        raise InspectError("empty Hugging Face URL")
    text = text.split("?", 1)[0].rstrip("/")
    match = re.match(r"https?://huggingface\.co/([^/]+)(?:/([^/]+))?(?:/tree/([^/]+))?", text)
    if match:
        owner, name, rev = match.group(1), match.group(2), match.group(3)
        if owner in {"datasets", "spaces", "docs", "login"}:
            raise InspectError("only model repositories are accepted")
        if name:
            return f"{owner}/{name}", rev
        return owner, rev
    if re.match(r"^[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)?$", text):
        return text, None
    raise InspectError("not a Hugging Face model URL or repo id")


def _get(url: str, timeout: float = 30, headers: dict | None = None, max_bytes: int = MAX_INDEX_BYTES) -> tuple[int, dict, bytes]:
    req_headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    request = urllib.request.Request(url, headers=req_headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            resp_headers = dict(response.headers)
            body = response.read(max_bytes + 1)
    except urllib.error.HTTPError as exc:
        detail = exc.read(500).decode("utf-8", "replace")
        raise InspectError(f"HTTP {exc.code} for {url}: {detail[:300]}") from exc
    except urllib.error.URLError as exc:
        raise InspectError(f"network error for {url}: {exc.reason}") from exc
    if len(body) > max_bytes:
        raise InspectError(f"response exceeds {max_bytes} bytes: {url}")
    return status, resp_headers, body


def _json(url: str, timeout: float = 30) -> Any:
    _, _, body = _get(url, timeout=timeout)
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise InspectError(f"invalid JSON from {url}") from exc


def inspect_repo(source: str, timeout: float = 30) -> dict:
    repo_id, requested_rev = parse_hf_url(source)
    info = _json(f"https://huggingface.co/api/models/{urllib.parse.quote(repo_id)}", timeout)
    sha = info.get("sha")
    if not sha:
        raise InspectError("model API did not return a commit SHA")
    revision = requested_rev or sha
    if requested_rev and requested_rev != sha:
        # Pin the requested ref by resolving it. A branch name is not a frozen revision.
        resolved = _json(
            f"https://huggingface.co/api/models/{urllib.parse.quote(repo_id)}/revision/{urllib.parse.quote(requested_rev)}",
            timeout,
        )
        revision = resolved.get("sha")
        if not revision:
            raise InspectError("requested revision could not be pinned")
        info = resolved
    card = info.get("cardData") or {}
    license_id = card.get("license") or _tag_license(info.get("tags") or [])
    config = _read_config(repo_id, revision, timeout)
    siblings = info.get("siblings") or []
    files = [item.get("rfilename") for item in siblings if item.get("rfilename")]
    model_index = None
    if "model_index.json" in files:
        try:
            _, _, raw = _get(f"https://huggingface.co/{urllib.parse.quote(repo_id)}/resolve/{revision}/model_index.json", timeout=timeout, max_bytes=MAX_CONFIG_BYTES)
            candidate = json.loads(raw.decode("utf-8"))
            if isinstance(candidate, dict):
                model_index = candidate
        except (InspectError, ValueError):
            pass
    weight_files = [name for name in files if name.endswith((".safetensors", ".gguf", ".bin"))]
    from hfqa.image_components import component_files, extract_component_references
    readme = ''
    image_hint = (info.get('pipeline_tag') in {'text-to-image','image-to-image','image-text-to-image','text-to-video','image-to-video'}
                  or info.get('library_name') == 'diffusers' or 'image' in repo_id.lower())
    if image_hint and 'README.md' in files:
        try:
            _, _, raw = _get(f'https://huggingface.co/{urllib.parse.quote(repo_id)}/resolve/{revision}/README.md', timeout=timeout, max_bytes=262144)
            readme = raw.decode('utf-8', 'replace')
        except InspectError:
            pass
    index = _read_index(repo_id, revision, files, timeout)
    tensor_sample = None
    if not index and any(name.endswith(".safetensors") and "index" not in name for name in files):
        first = next(name for name in files if name.endswith(".safetensors"))
        tensor_sample = read_safetensors_header(repo_id, revision, first, timeout)
    safetensors = info.get("safetensors") or {}
    parameters = safetensors.get("total")
    gguf = info.get("gguf") or {}
    card = _absolute_license_link(card, repo_id, revision)
    license_decision = classify_license(license_id, card)
    from hfqa.provenance import identify
    return {
        "repo_id": repo_id,
        "model_provenance": identify(info,repo_id,revision),
        "requested_revision": requested_rev,
        "revision": revision,
        "private": bool(info.get("private")),
        "gated": bool(info.get("gated")),
        "pipeline_tag": info.get("pipeline_tag"),
        "library_name": info.get("library_name"),
        "model_index": model_index,
        "component_files": component_files(files),
        "component_references": extract_component_references(readme, card),
        "config": {key: config[key] for key in ("_class_name", "model_type", "architectures") if key in (config or {})},
        "model_type": (config or {}).get("model_type") or gguf.get("architecture"),
        "architectures": (config or {}).get("architectures") or [],
        "num_hidden_layers": (config or {}).get("num_hidden_layers"),
        "num_experts": (config or {}).get("num_experts") or (config or {}).get("num_local_experts"),
        "num_experts_per_tok": (config or {}).get("num_experts_per_tok"),
        "hidden_size": (config or {}).get("hidden_size"),
        "vocab_size": (config or {}).get("vocab_size"),
        "max_position_embeddings": (config or {}).get("max_position_embeddings") or gguf.get("context_length"),
        "torch_dtype": (config or {}).get("torch_dtype"),
        "parameters": parameters,
        "used_storage": info.get("usedStorage"),
        "license": license_id,
        "license_name": card.get("license_name"),
        "license_link": card.get("license_link"),
        "license_decision": license_decision,
        "custom_code": "custom_code" in (info.get("tags") or []) or bool((config or {}).get("auto_map")),
        "trust_remote_code_executed": False,
        "files": files[:200],
        "file_count": len(files),
        "weight_files": weight_files[:80],
        "formats": detect_formats(files, config, gguf),
        "gguf_architecture": gguf.get("architecture"),
        "tensor_index": index,
        "safetensors_header_sample": tensor_sample,
        "has_chat_template": bool((config or {}).get("chat_template") or gguf.get("chat_template")),
        "source_url": f"https://huggingface.co/{repo_id}",
        "card_excerpt_untrusted": None,
    }


def _tag_license(tags: list[str]) -> str | None:
    for tag in tags:
        if tag.startswith("license:"):
            return tag.split(":", 1)[1]
    return None


def _read_config(repo_id: str, revision: str, timeout: float) -> dict | None:
    url = (
        f"https://huggingface.co/{urllib.parse.quote(repo_id)}/resolve/"
        f"{urllib.parse.quote(revision)}/config.json"
    )
    try:
        _, _, body = _get(url, timeout=timeout, headers={"Accept": "application/json"}, max_bytes=MAX_CONFIG_BYTES)
    except InspectError:
        return None
    try:
        data = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _read_index(repo_id: str, revision: str, files: list[str], timeout: float) -> dict | None:
    name = next((item for item in files if item.endswith(".safetensors.index.json")), None)
    if not name:
        return None
    url = (
        f"https://huggingface.co/{urllib.parse.quote(repo_id)}/resolve/"
        f"{urllib.parse.quote(revision)}/{urllib.parse.quote(name)}"
    )
    try:
        _, _, body = _get(url, timeout=timeout, max_bytes=MAX_INDEX_BYTES)
        data = json.loads(body.decode("utf-8"))
    except (InspectError, json.JSONDecodeError):
        return {"error": "index unreadable", "path": name}
    weight_map = data.get("weight_map") or {}
    dtypes: dict[str, int] = {}
    shapes_sample = []
    metadata = data.get("metadata") or {}
    # Index files usually do not contain shapes. Report coverage only.
    return {
        "path": name,
        "tensor_count": len(weight_map),
        "shard_count": len(set(weight_map.values())),
        "total_size": metadata.get("total_size"),
        "dtype_counts": dtypes,
        "shape_sample": shapes_sample,
        "names_sample": list(weight_map)[:12],
    }


def read_safetensors_header(repo_id: str, revision: str, path: str, timeout: float = 30) -> dict:
    """Read only the safetensors JSON header via HTTP ranges. No weight download."""
    url = (
        f"https://huggingface.co/{urllib.parse.quote(repo_id)}/resolve/"
        f"{urllib.parse.quote(revision)}/{urllib.parse.quote(path)}"
    )
    status, headers, prefix = _get(
        url, timeout=timeout, headers={"Range": "bytes=0-7", "Accept": "application/octet-stream"}, max_bytes=16
    )
    if status not in (200, 206) or len(prefix) < 8:
        return {"path": path, "error": f"header prefix unavailable (status {status})"}
    header_len = struct.unpack("<Q", prefix[:8])[0]
    if header_len <= 0 or header_len > MAX_SAFETENSORS_HEADER:
        return {"path": path, "error": f"refusing header length {header_len}"}
    end = 8 + header_len - 1
    _, _, blob = _get(
        url,
        timeout=timeout,
        headers={"Range": f"bytes=8-{end}", "Accept": "application/octet-stream"},
        max_bytes=header_len + 8,
    )
    try:
        header = json.loads(blob[:header_len].decode("utf-8"))
    except json.JSONDecodeError:
        return {"path": path, "error": "safetensors header is not JSON"}
    tensors = {k: v for k, v in header.items() if k != "__metadata__" and isinstance(v, dict)}
    dtype_counts: dict[str, int] = {}
    for spec in tensors.values():
        dtype = str(spec.get("dtype") or "unknown")
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    sample = []
    for name, spec in list(tensors.items())[:8]:
        sample.append({"name": name, "dtype": spec.get("dtype"), "shape": spec.get("shape")})
    return {
        "path": path,
        "tensor_count": len(tensors),
        "dtype_counts": dtype_counts,
        "shape_sample": sample,
        "metadata_keys": list((header.get("__metadata__") or {}).keys())[:20],
        "weights_downloaded": False,
    }


def detect_formats(files: list[str], config: dict | None, gguf: dict) -> list[str]:
    found = []
    joined = " ".join(files).lower()
    if any(name.endswith(".gguf") for name in files) or gguf:
        found.append("gguf")
    if any(name.endswith(".safetensors") for name in files):
        found.append("safetensors")
    if "mxfp8" in joined or (config or {}).get("quantization_config", {}).get("quant_method") == "mxfp8":
        found.append("mxfp8")
    if "nvfp4" in joined:
        found.append("nvfp4")
    if "fp8" in joined:
        found.append("fp8")
    dtype = str((config or {}).get("torch_dtype") or "").lower()
    if dtype:
        found.append(dtype)
    return found


def _absolute_license_link(card: dict, repo_id: str, revision: str | None) -> dict:
    """`license_link: LICENSE` names a file of the repository itself; make it an address that can be read."""
    link = str((card or {}).get("license_link") or "").strip()
    if not link or link.startswith(("http://", "https://")) or not repo_id or ".." in link or link.startswith("/"):
        return card
    return {**card, "license_link": f"https://huggingface.co/{repo_id}/resolve/{revision or 'main'}/{urllib.parse.quote(link)}"}


def classify_license(license_id: str | None, card: dict) -> dict:
    normalized = (license_id or "").strip().lower()
    blob = json.dumps(card, ensure_ascii=False).lower()
    # A vendor license published as `other` can be approved by the operator once, by its license name.
    if normalized == "other" and (card or {}).get("license_name"):
        from hfqa.license_approvals import is_approved
        from hfqa import license_review
        approved = is_approved(card.get("license_name"))
        automatic = None
        if not approved:
            # The operator wants this to run by itself: a strict text check of the vendor license, logged with its evidence.
            approved = automatic = license_review.auto_decide(card.get("license_name"), card.get("license_link"))
        if approved:
            how = "approved automatically after a license text check" if (automatic or approved.get("automatic")) else "approved by the operator"
            return {"redistributable": True, "reason": f"{how}: {approved['license_name']} ({approved['license_link']})",
                    "operator_approved": not (automatic or approved.get("automatic")), "automatic": bool(automatic or approved.get("automatic")),
                    "license_name": approved["license_name"], "license_link": approved["license_link"]}
        review = license_review.review_for(card.get("license_name"))
        if review:
            restricted = any(hint in blob or hint in normalized for hint in RESTRICTED_HINTS)
            return {"redistributable": False, "reason": f"license {card.get('license_name')}: automatic check says {review['verdict']} ({'; '.join(review.get('reasons') or [])})",
                    "automatic_review": review, "restricted_hint": restricted}
    restricted = any(hint in blob or hint in normalized for hint in RESTRICTED_HINTS)
    if not normalized:
        return {"redistributable": False, "reason": "license not declared; publication blocked"}
    if restricted or normalized not in REDISTRIBUTABLE:
        return {
            "redistributable": False,
            "reason": f"license {normalized or 'undeclared'} is not in the automatic redistribution allow-list",
        }
    return {"redistributable": True, "reason": f"{normalized} allows redistribution with attribution"}
