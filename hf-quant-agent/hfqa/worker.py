"""Confined stock requant sources and resumable, size-bounded public downloads."""

from __future__ import annotations

from pathlib import Path


MAX_AUTO_DOWNLOAD_BYTES = 800 * 1024 * 1024


def head_content_length(url: str, timeout: float = 30) -> int | None:
    import urllib.request
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "hf-quant-agent"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            length = response.headers.get("content-length")
            return int(length) if length else None
    except Exception:
        return None


def download_decision(content_length: int | None, cap: int = MAX_AUTO_DOWNLOAD_BYTES) -> dict:
    """Decide whether an automatic GGUF download is small enough. Never starts the download."""
    if content_length is None or content_length <= 0:
        return {"allowed": False, "reason": "content length is unknown; download not started"}
    if content_length > cap:
        return {
            "allowed": False,
            "reason": f"file is {content_length} bytes, above the {cap} byte automatic download cap",
            "bytes": content_length,
        }
    return {"allowed": True, "reason": "file is within the automatic download cap", "bytes": content_length}


ALLOWED_ROOTS = (
    Path("/home/ubuntu/hfqa/work"),
    Path("/var/lib/hf-quant-agent"),
)


def find_local_gguf(directory: Path, names: list[str]) -> Path | None:
    """Return an already present GGUF. Never downloads."""
    preferred = [name for name in names if "q8_0" in name.lower()] + list(names)
    seen = set()
    for name in preferred:
        if name in seen:
            continue
        seen.add(name)
        candidate = directory / name
        try:
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
        except PermissionError:
            continue
    return None


def allow_source(path: str) -> Path:
    resolved = Path(path).resolve()
    if not any(resolved == root or root in resolved.parents for root in ALLOWED_ROOTS):
        raise ValueError("source path is outside the worker workspace")
    return resolved


def worker_decision(job: dict, source: Path | None = None, assume_present: bool = False) -> dict:
    state = job.get("state")
    route = ((job.get("plan") or {}).get("route"))
    if state == "COMPLETED":
        return {"allowed": False, "reason": "job already completed; existing output is not recomputed"}
    if state != "WAITING_FOR_WORKER":
        return {"allowed": False, "reason": f"worker start is only allowed from WAITING_FOR_WORKER, not {state}"}
    if route not in {"native_gguf_requant", "oracle_local_requant"}:
        return {"allowed": False, "reason": f"route {route} is not a stock requant"}
    if (job.get("plan") or {}).get("allow_new_patch"):
        return {"allowed": False, "reason": "custom patch is not allowed"}
    if not assume_present and (source is None or not source.is_file() or source.stat().st_size <= 0):
        return {
            "allowed": False,
            "reason": "source GGUF is not already on this host; refusing to download weights automatically",
        }
    return {
        "allowed": True,
        "reason": "stock llama-quantize may run on the local file",
        "source": str(source) if source else None,
        "bytes": None if assume_present or source is None else source.stat().st_size,
    }

# Small public GGUF downloads are performed by the worker, never the API process.
def source_descriptor(analysis: dict, timeout: float = 30) -> dict:
    import re
    import urllib.parse
    from hfqa.inspect import _json
    sibling = analysis.get("native_gguf_repo") or {
        "repo_id": analysis.get("repo_id"), "revision": analysis.get("revision"),
        "gguf_files": analysis.get("weight_files") or [],
    }
    repo, revision = sibling.get("repo_id"), sibling.get("revision")
    if not repo or not re.fullmatch(r"[a-fA-F0-9]{40}", revision or ""):
        raise ValueError("GGUF source must have a pinned full commit SHA")
    info = _json(f"https://huggingface.co/api/models/{urllib.parse.quote(repo, safe='/')}/revision/{revision}?blobs=true", timeout)
    files = info.get("siblings") or []
    candidates = [item for item in files if str(item.get("rfilename") or "").lower().endswith(".gguf")
                  and re.search(r"(?:^|[-_.])(q8_0|f16|bf16)(?:[-_.]|$)", str(item.get("rfilename") or "").lower())
                  and not re.search(r"-\d{5}-of-\d{5}", str(item.get("rfilename") or ""))]
    candidates.sort(key=lambda item: ("q8_0" not in item["rfilename"].lower(), item["rfilename"]))
    for item in candidates:
        lfs = item.get("lfs") or {}
        size = item.get("size") or lfs.get("size")
        digest = lfs.get("sha256") or lfs.get("oid")
        if digest and digest.startswith("sha256:"):
            digest = digest[7:]
        if not download_decision(size)["allowed"] or not re.fullmatch(r"[a-fA-F0-9]{64}", digest or ""):
            continue
        return {"repo_id": repo, "revision": revision, "filename": item["rfilename"],
                "size": int(size), "sha256": digest.lower(),
                "architecture": sibling.get("architecture") or analysis.get("model_type"),
                "license_decision": sibling.get("license_decision") or analysis.get("license_decision")}
    raise ValueError("no single Q8_0/F16/BF16 GGUF with verified SHA-256 within 800 MiB; a larger worker is required")


def source_destination(descriptor: dict) -> Path:
    import os
    import re
    from pathlib import PurePosixPath
    repo, revision, name = descriptor["repo_id"], descriptor["revision"], descriptor["filename"]
    if not re.fullmatch(r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)?", repo) or not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
        raise ValueError("invalid source identity")
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or "\\" in name or ":" in name or not name.lower().endswith(".gguf"):
        raise ValueError("invalid GGUF filename")
    root = Path(os.environ.get("HFQA_SOURCE_DIR", "/var/lib/hf-quant-agent/sources")).resolve()
    target = (root / repo.replace("/", "--") / revision / Path(*rel.parts)).resolve()
    if root not in target.parents:
        raise ValueError("GGUF path escapes source cache")
    return target


def prepare_source(ticket: dict, job: dict, should_stop=None) -> Path:
    import json
    import os
    import shutil
    import time
    import urllib.request
    import urllib.parse
    from hfqa.publish import sha256_file
    desc = ticket.get("download")
    if not desc:
        local = allow_source(ticket["source"])
        size = local.stat().st_size
        if not download_decision(size)["allowed"]:
            raise ValueError("local source exceeds small worker cap")
        if job.get("publish"):
            raise ValueError("publication requires a pinned verified source descriptor")
        if shutil.disk_usage(local.parent).free < size * (len(job.get("quants") or []) + 1) + 2 * 1024**3:
            raise RuntimeError("insufficient free disk for local outputs and DNS reserve")
        return local
    size, digest = int(desc["size"]), desc["sha256"]
    if not download_decision(size)["allowed"]:
        raise ValueError("download exceeds automatic limit")
    expected = source_descriptor(job.get("analysis") or {})
    if any(desc.get(k) != expected.get(k) for k in ("repo_id", "revision", "filename", "size", "sha256")):
        raise ValueError("download descriptor does not match pinned Hub source")
    destination = source_destination(desc)
    destination.parent.mkdir(parents=True, exist_ok=True)
    required = size * (len(job.get("quants") or []) + 2) + 2 * 1024**3
    if shutil.disk_usage(destination.parent).free < required:
        raise RuntimeError("insufficient free disk for source, outputs and 2 GiB DNS reserve")
    if destination.is_file() and destination.stat().st_size == size and sha256_file(destination) == digest:
        return destination
    # Adopt a previously downloaded flat cache only after checking the pinned hash.
    legacy = Path(os.environ.get("HFQA_SOURCE_DIR", "/var/lib/hf-quant-agent/sources")) / Path(desc["filename"]).name
    if legacy.is_file() and legacy.stat().st_size == size and sha256_file(legacy) == digest:
        temporary = destination.with_suffix(".adopting")
        try:
            os.link(legacy, temporary)
        except OSError:
            shutil.copyfile(legacy, temporary)
        os.replace(temporary, destination)
        destination.with_suffix(".source.json").write_text(json.dumps(desc, indent=2), encoding="utf-8")
        return destination
    deadline = time.monotonic() + 900
    partial = destination.with_suffix(destination.suffix + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        raise ValueError("partial download exceeds expected size")
    if offset < size:
        headers = {"User-Agent": "hf-quant-agent", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        url = f"https://huggingface.co/{urllib.parse.quote(desc['repo_id'], safe='/')}/resolve/{desc['revision']}/{urllib.parse.quote(desc['filename'], safe='/')}"
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
            if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                offset = 0
                if response.status == 206:
                    raise ValueError("invalid partial response")
            with partial.open("ab" if offset else "wb") as handle:
                count = offset
                while True:
                    if time.monotonic() > deadline:
                        raise TimeoutError("download deadline reached; partial retained for retry")
                    if should_stop and should_stop():
                        raise InterruptedError("job paused or cancelled during download")
                    chunk = response.read(min(1024 * 1024, size - count + 1))
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > size:
                        raise ValueError("server exceeded pinned download size")
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
    if partial.stat().st_size != size:
        raise IOError("download incomplete; partial retained for retry")
    if sha256_file(partial) != digest:
        # A corrupt completed partial must not be reused on retry.
        partial.unlink(missing_ok=True)
        raise ValueError("source size or SHA-256 verification failed")
    os.replace(partial, destination)
    provenance = destination.with_suffix(".source.json")
    provenance.write_text(json.dumps(desc, indent=2), encoding="utf-8")
    return destination
