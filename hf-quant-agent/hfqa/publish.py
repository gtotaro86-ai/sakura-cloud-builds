"""Model-card generation and gated Hugging Face upload."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_readme(analysis: dict, plan: dict, validation: dict, files: list[dict]) -> str:
    repo = analysis.get("repo_id") or "unknown"
    revision = analysis.get("revision") or "unpinned"
    quants = ", ".join(plan.get("quants") or []) or "none"
    license_id = analysis.get("license") or "undeclared"
    arch = analysis.get("model_type") or analysis.get("gguf_architecture") or "untested"
    file_rows = "\n".join(
        f"| `{item['name']}` | {item.get('bytes', 'n/a')} | `{item.get('sha256', 'n/a')}` | {item.get('quant', '')} |"
        for item in files
    ) or "| none | | | |"
    structure = validation.get("structure_valid")
    runtime = validation.get("runtime_compatible")
    inference = validation.get("inference_tested")
    gguf_source = validation.get("source_descriptor") or {}
    source_structure = validation.get("source_structure") or {}
    def histogram(report):
        return ", ".join(f"{name}: {count}" for name, count in sorted((report.get("tensor_type_histogram") or {}).items())) or "unmeasured"
    tensor_rows = "\n".join(
        f"| `{item['name']}` | {histogram(item.get('quantization') or {})} | "
        f"{(item.get('quantization') or {}).get('fallback_tensor_count') if (item.get('quantization') or {}).get('fallback_tensor_count') is not None else 'unmeasured; inspect quantizer log'} | "
        f"`{(item.get('quantization') or {}).get('tensor_name_set_sha256') or 'unmeasured'}` |"
        for item in files
    ) or "| none | unmeasured | unmeasured | unmeasured |"
    runtime_rows = "\n".join(
        f"| `{item['name']}` | `{(item.get('inference') or {}).get('runtime_build') or 'unreported'}` | "
        f"`{(item.get('inference') or {}).get('runtime_sha256') or 'unmeasured'}` |"
        for item in files
    ) or "| none | unreported | unmeasured |"
    return f"""---
license: {license_id if license_id != "undeclared" else "other"}
base_model: {repo}
hfqa_source_revision: {revision}
tags:
- gguf
- quantization
---

# GGUF quantization of {repo}

Independent community quantization. Not an official release of the upstream author.

## 1. Model Overview

This repository is produced by the local HF Quant Agent control plane. It records only checks that actually ran. Unmeasured quality, speed and compatibility are marked untested.

## 2. Original Model / Attribution

- Source: [{repo}](https://huggingface.co/{repo})
- Publisher of this quantization: `webmp3`, when a release is actually uploaded
- Upstream authors retain their license and attribution
- GGUF source publisher: `{gguf_source.get("repo_id") or "local source, upstream provenance unverified"}`
- GGUF source commit: `{gguf_source.get("revision") or "unverified"}`
- GGUF source filename: `{gguf_source.get("filename") or "unverified"}`
- GGUF source SHA-256: `{gguf_source.get("sha256") or validation.get("source_sha256") or "unverified"}`
- GGUF source provenance verified: `{validation.get("source_verified") is True}`

## 3. Source Revision

Pinned revision: `{revision}`

Jobs are bound to this SHA. A later upstream commit does not silently change a running or completed job.

## 4. Supported GGUF Quantizations

Requested: {quants}

| File | Bytes | SHA-256 | Quant |
|---|---:|---|---|
{file_rows}

## 5. Quantization Methodology

Route: `{plan.get("route")}`

Adapter: `{plan.get("adapter_id") or "none"}`

This route performs **existing-GGUF requantization** from the pinned Q8_0/F16/BF16 source GGUF identified above. This job did not convert the upstream BF16/safetensors checkpoint. Source tensor storage measured in the GGUF: {histogram(source_structure)}.

**No importance-matrix calibration was run by this job.** Requantizing a previously quantized source can add quantization error. Estimated sizes are planning ratios, not measurements, unless a file byte count is listed above.

The target quant label describes the selected stock quantizer preset. Stock llama.cpp may retain floating-point tensors or use fallback tensor types; every tensor is not necessarily stored in the named target type. The table records measured output tensor types and only reports a fallback count when the quantizer printed one.

Source tensor count: `{source_structure.get("tensor_count", "unmeasured")}`. Source full tensor-name set SHA-256: `{source_structure.get("tensor_name_set_sha256") or "unmeasured"}`. New and reused outputs must retain the source tensor count and complete name set.

| File | Measured tensor-type counts | Quantizer-reported fallback tensors | Full tensor-name set SHA-256 |
|---|---|---:|---|
{tensor_rows}

## 6. Architecture Information

- model_type / architecture: `{arch}`
- parameters: `{analysis.get("parameters") or "not reported by the Hub metadata"}`
- experts: `{analysis.get("num_experts") or "n/a or not declared"}`
- source formats observed: {", ".join(analysis.get("formats") or []) or "unknown"}
- custom code present: `{analysis.get("custom_code")}`
- `trust_remote_code` executed: `{analysis.get("trust_remote_code_executed")}`

## 7. Conversion and Runtime Compatibility

- GGUF structure valid: `{structure}`
- Runtime load tested: `{runtime}`
- Inference tested: `{inference}`

A structurally valid GGUF is not automatically a working model.

| File | Runtime build reported by executable | Runtime executable SHA-256 |
|---|---|---|
{runtime_rows}

## 8. Download and Usage Instructions

Use a llama.cpp build that supports this architecture. Example, after you confirm compatibility:

```bash
llama-cli -m <file>.gguf -p "Hello" -n 32
```

This card does not claim that every runtime can load the file.

## 9. Validation Results

{validation.get("summary") or "No validation summary was recorded."}

## 10. Limitations

- No coding benchmark, perplexity or quality score is claimed unless a measured value is copied from this job's own log.
- Shard resume and upload verification apply only to artifacts listed in this card.
- Oracle DNS services take priority over local quantization. Large jobs are not run on the 2-core Oracle VM.

## 11. License Information

Source license: `{license_id}`

Decision: {(analysis.get("license_decision") or {}).get("reason") or "not evaluated"}

If the license does not allow redistribution, this card must not be uploaded.

## 12. Reproduction Information

- Control plane: HF Quant Agent
- Planner route: `{plan.get("route")}`
- Workflow: `{plan.get("workflow") or "none"}`
- Blockers: {"; ".join(plan.get("blockers") or []) or "none recorded"}
"""


def fetch_release_status(repo_id: str, expected: dict[str, int], timeout: float = 30) -> dict:
    """Read a public Hub repo and compare file sizes. Does not upload."""
    import json
    import urllib.request

    request = urllib.request.Request(
        f"https://huggingface.co/api/models/{repo_id}",
        headers={"User-Agent": "hf-quant-agent"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        info = json.loads(response.read().decode())
    names = {item.get("rfilename") for item in info.get("siblings") or []}
    remote: dict[str, int | None] = {}
    for name in expected:
        if name not in names:
            continue
        head = urllib.request.Request(
            f"https://huggingface.co/{repo_id}/resolve/main/{name}",
            method="HEAD",
            headers={"User-Agent": "hf-quant-agent"},
        )
        with urllib.request.urlopen(head, timeout=timeout) as response:
            length = response.headers.get("content-length")
            remote[name] = int(length) if length else None
    gaps = release_gaps(remote, expected)
    return {
        "repo": repo_id,
        "url": f"https://huggingface.co/{repo_id}",
        "sha": info.get("sha"),
        "architecture": (info.get("gguf") or {}).get("architecture"),
        "complete": gaps["complete"],
        "gaps": gaps,
    }


def release_gaps(remote_sizes: dict[str, int | None], expected: dict[str, int]) -> dict:
    """Compare a remote file listing with local validated artifacts. Do not recompute matches."""
    present = []
    missing = []
    mismatched = []
    for name, size in expected.items():
        remote = remote_sizes.get(name)
        if remote is None:
            missing.append(name)
        elif int(remote) != int(size):
            mismatched.append({"name": name, "remote": int(remote), "local": int(size)})
        else:
            present.append(name)
    return {
        "complete": not missing and not mismatched and bool(expected),
        "present": present,
        "missing": missing,
        "mismatched": mismatched,
    }


def publication_gate(requested: bool, analysis: dict, plan: dict, validation: dict, files: list[Path]) -> None:
    """Fail closed before importing the client or performing any external write."""
    if requested is not True:
        raise RuntimeError("publication was not requested by the job")
    if (analysis.get("license_decision") or {}).get("redistributable") is not True:
        raise RuntimeError("source license does not permit publication")
    if plan.get("publish_allowed") is not True or plan.get("allow_new_patch"):
        raise RuntimeError("planner has not qualified publication")
    if not analysis.get("revision") or analysis.get("trust_remote_code_executed"):
        raise RuntimeError("publication requires a pinned source and no remote-code execution")
    source = validation.get("source_descriptor") or {}
    expected_source = analysis.get("native_gguf_repo") or analysis
    if (validation.get("source_verified") is not True
            or source.get("repo_id") != expected_source.get("repo_id")
            or source.get("revision") != expected_source.get("revision")
            or not re.fullmatch(r"[a-fA-F0-9]{40}", source.get("revision") or "")
            or not re.fullmatch(r"[a-fA-F0-9]{64}", source.get("sha256") or "")
            or source.get("sha256") != validation.get("source_sha256")):
        raise RuntimeError("publication requires verified pinned GGUF source provenance")
    if not all(validation.get(key) is True for key in ("structure_valid", "runtime_compatible", "inference_tested")):
        raise RuntimeError("required structural and runtime validations are incomplete")
    artifacts = validation.get("artifacts") or []
    if not files or len(files) != len(artifacts) or len({p.name for p in files}) != len(files):
        raise RuntimeError("release artifact set is incomplete or ambiguous")
    if set(plan.get("quants") or []) != {item.get("quant") for item in artifacts}:
        raise RuntimeError("release does not contain every requested quant")
    for path in files:
        matching = [item for item in artifacts if item.get("name") == path.name]
        if len(matching) != 1:
            raise RuntimeError("artifact is not uniquely validated")
        item = matching[0]
        inference = item.get("inference") or {}
        if item.get("source_sha256") != source["sha256"]:
            raise RuntimeError("artifact is not bound to the verified GGUF source")
        digest = sha256_file(path)
        if item.get("sha256") != digest or inference.get("model_sha256") != digest:
            raise RuntimeError("artifact changed after validation")
        if not all(item.get(key) is True for key in ("structure_valid", "runtime_compatible", "inference_tested")):
            raise RuntimeError("artifact validation did not pass")


def _remote_matches(item, path: Path | None = None, payload: bytes | None = None) -> bool:
    if item is None:
        return False
    size = path.stat().st_size if path else len(payload)
    if getattr(item, "size", None) != size:
        return False
    lfs = getattr(item, "lfs", None)
    lfs_digest = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    if lfs_digest:
        digest = sha256_file(path) if path else hashlib.sha256(payload).hexdigest()
        return digest == lfs_digest
    # Small Hub files use Git blob object IDs, not SHA-256.
    digest = hashlib.sha1(f"blob {size}\0".encode())
    if path:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    else:
        digest.update(payload)
    return getattr(item, "blob_id", None) == digest.hexdigest()


def upload_verified_release(
    repo_id: str, readme: str, files: list[Path], token: str, private: bool = False,
    *, requested: bool = False, analysis: dict | None = None, plan: dict | None = None,
    validation: dict | None = None, before_upload=None,
) -> dict:
    """Resume matching remote artifacts and verify all hashes plus the model card."""
    publication_gate(requested, analysis or {}, plan or {}, validation or {}, files)
    if not re.fullmatch(r"webmp3/[A-Za-z0-9][A-Za-z0-9._-]*", repo_id):
        raise RuntimeError("release target must be in the configured webmp3 namespace")
    if not token:
        raise RuntimeError("HF token missing")
    if not files:
        raise RuntimeError("refusing to publish a release with no GGUF files")
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    if before_upload:
        before_upload()
    identity = api.whoami()
    if identity.get("name") != "webmp3":
        raise RuntimeError("HF token does not belong to the configured webmp3 publisher")
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    info = api.model_info(repo_id, files_metadata=True)
    remote = {item.rfilename: item for item in info.siblings or []}
    if "README.md" in remote:
        card = getattr(info, "card_data", None) or {}
        base = card.get("base_model")
        if base not in (analysis.get("repo_id"), [analysis.get("repo_id")]) or card.get("hfqa_source_revision") != analysis.get("revision"):
            raise RuntimeError("existing release has no matching pinned source provenance; refusing overwrite")
    uploaded = []
    skipped = []
    operations = []
    for path in files:
        if before_upload:
            before_upload()
        if _remote_matches(remote.get(path.name), path):
            skipped.append(path.name)
            continue
        operations.append(CommitOperationAdd(path_in_repo=path.name, path_or_fileobj=str(path)))
        uploaded.append(path.name)
    if before_upload:
        before_upload()
    readme_bytes = readme.encode("utf-8")
    if not _remote_matches(remote.get("README.md"), payload=readme_bytes):
        operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme_bytes))
    if operations:
        if before_upload:
            before_upload()
        # Hub uploads the blobs before publishing this single commit. All quants
        # and the final model card become visible together; matching blobs resume.
        api.create_commit(repo_id=repo_id, operations=operations,
                          commit_message="Publish complete validated GGUF release", parent_commit=info.sha)
    if before_upload:
        before_upload()
    info = api.model_info(repo_id, files_metadata=True)
    remote = {item.rfilename: item for item in info.siblings or []}
    mismatched = [path.name for path in files if not _remote_matches(remote.get(path.name), path)]
    if not _remote_matches(remote.get("README.md"), payload=readme_bytes):
        mismatched.append("README.md")
    if mismatched or not info.sha:
        raise RuntimeError(f"upload hash verification failed for {mismatched}")
    return {
        "repo_id": repo_id,
        "revision": info.sha,
        "url": f"https://huggingface.co/{repo_id}",
        "files": [path.name for path in files],
        "uploaded": uploaded,
        "skipped": skipped,
        "readme_verified": True,
        "verified": True,
    }
