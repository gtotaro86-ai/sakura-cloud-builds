"""Local llama-quantize wrapper. Arguments are a list, never a shell string."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from hfqa.gguf import validate_gguf

ALLOWED_QUANTS = {"Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M"}


def _require_same_tensors(source: dict, output: dict) -> None:
    if (source["tensor_count"] != output["tensor_count"]
            or source["tensor_name_set_sha256"] != output["tensor_name_set_sha256"]):
        raise ValueError("quantized output tensor count/name set differs from source")


def _quantization_logs(stdout_path: Path, stderr_path: Path) -> dict:
    lines = []
    count = None
    for path in (stdout_path, stderr_path):
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "fallback" not in line.lower():
                    continue
                lines.append(line.rstrip()[:1000])
                match = re.search(r"(\d+)\s+(?:of\s+\d+\s+)?tensor(?:\(s\)|s)?\s+required\s+fallback", line, re.I)
                if match:
                    count = int(match.group(1))
    return {"quantize_stdout_log": str(stdout_path), "quantize_stderr_log": str(stderr_path),
            "fallback_tensor_count": count, "fallback_warning_lines": lines[-30:]}


def quantize_file(
    quantize_bin: str | Path,
    source: str | Path,
    output: str | Path,
    quant: str,
    timeout_seconds: int = 3600,
    threads: int = 1,
    expected_sha256: str | None = None,
    expected_architecture: str | None = None,
) -> dict:
    if quant not in ALLOWED_QUANTS:
        raise ValueError(f"refusing quant {quant}")
    binary = Path(quantize_bin)
    src = Path(source)
    dst = Path(output)
    binary = binary.resolve()
    src = src.resolve()
    dst = dst.resolve()
    if src == dst:
        raise ValueError("source and output must differ")
    if not binary.is_file():
        raise FileNotFoundError(binary)
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = dst.with_name(dst.name + ".quant.json")
    stdout_path = dst.with_name(dst.name + ".quant.stdout.log")
    stderr_path = dst.with_name(dst.name + ".quant.stderr.log")
    source_report = validate_gguf(src, expected_architecture)
    if not source_report["structure_valid"]:
        raise ValueError(source_report["note"])
    if dst.exists():
        if not expected_sha256 and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest.get("quant") == quant and manifest.get("source_sha256") == _sha256(src)
                    and manifest.get("quantizer_sha256") == _sha256(binary)):
                expected_sha256 = manifest.get("sha256")
        if not expected_sha256 or _sha256(dst) != expected_sha256:
            raise FileExistsError("existing output has no matching durable checkpoint; refusing overwrite")
        report = validate_gguf(dst, expected_architecture)
        if not report["structure_valid"]:
            raise ValueError(report["note"])
        _require_same_tensors(source_report, report)
        report.update(_quantization_logs(stdout_path, stderr_path))
        report["skipped"] = True
        report["reason"] = "output already exists; not recomputed"
        report["sha256"] = _sha256(dst)
        return report
    env = os.environ.copy()
    lib_dir = str(binary.parent)
    env["LD_LIBRARY_PATH"] = lib_dir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    partial = dst.with_name(dst.name + ".partial.gguf")
    command = [str(binary), "--allow-requantize", str(src), str(partial), quant, str(threads)]
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        completed = subprocess.run(command, check=False, stdout=stdout, stderr=stderr,
                                   timeout=timeout_seconds, env=env, stdin=subprocess.DEVNULL)
    if completed.returncode != 0 or not partial.is_file():
        raise RuntimeError(
            f"llama-quantize failed (exit {completed.returncode}); logs: {stderr_path}, {stdout_path}"
        )
    report = validate_gguf(partial, expected_architecture or source_report["architecture"])
    if not report["structure_valid"]:
        raise ValueError(report["note"])
    _require_same_tensors(source_report, report)
    report.update(_quantization_logs(stdout_path, stderr_path))
    with partial.open("r+b") as handle:
        os.fsync(handle.fileno())
    manifest = {"quant": quant, "source_sha256": _sha256(src),
                "quantizer_sha256": _sha256(binary), "sha256": _sha256(partial)}
    manifest_partial = manifest_path.with_suffix(".json.partial")
    with manifest_partial.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle)
        handle.flush()
        os.fsync(handle.fileno())
    manifest_partial.replace(manifest_path)
    partial.replace(dst)
    report["path"] = str(dst)
    report["sha256"] = _sha256(dst)
    report["skipped"] = False
    report["quant"] = quant
    report["source"] = str(src)
    report["command"] = [binary.name, "--allow-requantize", src.name, dst.name, quant, str(threads)]
    return report


def test_inference(
    runtime_bin: str | Path,
    model: str | Path,
    log_dir: str | Path,
    timeout_seconds: int = 180,
) -> dict:
    """One bounded CPU inference, using the configured stock llama-cli executable.

    Captures real stdout/stderr as evidence. Passing is only a smoke check and
    does not assert quality, correctness, perplexity or benchmark scores.
    """
    binary = Path(runtime_bin).resolve()
    model_path = Path(model).resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)
    if not validate_gguf(model_path)["structure_valid"]:
        raise ValueError("invalid model structure")
    logs = Path(log_dir)
    logs.mkdir(parents=True, exist_ok=True)
    prompt = "The capital of France is"
    command = [str(binary), "-m", str(model_path), "-p", prompt,
               "-n", "16", "-c", "128", "--temp", "0", "-t", "1", "-ngl", "0",
               "--single-turn", "--no-display-prompt"]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(binary.parent) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    # Files bound memory use even if a runtime emits excessive diagnostic output.
    stdout_path = logs / (model_path.stem + ".stdout.log")
    stderr_path = logs / (model_path.stem + ".stderr.log")
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        completed = subprocess.run(command, check=False, stdin=subprocess.DEVNULL,
                                   stdout=stdout, stderr=stderr, timeout=timeout_seconds, env=env)
    with stdout_path.open("rb") as stdout:
        output = stdout.read(32768).decode("utf-8", "replace")
    clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", output)
    clean = re.sub(r"\[(?:end of text|end of generation)\]|<\|(?:eot_id|im_end)\|>", "", clean)
    with stderr_path.open("rb") as stderr:
        diagnostic = stderr.read(65536).decode("utf-8", "replace")
    build = re.search(r"(?:build|version)\s*[:=]\s*([^\r\n]{1,200})", diagnostic + "\n" + output, re.I)
    if completed.returncode != 0 or not clean.strip():
        raise RuntimeError(f"stock runtime inference failed (exit {completed.returncode}); logs: {stderr_path}")
    return {
        "runtime_compatible": True, "inference_tested": True,
        "runtime_binary": str(binary), "runtime_sha256": _sha256(binary),
        "runtime_build": build.group(1).strip() if build else "unreported by runtime; identified by binary SHA-256",
        "model_sha256": _sha256(model_path), "returncode": completed.returncode,
        "prompt": prompt, "completion": clean.strip()[:4000],
        "context": 128, "predict": 16, "threads": 1,
        "stdout_log": str(stdout_path), "stderr_log": str(stderr_path),
        "command": command, "note": "One smoke prompt; no quality benchmark is claimed.",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
