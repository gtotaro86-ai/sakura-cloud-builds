"""Bounded local text calibration using stock llama-imatrix; no quality claim.

The fixed calibration prose below was authored for HFQA and is dedicated to the
public domain under CC0-1.0. It contains no downloaded/user/private material.
Legacy .dat structure follows llama.cpp b11669 common/imatrix-loader.cpp.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import subprocess
from pathlib import Path

from hfqa.gguf import validate_gguf

MAX_SOURCE_BYTES = 800 * 1024**2
SETTINGS = {"threads": 1, "context": 128, "chunks": 4, "batch": 128,
            "timeout_seconds": 900, "format": "dat", "gpu_layers": 0}
# Conservative initial qualification scope; other architectures fall back explicitly.
SUPPORTED_ARCHITECTURES = frozenset({"qwen2", "llama"})
DATASET_ID = "hfqa-original-calibration-v1"
DATASET_LICENSE = "CC0-1.0"
_PROSE = """A small library opens beside the river. Each morning a reader returns a book,
and the librarian records the title, date, and shelf number. Clear records help
the next visitor find the right volume. The windows face a garden of green trees.

Water can exist as a solid, liquid, or gas. Heating changes its state while the
amount of matter is conserved in a closed container. A careful experiment lists
its measurements and units, repeats the observation, and describes uncertainty.

To prepare a meal, wash vegetables and cut them into even pieces. Warm a pan,
add the ingredients, stir gently, and taste before serving. Write down a useful
recipe so another cook can follow the same steps with the same quantities.

A computer program reads a list of numbers and returns their sum. Start with
zero, add each number once, and return the accumulated value. For an empty list
the result is zero. For the list one, two, and three, the result is six.

Eine Reise beginnt mit einem Plan. Der Zug faehrt am Morgen ab und erreicht die
Stadt am Mittag. Am Bahnhof stehen Wegweiser. Die Reisenden lesen die Anzeige,
suchen den richtigen Bahnsteig und lassen anderen Menschen genug Platz.

In a workshop, a builder measures a wooden board twice before cutting it.
The drawing shows the length, width, and position of every joint. When a part
does not fit, compare it with the drawing and adjust the measurement carefully.

Explain a difficult idea with familiar words and a concrete example. Separate
observations from guesses. If evidence is missing, say what remains unknown.
An answer becomes more useful when its assumptions and limits are visible.

The night sky changes with the seasons. Earth rotates on its axis and travels
around the Sun. A map can mark the stars that are visible from a chosen place.
Clouds may hide them, so an observer records the weather along with the time.
"""
DATASET = ((_PROSE + "\n") * 4).encode("utf-8")
DATASET_SHA256 = hashlib.sha256(DATASET).hexdigest()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(matrix: Path) -> Path:
    return matrix.with_name(matrix.name + ".json")


def _validate_dat(path: Path) -> dict:
    """Reject empty/truncated/NaN/negative/duplicate legacy data before native use."""
    if not 16 <= path.stat().st_size <= 64 * 1024**2:
        raise ValueError("invalid imatrix size")
    with path.open("rb") as handle:
        def read(n):
            data = handle.read(n)
            if len(data) != n:
                raise ValueError("truncated imatrix")
            return data
        def integer():
            return struct.unpack("<i", read(4))[0]
        entries = integer()
        if not 1 <= entries <= 10000:
            raise ValueError("invalid imatrix entry count")
        names = set()
        for _ in range(entries):
            length = integer()
            if not 1 <= length <= 1024:
                raise ValueError("invalid imatrix tensor name")
            name = read(length).decode("utf-8", "strict")
            if name in names or "\0" in name:
                raise ValueError("duplicate or invalid imatrix tensor name")
            names.add(name)
            calls, count = integer(), integer()
            if not 1 <= calls <= SETTINGS["chunks"] or not 1 <= count <= 4_000_000:
                raise ValueError("invalid imatrix counts")
            positive = False
            for (value,) in struct.iter_unpack("<f", read(count * 4)):
                if not math.isfinite(value) or value < 0:
                    raise ValueError("nonfinite or negative imatrix value")
                positive |= value > 0
            if not positive:
                raise ValueError("empty imatrix tensor data")
        chunks = integer()
        length = integer()
        if chunks != SETTINGS["chunks"] or not 1 <= length <= 8192:
            raise ValueError("imatrix calibration chunk count or dataset missing")
        dataset = read(length).decode("utf-8", "strict")
        if handle.read(1):
            raise ValueError("unexpected imatrix trailing data")
    return {"entries": entries, "chunks_processed": chunks, "dataset_path": dataset}


def validate_imatrix(matrix: str | Path, source: str | Path, *, tool: str | Path | None = None) -> dict:
    path, src = Path(matrix).resolve(), Path(source).resolve()
    manifest = json.loads(_manifest_path(path).read_text(encoding="utf-8"))
    if (manifest.get("schema") != 1 or manifest.get("source_sha256") != sha256(src)
            or manifest.get("dataset_sha256") != DATASET_SHA256
            or manifest.get("settings") != SETTINGS
            or manifest.get("matrix_sha256") != sha256(path)
            or manifest.get("actual_used_method") != "imatrix"
            or not isinstance(manifest.get("tool_sha256"), str)
            or len(manifest["tool_sha256"]) != 64):
        raise ValueError("imatrix manifest/checksum/source mismatch")
    if tool is not None and manifest["tool_sha256"] != sha256(tool):
        raise ValueError("imatrix calibration tool mismatch")
    parsed = _validate_dat(path)
    if manifest.get("entries") != parsed["entries"] or manifest.get("chunks_processed") != parsed["chunks_processed"]:
        raise ValueError("imatrix manifest structure mismatch")
    return {**manifest, "path": str(path)}


def prepare_imatrix(source: str | Path, output_dir: str | Path, binary: str | Path,
                    *, quality_mode: str = "normal", text_only: bool = True) -> dict:
    """Optional support failures fall back; execution/corruption failures propagate."""
    if quality_mode not in {"normal", "imatrix_if_supported"}:
        raise ValueError("unknown quality mode")
    base = {"requested_quality_mode": quality_mode, "actual_used_method": "normal"}
    if quality_mode == "normal":
        return base
    src, tool = Path(source).resolve(), Path(binary).resolve()
    reason = None
    if not text_only:
        reason = "imatrix supports text-only models in this worker"
    elif src.stat().st_size > MAX_SOURCE_BYTES:
        reason = "imatrix source exceeds the 800 MiB local calibration limit"
    else:
        structure = validate_gguf(src)
        if not structure["structure_valid"]:
            raise ValueError("invalid imatrix source")
        if structure["architecture"] not in SUPPORTED_ARCHITECTURES:
            reason = "imatrix architecture is outside the qualified local scope"
        elif not tool.is_file() or not os.access(tool, os.X_OK):
            reason = "stock llama-imatrix executable is unavailable"
    if reason:
        return {**base, "fallback_reason": reason}
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    matrix = directory / "importance.dat"
    if matrix.exists() or _manifest_path(matrix).exists():
        return {**validate_imatrix(matrix, src, tool=tool), "reused": True,
                "requested_quality_mode": quality_mode}
    dataset = directory / (DATASET_ID + ".txt")
    if dataset.exists() and dataset.read_bytes() != DATASET:
        raise ValueError("calibration dataset checksum mismatch")
    dataset.write_bytes(DATASET)
    partial = matrix.with_name(matrix.name + ".partial")
    partial.unlink(missing_ok=True)
    stdout_path, stderr_path = directory / "imatrix.stdout.log", directory / "imatrix.stderr.log"
    command = [str(tool), "-m", str(src), "-f", str(dataset), "-o", str(partial),
               "--output-format", "dat", "--no-ppl", "-t", "1", "-tb", "1",
               "-c", "128", "-b", "128", "-ub", "128", "--chunks", "4", "-ngl", "0"]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(tool.parent) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        completed = subprocess.run(command, check=False, stdin=subprocess.DEVNULL,
                                   stdout=stdout, stderr=stderr, timeout=900, env=env, cwd=directory)
    if completed.returncode != 0 or not partial.is_file():
        raise RuntimeError(f"llama-imatrix failed (exit {completed.returncode}); logs: {stderr_path}")
    parsed = _validate_dat(partial)
    manifest = {"schema": 1, "actual_used_method": "imatrix", "source_sha256": sha256(src),
                "tool_sha256": sha256(tool), "dataset_id": DATASET_ID, "dataset_license": DATASET_LICENSE,
                "dataset_sha256": DATASET_SHA256, "settings": SETTINGS,
                "matrix_sha256": sha256(partial), **parsed,
                "stdout_log": str(stdout_path), "stderr_log": str(stderr_path),
                "note": "Short fixed text calibration; no quality improvement or benchmark is claimed."}
    with partial.open("r+b") as handle:
        os.fsync(handle.fileno())
    manifest_tmp = _manifest_path(matrix).with_suffix(".json.partial")
    with manifest_tmp.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(matrix)
    manifest_tmp.replace(_manifest_path(matrix))
    return {**manifest, "path": str(matrix), "reused": False, "requested_quality_mode": quality_mode}
