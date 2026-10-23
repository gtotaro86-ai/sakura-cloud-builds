"""Durable stock GGUF worker with resumable validation and gated publication."""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hfqa.gguf import validate_gguf
from hfqa.heretic import release_plan
from hfqa.jobs import JobStore, TransitionError
from hfqa.publish import publication_gate, render_readme, sha256_file, upload_verified_release
from hfqa.quantize import ALLOWED_QUANTS, quantize_file, test_inference
from hfqa.security import redact
from hfqa.worker import prepare_source

QUEUE = Path(os.environ.get("HFQA_QUEUE_DIR", "/var/lib/hf-quant-agent/queue"))
DB = Path(os.environ.get("HFQA_DB_PATH", "/var/lib/hf-quant-agent/data/jobs.sqlite"))
QUANTIZE = Path(os.environ.get("HFQA_QUANTIZE_BIN", "/home/ubuntu/hfqa/runtime/llama-b11669/llama-quantize"))
RUNTIME = Path(os.environ.get("HFQA_RUNTIME_BIN", str(QUANTIZE.with_name("llama-cli"))))
ACTIVE = {"WAITING_FOR_WORKER", "QUANTIZING", "VALIDATING", "UPLOADING", "VERIFYING"}

class WorkerStopped(RuntimeError):
    pass

class WorkerBusy(RuntimeError):
    pass


def _check_active(store, job_id, owner):
    job = store.get(job_id)
    if not job or job["state"] not in ACTIVE:
        raise WorkerStopped("job is paused, cancelled or no longer active")
    if not store.renew_claim(job_id, owner, ttl_seconds=3600):
        raise WorkerStopped("worker lease was lost")
    return job


def _stage(store, job_id, owner, target, message):
    job = _check_active(store, job_id, owner)
    if job["state"] != target:
        store.transition(job_id, target, message, expected_state=job["state"])


def _read_token():
    token_file = os.environ.get("HFQA_HF_TOKEN_FILE")
    if token_file:
        path = Path(token_file)
        if os.name != "nt" and path.stat().st_mode & 0o077:
            raise RuntimeError("HF token file must be private to its owner")
        token = path.read_text(encoding="utf-8").strip()
    else:
        token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("publication requested but HF token is missing")
    return token


def process_ticket(ticket_path, store, quantize_bin=QUANTIZE, runtime_bin=RUNTIME, output_root=None):
    ticket_path = Path(ticket_path)
    ticket = json.loads(ticket_path.read_text(encoding="utf-8"))
    job_id = ticket.get("job_id", "")
    if not re.fullmatch(r"[a-f0-9]{16}", job_id):
        raise ValueError("invalid worker job id")
    job = store.get(job_id)
    if not job:
        raise ValueError("ticket references an unknown job")
    if job["state"] in {"COMPLETED", "CANCELLED"}:
        ticket_path.unlink(missing_ok=True)
        return job
    if job["state"] not in ACTIVE:
        return job
    owner = uuid.uuid4().hex
    if not store.claim(job_id, owner, ttl_seconds=3600):
        raise WorkerBusy("job already has a live worker lease")
    try:
        return _process(ticket, ticket_path, store, owner, Path(quantize_bin), Path(runtime_bin), output_root)
    except (WorkerStopped, TransitionError):
        return store.get(job_id)
    except Exception as exc:
        current = store.get(job_id)
        if current and current["state"] in ACTIVE:
            error = redact(f"{type(exc).__name__}: {exc}")[:2000]
            try:
                store.update_fields(job_id, error=error, expected_state=current["state"])
                store.transition(job_id, "BLOCKED", "worker stopped; checkpoints retained", {"error": error}, expected_state=current["state"])
            except TransitionError:
                pass
        return store.get(job_id)
    finally:
        store.release_claim(job_id, owner)


def _process(ticket, ticket_path, store, owner, quantize_bin, runtime_bin, output_root):
    job_id = ticket["job_id"]
    check = lambda: _check_active(store, job_id, owner)
    stage = lambda target, message: _stage(store, job_id, owner, target, message)
    save = lambda **fields: store.update_fields(job_id, expected_state=check()["state"], **fields)
    job = check()
    plan, analysis = job.get("plan") or {}, job.get("analysis") or {}
    if plan.get("route") not in {"native_gguf_requant", "oracle_local_requant"} or plan.get("allow_new_patch"):
        raise RuntimeError("worker only executes qualified stock GGUF routes")
    if job.get("variants", ["normal"]) != ["normal"]:
        raise RuntimeError("Heretic variants require a separate qualified worker")
    quants = job.get("quants") or []
    if not quants or len(set(quants)) != len(quants) or any(q not in ALLOWED_QUANTS for q in quants):
        raise RuntimeError("invalid quantization set")
    if not job.get("revision") or analysis.get("revision") != job["revision"]:
        raise RuntimeError("job source is not pinned to its inspected revision")
    if not quantize_bin.is_file() or not runtime_bin.is_file():
        raise RuntimeError("stock quantizer and runtime must both be installed")
    def should_stop():
        try:
            check()
            return False
        except WorkerStopped:
            return True
    source = prepare_source(ticket, job, should_stop=should_stop)
    check()
    structure = validate_gguf(source)
    if not structure["structure_valid"]:
        raise RuntimeError("source structural validation failed")
    source_digest = sha256_file(source)
    descriptor = ticket.get("download")
    if not descriptor and job["publish"]:
        provenance = source.with_suffix(".source.json")
        if provenance.is_file():
            descriptor = json.loads(provenance.read_text(encoding="utf-8"))
            from hfqa.worker import source_descriptor
            expected = source_descriptor(analysis)
            if any(descriptor.get(key) != expected.get(key) for key in ("repo_id", "revision", "filename", "size", "sha256")):
                raise RuntimeError("local source provenance does not match the pinned Hub artifact")
    source_verified = bool(descriptor and descriptor.get("sha256") == source_digest
                           and descriptor.get("size") == source.stat().st_size
                           and re.fullmatch(r"[a-fA-F0-9]{40}", descriptor.get("revision", "")))
    if job["publish"] and not source_verified:
        raise RuntimeError("publication requires verified pinned GGUF source provenance")
    result = dict(job.get("result") or {})
    if result.get("source_sha256") and result["source_sha256"] != source_digest:
        raise RuntimeError("source changed since the durable checkpoint")
    result.update({"source": str(source), "source_sha256": source_digest,
                   "source_verified": source_verified, "source_descriptor": descriptor,
                   "source_structure": structure, "published": False})
    save(result_json=result, error=None)
    try:
        from hfqa.agent import advise
    except ImportError:
        advise = None
    if advise and not result.get("worker_advised"):
        advise(store, job_id)
        check()
        result["worker_advised"] = True
        save(result_json=result)
    out_dir = Path(output_root or os.environ.get("HFQA_OUTPUT_DIR", str(source.parent / "hfqa-outputs"))) / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for quant in quants:
        check()
        output = out_dir / f"{source.stem}-{quant}-requant.gguf"
        checkpoint = next((s for s in store.shards(job_id) if s["quant"] == quant and s["shard_index"] == 0), None)
        detail = json.loads((checkpoint or {}).get("detail_json") or "{}")
        if checkpoint and checkpoint.get("path") and Path(checkpoint["path"]).resolve() != output.resolve():
            raise RuntimeError("checkpoint output path differs from worker workspace")
        if output.is_file() and checkpoint and checkpoint.get("checksum"):
            report = quantize_file(quantize_bin, source, output, quant, expected_sha256=checkpoint["checksum"], expected_architecture=structure["architecture"])
        else:
            if check()["state"] in {"UPLOADING", "VERIFYING"}:
                raise RuntimeError("upload-stage job has a missing quantization checkpoint")
            stage("QUANTIZING", f"stock quantization: {quant}")
            report = quantize_file(quantize_bin, source, output, quant, timeout_seconds=900, threads=1, expected_architecture=structure["architecture"])
            detail = {"structure": report, "source_sha256": source_digest}
            store.upsert_shard(job_id, quant, 0, "quantized", checksum=report["sha256"], nbytes=report["bytes"], path=str(output), detail=detail)
        check()
        inference = detail.get("inference") or {}
        if not (inference.get("inference_tested") is True and inference.get("runtime_compatible") is True and inference.get("model_sha256") == report["sha256"] and inference.get("runtime_sha256") == sha256_file(runtime_bin)):
            if check()["state"] in {"UPLOADING", "VERIFYING"}:
                raise RuntimeError("runtime changed or validation checkpoint is incomplete; retry the job to revalidate")
            stage("VALIDATING", f"stock runtime inference: {quant}")
            inference = test_inference(runtime_bin, output, out_dir / "logs")
            if not (inference.get("runtime_compatible") is True and inference.get("inference_tested") is True and inference.get("model_sha256") == report["sha256"]):
                raise RuntimeError("runtime validation did not prove inference for this output")
            detail.update({"structure": report, "inference": inference, "source_sha256": source_digest})
            store.upsert_shard(job_id, quant, 0, "validated", checksum=report["sha256"], nbytes=report["bytes"], path=str(output), detail=detail)
        check()
        artifacts.append({"name": output.name, "path": str(output), "quant": quant, "bytes": report["bytes"], "sha256": report["sha256"], "source_sha256": source_digest, "structure_valid": True, "runtime_compatible": True, "inference_tested": True, "inference": inference, "quantization": report})
    if check()["state"] not in {"UPLOADING", "VERIFYING"}:
        stage("VALIDATING", "all requested quants passed the stock runtime")
    validation = {"structure_valid": True, "runtime_compatible": True, "inference_tested": True,
                  "source_verified": source_verified, "source_sha256": source_digest, "source_descriptor": descriptor,
                  "source_structure": structure,
                  "artifacts": artifacts, "summary": f"All {len(artifacts)} requested quants passed structure and one real CPU inference each. No quality benchmark was run."}
    result = dict(store.get(job_id).get("result") or {})
    result.update(validation)
    save(result_json=result)
    check()
    if job["publish"]:
        files = [Path(item["path"]) for item in artifacts]
        publication_gate(True, analysis, plan, validation, files)
        token = _read_token()
        readme = render_readme(analysis, plan, validation, artifacts)
        (out_dir / "README.md").write_text(readme, encoding="utf-8")
        target = plan.get("release_repo_id") or release_plan(job["repo_id"], ["normal"])[0]["repo_id"]
        stage("UPLOADING", "uploading validated release artifacts")
        release = upload_verified_release(target, readme, files, token, requested=True, analysis=analysis, plan=plan, validation=validation, before_upload=check)
        stage("VERIFYING", "remote artifacts and model-card hashes verified")
        if release.get("verified") is not True or release.get("readme_verified") is not True:
            raise RuntimeError("remote release verification is incomplete")
        result.update({"published": True, "release": release})
        save(result_json=result)
    final = check()
    store.transition(job_id, "COMPLETED", "all requested artifacts validated" + (" and publication verified" if job["publish"] else "; publication not requested"), expected_state=final["state"])
    ticket_path.unlink(missing_ok=True)
    return store.get(job_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=QUEUE)
    parser.add_argument("--db", type=Path, default=DB)
    parser.add_argument("--quantize-bin", type=Path, default=QUANTIZE)
    parser.add_argument("--runtime-bin", type=Path, default=RUNTIME)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if hasattr(os, "nice"):
        os.nice(15)
    tickets = sorted(args.queue.glob("*.json")) if args.queue.is_dir() else []
    if not tickets:
        print("no tickets")
        return
    with JobStore(args.db) as store:
        for ticket_path in tickets:
            try:
                result = process_ticket(ticket_path, store, args.quantize_bin, args.runtime_bin, args.output_dir)
                print(json.dumps({"job_id": result["id"], "state": result["state"], "published": (result.get("result") or {}).get("published", False)}))
                if result["state"] == "COMPLETED":
                    break
            except WorkerBusy:
                continue
            except Exception as exc:
                print(redact(f"invalid ticket {ticket_path.name}: {type(exc).__name__}: {exc}")[:2000], file=sys.stderr)

if __name__ == "__main__":
    main()
