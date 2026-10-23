"""Tailscale-only FastAPI control plane. Tokens stay in environment files, never in responses."""

from __future__ import annotations

import json
import os
import hmac
import time
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from hfqa import __version__
from hfqa.inspect import InspectError, parse_hf_url
from hfqa.jobs import JobStore, TERMINAL, TransitionError
from hfqa.heretic import normalize_variants
from hfqa.publish import fetch_release_status
from hfqa.worker import allow_source, worker_decision
from hfqa.orchestrator import Orchestrator, default_quants
from hfqa.security import redact

DATA = Path(os.environ.get("HFQA_DATA", "data"))
def _load_token() -> str:
    direct = os.environ.get("HFQA_API_TOKEN", "").strip()
    if direct:
        return direct
    path = os.environ.get("HFQA_API_TOKEN_FILE", "")
    if path and Path(path).is_file():
        return Path(path).read_text(encoding="utf-8").strip()
    return ""


TOKEN = _load_token()
store = JobStore(DATA / "jobs.sqlite")
orchestrator = Orchestrator(store, DATA / "work", DATA / "budget.json")
app = FastAPI(title="HF Quant Agent", version=__version__)


class WorkRequest(BaseModel):
    source_path: str


class JobCreate(BaseModel):
    url: str
    quants: list[str] = Field(default_factory=lambda: ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M"])
    variants: list[str] = Field(default_factory=lambda: ["normal"])
    publish: bool = False


class BudgetUpdate(BaseModel):
    enabled: bool | None = None
    require_confirmation: bool | None = None
    per_job_usd: float | None = Field(default=None, ge=0, le=100)
    monthly_usd: float | None = Field(default=None, ge=0, le=1000)


def require_token(authorization: str | None = Header(default=None)) -> None:
    if not TOKEN:
        raise HTTPException(503, "API token is not configured")
    if not hmac.compare_digest(authorization or "", f"Bearer {TOKEN}"):
        raise HTTPException(401, "unauthorized")


@app.get("/releases/smoke", dependencies=[Depends(require_token)])
def smoke_release() -> dict:
    return fetch_release_status(
        "webmp3/hfqa-smoke-Qwen2.5-0.5B-Instruct-requant-GGUF",
        {
            "qwen2.5-0.5b-instruct-Q4_K_M-requant.gguf": 491400032,
            "qwen2.5-0.5b-instruct-Q5_K_M-requant.gguf": 522186592,
            "README.md": 3722,
        },
    )


@app.get("/health")
def health() -> dict:
    return {"ok": True, "version": __version__, "service": "hf-quant-agent"}


@app.post("/jobs", dependencies=[Depends(require_token)])
def create_job(body: JobCreate) -> dict:
    try:
        repo_id, revision = parse_hf_url(body.url)
        quants = default_quants(body.quants)
        variants = normalize_variants(body.variants)
    except (InspectError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    job, created = store.create(body.url, repo_id, quants, publish=body.publish, revision=revision, variants=variants)
    return {"job": _public(job), "created": created}


@app.post("/jobs/{job_id}/run", dependencies=[Depends(require_token)])
def run_job(job_id: str) -> dict:
    if store.get(job_id) is None:
        raise HTTPException(404, "job not found")
    try:
        job = orchestrator.run(job_id, resources=_resources())
    except TransitionError:
        return {"job": _public(store.get(job_id))}
    except Exception as exc:  # noqa: BLE001
        store.update_fields(job_id, error=redact(str(exc)))
        current = store.get(job_id)
        if current and current["state"] in {"INSPECTING", "PLANNING", "DEVELOPING"}:
            try:
                store.transition(job_id, "FAILED", redact(str(exc)))
            except Exception:
                pass
        raise HTTPException(500, redact(str(exc))) from exc
    return {"job": _public(job)}


@app.post("/jobs/{job_id}/work", dependencies=[Depends(require_token)])
def start_work(job_id: str, body: WorkRequest) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    try:
        source = allow_source(body.source_path)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    assume_present = False
    try:
        visible = source if source.is_file() else None
    except PermissionError:
        visible = source
        assume_present = True
    decision = worker_decision(job, visible, assume_present=assume_present)
    if not decision["allowed"]:
        raise HTTPException(409, decision["reason"])
    queue = Path("/var/lib/hf-quant-agent/queue")
    queue.mkdir(parents=True, exist_ok=True)
    ticket = {
        "job_id": job_id,
        "source": str(source),
        "quants": job["quants"],
        "publish": bool(job["publish"]),
    }
    tmp = queue / f"{job_id}.tmp"
    tmp.write_text(json.dumps(ticket), encoding="utf-8")
    os.replace(tmp, queue / f"{job_id}.json")
    store.event(job_id, job["state"], job["state"], "worker ticket queued", {"quants": ticket["quants"]})
    return {"queued": True, "started": False, "ticket": ticket}


@app.get("/jobs", dependencies=[Depends(require_token)])
def list_jobs() -> dict:
    return {"jobs": [_public(job) for job in store.list_jobs()]}


@app.get("/jobs/{job_id}", dependencies=[Depends(require_token)])
def get_job(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return {"job": _public(job), "events": store.events_for(job_id), "shards": store.shards(job_id)}


@app.get("/jobs/{job_id}/card", dependencies=[Depends(require_token)])
def card(job_id: str) -> dict:
    if store.get(job_id) is None:
        raise HTTPException(404, "job not found")
    job = store.get(job_id)
    result = job.get("result") or {}
    if result.get("artifacts"):
        from hfqa.publish import render_readme
        return {"markdown": render_readme(job.get("analysis") or {}, job.get("plan") or {}, result, result["artifacts"])}
    return {"markdown": orchestrator.render_card(job_id)}


@app.post("/jobs/{job_id}/pause", dependencies=[Depends(require_token)])
def pause(job_id: str) -> dict:
    return _simple_transition(job_id, "PAUSED", "paused by operator")


@app.post("/jobs/{job_id}/cancel", dependencies=[Depends(require_token)])
def cancel(job_id: str) -> dict:
    return _simple_transition(job_id, "CANCELLED", "cancelled by operator")


@app.post("/jobs/{job_id}/resume", dependencies=[Depends(require_token)])
def resume(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job["state"] != "PAUSED":
        raise HTTPException(409, "job is not paused")
    target = job.get("paused_from") or "PLANNING"
    try:
        store.transition(job_id, target, "resumed by operator")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(409, str(exc)) from exc
    if target in {"CREATED", "INSPECTING", "PLANNING", "WAITING_FOR_WORKER"}:
        return run_job(job_id)
    return {"job": _public(store.get(job_id))}


@app.get("/system", dependencies=[Depends(require_token)])
def system() -> dict:
    return {"resources": _resources(), "budget": orchestrator.budget.data, "spark": spark_status()}


@app.post("/budget", dependencies=[Depends(require_token)])
def update_budget(body: BudgetUpdate) -> dict:
    data = orchestrator.budget.data
    for key in ("enabled", "require_confirmation", "per_job_usd", "monthly_usd"):
        value = getattr(body, key)
        if value is not None:
            data[key] = value
    if float(data.get("per_job_usd") or 0) > 0.50 and os.environ.get("HFQA_ALLOW_HIGHER_BUDGET") != "1":
        data["per_job_usd"] = 0.50
    if float(data.get("monthly_usd") or 0) > 3.0 and os.environ.get("HFQA_ALLOW_HIGHER_BUDGET") != "1":
        data["monthly_usd"] = 3.0
    orchestrator.budget.save()
    return {"budget": data}


def _simple_transition(job_id: str, state: str, message: str) -> dict:
    if store.get(job_id) is None:
        raise HTTPException(404, "job not found")
    try:
        store.transition(job_id, state, message)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(409, str(exc)) from exc
    return {"job": _public(store.get(job_id))}


def _public(job: dict) -> dict:
    return {
        "id": job["id"],
        "source_url": job["source_url"],
        "repo_id": job["repo_id"],
        "revision": job["revision"],
        "quants": job["quants"],
        "variants": job.get("variants") or ["normal"],
        "state": job["state"],
        "publish": job["publish"],
        "error": job.get("error"),
        "analysis": job.get("analysis"),
        "plan": job.get("plan"),
        "result": job.get("result"),
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }


def _resources() -> dict:
    path = Path(os.environ.get("HFQA_DISK_PATH", "/"))
    try:
        usage = os.statvfs(path) if hasattr(os, "statvfs") else None
    except OSError:
        usage = None
    if usage:
        free = usage.f_bavail * usage.f_frsize
        total = usage.f_blocks * usage.f_frsize
    else:
        import shutil
        disk = shutil.disk_usage(path if path.exists() else Path.cwd())
        free, total = disk.free, disk.total
    meminfo = Path("/proc/meminfo")
    available = None
    total_ram = None
    if meminfo.exists():
        values = {}
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            values[key] = int(rest.strip().split()[0]) * 1024
        available = values.get("MemAvailable")
        total_ram = values.get("MemTotal")
    return {
        "free_disk_bytes": free,
        "total_disk_bytes": total,
        "available_ram_bytes": available,
        "total_ram_bytes": total_ram,
        "dns_services_must_stay_up": True,
        "cpu_load": os.getloadavg()[0] if hasattr(os, "getloadavg") else None,
    }


def spark_status() -> dict:
    import urllib.request
    host = os.environ.get("HFQA_SPARK_URL", "http://127.0.0.1:8088")
    model = os.environ.get("HFQA_SPARK_MODEL", "bartowski/Spark-X2.6-4B-GGUF Q4_K_M")
    probed = False
    error = None
    try:
        with urllib.request.urlopen(host.rstrip("/") + "/health", timeout=2) as response:
            probed = response.status == 200
    except Exception as exc:  # noqa: BLE001
        error = type(exc).__name__
    return {"model": model, "public": False, "loopback_only": True, "healthy": probed, "error": error}


@app.post("/jobs/{job_id}/retry", dependencies=[Depends(require_token)])
def retry_job(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job["state"] not in {"FAILED", "BLOCKED"}:
        raise HTTPException(409, "retry is allowed only for failed or blocked jobs")
    store.retry(job_id)
    return run_job(job_id)


@app.get("/jobs/{job_id}/events", dependencies=[Depends(require_token)])
def stream_events(job_id: str, after: int = 0) -> StreamingResponse:
    if store.get(job_id) is None:
        raise HTTPException(404, "job not found")
    def generate():
        cursor = after
        # Bounded stream; client reconnects with the last delivered event id.
        for _ in range(30):
            for event in store.events_for(job_id):
                if event["id"] > cursor:
                    cursor = event["id"]
                    yield f"id: {cursor}\nevent: progress\ndata: {json.dumps(event)}\n\n"
            if store.get(job_id)["state"] in TERMINAL:
                return
            yield ": heartbeat\n\n"
            time.sleep(2)
    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
