"""Deterministic control loop. Spark may advise; it cannot skip validation gates."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from hfqa.escalation import BudgetManager, DeepSeekClient, EscalationDisabled
from hfqa.githubmon import list_runs, runs_for_workflow
from hfqa.inspect import find_native_gguf_repo, inspect_repo
from hfqa.jobs import JobStore, TERMINAL
from hfqa.planner import normalize_quants, plan
from hfqa.publish import render_readme
from hfqa.security import as_untrusted_data, redact
from hfqa.worker import source_descriptor, source_destination

MAX_LOCAL_REPAIRS = 2


class Orchestrator:
    def __init__(self, store: JobStore, work_dir: Path, budget_path: Path | None = None):
        self.store = store
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.budget = BudgetManager(budget_path or (self.work_dir / "budget.json"))

    def run(self, job_id: str, resources: dict | None = None, runs: list[dict] | None = None) -> dict:
        owner = "planner-" + uuid.uuid4().hex
        if not self.store.claim(job_id, owner, ttl_seconds=300):
            current = self.store.get(job_id)
            if current is None:
                raise KeyError(job_id)
            return current
        try:
            return self._run(job_id, resources, runs)
        finally:
            self.store.release_claim(job_id, owner)

    def _run(self, job_id: str, resources: dict | None = None, runs: list[dict] | None = None) -> dict:
        job = self.store.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job["state"] in TERMINAL:
            return job
        if job["state"] == "PAUSED":
            return job
        if job["state"] == "CREATED":
            self.store.transition(job_id, "INSPECTING", "reading Hugging Face metadata", expected_state="CREATED")
        if job["state"] == "WAITING_FOR_WORKER":
            if (job.get("plan") or {}).get("route") == "existing_pipeline":
                return self._existing_pipeline(job_id, runs)
            self._queue_local_source(job_id)
            return self.store.get(job_id)
        if job["state"] in {"QUANTIZING", "VALIDATING", "UPLOADING", "VERIFYING"}:
            return job
        inspected = self._inspect(job_id)
        if inspected and inspected["id"] != job_id:
            return inspected
        job = self.store.get(job_id)
        if job["state"] in TERMINAL:
            return job
        if job["state"] == "INSPECTING":
            self.store.transition(job_id, "PLANNING", "building conversion plan", expected_state="INSPECTING")
        self._plan(job_id, resources or {}, runs)
        job = self.store.get(job_id)
        route = (job.get("plan") or {}).get("route")
        if "heretic" in (job.get("variants") or []):
            self.store.transition(job_id, "BLOCKED", "Heretic requires a configured GPU worker; no transformed checkpoint exists")
            return self.store.get(job_id)
        if job["state"] in TERMINAL:
            return job
        if route == "existing_pipeline":
            return self._existing_pipeline(job_id, runs)
        if route == "agent_runtime":
            self.store.transition(job_id, "BLOCKED", "Spark is the agent runtime, not a quantization release target")
            return self.store.get(job_id)
        if route in {"develop", "needs_converter", "github_standard_unqualified", "blocked"}:
            return self._develop_or_block(job_id)
        if route in {"oracle_local_requant", "native_gguf_requant"}:
            sibling = ((job.get("analysis") or {}).get("native_gguf_repo") or {}).get("repo_id")
            self.store.transition(
                job_id,
                "WAITING_FOR_WORKER",
                "native GGUF requant is eligible; no patch and no weight download in the planner",
                {"gguf_repo": sibling},
            )
            self._queue_local_source(job_id)
            return self.store.get(job_id)
        self.store.transition(job_id, "BLOCKED", f"unhandled route {route}")
        return self.store.get(job_id)

    def _queue_local_source(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job["state"] != "WAITING_FOR_WORKER":
            return
        try:
            descriptor = source_descriptor(job.get("analysis") or {})
            source = source_destination(descriptor)
        except Exception as exc:
            self.store.transition(job_id, "BLOCKED", redact(str(exc)))
            return
        queue = Path(os.environ.get("HFQA_QUEUE_DIR", "/var/lib/hf-quant-agent/queue"))
        queue.mkdir(parents=True, exist_ok=True)
        ticket = {"job_id": job_id, "source": str(source), "download": descriptor,
                  "quants": job["quants"], "publish": bool(job["publish"])}
        temporary = queue / f"{job_id}.tmp"
        temporary.write_text(json.dumps(ticket), encoding="utf-8")
        os.replace(temporary, queue / f"{job_id}.json")
        self.store.event(job_id, job["state"], job["state"], "pinned source queued for bounded worker", {
            "quants": job["quants"], "source_revision": descriptor["revision"],
            "source_bytes": descriptor["size"], "cached": source.is_file()})

    def _inspect(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job.get("analysis") and job.get("revision"):
            return
        analysis = inspect_repo(job["source_url"])
        if "gguf" not in (analysis.get("formats") or []):
            try:
                sibling = find_native_gguf_repo(analysis["repo_id"])
            except Exception as exc:  # noqa: BLE001
                sibling = None
                self.store.event(job_id, job["state"], job["state"], "gguf sibling lookup failed", {"error": redact(str(exc))})
            if sibling:
                analysis["native_gguf_repo"] = sibling
        analysis["card_excerpt_untrusted"] = as_untrusted_data(json.dumps({
            "license": analysis.get("license"),
            "model_type": analysis.get("model_type"),
        }))
        canonical, pinned_here = self.store.pin_revision(
            job_id,
            revision=analysis["revision"],
            repo_id=analysis["repo_id"],
            analysis_json=analysis,

        )
        if not pinned_here:
            return canonical
        self.store.event(job_id, job["state"], job["state"], "inspection pinned", {
            "repo_id": analysis["repo_id"],
            "revision": analysis["revision"],
            "model_type": analysis.get("model_type"),
        })

    def _plan(self, job_id: str, resources: dict, runs: list[dict] | None) -> None:
        job = self.store.get(job_id)
        if runs is None:
            try:
                runs = list_runs()
            except Exception as exc:  # noqa: BLE001 - planning must record the failure, not crash the API
                runs = []
                self.store.event(job_id, job["state"], job["state"], "github status unavailable", {"error": redact(str(exc))})
        built = plan(
            job["analysis"],
            job["quants"],
            resources=resources,
            active_workflows=runs,
            variants=job.get("variants") or ["normal"],
        )
        self.store.update_fields(job_id, plan_json=built)
        self.store.event(job_id, job["state"], job["state"], "plan ready", {"route": built["route"], "blockers": built["blockers"]})

    def _existing_pipeline(self, job_id: str, runs: list[dict] | None) -> dict:
        job = self.store.get(job_id)
        workflow = (job.get("plan") or {}).get("workflow")
        observed = runs if runs is not None else []
        if not observed:
            try:
                observed = list_runs()
            except Exception as exc:  # noqa: BLE001
                self.store.transition(job_id, "BLOCKED", "could not read GitHub Actions", {"error": redact(str(exc))})
                return self.store.get(job_id)
        matched = runs_for_workflow(workflow, observed) if workflow else []
        active = [item for item in matched if item.get("status") in {"queued", "in_progress", "waiting"}]
        succeeded = [item for item in matched if item.get("conclusion") == "success"]
        failed = [item for item in matched if item.get("conclusion") in {"failure", "cancelled", "timed_out"}]
        summary = {
            "active": active[:3],
            "recent_success": succeeded[:2],
            "recent_failure": failed[:2],
            "dispatch": "not sent",
            "reason": "existing pipeline is monitored read-only; a new full build is not started automatically",
        }
        self.store.update_fields(job_id, result_json={"github": summary, "published": False})
        if active:
            if job["state"] == "WAITING_FOR_WORKER":
                self.store.event(job_id, job["state"], job["state"], "existing GitHub run remains active", summary)
            else:
                self.store.transition(job_id, "WAITING_FOR_WORKER", "existing GitHub run is active; not recomputing", summary)
        elif succeeded:
            self.store.transition(job_id, "BLOCKED", "historical successful workflow observed; source revision and release artifacts are not verified", summary)
        else:
            self.store.transition(job_id, "BLOCKED", "no successful run to reuse and dispatch is disabled", summary)
        return self.store.get(job_id)

    def _develop_or_block(self, job_id: str) -> dict:
        job = self.store.get(job_id)
        if job["state"] == "PLANNING":
            self.store.transition(job_id, "DEVELOPING", "no qualified adapter; preparing a bounded development brief")
        brief = self._write_brief(job_id)
        self.store.event(job_id, "DEVELOPING", "DEVELOPING", "native-only policy prevents custom converter patches", {
            "brief": str(brief), "repair_attempts": 0,
        })
        self._maybe_escalate(job_id, brief)
        job = self.store.get(job_id)
        if job["state"] not in TERMINAL:
            self.store.transition(
                job_id,
                "BLOCKED",
                "stock converter is not qualified for this route; no converter patch was attempted",
                {"brief": str(brief)},
            )
        return self.store.get(job_id)

    def _write_brief(self, job_id: str) -> Path:
        job = self.store.get(job_id)
        folder = self.work_dir / "adapters" / job_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "ADAPTER_BRIEF.json"
        payload = {
            "repo_id": job["repo_id"],
            "revision": job["revision"],
            "model_type": (job.get("analysis") or {}).get("model_type"),
            "formats": (job.get("analysis") or {}).get("formats"),
            "plan": job.get("plan"),
            "trust_remote_code": False,
            "status": "not_qualified",
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _maybe_escalate(self, job_id: str, brief: Path) -> None:
        client = DeepSeekClient(self.budget, transport=None)
        try:
            client.complete({"brief": str(brief)}, job_spent=0.0, confirmed=False)
        except EscalationDisabled as exc:
            self.store.event(job_id, "DEVELOPING", "DEVELOPING", "DeepSeek not called", {"reason": str(exc)})

    def render_card(self, job_id: str) -> str:
        job = self.store.get(job_id)
        validation = {
            "structure_valid": False,
            "runtime_compatible": None,
            "inference_tested": False,
            "summary": "No GGUF has been validated for this job.",
        }
        return render_readme(job.get("analysis") or {}, job.get("plan") or {}, validation, [])


def default_quants(raw: list[str] | None) -> list[str]:
    return normalize_quants(raw)
