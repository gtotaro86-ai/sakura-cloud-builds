"""Real Hugging Face inspection and planning. Does not publish or dispatch."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hfqa.inspect import inspect_repo
from hfqa.jobs import JobStore
from hfqa.orchestrator import Orchestrator

SAMPLES = [
    "https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash",
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct",
    "https://huggingface.co/bartowski/Spark-X2.6-4B-GGUF",
]


def main() -> None:
    data = ROOT / "data" / "live"
    store = JobStore(data / "jobs.sqlite")
    orch = Orchestrator(store, data / "work", data / "budget.json")
    results = []
    for url in SAMPLES:
        analysis = inspect_repo(url)
        job, created = store.create(url, analysis["repo_id"], ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M"], revision=analysis["revision"])
        if created or job["state"] == "CREATED":
            finished = orch.run(job["id"], resources={"free_disk_bytes": 36 * 1024**3, "available_ram_bytes": 8 * 1024**3})
        else:
            finished = job
        results.append({
            "url": url,
            "created": created,
            "state": finished["state"],
            "revision": finished.get("revision"),
            "model_type": (finished.get("analysis") or {}).get("model_type"),
            "formats": (finished.get("analysis") or {}).get("formats"),
            "license": (finished.get("analysis") or {}).get("license_decision"),
            "route": (finished.get("plan") or {}).get("route"),
            "blockers": (finished.get("plan") or {}).get("blockers"),
            "result": finished.get("result"),
        })
        print(json.dumps(results[-1], indent=2)[:2000])
    (data / "live-summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    store.close()


if __name__ == "__main__":
    main()
