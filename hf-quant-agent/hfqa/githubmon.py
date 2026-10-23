"""Read public GitHub Actions status. Dispatch is opt-in and never implied by a plan."""

from __future__ import annotations

import json
import urllib.request

DEFAULT_REPO = "gtotaro86-ai/sakura-cloud-builds"


def list_runs(repo: str = DEFAULT_REPO, limit: int = 15, timeout: float = 30) -> list[dict]:
    url = f"https://api.github.com/repos/{repo}/actions/runs?per_page={limit}"
    request = urllib.request.Request(url, headers={"User-Agent": "hf-quant-agent", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    runs = []
    for item in payload.get("workflow_runs") or []:
        runs.append({
            "id": item.get("id"),
            "name": item.get("name"),
            "status": item.get("status"),
            "conclusion": item.get("conclusion"),
            "html_url": item.get("html_url"),
            "path": item.get("path"),
            "head_sha": item.get("head_sha"),
            "created_at": item.get("created_at"),
            "updated_at": item.get("updated_at"),
            "display_title": item.get("display_title"),
        })
    return runs


def runs_for_workflow(workflow_file: str, runs: list[dict]) -> list[dict]:
    return [run for run in runs if str(run.get("path") or "").endswith(workflow_file)]
