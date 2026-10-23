"""Bounded local Spark tool loop. Deterministic gates retain final authority."""
from __future__ import annotations
import json
import os
import urllib.request
from hfqa.security import redact

TOOLS = [{"type": "function", "function": {
    "name": "read_pinned_analysis", "description": "Read the pinned, validated source analysis and quantization plan for the current job.",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}}]


def advise(store, job_id: str, transport=None) -> dict:
    job = store.get(job_id)
    if any(e["message"] == "Spark planning result" for e in store.events_for(job_id)):
        return {"reused": True}
    if transport is None and os.environ.get("HFQA_SPARK_ENABLED", "1") != "1":
        return {"enabled": False}
    base = os.environ.get("HFQA_SPARK_URL", "http://127.0.0.1:8088").rstrip("/")
    if base != "http://127.0.0.1:8088":
        raise ValueError("Spark may only use the approved loopback service")
    def post(payload):
        if transport:
            return transport(payload)
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=55) as response:
            return json.loads(response.read(128000))
    common = {"model": "spark-x2.6-4b", "tools": TOOLS, "temperature": 0,
              "max_tokens": 48, "chat_template_kwargs": {"enable_thinking": False}}
    messages = [{"role": "user", "content": "Call read_pinned_analysis. Then state the safe next step. Native stock llama.cpp only."}]
    try:
        first = post({**common, "messages": messages, "tool_choice": "required"})
        msg = first["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if len(calls) != 1 or calls[0].get("function", {}).get("name") != "read_pinned_analysis":
            raise ValueError("Spark did not choose the permitted read-only tool")
        args = json.loads(calls[0]["function"].get("arguments") or "{}")
        if args != {}:
            raise ValueError("tool accepts no model-controlled arguments")
        analysis, plan = job.get("analysis") or {}, job.get("plan") or {}
        data = {"repo": analysis.get("repo_id"), "revision": job.get("revision"),
                "architecture": analysis.get("model_type"), "route": plan.get("route"),
                "quants": job["quants"], "allow_patch": False, "publish": job["publish"],
                "publication_gate": plan.get("publish_allowed")}
        store.event(job_id, job["state"], job["state"], "Spark tool executed", {"tool": "read_pinned_analysis", "result": data})
        messages += [msg, {"role": "tool", "tool_call_id": calls[0].get("id"), "content": json.dumps(data)}]
        second = post({**common, "messages": messages, "tool_choice": "none"})
        summary = redact(second["choices"][0]["message"].get("content") or "")[:1000]
        result = {"ok": True, "tool": "read_pinned_analysis", "advice": summary, "decision_authority": "validated deterministic plan"}
    except Exception as exc:
        result = {"ok": False, "error": redact(str(exc))[:400], "fallback": "validated deterministic plan"}
    current = store.get(job_id)
    store.event(job_id, current["state"], current["state"], "Spark planning result", result)
    return result
