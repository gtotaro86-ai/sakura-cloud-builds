"""Record the already observed Oracle inference. Does not rerun the model."""

import sys
from pathlib import Path

sys.path.insert(0, "/opt/hf-quant-agent")
from hfqa.jobs import JobStore

store = JobStore("/var/lib/hf-quant-agent/data/jobs.sqlite")
job = store.get("41b5b48696474a42")
result = dict(job.get("result") or {})
result["inference_tested"] = True
result["inference"] = {
    "runtime": "llama.cpp b11669-620d2225c",
    "host": "oracle",
    "prompt": "The capital of France is",
    "completion": "The capital of France is Paris.",
    "prompt_tokens_per_second": 48.4,
    "generation_tokens_per_second": 23.6,
    "context": 64,
    "predict": 8,
    "threads": 1,
    "note": "One short prompt only. Not a quality benchmark.",
}
store.update_fields("41b5b48696474a42", result_json=result)
store.event("41b5b48696474a42", "COMPLETED", "COMPLETED", "oracle inference recorded", result["inference"])
store.close()
print("recorded")
