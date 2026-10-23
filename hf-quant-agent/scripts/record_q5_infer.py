"""Record the observed Oracle Q5 inference. Does not rerun the model."""

import sys

sys.path.insert(0, "/opt/hf-quant-agent")
from hfqa.jobs import JobStore

store = JobStore("/var/lib/hf-quant-agent/data/jobs.sqlite")
job = store.get("6a00822c33584de0")
result = dict(job.get("result") or {})
result["inference_tested"] = True
result["published"] = False
result["inference"] = {
    "runtime": "llama.cpp b11669-620d2225c",
    "host": "oracle",
    "prompt": "The capital of France is",
    "completion": "The capital of France is Paris.",
    "prompt_tokens_per_second": 26.3,
    "generation_tokens_per_second": 15.3,
    "context": 64,
    "predict": 8,
    "threads": 1,
    "quant": "Q5_K_M",
    "note": "One short prompt only. Not a quality benchmark.",
}
store.update_fields("6a00822c33584de0", result_json=result)
store.event("6a00822c33584de0", "COMPLETED", "COMPLETED", "oracle Q5 inference recorded", result["inference"])
store.close()
print("recorded")
