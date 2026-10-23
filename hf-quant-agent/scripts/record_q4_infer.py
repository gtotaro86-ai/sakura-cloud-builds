"""Record the observed Oracle Q4 inference. Does not rerun the model."""

import sys

sys.path.insert(0, "/opt/hf-quant-agent")
from hfqa.jobs import JobStore

store = JobStore("/var/lib/hf-quant-agent/data/jobs.sqlite")
job = store.get("45b39bf3b2034702")
result = dict(job.get("result") or {})
result["inference_tested"] = True
result["published"] = False
result["inference"] = {
    "runtime": "llama.cpp b11669-620d2225c",
    "host": "oracle",
    "prompt": "The capital of France is",
    "completion": "The capital of France is Paris.",
    "prompt_tokens_per_second": 29.1,
    "generation_tokens_per_second": 17.2,
    "context": 64,
    "predict": 8,
    "threads": 1,
    "quant": "Q4_K_M",
    "note": "One short prompt only. Not a quality benchmark.",
}
store.update_fields("45b39bf3b2034702", result_json=result)
store.event("45b39bf3b2034702", "COMPLETED", "COMPLETED", "oracle Q4 inference recorded", result["inference"])
store.close()
print("recorded")
