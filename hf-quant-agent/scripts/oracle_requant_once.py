"""One bounded stock requant on the controller. Does not publish or patch."""

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/opt/hf-quant-agent")

from hfqa.jobs import JobStore
from hfqa.quantize import quantize_file

JOB_ID = "41b5b48696474a42"
SOURCE_URL = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf"
EXPECTED_BYTES = 675710816
WORK = Path("/home/ubuntu/hfqa/work/qwen-0.5b")
SOURCE = WORK / "qwen2.5-0.5b-instruct-q8_0.gguf"
OUTPUT = WORK / "qwen2.5-0.5b-instruct-Q2_K-requant.gguf"
QUANTIZE = Path("/home/ubuntu/hfqa/runtime/llama-b11669/llama-quantize")
DB = Path("/var/lib/hf-quant-agent/data/jobs.sqlite")


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    if not SOURCE.exists() or SOURCE.stat().st_size != EXPECTED_BYTES:
        subprocess.run(
            ["curl", "-L", "--retry", "5", "-C", "-", "-o", str(SOURCE), SOURCE_URL],
            check=True,
        )
    if SOURCE.stat().st_size != EXPECTED_BYTES:
        raise SystemExit(f"source size {SOURCE.stat().st_size} != {EXPECTED_BYTES}")
    os.nice(15)
    report = quantize_file(QUANTIZE, SOURCE, OUTPUT, "Q2_K", timeout_seconds=900, threads=1)
    if not report.get("structure_valid"):
        raise SystemExit(f"invalid gguf: {report}")
    store = JobStore(DB)
    job = store.get(JOB_ID)
    if job and job["state"] == "WAITING_FOR_WORKER":
        store.transition(JOB_ID, "QUANTIZING", "stock llama-quantize started", {"threads": 1})
        store.upsert_shard(JOB_ID, "Q2_K", 0, "done", checksum=report["sha256"], nbytes=report["bytes"], path=str(OUTPUT))
        store.transition(JOB_ID, "VALIDATING", "gguf structure checked", {"architecture": report.get("architecture")})
        store.update_fields(JOB_ID, result_json={
            "published": False,
            "quant": report,
            "inference_tested": False,
            "runtime": "stock llama.cpp b11669",
        })
        store.transition(JOB_ID, "COMPLETED", "structural requant complete; inference not run on Oracle", {"bytes": report["bytes"]})
    store.close()
    print({
        "bytes": report["bytes"],
        "architecture": report.get("architecture"),
        "tensor_count": report.get("tensor_count"),
        "sha256": report["sha256"],
        "structure_valid": report["structure_valid"],
        "inference_tested": report["inference_tested"],
        "job": JOB_ID,
    })


if __name__ == "__main__":
    main()
