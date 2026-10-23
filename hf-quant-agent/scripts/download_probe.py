"""One bounded real download probe into an isolated cache; no quantization or upload."""
import json, os
from pathlib import Path
from hfqa.jobs import JobStore
from hfqa.worker import source_descriptor, prepare_source
os.environ['HFQA_SOURCE_DIR']='/var/lib/hf-quant-agent/download-probe'
with JobStore('/var/lib/hf-quant-agent/data/jobs.sqlite') as store:
    jid=Path('/var/lib/hf-quant-agent/e2e-job-id').read_text().strip()
    job=store.get(jid)
    descriptor=source_descriptor(job['analysis'])
    path=prepare_source({'download':descriptor},job)
    print(json.dumps({'downloaded':str(path),'bytes':path.stat().st_size,'sha256':descriptor['sha256']}))
