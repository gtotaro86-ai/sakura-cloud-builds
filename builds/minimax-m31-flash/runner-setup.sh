#!/usr/bin/env bash
set -euo pipefail
python -m venv convert-env
convert-env/bin/pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
convert-env/bin/pip install --no-cache-dir -r requirements-convert.txt
python -m venv upload-env
upload-env/bin/pip install --no-cache-dir -r requirements-upload.txt
# Only runner-owned pip download cache, not toolchains or user files.
python -m pip cache purge || true
python - <<'PY'
import shutil,json
from source import inventory
needed=max(s['size'] for s in inventory()['weights'])+512*1024**2
free=shutil.disk_usage('.').free
print(json.dumps({'runner_free_bytes':free,'required_one_source_shard_bytes':needed}))
if free<needed:raise SystemExit('Insufficient standard-runner disk; no paid resource fallback')
PY
