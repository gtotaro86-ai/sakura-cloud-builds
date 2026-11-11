"""Phase A of a llama.cpp FORK build, on Modal (docs/FORK-BUILDS.md): the only place where the fork's code is built and run.

The GitHub job (the attested one) starts this function with scripts/run_fork_on_modal.py, which holds a Modal token but no Hugging Face key and runs nothing of the fork.
The function itself has no secret at all and cannot reach other Modal resources: it clones the pinned commit of the fork, builds the four tools, runs the quantizations and the
inference test (scripts/fork_phase.py) and leaves the files in a Modal Volume. Nothing it reports is trusted: the GitHub job downloads the files, and phase B
(scripts/run_imatrix_worker.py --phase stage) hashes them again, measures the tensor list against the source itself and only then uploads.

Return values of a Modal function are normally unpickled by the caller, which would let code in the container run code on the GitHub runner. Therefore the function returns
nothing but a short text through the restricted (non-pickle) channel and every exception is caught inside; the real result is `status.json` in the Volume, read as plain JSON.
"""
from __future__ import annotations

from pathlib import Path

import modal

from scripts import fork_phase

APP_NAME = 'hfqa-fork-build'
VOLUME_NAME = 'hfqa-fork-transfer'
TIMEOUT_S = 5 * 3600

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version='3.12')
    .apt_install('git', 'cmake', 'build-essential', 'ca-certificates')
    .pip_install('numpy==2.2.6', 'huggingface_hub==2.3.0', 'hf-xet==1.7.0')
    .add_local_dir(str(Path(__file__).resolve().parents[1]), fork_phase.APP_DIR, ignore=['**/.git', '**/__pycache__'])
)


@app.function(image=image, volumes={fork_phase.MOUNT: volume}, timeout=TIMEOUT_S, cpu=16, memory=32768, restrict_modal_access=True, max_containers=2, retries=0,
              include_source=True, _experimental_restrict_output=True)
def phase_a(manifest_json: str, job_id: str, github_env_json: str = '{}') -> str:
    """Never raises and returns only 'done' or 'failed'; the outcome is in <volume>/<job id>/status.json."""
    status = fork_phase.run_and_report(manifest_json, job_id, github_env_json, Path('/tmp/hfqa-fork'))
    try:
        volume.commit()
    except Exception:   # noqa: BLE001 - the volume is committed when the container ends anyway; the client waits for status.json
        pass
    return 'done' if status.get('ok') else 'failed'
