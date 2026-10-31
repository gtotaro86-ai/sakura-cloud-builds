"""Create a matrix and pinned evidence for a local, no-upload stream proof.

Only the already-qualified single-file experimental runtime is allowed.
This still requires a materialized calibration model; it does not solve large
model calibration or authorize a release.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hfqa.imatrix_provenance import validate_matrix_provenance
from hfqa.runner_imatrix import make_matrix
from hfqa.runner_release import sha256_file
from hfqa.stream_imatrix import EXPERIMENTAL_LLAMA_REVISION
from scripts.run_stream_worker import atomic_json, read_manifest


def prepare(manifest_path, source_path, imatrix_tool, work, *, cpus=1):
    manifest, source = read_manifest(manifest_path)
    spec = manifest.get('imatrix') or {}
    if spec.get('method') != 'stream_imatrix' or manifest.get('runtime_revision') != EXPERIMENTAL_LLAMA_REVISION:
        raise ValueError('experimental stream manifest and qualified runtime required')
    if source.get('sources'):
        raise ValueError('one original source file is required for this proof')
    source_path, work = Path(source_path), Path(work)
    if source_path.stat().st_size != source['size'] or sha256_file(source_path) != source['sha256']:
        raise ValueError('calibration source does not match the pinned original')
    work.mkdir(parents=True, exist_ok=True)
    result = make_matrix(source_path, {'imatrix': Path(imatrix_tool)}, work, max(1, min(cpus, 4)))
    # Re-check the source after calibration to guard against accidental changes.
    if source_path.stat().st_size != source['size'] or sha256_file(source_path) != source['sha256']:
        raise ValueError('calibration source changed during matrix generation')
    evidence = dict(result['evidence'], source_sha256=source['sha256'], source_sha256_verified=True,
                    runtime_revision=manifest['runtime_revision'])
    validate_matrix_provenance(evidence, source=source,
        calibration={'id': spec.get('dataset_id'), 'sha256': spec.get('dataset_sha256'), 'context': spec.get('context')},
        runtime_revision=manifest['runtime_revision'])
    atomic_json(work / 'matrix-evidence.json', evidence)
    prepared = {'matrix': str(result['path']), 'evidence': str(work / 'matrix-evidence.json'), 'published': False}
    atomic_json(work / 'prepare-result.json', prepared)  # a workflow step reads the matrix path from here
    return prepared


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--imatrix-tool', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.manifest, a.source, a.imatrix_tool, a.work, cpus=os.cpu_count() or 1)))
