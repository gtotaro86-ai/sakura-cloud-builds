"""Runner step of the weighted stream release: ONE importance matrix for the whole job, from the files Oracle pinned in the manifest (no secrets, no upload).

The calibration basis is decided on Oracle (hfqa.matrix_basis) and written into the manifest: the pinned source itself, or the Q8_0 set of the same
repository revision when the source is too big for the disk. Every file is downloaded, checked against its pinned size and SHA-256, and the stock
`llama-imatrix` runs over them in packed passes (hfqa.runner_imatrix.make_matrix, measured: 72B Q8_0 in 27 minutes on the free runner). The evidence binds the
matrix to the identity of the pinned SOURCE and names the files it was really computed on; the quant jobs download matrix and evidence as an artifact.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hfqa.imatrix_provenance import validate_matrix_provenance
from hfqa.runner_imatrix import make_matrix
from hfqa.runner_release import open_source, sha256_file, source_structure
from hfqa.source_identity import identity
from hfqa.stream_imatrix import EXPERIMENTAL_LLAMA_REVISION
from scripts.run_stream_worker import atomic_json, read_manifest


def fetch_verified(descriptors: list[dict], directory: Path, *, download=None, log=print) -> Path:
    """Download every file of the set, check size and SHA-256, return the path of the first part (llama.cpp finds the others next to it)."""
    from huggingface_hub import hf_hub_download
    download = download or (lambda item, target: Path(hf_hub_download(repo_id=item['repo_id'], filename=item['filename'], revision=item['revision'], local_dir=str(target), token=False)))
    directory.mkdir(parents=True, exist_ok=True)
    first = None
    for item in sorted(descriptors, key=lambda d: d.get('split_index', 1)):
        path = Path(download(item, directory))
        if path.stat().st_size != item['size'] or sha256_file(path) != item['sha256']:
            raise RuntimeError(f"{item['filename']} differs from the pinned descriptor")
        log(json.dumps({'event': 'basis_file_verified', 'file': item['filename'], 'bytes': item['size']}), flush=True)
        first = first or path
    return first


def prepare(manifest_path, imatrix_tool, work, *, cpus=1, download=None, structure=None, log=print) -> dict:
    manifest, source = read_manifest(manifest_path)
    spec = manifest.get('imatrix') or {}
    if spec.get('method') != 'stream_imatrix' or manifest.get('runtime_revision') != EXPERIMENTAL_LLAMA_REVISION:
        raise ValueError('a weighted stream manifest with the qualified runtime is required')
    pinned = identity(source)
    basis = manifest.get('matrix_basis')
    if not isinstance(basis, dict) or basis.get('kind') not in {'source', 'q8_0_sibling'}:
        raise ValueError('the manifest names no calibration basis')
    files = (source.get('sources') or [source]) if basis['kind'] == 'source' else basis.get('files')
    if not files or any(not isinstance(item, dict) for item in files):
        raise ValueError('the calibration basis has no files')
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    evidence_basis = {'kind': 'source'}
    if basis['kind'] == 'q8_0_sibling':
        names = (structure or (lambda descriptors: source_structure(open_source({'sources': descriptors} if len(descriptors) > 1 else descriptors[0]))['tensor_name_set_sha256']))(files)
        if names != basis.get('tensor_name_set_sha256'):
            raise ValueError('the Q8_0 calibration files do not hold the tensors Oracle pinned')
        evidence_basis = {'kind': 'q8_0_sibling', 'tensor_name_set_sha256': names,
                          'files': [{key: item[key] for key in ('filename', 'size', 'sha256')} for item in files]}
    first = fetch_verified(files, work / 'basis', download=download, log=log)
    result = make_matrix(first, {'imatrix': Path(imatrix_tool)}, work, max(1, min(cpus, 4)), large=True, log=log)
    evidence = dict(result['evidence'], source_sha256=pinned['sha256'], source_sha256_verified=True, runtime_revision=manifest['runtime_revision'],
                    calibration_basis=evidence_basis)
    validate_matrix_provenance(evidence, source=pinned,
                               calibration={'id': spec.get('dataset_id'), 'sha256': spec.get('dataset_sha256'), 'context': spec.get('context')},
                               runtime_revision=manifest['runtime_revision'])
    atomic_json(work / 'matrix-evidence.json', evidence)
    prepared = {'matrix': str(result['path']), 'evidence': str(work / 'matrix-evidence.json'), 'published': False}
    atomic_json(work / 'prepare-result.json', prepared)
    return prepared


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--imatrix-tool', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.imatrix_tool, args.work, cpus=os.cpu_count() or 1)))
