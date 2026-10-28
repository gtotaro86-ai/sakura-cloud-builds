"""Measure-only run: how far are given GGUF files from their full-precision source, on the pinned evaluation texts, with llama.cpp built from an exact commit.

Nothing is published and no secret is used: public files are downloaded at pinned revisions, the same `llama-perplexity` measurement as in a release
(`hfqa/runner_quality.py`) is run for all of them under identical conditions, and the numbers are written to one JSON file. It exists to compare quantizations
(ours and others') fairly, and to give an existing repository the measurement its card lacks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hfqa import runner_quality, source_builds
from scripts.run_imatrix_worker import source_build_evidence
from hfqa.measurement_spec import validate_manifest, file_descriptor, fingerprint
from hfqa.gguf import validate_gguf
from hfqa.remote_gguf import inspect_remote_gguf

MAX_CANDIDATES = 12
MAX_BYTES = 40 * 10 ** 9

def compatibility_signature(path):
    class Reader:
        def __init__(self,path):self.path=path;self.size=path.stat().st_size
        def read_range(self,offset,count):
            if not 0<=count<=8*1024**2 or offset<0 or offset+count>self.size:raise ValueError('GGUF header read outside bounds')
            with self.path.open('rb') as handle:handle.seek(offset);return handle.read(count)
    remote=inspect_remote_gguf({'size':path.stat().st_size},reader=Reader(path))
    arch=remote.metadata.get('general.architecture')
    # Perplexity scores raw text, so a different chat template does not change this experiment.
    identity=[(key,kind,hashlib.sha256(raw).hexdigest()) for key,kind,raw in remote.metadata_entries if key.startswith('tokenizer.ggml.') or key.startswith(str(arch)+'.')]
    vocab=next((struct.unpack_from('<Q',raw,4)[0] for key,kind,raw in remote.metadata_entries if key=='tokenizer.ggml.tokens' and kind==9 and len(raw)>=12),0)
    if not 0<vocab<=2_000_000:raise ValueError('reference/candidate vocabulary size cannot be bounded')
    return {'architecture':arch,'vocabulary_size':vocab,'tensor_shapes_sha256':fingerprint(sorted((t.name,list(t.shape)) for t in remote.tensors)),
            'model_metadata_sha256':fingerprint(sorted(identity))}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while chunk := handle.read(1024 ** 2):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(item: dict, folder: Path) -> Path:
    from huggingface_hub import hf_hub_download
    if not (isinstance(item.get('revision'), str) and len(item['revision']) == 40 and isinstance(item.get('repo_id'), str) and isinstance(item.get('filename'), str)):
        raise ValueError('every file must be pinned to an exact revision')
    file_descriptor(item)
    path = Path(hf_hub_download(item['repo_id'], item['filename'], revision=item['revision'], local_dir=folder, token=False))
    if item.get('size') is not None and path.stat().st_size != item['size']:
        raise ValueError(f"{item['filename']}: size differs from the pinned size")
    if _sha256(path)!=item['sha256']:raise ValueError('download differs from the pinned SHA-256')
    return path


def run(args) -> dict:
    if Path(args.manifest).stat().st_size>100000:raise ValueError('measurement manifest exceeds dispatch budget')
    manifest = validate_manifest(json.loads(Path(args.manifest).read_text(encoding='utf-8')))
    if not source_builds.is_source(manifest.get('runtime')) or not isinstance(manifest.get('candidates'), list) or not 0 < len(manifest['candidates']) <= MAX_CANDIDATES:
        raise ValueError('the manifest needs a source build and between one and twelve candidate files')
    labels = [item.get('label') for item in manifest['candidates']]
    if len(set(labels)) != len(labels) or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError('candidate labels must be unique')
    if sum(int(item.get('size') or 0) for item in manifest['candidates']) + int(manifest['source'].get('size') or 0) > MAX_BYTES:
        raise ValueError('the files are too large for one runner job')
    build = source_build_evidence(manifest, args.runtime_dir, args.build_receipt)
    work = Path(args.work)
    work.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(work).free<sum(i['size'] for i in [manifest['source'],*manifest['candidates']])+2*1024**3:raise RuntimeError('insufficient measured free disk for the pinned input set and reserve')
    source_path = fetch(manifest['source'], work / 'source')
    structure=validate_gguf(source_path)
    if not set(structure['tensor_type_histogram'])<={'F32','F16','BF16'}:raise ValueError('measurement reference must be full precision, not a quantized model')
    source_compatibility=compatibility_signature(source_path)
    # Conservative FP32 bound for the source probability cache plus bookkeeping.
    cache_budget=source_compatibility['vocabulary_size']*runner_quality.CONTEXT*runner_quality.chunks_for(manifest['source']['size'])*4+128*1024**2
    if shutil.disk_usage(work).free<sum(i['size'] for i in manifest['candidates'])+cache_budget+2*1024**3:raise RuntimeError('insufficient free disk for candidates, source probability cache and reserve')
    source_sha = _sha256(source_path)
    if manifest['source'].get('sha256') and source_sha != manifest['source']['sha256']:
        raise ValueError('the source file differs from its pinned SHA-256')
    outputs, files,compatibility = {}, {},{}
    for item in manifest['candidates']:
        path = fetch(item, work / 'files' / item['label'])
        candidate=validate_gguf(path,structure['architecture'])
        if any(candidate[k]!=structure[k] for k in ('tensor_count','tensor_name_set_sha256')):raise ValueError('candidate and reference have different tensor catalogues')
        signature=compatibility_signature(path)
        if signature!=source_compatibility:raise ValueError('candidate tensor shapes, tokenizer or architecture metadata differ from reference')
        compatibility[item['label']]=signature
        outputs[item['label']] = path
        files[item['label']] = {'repo_id': item['repo_id'], 'revision': item['revision'], 'filename': item['filename'], 'bytes': path.stat().st_size, 'sha256': _sha256(path)}
    tool = Path(args.runtime_dir) / 'llama-perplexity'
    if shutil.disk_usage(work).free<cache_budget+2*1024**3:raise RuntimeError('insufficient free disk for source probability cache and reserve')
    quality = runner_quality.measure(source_path, outputs, tool, work, os.cpu_count() or 1, source_path.stat().st_size)
    result = {'schema': 'hfqa-measure-1', 'runtime': build, 'source': {**{k: manifest['source'].get(k) for k in ('repo_id', 'revision', 'filename')}, 'bytes': source_path.stat().st_size, 'sha256': source_sha},
              'files': files, 'quality': quality, 'spec': runner_quality.SPEC,'manifest_sha256':fingerprint(manifest),
              'compatibility':{'source':source_compatibility,'candidates':compatibility},'source_probability_cache_budget_bytes':cache_budget,'chat_template_used':False}
    if _sha256(source_path)!=source_sha or any(_sha256(outputs[label])!=item['sha256'] for label,item in files.items()):raise ValueError('measurement inputs changed during evaluation')
    if _sha256(tool)!=build['binaries']['llama-perplexity']:raise ValueError('measurement tool changed during evaluation')
    folder = work / 'receipts'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'measure-results.json').write_text(json.dumps(result, indent=1, sort_keys=True, default=str), encoding='utf-8')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--runtime-dir', type=Path, required=True)
    parser.add_argument('--build-receipt', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:
        print(f'{type(error).__name__}: {str(error)[:1500]}', file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps({'event': 'measured', 'files': sorted(result['files']), 'skipped': (result['quality'] or {}).get('skipped')}), flush=True)


if __name__ == '__main__':
    main()
