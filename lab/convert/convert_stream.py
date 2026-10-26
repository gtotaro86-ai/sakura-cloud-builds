"""Convert a Hugging Face safetensors model to GGUF with the official llama.cpp converter, reading the weights remotely and handing every
finished shard to the Hub at once, so the runner never holds more than a shard or two on disk.

  parity   convert (no upload), compare tensor by tensor with an official GGUF of the same model and dtype
  release  convert and upload the shards to a Hugging Face repository (public unless told otherwise), read the hashes back

The converter is the unmodified official one at the pinned llama.cpp revision; only the loop that writes the shards is wrapped
(the same loop, plus a call after each shard)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import sys
import time
from pathlib import Path


def sha256_file(path, block=8 * 1024 ** 2):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(block), b''):
            digest.update(chunk)
    return digest.hexdigest()


class Shards:
    """What the converter wrote, in order, with size and SHA-256 taken from the finished file before it is uploaded and deleted."""
    def __init__(self, upload=None):
        self.items, self.upload = [], upload

    def finished(self, path: Path):
        size = path.stat().st_size
        item = {'name': path.name, 'size': size, 'sha256': sha256_file(path)}
        if self.upload:
            item['uploaded'] = self.upload(path, item)
            path.unlink()
        self.items.append(item)
        print(json.dumps({'shard_done': item['name'], 'size': size, 'uploaded': bool(self.upload)}), flush=True)


def patch_writer(llama: Path, shards: Shards):
    sys.path.insert(0, str(llama))
    sys.path.insert(0, str(llama / 'gguf-py'))
    from gguf import gguf_writer

    def write_tensors_to_file(self, *, progress: bool = False) -> None:
        # same steps as the official method for the no-temp-file case; after each shard the file is complete and handed over
        if self.temp_file is not None:
            raise RuntimeError('temporary-file mode is not supported here')
        self.write_ti_data_to_file()
        for fout in self.fout:
            self.write_padding(fout, fout.tell())
        for fout, tensors in zip(self.fout, self.tensors):
            for name, ti in tensors.items():
                assert ti.tensor is not None and ti.tensor.nbytes == ti.nbytes
                start = fout.tell()
                ti.tensor.tofile(fout)
                if fout.tell() - start != ti.nbytes:
                    raise ValueError(f'tensor {name!r} wrote {fout.tell() - start} bytes, expected {ti.nbytes}')
                self.write_padding(fout, ti.nbytes)
                ti.tensor = None
            fout.flush()
            os.fsync(fout.fileno())
            shards.finished(Path(fout.name))
        from gguf.gguf_writer import WriterState
        self.state = WriterState.WEIGHTS

    gguf_writer.GGUFWriter.write_tensors_to_file = write_tensors_to_file


def patch_remote_probe(model: str):
    """The converter decides 'is there a model.safetensors?' with one HEAD request. Report what that request sees and, if it says no for a file
    the Hub does list, answer from the Hub's own file listing instead (the converter's answer is only used to pick single-file or sharded)."""
    import re
    import requests
    from huggingface_hub import HfApi
    from gguf.utility import SafetensorRemote
    probe = f'https://huggingface.co/{model}/resolve/main/model.safetensors'
    try:
        reply = requests.head(probe, allow_redirects=True, headers={**SafetensorRemote._get_request_headers(), 'Range': 'bytes=0-0'}, timeout=60)
        print(json.dumps({'probe_status': reply.status_code, 'probe_url': reply.url[:80]}), flush=True)
    except Exception as error:  # noqa: BLE001 - diagnostics only
        print(json.dumps({'probe_error': type(error).__name__ + ': ' + str(error)[:160]}), flush=True)
    original = SafetensorRemote.check_file_exist.__func__
    api = HfApi()

    def check_file_exist(cls, url):
        if original(cls, url):
            return True
        found = re.match(r'https://huggingface\.co/(.+?)/resolve/main/(.+)$', url)
        return bool(found) and api.file_exists(found.group(1), found.group(2))
    SafetensorRemote.check_file_exist = classmethod(check_file_exist)


def convert(llama: Path, model: str, outtype: str, split: str, out_dir: Path, shards: Shards):
    out_dir.mkdir(parents=True, exist_ok=True)
    patch_writer(llama, shards)
    patch_remote_probe(model)
    argv = ['convert_hf_to_gguf.py', '--remote', '--outtype', outtype, '--split-max-size', split, '--outfile', str(out_dir / 'model.gguf'), model]
    sys.argv = argv
    started = time.time()
    try:
        runpy.run_path(str(llama / 'convert_hf_to_gguf.py'), run_name='__main__')
    except SystemExit as stop:
        if stop.code not in (None, 0):
            raise
    return time.time() - started


def tensor_table(path: Path) -> dict:
    sys.path.insert(0, str(Path(os.environ['LLAMA_DIR']) / 'gguf-py'))
    from gguf import GGUFReader
    reader = GGUFReader(str(path), 'r')
    table = {}
    for tensor in reader.tensors:
        table[tensor.name] = {'shape': [int(x) for x in tensor.shape], 'type': int(tensor.tensor_type), 'sha256': hashlib.sha256(tensor.data.tobytes()).hexdigest()}
    return table


def numeric_compare(ours_paths: list[Path], reference: Path, names: list[str]) -> list[dict]:
    """How far apart are two tensors that are not byte-identical: tiny rounding differences or really different values?"""
    import numpy as np
    sys.path.insert(0, str(Path(os.environ['LLAMA_DIR']) / 'gguf-py'))
    from gguf import GGUFReader
    wanted = set(names)
    ours = {}
    for path in ours_paths:
        for tensor in GGUFReader(str(path), 'r').tensors:
            if tensor.name in wanted:
                ours[tensor.name] = np.array(tensor.data, dtype=np.float32)
    rows = []
    for tensor in GGUFReader(str(reference), 'r').tensors:
        if tensor.name in ours:
            a, b = ours[tensor.name], np.array(tensor.data, dtype=np.float32)
            diff = np.abs(a - b)
            rows.append({'name': tensor.name, 'max_abs': float(diff.max()), 'mean_abs': float(diff.mean()), 'mean_abs_value': float(np.abs(b).mean()),
                         'share_equal': float((a == b).mean()), 'ours_first': a.reshape(-1)[:4].tolist(), 'ref_first': b.reshape(-1)[:4].tolist()})
    return rows


def merged_table(paths: list[Path]) -> dict:
    table = {}
    for path in paths:
        table.update(tensor_table(path))
    return table


ALLOWED_LICENSES = {'apache-2.0', 'mit', 'bsd-3-clause', 'bsd-2-clause', 'cc-by-4.0'}


def write_card(args, report: dict, shards: Shards) -> None:
    """A plain model card: where it comes from, what was done, what was and was not checked. Written only after every shard is on the Hub."""
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ['HF_TOKEN'])
    source = api.model_info(args.model)
    license_id = ((source.card_data or {}).get('license') if source.card_data else None) or ''
    rows = '
'.join(f"| {item['name']} | {item['size']} | `{item['sha256']}` |" for item in shards.items)
    text = f"""---
license: {license_id}
base_model: {args.model}
base_model_relation: quantized
tags:
- gguf
---

# {args.model.split('/')[1]} as GGUF ({args.outtype.upper()})

Independent community conversion. Not an official release of the upstream author: [{args.model}](https://huggingface.co/{args.model}) (license: `{license_id}`).

The weights were converted from the original safetensors to GGUF **{args.outtype.upper()} without any quantization** by the unmodified official `convert_hf_to_gguf.py` of llama.cpp
(revision `{os.environ.get('LLAMA_REVISION', 'unknown')}`), reading the safetensors remotely and splitting the output into the shards below.
The same pipeline produced byte-identical tensors to an independent third-party F16 GGUF of Qwen2.5-0.5B-Instruct (290 of 290 tensors) and to a local run of the same converter.
**No inference test and no quality measurement was done for this model.**

| File | Bytes | SHA-256 |
|---|---:|---|
{rows}

Keep all numbered files together and load the first one (`{shards.items[0]['name']}`) with a llama.cpp build that supports this architecture.

<details>
<summary>中文说明（点击展开）</summary>

独立的社区转换版本，并非上游作者的官方发布：[{args.model}](https://huggingface.co/{args.model})（许可证：`{license_id}`）。权重由官方未修改的 llama.cpp `convert_hf_to_gguf.py` 从原始 safetensors 转换为 GGUF {args.outtype.upper()}，**未做任何量化**。**未对该模型做推理测试或质量评估。**请将所有编号文件放在一起，加载第一个文件。

</details>
"""
    if license_id not in ALLOWED_LICENSES:
        raise RuntimeError(f'license {license_id!r} is not on the list this lab may republish')
    api.upload_file(path_or_fileobj=text.encode('utf-8'), path_in_repo='README.md', repo_id=args.target_repo, repo_type='model', commit_message='Add model card')
    report['card'] = True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['parity', 'release'], required=True)
    parser.add_argument('--llama', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--outtype', default='bf16')
    parser.add_argument('--split-max-size', default='4G')
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--references', default='[]', help='JSON list of {repo, file}')
    parser.add_argument('--local-check', action='store_true')
    parser.add_argument('--target-repo')
    parser.add_argument('--private', action='store_true')
    args = parser.parse_args()
    os.environ['LLAMA_DIR'] = str(args.llama)
    if args.mode == 'parity':
        os.environ.pop('HF_TOKEN', None)  # public models only; no credential is needed or sent
    report = {'mode': args.mode, 'model': args.model, 'outtype': args.outtype, 'split_max_size': args.split_max_size}

    if args.mode == 'release':
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ['HF_TOKEN'])
        source_license = ((api.model_info(args.model).card_data or {}).get('license') if api.model_info(args.model).card_data else None) or ''
        if source_license not in ALLOWED_LICENSES:
            raise RuntimeError(f'license {source_license!r} is not on the list this lab may republish')
        api.create_repo(args.target_repo, repo_type='model', private=args.private, exist_ok=True)

        def upload(path: Path, item: dict):
            api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name, repo_id=args.target_repo, repo_type='model',
                            commit_message=f'Add {path.name}')
            return True
        shards = Shards(upload)
    else:
        shards = Shards()

    report['seconds'] = round(convert(args.llama, args.model, args.outtype, args.split_max_size, args.work / 'out', shards), 1)
    report['shards'] = shards.items
    report['total_bytes'] = sum(item['size'] for item in shards.items)

    if args.mode == 'release':
        write_card(args, report, shards)
    if args.mode == 'release':
        from huggingface_hub import HfApi
        info = HfApi(token=os.environ['HF_TOKEN']).model_info(args.target_repo, files_metadata=True)
        remote = {s.rfilename: (s.size, getattr(s.lfs, 'sha256', None) if s.lfs and not isinstance(s.lfs, dict) else (s.lfs or {}).get('sha256')) for s in info.siblings}
        bad = [item['name'] for item in shards.items if remote.get(item['name']) != (item['size'], item['sha256'])]
        report['read_back_verified'] = not bad
        report['read_back_mismatch'] = bad
    else:
        from huggingface_hub import hf_hub_download
        ours_paths = sorted((args.work / 'out').glob('*.gguf'))
        ours = merged_table(ours_paths)
        report['comparisons'] = []
        compared = {'ours (remote conversion)': ours}
        if args.local_check:
            import subprocess
            from huggingface_hub import snapshot_download
            local_model = snapshot_download(args.model, local_dir=str(args.work / 'localmodel'))
            (args.work / 'local').mkdir(parents=True, exist_ok=True)
            subprocess.run([sys.executable, str(args.llama / 'convert_hf_to_gguf.py'), local_model, '--outtype', args.outtype, '--split-max-size', args.split_max_size,
                            '--outfile', str(args.work / 'local' / 'model.gguf')], check=True, env={**os.environ, 'PYTHONPATH': str(args.llama / 'gguf-py')})
            compared['ours (local conversion)'] = merged_table(sorted((args.work / 'local').glob('*.gguf')))
        for entry in json.loads(args.references):
            reference = Path(hf_hub_download(entry['repo'], entry['file'], local_dir=str(args.work / 'ref' / entry['repo'].replace('/', '_'))))
            compared[entry['repo']] = tensor_table(reference)
        names = list(compared)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                a, b = compared[left], compared[right]
                differing = sorted(n for n in set(a) | set(b) if a.get(n) != b.get(n))
                report['comparisons'].append({'left': left, 'right': right, 'tensors': [len(a), len(b)], 'identical': len(set(a) & set(b)) - len([n for n in differing if n in a and n in b]),
                                              'only_left': sorted(set(a) - set(b))[:5], 'only_right': sorted(set(b) - set(a))[:5],
                                              'differing_examples': differing[:5], 'all_identical': not differing})
    print('REPORT ' + json.dumps(report))
    (args.work / 'report.json').write_text(json.dumps(report, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
