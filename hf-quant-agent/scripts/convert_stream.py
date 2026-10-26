#!/usr/bin/env python3
"""Runner step of the BF16 conversion: Hugging Face safetensors -> GGUF (BF16), uploaded part by part, with a receipt.

The converter is the unmodified official `convert_hf_to_gguf.py` of the pinned llama.cpp revision. It reads the safetensors over HTTP
(`--remote`), writes its output as split GGUF files, and the only change is a wrapper around the loop that writes the parts: after a
part is complete its SHA-256 is taken, the file goes to the Hub and is deleted, so the runner never holds more than one part.

Before anything is uploaded: the manifest, the target namespace, the license decision, that the source revision is still the pinned one, that the
disk can hold one part, and that the target repository holds nothing but our own earlier parts. A card saying "being filled" goes up first, so the
repository is never public without one. A repeated run (after a timeout) skips every part the Hub already holds with the same size and SHA-256 and
refuses a part that differs. After the last part: the source is checked once more, the final card replaces the first, and every part's size and
LFS SHA-256 is read back from the Hub; the repository must hold exactly the parts of the receipt and no other GGUF.
Needs: torch (CPU), numpy, sentencepiece, transformers, protobuf, safetensors, requests, huggingface_hub, hf-xet, and the llama.cpp source tree."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import runpy
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
SCHEMA = 'hfqa.convert-receipt/1'
REPO = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}')
REVISION = re.compile(r'[a-f0-9]{40}')
STEM = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,120}')
OUTTYPES = {'bf16', 'f16'}
PARITY_REVISION = '620d2225cc436ab0b001e3200e590e3b286ef9ab'  # the llama.cpp revision with which the byte-identity check was made


class ConvertError(RuntimeError):
    pass


def sha256_file(path, block=8 * 1024 ** 2):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(block), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(manifest: dict) -> dict:
    need = {'job_id', 'model_repo_id', 'model_revision', 'outtype', 'split_max_size', 'target_repo_id', 'llama_revision', 'output_stem', 'source_license'}
    if not isinstance(manifest, dict) or need - set(manifest):
        raise ConvertError('manifest is incomplete')
    if not (REPO.fullmatch(manifest['model_repo_id']) and REPO.fullmatch(manifest['target_repo_id']) and REVISION.fullmatch(manifest['model_revision'])
            and REVISION.fullmatch(manifest['llama_revision']) and STEM.fullmatch(manifest['output_stem']) and re.fullmatch(r'[0-9]{1,3}G', manifest['split_max_size'])):
        raise ConvertError('manifest has an invalid field')
    if manifest['outtype'] not in OUTTYPES:
        raise ConvertError('output type is not allowed')
    if (manifest.get('source_license') or {}).get('redistributable') is not True:
        raise ConvertError('the license decision does not allow redistribution')
    if manifest['target_repo_id'].split('/')[0] != 'webmp3' or manifest.get('private'):
        raise ConvertError('the pinned worker is qualified only for public webmp3 targets')
    if manifest['model_repo_id'] == manifest['target_repo_id']:
        raise ConvertError('source and target are the same repository')
    return manifest


def part_pattern(manifest: dict):
    return re.compile(re.escape(f"{manifest['output_stem']}-{manifest['outtype'].upper()}") + r'(?:-\d{5}-of-\d{5})?\.gguf')


def split_bytes(manifest: dict) -> int:
    return int(manifest['split_max_size'][:-1]) * 1000 ** 3  # llama.cpp counts G as 10^9


class Shards:
    """Finished parts in order, each with size and SHA-256 taken from the complete file before it is handed over and deleted."""
    def __init__(self, upload):
        self.items, self.upload = [], upload

    def finished(self, path: Path):
        item = {'name': path.name, 'size': path.stat().st_size, 'sha256': sha256_file(path)}
        self.upload(path, item)
        path.unlink()
        self.items.append(item)
        print(json.dumps({'shard_done': item['name'], 'size': item['size']}), flush=True)


def patch_writer(llama: Path, shards: Shards):
    sys.path.insert(0, str(llama))
    sys.path.insert(0, str(llama / 'gguf-py'))
    from gguf import gguf_writer

    def write_tensors_to_file(self, *, progress: bool = False) -> None:
        # the official loop for the no-temp-file case; after each finished part the file is handed over
        if self.temp_file is not None:
            raise ConvertError('temporary-file mode is not supported')
        self.write_ti_data_to_file()
        for fout in self.fout:
            self.write_padding(fout, fout.tell())
        for fout, tensors in zip(self.fout, self.tensors):
            for name, info in tensors.items():
                if info.tensor is None or info.tensor.nbytes != info.nbytes:
                    raise ConvertError(f'tensor {name!r} is not ready')
                start = fout.tell()
                info.tensor.tofile(fout)
                if fout.tell() - start != info.nbytes:
                    raise ConvertError(f'tensor {name!r} wrote {fout.tell() - start} bytes, expected {info.nbytes}')
                self.write_padding(fout, info.nbytes)
                info.tensor = None
            fout.flush()
            os.fsync(fout.fileno())
            shards.finished(Path(fout.name))
        self.state = gguf_writer.WriterState.WEIGHTS

    gguf_writer.GGUFWriter.write_tensors_to_file = write_tensors_to_file


def patch_remote_probe():
    """The converter's HEAD probe for model.safetensors is only used to choose single-file or sharded layout; answer from the Hub's own listing when it says no."""
    from huggingface_hub import HfApi
    from gguf.utility import SafetensorRemote
    original = SafetensorRemote.check_file_exist.__func__
    api = HfApi()

    def check_file_exist(cls, url):
        if original(cls, url):
            return True
        found = re.match(r'https://huggingface\.co/(.+?)/resolve/main/(.+)$', url)
        return bool(found) and api.file_exists(found.group(1), found.group(2))
    SafetensorRemote.check_file_exist = classmethod(check_file_exist)


def front_matter(manifest: dict, license_id: str, extra: dict) -> str:
    meta = ''.join(f'\n{key}: {extra[key]}' for key in ('license_name', 'license_link') if extra.get(key))
    return f'''---
language:
- en
- zh
license: {license_id or 'other'}{meta}
base_model: {manifest['model_repo_id']}
base_model_relation: quantized
hfqa_source_revision: {manifest['model_revision']}
hfqa_converter_revision: {manifest['llama_revision']}
tags:
- gguf
---
'''


def placeholder_card(manifest: dict, license_id: str, extra: dict) -> str:
    """The card the repository carries while the parts are still being uploaded."""
    model, kind = manifest['model_repo_id'], manifest['outtype'].upper()
    return front_matter(manifest, license_id, extra) + f'''
# {manifest['output_stem']} · GGUF {kind} (being filled)

**This repository is still being filled and may be incomplete. Do not use it until this notice is gone.**

Independent community conversion of [{model}](https://huggingface.co/{model}) (commit `{manifest['model_revision']}`, license `{license_id or 'see original'}`) to GGUF {kind}, not quantized. Not an official release of the upstream author.

<details>
<summary>中文说明（点击展开）</summary>

**此仓库仍在上传中，内容可能不完整，在此提示消失之前请勿使用。**独立的社区转换版本，并非上游作者的官方发布：[{model}](https://huggingface.co/{model})。

</details>
'''


def card(manifest: dict, license_id: str, shards: list[dict], extra: dict | None = None) -> tuple[str, str]:
    """(English, Chinese) model card of the intermediate: what it is, where it comes from, what was and was not checked."""
    extra = extra or {}
    name, model, kind = manifest['output_stem'], manifest['model_repo_id'], manifest['outtype'].upper()
    limit_gb = int(manifest['split_max_size'][:-1])
    total = sum(item['size'] for item in shards)
    rows = chr(10).join(f"| `{item['name']}` | {item['size']} | `{item['sha256']}` |" for item in shards)
    parts = ('a single file' if len(shards) == 1 else f'{len(shards)} numbered files of at most {limit_gb} GB each; keep them together and load the first one')
    parts_zh = ('单个文件' if len(shards) == 1 else f'{len(shards)} 个带编号的文件，每个不超过 {limit_gb} GB；请放在一起并加载第一个')
    if manifest['llama_revision'] == PARITY_REVISION:
        parity = ('The same procedure produced tensors byte-identical to a local run of the same converter and to an independent third-party F16 GGUF for one reference model '
                  '(Qwen2.5-0.5B-Instruct, 290 of 290 tensors). For other models that has not been compared.')
        parity_zh = '同一流程在一个参考模型（Qwen2.5-0.5B-Instruct，290/290 个张量）上与同一转换器的本地运行结果、以及与独立第三方 F16 GGUF 逐字节一致；其他模型尚未比较。'
    else:
        parity = ('A byte-identity comparison with a local run and a third-party GGUF was made for one reference model (Qwen2.5-0.5B-Instruct) with another llama.cpp revision '
                  f'(`{PARITY_REVISION}`); it has not been repeated with this revision or for this model.')
        parity_zh = f'曾在另一个 llama.cpp 修订（`{PARITY_REVISION}`）下，对一个参考模型（Qwen2.5-0.5B-Instruct）做过与本地运行结果和第三方 GGUF 的逐字节比较；本修订及本模型尚未重复比较。'
    english = front_matter(manifest, license_id, extra) + f'''
# {name} · GGUF {kind}

Independent community conversion of [{model}](https://huggingface.co/{model}) (commit `{manifest['model_revision']}`, license `{license_id or 'see original'}`). Not an official release of the upstream author.

This is the full-precision {kind} GGUF, **not quantized**: {total / 1e9:.1f} GB in total, as {parts}.

| File | Bytes | SHA-256 |
|---|---:|---|
{rows}

## How it was made

The weights were read from the original safetensors and written to GGUF {kind} by the unmodified official `convert_hf_to_gguf.py` of llama.cpp (revision `{manifest['llama_revision']}`). Every file's SHA-256 was read back from the Hub after the upload.
{parity}
**No inference test and no quality measurement was done on this file.** Smaller quantizations made from it are tested separately.

## Usage

```bash
llama-cli -m <first file>.gguf -p "Hello" -n 32
```

Needs a llama.cpp build that supports this architecture.

<details>
<summary>中文说明（点击展开）</summary>

独立的社区转换版本，并非上游作者的官方发布：[{model}](https://huggingface.co/{model})（提交 `{manifest['model_revision']}`，许可证 `{license_id or '见原始模型'}`）。

这是全精度 {kind} GGUF，**未做量化**：共 {total / 1e9:.1f} GB，{parts_zh}。权重由官方未修改的 llama.cpp（修订 `{manifest['llama_revision']}`）的 `convert_hf_to_gguf.py` 从原始 safetensors 转换而来，上传后已从 Hub 回读每个文件的 SHA-256。**未对此文件做推理测试或质量评估。**

</details>
'''
    chinese = f'''# {name} · GGUF {kind}（中文说明）

[English version](README.md)

独立的社区转换版本，并非上游作者的官方发布：[{model}](https://huggingface.co/{model})（提交 `{manifest['model_revision']}`，许可证 `{license_id or '见原始模型'}`）。

这是全精度 {kind} GGUF，**未做量化**：共 {total / 1e9:.1f} GB，{parts_zh}。权重由官方未修改的 llama.cpp（修订 `{manifest['llama_revision']}`）的 `convert_hf_to_gguf.py` 从原始 safetensors 转换而来，上传后已从 Hub 回读每个文件的 SHA-256。{parity_zh}**未对此文件做推理测试或质量评估。**

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
{rows}
'''
    return english, chinese


def remote_files(api, repo: str) -> dict:
    """{name: (size, LFS sha256 or None)} of the repository's main branch as the Hub reports it."""
    info = api.model_info(repo, files_metadata=True)
    table = {}
    for entry in info.siblings or []:
        lfs = entry.lfs
        digest = (lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)) if lfs else None
        table[entry.rfilename] = (entry.size, digest)
    return table, info.sha


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--llama', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = validate_manifest(json.loads(args.manifest.read_text(encoding='utf-8')))
    token = os.environ.get('HF_TOKEN', '').strip()  # a secret often carries a trailing newline, an illegal header value
    if not token or not token.isascii() or any(ch.isspace() for ch in token):
        raise ConvertError('HF_TOKEN must be one ASCII token; value suppressed')
    os.environ['HF_TOKEN'] = token
    from huggingface_hub import HfApi
    from hfqa.target_policy import authorize_target  # Oracle approved the llama.cpp revision when it wrote the manifest
    api = HfApi(token=token)
    target, work = manifest['target_repo_id'], args.work
    authorize_target(target, api.whoami())
    work.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(work).free
    if free < split_bytes(manifest) * 1.1 + 3 * 10 ** 9:
        raise ConvertError(f'DISK_TOO_SMALL: {free / 1e9:.0f} GB free, one part of {manifest["split_max_size"]} plus working room does not fit')
    source = api.model_info(manifest['model_repo_id'])
    if source.sha != manifest['model_revision']:
        raise ConvertError('SOURCE_MOVED: the original repository is no longer at the pinned revision')
    card_data = getattr(source, 'card_data', None)
    card_data = dict(card_data.to_dict()) if card_data is not None and hasattr(card_data, 'to_dict') else dict(card_data or {})
    license_id = card_data.get('license')
    if isinstance(license_id, list):
        license_id = license_id[0] if license_id else None
    extra = {key: card_data.get(key) for key in ('license_name', 'license_link')}

    pattern = part_pattern(manifest)
    existing = {}
    if api.repo_exists(target):
        table, _ = remote_files(api, target)
        existing = {name: value for name, value in table.items() if name.endswith('.gguf')}
        foreign = [name for name in existing if not pattern.fullmatch(name)]
        if foreign:
            raise ConvertError('TARGET_NOT_EMPTY: the target repository holds GGUF files that are not parts of this conversion: ' + ', '.join(foreign[:3]))
    api.create_repo(target, repo_type='model', private=False, exist_ok=True)
    api.upload_file(path_or_fileobj=placeholder_card(manifest, license_id or '', extra).encode('utf-8'), path_in_repo='README.md', repo_id=target, repo_type='model',
                    commit_message='Add provisional card')

    def upload(path: Path, item: dict):
        held = existing.get(path.name)
        if held is not None:
            if held != (item['size'], item['sha256']):
                raise ConvertError(f'TARGET_DIFFERS: {path.name} already exists with different content')
            print(json.dumps({'shard_already_on_hub': path.name}), flush=True)
            return
        api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name, repo_id=target, repo_type='model', commit_message=f'Add {path.name}')
    shards = Shards(upload)
    (work / 'out').mkdir(parents=True, exist_ok=True)
    patch_writer(args.llama, shards)
    patch_remote_probe()
    started = time.time()
    sys.argv = ['convert_hf_to_gguf.py', '--remote', '--outtype', manifest['outtype'], '--split-max-size', manifest['split_max_size'],
                '--outfile', str(work / 'out' / f"{manifest['output_stem']}-{manifest['outtype'].upper()}.gguf"), manifest['model_repo_id']]
    try:
        runpy.run_path(str(args.llama / 'convert_hf_to_gguf.py'), run_name='__main__')
    except SystemExit as stop:
        if stop.code not in (None, 0):
            raise
    seconds = round(time.time() - started, 1)
    if not shards.items:
        raise ConvertError('the converter wrote no file')
    if api.model_info(manifest['model_repo_id']).sha != manifest['model_revision']:
        raise ConvertError('SOURCE_MOVED: the original repository changed while it was being converted; the files do not match the pinned revision')

    english, chinese = card(manifest, license_id or '', shards.items, extra)
    for name, text in (('README.md', english), ('README_zh.md', chinese)):
        api.upload_file(path_or_fileobj=text.encode('utf-8'), path_in_repo=name, repo_id=target, repo_type='model', commit_message=f'Add {name}')
    remote, hub_revision = remote_files(api, target)
    bad = [item['name'] for item in shards.items if remote.get(item['name']) != (item['size'], item['sha256'])]
    extra_parts = sorted(name for name in remote if name.endswith('.gguf') and name not in {item['name'] for item in shards.items})
    if bad or extra_parts or 'README.md' not in remote or 'README_zh.md' not in remote:
        raise ConvertError(f'read-back verification failed: differing {bad}, unexpected {extra_parts}')
    receipt = {'schema': SCHEMA, 'job_id': manifest['job_id'], 'model_repo_id': manifest['model_repo_id'], 'model_revision': manifest['model_revision'],
               'outtype': manifest['outtype'], 'llama_revision': manifest['llama_revision'], 'target_repo_id': target, 'hub_revision': hub_revision,
               'shards': shards.items, 'total_bytes': sum(item['size'] for item in shards.items), 'license': license_id, 'seconds': seconds,
               'read_back_verified': True, 'readme': True, 'inference_tested': False,
               'runner': {key: os.environ.get(key) for key in ('GITHUB_RUN_ID', 'GITHUB_REPOSITORY', 'GITHUB_SHA')}}
    (work / 'convert-receipt.json').write_text(json.dumps(receipt, indent=1), encoding='utf-8')
    print('RECEIPT ' + json.dumps({k: receipt[k] for k in ('hub_revision', 'total_bytes', 'seconds')}))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except ConvertError as error:
        print(f'CONVERT_FAILED {error}')
        sys.exit(1)
