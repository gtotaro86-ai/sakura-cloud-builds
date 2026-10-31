"""Value parity at scale: a released streamed quant against the stock quantizer with the very same matrix (measurement only, nothing is written to the Hub).

The source (split, as pinned) is downloaded, the unchanged `llama-quantize --imatrix` makes the reference file on this machine, and every tensor of the
streamed file (read from the Hub by range requests, hashed on the fly, never stored) is compared with the same tensor of the reference: type and SHA-256 of the data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hfqa.remote_gguf import RangeReader, inspect_remote_gguf


def local_digests(path: Path) -> dict:
    from gguf import GGUFReader
    return {tensor.name: (tensor.tensor_type.name, hashlib.sha256(tensor.data.tobytes()).hexdigest()) for tensor in GGUFReader(path).tensors}


def remote_digests(descriptor: dict, token: str | None) -> dict:
    from hfqa.gguf import TENSOR_TYPES
    from hfqa.quant_plan import authorized_opener
    reader = RangeReader(descriptor, opener=authorized_opener(token)) if token else RangeReader(descriptor)
    info = inspect_remote_gguf(descriptor, reader=reader, allow_split=True)
    digests = {}
    chunk = 8 * 1024 ** 2
    for tensor in info.tensors:
        digest = hashlib.sha256()
        offset, end = info.data_offset + tensor.offset, info.data_offset + tensor.offset + tensor.nbytes
        while offset < end:
            count = min(chunk, end - offset)
            digest.update(reader.read_range(offset, count))
            offset += count
        digests[tensor.name] = (TENSOR_TYPES[tensor.type], digest.hexdigest())
    return digests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool-dir', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download
    request = json.loads(args.request.read_text(encoding='utf-8'))
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(args.tool_dir) + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    args.work.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    first = None
    for item in sorted(request['sources'], key=lambda d: d['split_index']):
        path = Path(hf_hub_download(repo_id=item['repo_id'], filename=item['filename'], revision=item['revision'], local_dir=str(args.work / 'src'), token=False))
        first = first or path
    download_seconds = round(time.monotonic() - started, 1)
    reference = args.work / 'reference.gguf'
    t0 = time.monotonic()
    done = subprocess.run([str(args.tool_dir / 'llama-quantize'), '--imatrix', str(args.matrix), str(first), str(reference), request['preset'], str(os.cpu_count() or 4)],
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, env=env, timeout=4 * 3600, check=False)
    quantize_seconds = round(time.monotonic() - t0, 1)
    report = {'preset': request['preset'], 'download_seconds': download_seconds, 'stock_quantize_seconds': quantize_seconds, 'stock_returncode': done.returncode}
    if done.returncode != 0:
        report['stock_tail'] = done.stderr[-500:]
        args.report.write_text(json.dumps(report, indent=1), encoding='utf-8')
        raise SystemExit(1)
    for item in request['sources']:
        (args.work / 'src' / item['filename']).unlink(missing_ok=True)
    t1 = time.monotonic()
    stock = local_digests(reference)
    target = request['target']
    streamed = remote_digests({'repo_id': target['repo_id'], 'revision': target['revision'], 'filename': target['filename'], 'size': target['size']}, (os.environ.get('HF_TOKEN') or '').strip() or None)  # the repository secret carries a trailing newline
    different = sorted(name for name in set(stock) & set(streamed) if stock[name] != streamed[name])
    report.update(compare_seconds=round(time.monotonic() - t1, 1), tensors_stock=len(stock), tensors_streamed=len(streamed),
                  only_in_stock=sorted(set(stock) - set(streamed))[:10], only_in_streamed=sorted(set(streamed) - set(stock))[:10],
                  different_count=len(different), different_examples=[{'name': n, 'stock': stock[n][0], 'streamed': streamed[n][0]} for n in different[:12]],
                  identical=not different and set(stock) == set(streamed), reference_bytes=reference.stat().st_size)
    args.report.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('identical', 'different_count', 'tensors_stock', 'tensors_streamed')}), flush=True)


if __name__ == '__main__':
    main()
