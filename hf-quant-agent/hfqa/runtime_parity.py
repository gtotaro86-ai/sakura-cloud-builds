"""Stock parity for a candidate llama.cpp build: our streamed quantization must equal the build's own `llama-quantize`.

Both outputs are reduced to a catalogue of (tensor name -> type, shape, SHA-256 of the payload). Metadata keys that
differ between writers (for example the file layout) are deliberately not compared; every tensor byte is.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from hfqa.remote_gguf import inspect_remote_gguf

CHUNK = 8 * 1024 ** 2


class _FileReader:
    def __init__(self, path: Path):
        self.path, self.size = Path(path), Path(path).stat().st_size

    def read_range(self, start: int, count: int) -> bytes:
        with self.path.open('rb') as handle:
            handle.seek(start)
            return handle.read(count)


def tensor_catalogue(path) -> dict:
    path = Path(path)
    reader = _FileReader(path)
    descriptor = {'repo_id': 'local/file', 'revision': '0' * 40, 'filename': path.name, 'size': reader.size, 'sha256': '0' * 64}
    remote = inspect_remote_gguf(descriptor, reader=reader, allow_split=True)
    catalogue = {}
    for tensor in remote.tensors:
        digest, start, left = hashlib.sha256(), remote.data_offset + tensor.offset, tensor.nbytes
        while left:
            block = reader.read_range(start, min(CHUNK, left))
            if not block:
                raise ValueError('tensor payload is truncated: ' + tensor.name)
            digest.update(block)
            start += len(block)
            left -= len(block)
        catalogue[tensor.name] = {'type': tensor.type, 'shape': list(tensor.shape), 'sha256': digest.hexdigest()}
    return catalogue


def merged_catalogue(paths) -> dict:
    """Shards of one split output together make one catalogue; a tensor name must appear exactly once."""
    merged: dict = {}
    for path in paths:
        for name, entry in tensor_catalogue(path).items():
            if name in merged:
                raise ValueError('tensor appears in more than one shard: ' + name)
            merged[name] = entry
    return merged


def compare(streamed: dict, stock: dict) -> dict:
    only_streamed = sorted(set(streamed) - set(stock))
    only_stock = sorted(set(stock) - set(streamed))
    different = sorted(name for name in set(streamed) & set(stock) if streamed[name] != stock[name])
    return {'parity': not (only_streamed or only_stock or different), 'tensor_count': len(stock),
            'only_streamed': only_streamed[:20], 'only_stock': only_stock[:20], 'different': different[:20],
            'different_count': len(different)}


def stock_quantize(binary, source, output, quant: str, work_log, threads: int = 2, timeout: int = 1800) -> None:
    """The build's own llama-quantize, exactly as the Oracle path calls it."""
    command = [str(binary), '--allow-requantize', str(source), str(output), quant, str(threads)]
    with Path(work_log).open('wb') as log:
        done = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, timeout=timeout,
                              env={'LD_LIBRARY_PATH': str(Path(binary).resolve().parent), 'PATH': '/usr/bin:/bin'}, check=False)
    if done.returncode != 0 or not Path(output).is_file():
        raise RuntimeError(f'stock llama-quantize failed (exit {done.returncode}) for {quant}')
