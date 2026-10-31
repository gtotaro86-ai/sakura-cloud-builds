"""An independent check of WHICH TYPE every tensor of a streamed quantization must have, for any model, without its weights.

Our own streaming quantizer decides nothing about types by itself: it applies what llama.cpp's quantizer would choose. The unchanged stock `llama-quantize`
can tell that choice without any weights (`--dry-run`, measured on a hosted runner: identical for the real file and for a SKELETON of it, and identical to the
types of a real quantization, for three architectures). The skeleton is the real header of the source followed by a hole up to the real size: a few MB of
disk for a model of any size.

Oracle builds the skeleton from the pinned source's header (HTTP range, never the weights), asks the stock tool for the plan, reads the tensor types of
the staged output files from THEIR headers on the Hub (again range reads), and compares name by name. A runner's own statement is not involved.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from hfqa.gguf import TENSOR_TYPES
from hfqa.remote_gguf import MAX_RANGE_BYTES, RangeReader, inspect_remote_gguf

LINE = re.compile(r'\[\s*\d+/\s*\d+\]\s+(\S+)\s+-\s+\[[^\]]*\],\s+type\s*=\s*(\w+),\s*size\s*=\s*[\d.]+\s*MiB(?:\s*->\s*[\d.]+\s*MiB\s*\((\w+)\))?')
FULL_PRECISION = {'f32', 'f16', 'bf16'}
DRY_RUN_TIMEOUT_S = 300
DEFAULT_QUANTIZE = '/opt/hf-quant-agent/runtime/llama-b11669/llama-quantize'  # the stock runtime the service user can read (the same b11669 build the local workers use)


def authorized_opener(token: str):
    """A range-reader opener for a private target repository. The credential is an *unredirected* header: urllib sends it to the Hub only and never
    forwards it when the Hub redirects the file request to its storage host."""
    def opener(request, timeout=45):
        request.add_unredirected_header('Authorization', 'Bearer ' + token)
        return urllib.request.urlopen(request, timeout=timeout)
    return opener


def parse_plan(text: str) -> dict[str, str]:
    """{tensor name: target type} from the per-tensor lines of `llama-quantize --dry-run` (a tensor that is not converted keeps its type)."""
    plan = {}
    for line in text.splitlines():
        found = LINE.search(line)
        if found:
            plan[found.group(1)] = (found.group(3) or found.group(2)).upper()
    return plan


def build_skeleton(descriptor: dict, directory: Path, *, reader=None) -> Path:
    """The real header of one source file followed by a hole to its real size. Only the header is read from the Hub."""
    reader = reader or RangeReader(descriptor)
    info = inspect_remote_gguf(descriptor, reader=reader, allow_split=True)
    path = Path(directory) / Path(descriptor['filename']).name
    with path.open('wb') as handle:
        offset = 0
        while offset < info.data_offset:
            count = min(MAX_RANGE_BYTES, info.data_offset - offset)
            handle.write(reader.read_range(offset, count))
            offset += count
    os.truncate(path, int(descriptor['size']))
    return path


def expected_plan(descriptors: list[dict], quant: str, *, quantize_bin: str | Path | None = None, source_precision: str | None = None,
                  runner=subprocess.run, reader_factory=None, timeout: int = DRY_RUN_TIMEOUT_S) -> dict[str, str]:
    """The stock quantizer's own plan for these source files and this preset."""
    quantize = Path(quantize_bin or os.environ.get('HFQA_QUANTIZE_BIN') or DEFAULT_QUANTIZE)
    if not quantize.is_file():
        raise RuntimeError('the stock quantizer for the type plan is not installed')
    ordered = sorted(descriptors, key=lambda item: item.get('split_index', 1))
    with tempfile.TemporaryDirectory(prefix='hfqa-plan-') as tmp:
        skeletons = [build_skeleton(item, Path(tmp), reader=reader_factory(item) if reader_factory else None) for item in ordered]
        command = [str(quantize), '--dry-run'] + ([] if (source_precision or '').lower() in FULL_PRECISION else ['--allow-requantize']) + [str(skeletons[0]), quant]
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = str(quantize.parent) + os.pathsep + env.get('LD_LIBRARY_PATH', '')
        completed = runner(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout, env=env, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f'the stock dry run failed (exit {completed.returncode})')
        plan = parse_plan((completed.stdout or '') + '\n' + (completed.stderr or ''))
    if not plan:
        raise RuntimeError('the stock dry run printed no tensor plan')
    return plan


def staged_types(shards: list[dict], staging: dict, repo_id: str, *, reader_factory=None, token: str | None = None) -> dict[str, str]:
    """{tensor name: type} of the staged output files, from their own headers on the Hub (the commit that staging recorded)."""
    types: dict[str, str] = {}
    for shard in shards:
        descriptor = {'repo_id': repo_id, 'revision': staging['commit_oid'], 'filename': shard['name'], 'size': shard['size']}
        reader = reader_factory(descriptor) if reader_factory else RangeReader(descriptor, opener=authorized_opener(token)) if token else None
        info = inspect_remote_gguf(descriptor, reader=reader, allow_split=True)
        for tensor in info.tensors:
            if tensor.name in types:
                raise ValueError('a tensor appears in two output files')
            name = TENSOR_TYPES.get(tensor.type)
            if name is None:
                raise ValueError(f'an output tensor has an unknown type id {tensor.type}')
            types[tensor.name] = name.upper()
    return types


def differences(plan: dict[str, str], actual: dict[str, str]) -> list[str]:
    problems = []
    missing, extra = sorted(set(plan) - set(actual)), sorted(set(actual) - set(plan))
    if missing:
        problems.append(f'{len(missing)} planned tensors are missing from the output (for example {missing[0]})')
    if extra:
        problems.append(f'{len(extra)} output tensors are not in the plan (for example {extra[0]})')
    wrong = sorted(name for name in set(plan) & set(actual) if plan[name] != actual[name])
    if wrong:
        problems.append(f'{len(wrong)} tensors have another type than the stock quantizer would give (for example {wrong[0]}: {actual[wrong[0]]} instead of {plan[wrong[0]]})')
    return problems


def check_release(manifest: dict, quant: str, shards: list[dict], staging: dict, *, plan_provider=expected_plan, types_provider=staged_types) -> None:
    """Raise ValueError when the staged files do not carry the types the stock quantizer assigns for this source and preset."""
    plan = plan_provider(manifest['sources'], quant, source_precision=manifest.get('source_precision'))
    token = None
    if manifest.get('private'):
        from hfqa.providers import read_token
        token = read_token('huggingface')            # a private target is read with the operator's own credential, sent to the Hub only
    actual = types_provider(shards, staging, manifest['target_repo_id'], **({'token': token} if token else {}))
    problems = differences(plan, actual)
    if problems:
        raise ValueError('; '.join(problems))
