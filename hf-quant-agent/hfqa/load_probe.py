"""Load probe: does the exact llama.cpp build accept this model's structure? Answered in seconds, before hours of quantization.

A runtime can know an architecture name and still be unable to load a model that carries it: Kolibri 1.1 keeps `kolibri1` but has a different tensor set than
Kolibri 1, so b12623 stopped at `check_tensor_dims: tensor 'blk.0.ffn_gate_inp.weight' not found` only in the final inference test, after about two hours of streaming.

The probe reads only the header of each pinned source file (bounded HTTP Range reads, `hfqa.remote_gguf`), writes it to a local file of the full source size whose
tensor data region is a hole (a sparse file: it takes no disk space and holds only zeros) and lets the build load it once with a tiny generation. The loader checks
tensor names, shapes, hyper-parameters and the graph before it needs any real weight, so a structural mismatch fails at once. Nothing proves the model is *good*;
that stays the job of the real inference test. The result is one of

* `ok`           the build loaded the file and generated a token from zero weights;
* `rejected`     the build exited with a loader error (tensor missing, unknown architecture, ...): this runtime cannot be used for this model;
* `inconclusive` anything else (timeout, killed, a flag the build does not know, ...): the probe says nothing, the pipeline goes on as before.
"""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

from hfqa import safe_env
from hfqa.remote_gguf import MAX_HEADER_BYTES, MAX_RANGE_BYTES, RangeReader, inspect_remote_gguf

# Structural loader failures of llama.cpp: the build reads the file and finds that this model does not fit what it expects. Only these make the verdict `rejected`.
# The generic closing lines ("error loading model", "failed to load model") are NOT enough on their own: memory or file-system trouble prints them too, and a wrong
# "rejected" is expensive (the job waits for a newer build).
LOADER_ERRORS = re.compile(
    r"check_tensor_dims|tensor '[^']{1,200}' (?:not found|has wrong shape|has invalid)|unknown model architecture|wrong number of tensors|"
    r"done_getting_tensors|invalid (?:hparams|hyperparameter)", re.I)
DEFAULT_TIMEOUT = 240
MAX_REASON = 300


def write_probe_file(descriptor: dict, destination: Path, reader=None) -> dict:
    """The header of one pinned GGUF file followed by a hole up to the full size. Returns the facts about it."""
    reader = reader or RangeReader(descriptor)
    info = inspect_remote_gguf(descriptor, reader=reader, allow_split=True)
    header = info.data_offset
    if not 0 < header <= min(MAX_HEADER_BYTES, reader.size):
        raise ValueError('implausible GGUF header size')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('wb') as out:
        offset = 0
        while offset < header:
            count = min(MAX_RANGE_BYTES, header - offset)
            out.write(reader.read_range(offset, count))
            offset += count
        out.truncate(reader.size)  # the tensor data stays a hole (zeros, no disk space on a sparse-capable file system)
    return {'architecture': info.metadata.get('general.architecture'), 'header_bytes': header, 'source_bytes': reader.size,
            'tensors': len(info.tensors), 'file': destination.name}


def classify(returncode: int | None, output: str, timed_out: bool = False) -> dict:
    """Verdict from one run. Only a clean exit counts as ok and only a loader error as rejected."""
    if timed_out:
        return {'verdict': 'inconclusive', 'reason': 'the probe run timed out'}
    if returncode == 0:
        return {'verdict': 'ok', 'reason': None}
    if returncode is not None and returncode < 0:
        return {'verdict': 'inconclusive', 'reason': f'the probe process was killed by signal {-returncode}'}
    for line in output.splitlines():
        if LOADER_ERRORS.search(line):
            clean = re.sub(r'^\s*[\d.]+\s+[A-Z]\s+', '', re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', line)).strip()  # leave out the time stamp and level of the log line
            return {'verdict': 'rejected', 'reason': clean[:MAX_REASON]}
    return {'verdict': 'inconclusive', 'reason': f'the probe run exited with code {returncode} without a loader error'}


def run_cli(launcher, first_file: Path, log_dir: Path, timeout: int = DEFAULT_TIMEOUT, threads: int = 2) -> dict:
    """One bounded CPU load of the probe file. `launcher` is the llama-cli path (or a command list for tests)."""
    prefix = [str(part) for part in launcher] if isinstance(launcher, (list, tuple)) else [str(launcher)]
    binary = Path(prefix[0])
    # No weight repacking (it would copy the whole model into RAM), no warm-up, one token: the load is what is being probed.
    command = prefix + ['-m', str(first_file), '-p', 'hi', '-n', '1', '-c', '64', '--temp', '0', '-t', str(threads), '-ngl', '0',
                        '--single-turn', '--no-display-prompt', '--no-warmup', '--no-repack']
    if 'completion' in Path(prefix[-1]).name:
        command.append('-no-cnv')  # llama-completion aborts (SIGABRT) on a model with a custom chat template unless conversation mode is off (measured with Inkling 2 on b12623)
    env = safe_env.clean()
    env['LD_LIBRARY_PATH'] = str(binary.resolve().parent) + ':' + env.get('LD_LIBRARY_PATH', '')
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path, err_path = log_dir / 'load-probe.stdout.log', log_dir / 'load-probe.stderr.log'
    started = time.monotonic()
    timed_out = False
    try:
        with out_path.open('wb') as out, err_path.open('wb') as err:
            completed = subprocess.run(command, check=False, stdin=subprocess.DEVNULL, stdout=out, stderr=err, timeout=timeout, env=env)
        code = completed.returncode
    except subprocess.TimeoutExpired:
        code, timed_out = None, True
    except OSError as exc:  # the binary itself cannot start: nothing is known about the model
        out_path.write_text('', encoding='utf-8')
        err_path.write_text(f'{type(exc).__name__}: {exc}\n', encoding='utf-8')
        return {'verdict': 'inconclusive', 'reason': 'the probe binary could not be started', 'returncode': None, 'seconds': 0.0,
                'stdout_log': str(out_path), 'stderr_log': str(err_path)}
    text = err_path.read_bytes()[:65536].decode('utf-8', 'replace') + '\n' + out_path.read_bytes()[:16384].decode('utf-8', 'replace')
    verdict = classify(code, text, timed_out)
    return {**verdict, 'returncode': code, 'seconds': round(time.monotonic() - started, 1), 'stdout_log': str(out_path), 'stderr_log': str(err_path)}


def probe(descriptors: list[dict], launcher, work: Path, *, reader_factory=RangeReader, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Probe every shard of one pinned source set; the first shard is what llama.cpp is asked to load."""
    work = Path(work)
    shards = sorted(descriptors, key=lambda item: item.get('split_index', 1))
    files, facts = [], []
    try:
        for descriptor in shards:
            target = work / 'files' / descriptor['filename'].rsplit('/', 1)[-1]
            facts.append(write_probe_file(descriptor, target, reader_factory(descriptor)))
            files.append(target)
        result = run_cli(launcher, files[0], work / 'logs', timeout)
    finally:
        for path in files:
            path.unlink(missing_ok=True)  # the hole costs nothing, the header copy should not outlive the probe
    return {**result, 'architecture': facts[0]['architecture'], 'shards': len(facts), 'header_bytes': sum(f['header_bytes'] for f in facts),
            'source_bytes': sum(f['source_bytes'] for f in facts), 'tensors': sum(f['tensors'] for f in facts)}
