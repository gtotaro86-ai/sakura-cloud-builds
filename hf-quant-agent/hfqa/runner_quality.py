"""Measured deviation of quantized files from their full-precision source, with the stock llama.cpp `llama-perplexity` (public worker).

The tool stores the source's output probabilities for a fixed evaluation text once (`--kl-divergence-base`), then compares every quantized file with them
(`--kl-divergence`). What it reports is what the card shows: perplexity of the quant and of the source, the ratio of the two, the KL divergence of the
quant's next-token distribution from the source's, and how often both models rank the same token first.

The evaluation text is NOT the calibration text of the importance matrix: two public-domain books from Project Gutenberg (English and Chinese), fixed by
slice and SHA-256. If a text cannot be fetched or does not match its hash, the measurement is skipped and says why; it never stops a release.
No task benchmark is run; these numbers describe how far the quantized model's predictions drift from the original on this text, nothing more.
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

from hfqa.runner_release import sha256_file

# Pride and Prejudice (Jane Austen, 1813, public domain) and Hongloumeng (Cao Xueqin, 18th century, public domain); slices after the Gutenberg start marker.
EVAL_TEXTS = [
    {'name': 'english', 'url': 'https://www.gutenberg.org/cache/epub/1342/pg1342.txt', 'start': 50000, 'characters': 30000, 'bytes': 30476,
     'sha256': 'ba06f98b0ebe00b9e05434704ca558ef73c5b3eea41b5d5cd144621a19b38a4e'},
    {'name': 'chinese', 'url': 'https://www.gutenberg.org/cache/epub/24264/pg24264.txt', 'start': 50000, 'characters': 6000, 'bytes': 17404,
     'sha256': 'c4f3b6307d978bb5ef7603df95c62f936154f2991d7d00a4a9a9cb86bcbd764b'},
]
CONTEXT = 512
PASS_TIMEOUT_S = 25 * 60
SPEC = {'method': 'stock_perplexity_kld', 'context': CONTEXT, 'texts': [{k: t[k] for k in ('name', 'url', 'start', 'characters', 'bytes', 'sha256')} for t in EVAL_TEXTS]}


class Unavailable(RuntimeError):
    """The evaluation text could not be obtained or does not match its pinned hash."""


def chunks_for(source_bytes: int) -> int:
    """How many context windows to score: all of the text for small models, fewer for larger ones (CPU time grows with the model)."""
    return 64 if source_bytes <= 4 * 10 ** 9 else 12


def fetch_text(spec: dict, download=None) -> str:
    download = download or (lambda url: urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'hfqa-eval/1'}), timeout=60).read())
    try:
        raw = download(spec['url']).decode('utf-8').replace('\r\n', '\n')
    except Exception as error:  # noqa: BLE001 - any network or decoding problem means "unavailable", not "failed"
        raise Unavailable(f"the {spec['name']} evaluation text could not be fetched ({type(error).__name__})") from None
    marker = re.search(r'\*\*\* START OF[^\n]*\n', raw)
    if not marker:
        raise Unavailable(f"the {spec['name']} evaluation text has no start marker")
    piece = raw[marker.end():][spec['start']:spec['start'] + spec['characters']]
    import hashlib
    if hashlib.sha256(piece.encode('utf-8')).hexdigest() != spec['sha256']:
        raise Unavailable(f"the {spec['name']} evaluation text differs from its pinned hash")
    return piece


def evaluation_file(work: Path, spec: dict | None = None, download=None) -> tuple[Path, dict]:
    spec = spec or SPEC
    pieces = [fetch_text(item, download) for item in spec['texts']]
    path = work / 'evaluation.txt'
    path.write_text('\n\n'.join(pieces) + '\n', encoding='utf-8')
    return path, {'texts': [{k: t[k] for k in ('name', 'sha256', 'bytes')} for t in spec['texts']], 'file_sha256': sha256_file(path), 'context': spec['context']}


def _number(text: str, label: str, error: bool = False):
    # the tool pads its labels with a varying number of spaces ("Mean    KLD:", "99.0%   KLD:"): any run of white space matches
    pattern = r'\s+'.join(re.escape(part) for part in label.split())
    found = re.search(pattern + r'\s*:?\s*(-?[\d.]+(?:e[+-]?\d+)?)' + (r'\s*±\s*([\d.]+(?:e[+-]?\d+)?)' if error else ''), text, re.I)
    if not found:
        return None
    return (float(found.group(1)), float(found.group(2))) if error else float(found.group(1))


def parse_statistics(text: str) -> dict:
    """The numbers of one `--kl-divergence` pass; every value is None when the tool did not print it (the caller then records the pass as unparsed)."""
    def pair(label):
        value = _number(text, label, error=True)
        return {'value': value[0], 'error': value[1]} if value else None
    return {
        'ppl_quant': pair('Mean PPL(Q)'), 'ppl_source': pair('Mean PPL(base)'), 'ppl_ratio': pair('Mean PPL(Q)/PPL(base)'),
        'mean_kld': pair('Mean    KLD') or pair('Mean KLD'), 'median_kld': _number(text, 'Median  KLD') or _number(text, 'Median KLD'),
        'p99_kld': _number(text, '99.0%   KLD') or _number(text, '99.0% KLD'), 'max_kld': _number(text, 'Maximum KLD'),
        'same_top_token_percent': pair('Same top p'), 'rms_delta_p_percent': pair('RMS Δp'),
    }


def _run(command: list[str], logs: Path, name: str, env: dict) -> str:
    import subprocess
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / f'{name}.stdout.log').open('wb') as out:
        completed = subprocess.run(command, check=False, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, timeout=PASS_TIMEOUT_S, env=env)
    text = (logs / f'{name}.stdout.log').read_bytes().decode('utf-8', 'replace')
    if completed.returncode != 0:
        raise RuntimeError(f'llama-perplexity failed (exit {completed.returncode}); log: {logs / (name + ".stdout.log")}')
    return text


def measure(source_path: Path, quants: dict[str, Path], tool: Path, work: Path, cpus: int, source_bytes: int, *, run=_run, download=None, log=print) -> dict:
    """{'evidence': ..., 'per_quant': {quant: statistics}} or {'skipped': reason}. Any failure to measure is recorded, never raised: it must not stop a verified release."""
    started = time.monotonic()
    try:
        text_path, evidence = evaluation_file(work, download=download)
    except Unavailable as error:
        return {'skipped': str(error)}
    import os
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(Path(tool).parent) + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    chunks = chunks_for(source_bytes)
    base_log = work / 'quality-logs' / 'source.kld'
    base_log.parent.mkdir(parents=True, exist_ok=True)
    common = ['-f', str(text_path), '-c', str(CONTEXT), '-b', str(CONTEXT), '-ub', str(CONTEXT), '-t', str(cpus), '-ngl', '0', '--chunks', str(chunks)]
    try:
        base_text = run([str(tool), '-m', str(source_path), '--kl-divergence-base', str(base_log)] + common, work / 'quality-logs', 'source', env)
        per_quant = {}
        for quant, path in quants.items():
            out = run([str(tool), '-m', str(path), '--kl-divergence-base', str(base_log), '--kl-divergence'] + common, work / 'quality-logs', quant, env)
            statistics = parse_statistics(out)
            if statistics['mean_kld'] is None or statistics['ppl_ratio'] is None:
                return {'skipped': f'the output of llama-perplexity for {quant} could not be read'}
            per_quant[quant] = statistics
    except Exception as error:  # noqa: BLE001 - a failed or timed out measurement is a note on the card, not a failed release
        return {'skipped': f'the measurement did not finish ({type(error).__name__})'}
    evidence.update(chunks=chunks, tool_sha256=sha256_file(tool), seconds=round(time.monotonic() - started, 1), source_bytes=source_bytes)
    log(json.dumps({'event': 'quality', 'quants': sorted(per_quant), 'seconds': evidence['seconds']}), flush=True)
    return {'evidence': evidence, 'per_quant': per_quant}
