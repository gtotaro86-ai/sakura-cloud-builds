"""Runner-side load probe (see hfqa/load_probe.py): can the pinned llama.cpp build load the pinned source's structure?

Prints one JSON line with the verdict (ok | rejected | inconclusive). The exit code is 0 for every verdict; the workflow decides what to do
with it (a `rejected` verdict stops the run before any long step). Holds no Hub key: it reads only public files with HTTP Range requests.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hfqa import load_probe
from hfqa.security import redact
from scripts.run_stream_worker import atomic_json, read_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--llama-cli', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=load_probe.DEFAULT_TIMEOUT)
    args = parser.parse_args()
    manifest, source = read_manifest(args.manifest)
    descriptors = source.get('sources') or [source]
    try:
        result = load_probe.probe(descriptors, args.llama_cli, args.work, timeout=args.timeout)
    except Exception as error:  # noqa: BLE001 - a probe that cannot run must never stop a release by itself
        result = {'verdict': 'inconclusive', 'reason': redact(f'{type(error).__name__}: {error}')[:300]}
    Path(args.work).mkdir(parents=True, exist_ok=True)
    result['runtime_tag'] = ((manifest.get('runtime') or {}).get('tag'))
    result['source'] = f"{descriptors[0].get('repo_id')}@{str(descriptors[0].get('revision'))[:12]}"
    atomic_json(Path(args.work) / 'probe.json', result)
    print(json.dumps(result, sort_keys=True), flush=True)
    output = os.environ.get('GITHUB_OUTPUT')
    if output:
        reason = ''.join(ch if 32 <= ord(ch) < 127 else ' ' for ch in str(result.get('reason') or ''))[:200]  # one plain line: it becomes a job name
        with open(output, 'a', encoding='utf-8') as handle:
            handle.write('verdict=' + result['verdict'] + '\n' + 'reason=' + reason + '\n')


if __name__ == '__main__':
    main()
