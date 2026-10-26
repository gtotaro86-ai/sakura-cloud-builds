"""The qualification receipt of a candidate llama.cpp build: written by the runner, judged by Oracle.

A build is qualified only when, on the runner, (1) the official asset matched the pinned SHA-256, (2) the bridge was built
unpatched against exactly that commit, (3) every fixture quant streamed through that bridge equals the build's own
`llama-quantize` tensor for tensor, and (4) the stock `llama-cli` of that build generated text from every streamed quant.
"""
from __future__ import annotations

import hashlib
import json
import re

SCHEMA = 'hfqa-runtime-qualification/1'
REQUIRED_QUANTS = ('Q2_K', 'Q3_K_M', 'Q4_K_M', 'Q5_K_M')
HEX64 = re.compile(r'[a-f0-9]{64}')


class QualificationError(ValueError):
    """The receipt does not prove a qualified build."""


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build_receipt(*, manifest: dict, descriptor: dict, env: dict, bridge_receipt: dict, binaries: dict, results: list[dict]) -> dict:
    return {'schema': SCHEMA, 'runtime': descriptor, 'manifest_sha256': canonical_sha(manifest),
            'runner': {key: str(env.get(key, '')) for key in ('GITHUB_RUN_ID', 'GITHUB_REPOSITORY', 'GITHUB_SHA')},
            'bridge': {'runtime_revision': bridge_receipt.get('runtime_revision'), 'runtime_patched': bridge_receipt.get('runtime_patched'),
                       'bridge_sha256': bridge_receipt.get('bridge_sha256')},
            'binaries': binaries, 'source': manifest.get('source') or (manifest.get('sources') or [{}])[0], 'results': results}


def _need(condition, message):
    if not condition:
        raise QualificationError(message)


def validate(receipt, *, manifest: dict, descriptor: dict, run_id: int, repository: str, worker_revision: str) -> dict:
    try:
        return _validate(receipt, manifest, descriptor, run_id, repository, worker_revision)
    except QualificationError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError):
        raise QualificationError('qualification receipt has invalid field types') from None


def _validate(receipt, manifest, descriptor, run_id, repository, worker_revision):
    _need(isinstance(receipt, dict) and receipt.get('schema') == SCHEMA, 'unknown qualification receipt schema')
    runner = receipt.get('runner') or {}
    _need(runner.get('GITHUB_RUN_ID') == str(run_id) and runner.get('GITHUB_REPOSITORY') == repository and runner.get('GITHUB_SHA') == worker_revision,
          'receipt was not produced by the dispatched workflow run')
    _need(receipt.get('manifest_sha256') == canonical_sha(manifest), 'receipt was produced for a different manifest')
    _need(receipt.get('runtime') == descriptor, 'receipt names a different runtime asset')
    bridge = receipt.get('bridge') or {}
    _need(bridge.get('runtime_revision') == descriptor['revision'] and bridge.get('runtime_patched') is False
          and HEX64.fullmatch(str(bridge.get('bridge_sha256', ''))), 'bridge was not built unpatched against the pinned commit')
    binaries = receipt.get('binaries') or {}
    _need(all(HEX64.fullmatch(str(binaries.get(key, ''))) for key in ('llama_cli_sha256', 'llama_quantize_sha256', 'libllama_sha256', 'libggml_base_sha256')),
          'binary identities are incomplete')
    results = receipt.get('results')
    wanted = list(manifest.get('quants') or [])
    _need(isinstance(results, list) and set(REQUIRED_QUANTS) <= set(wanted) and sorted(str(item.get('quant')) for item in results) == sorted(wanted), 'quant set is incomplete')
    for item in results:
        parity = item.get('parity') or {}
        _need(parity.get('parity') is True and parity.get('different_count') == 0 and parity.get('only_streamed') == [] and parity.get('only_stock') == []
              and type(parity.get('tensor_count')) is int and parity['tensor_count'] > 0, f"stock parity failed for {item.get('quant')}")
        inference = item.get('inference') or {}
        _need(inference.get('returncode') == 0 and 'paris' in str(inference.get('completion', '')).lower()
              and descriptor['revision'][:9] in str(inference.get('runtime_build')), f"inference proof failed for {item.get('quant')}")
    return {'quants': sorted(wanted), 'bridge_sha256': bridge['bridge_sha256'], 'binaries': binaries,
            'tensor_counts': {item['quant']: item['parity']['tensor_count'] for item in results}}
