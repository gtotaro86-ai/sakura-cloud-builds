"""Runner-side parts of the runtime channel (also run in the public worker repository): descriptor, parity, receipt."""
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hfqa import runtime_parity, runtime_qualification, runtimes

REPOSITORY = 'gtotaro86-ai/sakura-cloud-builds'
TAG = 'b11890'
REVISION = 'd' * 40
SHA = 'e' * 64
WORKER = 'c' * 40
DESCRIPTOR = {'tag': TAG, 'revision': REVISION, 'url': runtimes.asset_url(TAG), 'sha256': SHA}
MANIFEST = {'source': {'repo_id': 'Qwen/x', 'filename': 'x.gguf'}, 'runtime_revision': REVISION, 'runtime': DESCRIPTOR,
            'quants': list(runtime_qualification.REQUIRED_QUANTS)}


class DescriptorTests(unittest.TestCase):
    def test_descriptor_must_be_an_official_fully_pinned_asset(self):
        self.assertEqual(runtimes.validate_descriptor(DESCRIPTOR), DESCRIPTOR)
        for bad in ({**DESCRIPTOR, 'url': 'https://example.org/llama.tar.gz'}, {**DESCRIPTOR, 'revision': 'main'}, {**DESCRIPTOR, 'sha256': 'abc'},
                    {**DESCRIPTOR, 'tag': 'latest'}, {**DESCRIPTOR, 'url': runtimes.asset_url('b11891')}, 'x', None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                runtimes.validate_descriptor(bad)

    def test_manifest_without_descriptor_means_the_baseline_only(self):
        self.assertEqual(runtimes.descriptor_for_manifest({'runtime_revision': runtimes.BASELINE_REVISION})['tag'], runtimes.BASELINE_TAG)
        with self.assertRaises(ValueError):
            runtimes.descriptor_for_manifest({'runtime_revision': REVISION})
        with self.assertRaises(ValueError):
            runtimes.descriptor_for_manifest({'runtime_revision': 'f' * 40, 'runtime': DESCRIPTOR})
        self.assertEqual(runtimes.descriptor_for_manifest(MANIFEST), DESCRIPTOR)

    def test_runner_scripts_keep_the_baseline_pin(self):
        from scripts import fetch_stream_runtime
        self.assertEqual((fetch_stream_runtime.URL, fetch_stream_runtime.DIGEST), (runtimes.baseline()['url'], runtimes.BASELINE_SHA256))


class ParityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    @staticmethod
    def gguf(path, payloads):
        def string(value):
            raw = value.encode()
            return struct.pack('<Q', len(raw)) + raw
        header = b'GGUF' + struct.pack('<IQQ', 3, len(payloads), 1) + string('general.architecture') + struct.pack('<I', 8) + string('llama')
        data = bytearray()
        for name, values in payloads.items():
            header += string(name) + struct.pack('<I', 1) + struct.pack('<Q', len(values)) + struct.pack('<IQ', 0, len(data))
            data += struct.pack('<' + 'f' * len(values), *values)
            data += bytes(-len(data) % 32)
        header += bytes(-len(header) % 32)
        Path(path).write_bytes(header + bytes(data))

    def test_identical_tensors_match_and_one_changed_value_is_found(self):
        self.gguf(self.root / 'a.gguf', {'x.weight': [1.0] * 32, 'y.weight': [2.0] * 32})
        self.gguf(self.root / 'b.gguf', {'y.weight': [2.0] * 32, 'x.weight': [1.0] * 32})  # order differs, content equal
        self.gguf(self.root / 'c.gguf', {'x.weight': [1.0] * 31 + [9.0], 'y.weight': [2.0] * 32})
        a, b, c = (runtime_parity.tensor_catalogue(self.root / f'{n}.gguf') for n in 'abc')
        self.assertTrue(runtime_parity.compare(a, b)['parity'])
        result = runtime_parity.compare(a, c)
        self.assertEqual((result['parity'], result['different']), (False, ['x.weight']))

    def test_shards_merge_and_a_tensor_in_two_shards_is_refused(self):
        self.gguf(self.root / 's1.gguf', {'x.weight': [1.0] * 32})
        self.gguf(self.root / 's2.gguf', {'y.weight': [2.0] * 32})
        self.assertEqual(sorted(runtime_parity.merged_catalogue([self.root / 's1.gguf', self.root / 's2.gguf'])), ['x.weight', 'y.weight'])
        with self.assertRaises(ValueError):
            runtime_parity.merged_catalogue([self.root / 's1.gguf', self.root / 's1.gguf'])


def good_receipt(manifest=MANIFEST, run_id=77):
    result = lambda quant: {'quant': quant, 'parity': {'parity': True, 'tensor_count': 290, 'only_streamed': [], 'only_stock': [], 'different': [], 'different_count': 0},
                            'inference': {'returncode': 0, 'completion': ' Paris.', 'runtime_build': 'version: 11890 (' + REVISION[:9] + ')'}}
    return {'schema': runtime_qualification.SCHEMA, 'runtime': DESCRIPTOR, 'manifest_sha256': runtime_qualification.canonical_sha(manifest),
            'runner': {'GITHUB_RUN_ID': str(run_id), 'GITHUB_REPOSITORY': REPOSITORY, 'GITHUB_SHA': WORKER},
            'bridge': {'runtime_revision': REVISION, 'runtime_patched': False, 'bridge_sha256': '1' * 64},
            'binaries': {k: '2' * 64 for k in ('llama_cli_sha256', 'llama_quantize_sha256', 'libllama_sha256', 'libggml_base_sha256')},
            'results': [result(q) for q in runtime_qualification.REQUIRED_QUANTS]}


class QualificationTests(unittest.TestCase):
    def check(self, mutate=None):
        receipt = good_receipt()
        if mutate:
            mutate(receipt)
        return runtime_qualification.validate(receipt, manifest=MANIFEST, descriptor=DESCRIPTOR, run_id=77, repository=REPOSITORY, worker_revision=WORKER)

    def test_complete_receipt_passes(self):
        self.assertEqual(self.check()['quants'], sorted(runtime_qualification.REQUIRED_QUANTS))

    def test_every_weakness_is_rejected(self):
        mutations = {
            'other run': lambda r: r['runner'].update(GITHUB_RUN_ID='1'),
            'other repository': lambda r: r['runner'].update(GITHUB_REPOSITORY='evil/repo'),
            'other worker': lambda r: r['runner'].update(GITHUB_SHA='0' * 40),
            'other manifest': lambda r: r.update(manifest_sha256='0' * 64),
            'other asset': lambda r: r.update(runtime={**DESCRIPTOR, 'sha256': '0' * 64}),
            'patched bridge': lambda r: r['bridge'].update(runtime_patched=True),
            'bridge for other commit': lambda r: r['bridge'].update(runtime_revision='0' * 40),
            'parity failed': lambda r: r['results'][0]['parity'].update(parity=False, different_count=1, different=['x']),
            'missing quant': lambda r: r['results'].pop(),
            'no paris': lambda r: r['results'][1]['inference'].update(completion='Berlin'),
            'wrong build': lambda r: r['results'][1]['inference'].update(runtime_build='version: 11669 (620d2225c)'),
            'missing binary hash': lambda r: r['binaries'].pop('llama_quantize_sha256'),
            'hostile types': lambda r: r.update(results='nope'),
        }
        for name, mutate in mutations.items():
            with self.subTest(name), self.assertRaises(runtime_qualification.QualificationError):
                self.check(mutate)


if __name__ == '__main__':
    unittest.main()
