"""Runner side of the hybrid release with local files standing in for the Hub (real pinned ggml, skipped where absent)."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.test_hybrid import LIB, Q3_K, Q5_K, Q6_K, F16, FileReader, _string, descriptor, write_gguf
import struct

REVISION = '620d2225cc436ab0b001e3200e590e3b286ef9ab'


class FakeHub:
    def __init__(self):
        self.files, self.commits, self.branches = {}, [], []

    def whoami(self):
        return {'name': 'webmp3', 'orgs': []}

    def create_repo(self, **kwargs):
        pass

    def model_info(self, repo_id, **kwargs):
        return SimpleNamespace(sha='a' * 40, private=False)

    def create_branch(self, repo_id, branch, revision=None, exist_ok=True):
        self.branches.append(branch)

    def create_commit(self, repo_id, revision, commit_message, operations):
        for op in operations:
            self.files[op.path_in_repo] = Path(op.path_or_fileobj).read_bytes()
        self.commits.append(revision)
        return SimpleNamespace(oid='b' * 40)

    def get_paths_info(self, repo_id, revision, paths):
        return [SimpleNamespace(path=p, size=len(self.files[p]), lfs={'sha256': hashlib.sha256(self.files[p]).hexdigest()}) for p in paths]


@unittest.skipUnless(LIB.is_file(), 'pinned ggml library not available here')
class HybridReleaseTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from hfqa import hybrid, hybrid_release
        from hfqa.remote_gguf import inspect_remote_gguf
        self.hybrid, self.release, self.inspect = hybrid, hybrid_release, inspect_remote_gguf
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        ggml = hybrid.Ggml(LIB, REVISION)
        rng = np.random.default_rng(3)
        cols, rows_a, rows_b = 256, 20, 12
        table = rng.normal(0, 0.05, (rows_a + rows_b, cols)).astype(np.float32)
        arch = [('general.architecture', 8, _string('deepseek41')), ('split.no', 2, struct.pack('<H', 8)), ('split.count', 2, struct.pack('<H', 9))]
        write_gguf(self.root / 'a.gguf', arch, [('t1', (cols, rows_a), Q6_K, ggml.convert(table[:rows_a].tobytes(), 0, Q6_K, cols))])
        write_gguf(self.root / 'b.gguf', arch, [('t2', (cols, rows_b), F16, table[rows_a:].astype('<f2').tobytes())])
        write_gguf(self.root / 'template.gguf', arch, [('blk.1.engram_embd.weight', (cols, rows_a + rows_b), Q5_K, ggml.convert(table.tobytes(), 0, Q5_K, cols))])
        self.files = {name: self.root / name for name in ('a.gguf', 'b.gguf', 'template.gguf')}
        descriptors = {name: descriptor(path, name) for name, path in self.files.items()}
        self.manifest = {
            'runtime_revision': REVISION, 'target_repo_id': 'webmp3/Hybrid-GGUF', 'source_license': {'redistributable': True},
            'hybrid': {'template': descriptors['template.gguf'], 'sources': {'a': descriptors['a.gguf'], 'b': descriptors['b.gguf']},
                       'outputs': [{'name': 'blk.1.engram_embd.weight', 'parts': [{'source': 'a', 'tensor': 't1'}, {'source': 'b', 'tensor': 't2'}]}],
                       'out_type': Q3_K, 'output_filename': 'Table-Q3_K.gguf', 'compat': {'own_tensor': 'blk.1.engram_embd.weight'}}}

    def remote(self, desc):
        path = self.files[desc['filename']]
        return self.inspect(desc, reader=FileReader(path), allow_split=True)

    def run_it(self, api=None, manifest=None):
        with patch.object(self.release, 'open_remote', self.remote):
            return self.release.run_hybrid(manifest or self.manifest, ggml_path=LIB, work=self.root / 'work', job_id='job1', api=api, log=lambda *a, **k: None)

    def test_receipt_describes_an_integrity_only_table_file(self):
        receipt = self.run_it()
        self.assertEqual(receipt['state'], 'HYBRID_BUILT')
        self.assertTrue(receipt['integrity_only'] and not receipt['full_inference_test_performed'] and not receipt['hub_main_changed'])
        self.assertTrue(receipt['compatibility']['passes'])
        self.assertTrue(receipt['verification']['structure_valid'] and receipt['verification']['kv_block_identical_to_backbone_template'])
        self.assertEqual(receipt['output']['sha256'], receipt['table']['sha256'])
        self.assertFalse((self.root / 'work' / 'Table-Q3_K.gguf').exists())  # disk is the scarce resource on a runner

    def test_staging_uploads_exactly_the_verified_bytes_to_a_branch(self):
        hub = FakeHub()
        receipt = self.run_it(api=hub)
        self.assertEqual(hub.branches, ['hfqa-staging-job1'])
        self.assertEqual(hashlib.sha256(hub.files['Table-Q3_K.gguf']).hexdigest(), receipt['output']['sha256'])
        self.assertTrue(receipt['staging']['lfs_sha256_verified'])
        self.assertEqual(receipt['staging']['commit_oid'], 'b' * 40)

    def test_a_table_that_is_not_the_backbones_table_aborts_before_any_conversion(self):
        import numpy as np
        ggml = self.hybrid.Ggml(LIB, REVISION)
        other = np.random.default_rng(77).normal(0, 0.05, (20, 256)).astype(np.float32)
        write_gguf(self.root / 'a.gguf', [('general.architecture', 8, _string('deepseek41'))], [('t1', (256, 20), Q6_K, ggml.convert(other.tobytes(), 0, Q6_K, 256))])
        self.manifest['hybrid']['sources']['a'] = descriptor(self.root / 'a.gguf', 'a.gguf')
        with self.assertRaises(RuntimeError) as caught:
            self.run_it()
        self.assertIn('does not match the backbone table', str(caught.exception))

    def test_a_source_that_changed_after_pinning_is_rejected(self):
        self.manifest['hybrid']['sources']['b'] = {**self.manifest['hybrid']['sources']['b'], 'sha256': '0' * 64}
        with self.assertRaises(self.hybrid.HybridError):
            self.run_it()


if __name__ == '__main__':
    unittest.main()
