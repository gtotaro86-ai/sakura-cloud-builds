"""Runner-side release step (portable): shard writing, verification, inference budget, staging. Fakes only."""
import hashlib, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace

from hfqa.gguf import write_minimal_gguf
from hfqa.quantize import inference_budget
from hfqa.runner_release import materialize, names_hash, sha256_file, stage_on_branch, verify_shards


class FakeStream:
    quant = 'Q4_K_M'

    def __init__(self, payloads, verified=True):
        self.payloads = payloads
        self.shards = [{'index': i} for i in range(len(payloads))]
        self.size = sum(len(p) for p in payloads)
        self.receipt = {'source_sha256_verified': verified, 'shards': [
            {'index': i, 'size': len(p), 'sha256': hashlib.sha256(p).hexdigest(), 'tensor_names': [f't{i}']} for i, p in enumerate(payloads)]}

    def __iter__(self):
        for index, payload in enumerate(self.payloads):
            for start in range(0, len(payload), 7):
                yield index, payload[start:start + 7]


class MaterializeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_shards_are_written_and_hashed_against_the_stream(self):
        files = materialize(FakeStream([b'a' * 30, b'b' * 20]), self.dir / 'out', 'model', log=lambda *a, **k: None)
        self.assertEqual([f['size'] for f in files], [30, 20])
        for item in files:
            self.assertEqual(sha256_file(item['path']), item['sha256'])
        self.assertEqual(sorted(p.suffix for p in (self.dir / 'out').iterdir()), ['.gguf', '.gguf'])

    def test_unverified_source_leaves_no_files_behind(self):
        with self.assertRaises(RuntimeError):
            materialize(FakeStream([b'x' * 10], verified=False), self.dir / 'out', 'model', log=lambda *a, **k: None)
        self.assertEqual(list((self.dir / 'out').iterdir()), [])

    def test_refuses_when_the_disk_cannot_hold_the_output(self):
        stream = FakeStream([b'x'])
        stream.size = 10 ** 18
        with self.assertRaises(OSError):
            materialize(stream, self.dir / 'out', 'model', log=lambda *a, **k: None)

    def test_never_overwrites_existing_shards(self):
        materialize(FakeStream([b'x' * 5]), self.dir / 'out', 'model', log=lambda *a, **k: None)
        with self.assertRaises(FileExistsError):
            materialize(FakeStream([b'x' * 5]), self.dir / 'out', 'model', log=lambda *a, **k: None)


class VerifyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'a.gguf'
        write_minimal_gguf(self.path, architecture='llama', tensor_name='token_embd.weight')
        self.item = {'path': str(self.path), 'name': 'a.gguf', 'tensor_names': ['token_embd.weight']}
        self.structure = {'architecture': 'llama', 'tensor_name_set_sha256': names_hash(['token_embd.weight'])}

    def tearDown(self):
        self.tmp.cleanup()

    def test_complete_valid_shard_set_passes_and_reports_types(self):
        report = verify_shards([self.item], self.structure, 1)[0]
        self.assertEqual(report['tensor_count'], 1)
        self.assertTrue(report['tensor_type_histogram'])

    def test_wrong_architecture_missing_tensor_or_wrong_count_fails(self):
        with self.assertRaises(RuntimeError):
            verify_shards([self.item], {**self.structure, 'architecture': 'qwen2'}, 1)
        with self.assertRaises(RuntimeError):
            verify_shards([{**self.item, 'tensor_names': ['other']}], self.structure, 1)
        with self.assertRaises(RuntimeError):
            verify_shards([self.item], self.structure, 2)


class BudgetTests(unittest.TestCase):
    def test_large_models_get_more_time_and_fewer_tokens(self):
        small = inference_budget(500 * 1024 ** 2, 2)
        large = inference_budget(24 * 1024 ** 3, 4)
        self.assertGreater(large['timeout_seconds'], 4 * 600)
        self.assertGreater(large['timeout_seconds'], small['timeout_seconds'])
        self.assertEqual((small['threads'], large['threads']), (2, 4))
        self.assertLess(large['predict'], small['predict'])


class RepackTests(unittest.TestCase):
    """Regression (Ornith-1.5-122B, Q2_K 45 GB on a 16 GB runner): llama-cli wanted a 27 GB repacked copy of the weights and the real inference failed after 4.5 h."""
    def run_with(self, model_bytes, memory_bytes):
        from unittest.mock import patch
        from hfqa import runner_release
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'm.gguf'
            path.write_bytes(b'x')
            item = {'name': 'm.gguf', 'path': str(path), 'size': model_bytes, 'sha256': sha256_file(path)}
            item['size'] = model_bytes
            seen = {}

            def fake(binary, model, logs, timeout, **kwargs):
                seen.update(kwargs)
                return {}
            with patch.object(runner_release, 'test_inference', fake):
                try:
                    runner_release.run_inference(Path('llama-cli'), [item], Path(tmp), 4, memory_bytes)
                except RuntimeError:
                    pass  # the shard-size check after the fake inference is not what is tested here
        return seen

    def test_a_model_that_does_not_fit_beside_its_repacked_copy_runs_without_repacking(self):
        self.assertIs(self.run_with(45 * 10 ** 9, 16 * 10 ** 9)['repack'], False)
        self.assertIs(self.run_with(2 * 10 ** 9, 16 * 10 ** 9)['repack'], True)  # ordinary models keep the proven default
        self.assertIs(self.run_with(45 * 10 ** 9, None)['repack'], True)  # unknown memory: nothing changes

    def test_the_stock_option_is_added_to_the_command_only_when_asked(self):
        from unittest.mock import patch
        from hfqa import quantize
        commands = []

        class Done:
            returncode = 0

        def fake_run(command, **kwargs):
            commands.append(command)
            kwargs['stdout'].write(b'Paris')
            return Done()
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / 'llama-cli'
            binary.write_bytes(b'#')
            model = Path(tmp) / 'm.gguf'
            write_minimal_gguf(model)
            with patch.object(quantize.subprocess, 'run', fake_run):
                for repack in (True, False):
                    quantize.test_inference(binary, model, Path(tmp) / 'logs', 10, repack=repack)
        self.assertNotIn('--no-repack', commands[0])
        self.assertIn('--no-repack', commands[1])


class FakeApi:
    def __init__(self, stored_sha=None):
        self.stored_sha = stored_sha
        self.commits = []
        self.branches = []

    def whoami(self):
        return {'name': 'webmp3', 'orgs': []}

    def create_repo(self, repo_id, repo_type='model', private=False, exist_ok=True):
        pass

    def model_info(self, repo_id, **kw):
        return SimpleNamespace(private=False, sha='a' * 40)

    def create_branch(self, repo_id, branch, revision=None, exist_ok=False):
        self.branches.append((branch, revision, exist_ok))

    def create_commit(self, repo_id, revision, commit_message, operations):
        self.commits.append((revision, [op.path_in_repo for op in operations]))
        self.ops = operations
        return SimpleNamespace(oid='b' * 40)

    def get_paths_info(self, repo_id, revision, paths):
        return [SimpleNamespace(path=item['name'], size=item['size'], lfs={'sha256': self.stored_sha or item['sha256']}) for item in self.files if item['name'] in paths]


class StageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.files = []
        for index, payload in enumerate((b'12345', b'123456'), 1):
            path = Path(self.tmp.name) / f'm-0000{index}-of-00002.gguf'
            path.write_bytes(payload)
            self.files.append({'name': path.name, 'path': str(path), 'size': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()})

    def tearDown(self):
        self.tmp.cleanup()

    def test_validated_shards_go_to_a_staging_branch_never_main(self):
        api = FakeApi()
        api.files = self.files
        result = stage_on_branch(api, 'webmp3/Model-GGUF', 'hfqa-staging-job', self.files, private=False, message='m')
        self.assertEqual(api.commits[0][0], 'hfqa-staging-job')
        self.assertEqual(api.branches[0][0], 'hfqa-staging-job')
        self.assertTrue(result['lfs_sha256_verified'])
        self.assertEqual(result['commit_oid'], 'b' * 40)

    def test_hub_holding_different_bytes_is_detected(self):
        api = FakeApi(stored_sha='e' * 64)
        api.files = self.files
        with self.assertRaises(RuntimeError):
            stage_on_branch(api, 'webmp3/Model-GGUF', 'hfqa-staging-job', self.files, private=False, message='m')

    def test_foreign_namespace_is_refused_before_any_write(self):
        api = FakeApi()
        api.files = self.files
        with self.assertRaises(RuntimeError):
            stage_on_branch(api, 'someoneelse/Model-GGUF', 'hfqa-staging-job', self.files, private=False, message='m')
        self.assertEqual(api.commits, [])


if __name__ == '__main__':
    unittest.main()
