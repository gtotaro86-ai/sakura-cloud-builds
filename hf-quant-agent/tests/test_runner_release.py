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
    def run_with(self, model_bytes, memory_bytes, *, stderr_texts=(), remaining=None):
        """Run run_inference with a fake runtime that fails once per entry of stderr_texts (the text goes to its stderr log); returns the calls and the result."""
        from unittest.mock import patch
        from hfqa import runner_release
        calls, queue = [], list(stderr_texts)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'm.gguf'
            path.write_bytes(b'x')
            item = {'name': 'm.gguf', 'path': str(path), 'size': model_bytes, 'sha256': sha256_file(path)}

            def fake(binary, model, logs, timeout, **kwargs):
                calls.append({'timeout': timeout, **kwargs})
                if queue:
                    log = Path(tmp) / f'err{len(calls)}.log'
                    log.write_text(queue.pop(0))
                    raise RuntimeError(f'stock runtime inference failed (exit 1); logs: {log}')
                return {'model_sha256': 'x'}
            with patch.object(runner_release, 'test_inference', fake):
                try:
                    result = runner_release.run_inference(Path('llama-cli'), [item], Path(tmp), 4, memory_bytes, remaining)
                except RuntimeError as error:
                    result = error
        return calls, result

    def test_the_proven_default_is_tried_first_and_the_retry_is_only_for_allocation_failures(self):
        calls, result = self.run_with(45 * 10 ** 9, 32 * 10 ** 9, stderr_texts=['E failed to allocate CPU_REPACK buffer of size 27044831232'])
        self.assertEqual([c['repack'] for c in calls], [True, False])  # failed on the allocation, then ran without repacking
        self.assertIs(result['repack_retried_without_repacking'], True)
        calls, result = self.run_with(2 * 10 ** 9, 16 * 10 ** 9)
        self.assertEqual([c['repack'] for c in calls], [True])  # an ordinary model never leaves the default
        self.assertIs(result['repack_retried_without_repacking'], False)
        calls, result = self.run_with(2 * 10 ** 9, 16 * 10 ** 9, stderr_texts=['E unknown tensor type'])
        self.assertEqual(len(calls), 1)  # any other failure is reported, not retried
        self.assertIsInstance(result, RuntimeError)
        calls, result = self.run_with(2 * 10 ** 9, 16 * 10 ** 9, stderr_texts=['E failed to allocate X', 'E failed to allocate Y'])
        self.assertEqual(len(calls), 2)  # and the retry happens at most once
        self.assertIsInstance(result, RuntimeError)

    def test_a_model_far_larger_than_the_memory_skips_the_hopeless_first_attempt(self):
        calls, _ = self.run_with(45 * 10 ** 9, 16 * 10 ** 9)
        self.assertEqual([c['repack'] for c in calls], [False])
        calls, _ = self.run_with(45 * 10 ** 9, None)  # unknown memory: the default
        self.assertEqual([c['repack'] for c in calls], [True])
        calls, _ = self.run_with(23 * 10 ** 9, 16 * 10 ** 9)  # Nanbeige Q5_K_M (1.4 x memory) passed with repacking
        self.assertEqual([c['repack'] for c in calls], [True])

    def test_the_inference_timeout_never_outlives_the_job(self):
        from hfqa.quantize import inference_budget
        full = inference_budget(45 * 10 ** 9, 4)['timeout_seconds']
        self.assertGreater(full, 6000)
        calls, _ = self.run_with(45 * 10 ** 9, 16 * 10 ** 9, remaining=1800)
        self.assertLessEqual(calls[0]['timeout'], 1800)
        calls, _ = self.run_with(45 * 10 ** 9, 16 * 10 ** 9, remaining=10)
        self.assertEqual(calls[0]['timeout'], 300)  # a floor, so that a late start still gets an honest attempt
        from hfqa import runner_release
        import time
        self.assertLess(runner_release.remaining_job_seconds(time.monotonic() - 3 * 3600), 350 * 60 - 3 * 3600)

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
