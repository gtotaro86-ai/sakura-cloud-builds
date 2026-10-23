"""Durability/state tests with explicit native, HTTP and Xet doubles.

No test in this file performs a real quantization, network call or publication.
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from hfqa.jobs import JobStore
from hfqa.streaming import shard_filename
from scripts import run_oracle_stream_ticket as worker


class OracleStreamTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = JobStore(self.root / 'jobs.sqlite')
        self.quants = ['Q2_K', 'Q3_K_M', 'Q4_K_M', 'Q5_K_M']
        self.source = {'repo_id': 'Qwen/test-GGUF', 'revision': 'b' * 40,
                       'filename': 'test-q8_0.gguf', 'size': 12345, 'sha256': 'c' * 64}
        self.manifest = {'source': self.source, 'quants': self.quants,
            'runtime_revision': worker.RUNTIME_REVISION,
            'source_license': {'redistributable': True, 'license': 'apache-2.0'},
            'target_repo_id': 'webmp3/test-GGUF', 'target_revision': 'main'}
        self.native = SimpleNamespace(identity={'runtime_revision': worker.RUNTIME_REVISION,
            'bridge_sha256': '1' * 64, 'ggml_sha256': '2' * 64, 'llama_sha256': '3' * 64})
        self.api = SimpleNamespace(whoami=Mock(return_value={'name': 'webmp3'}), create_repo=Mock(), create_commit=Mock())
        self.runs = []
        self.after_quant = None
        self.fail_quant = None
        worker.SHUTDOWN.clear()

    def tearDown(self):
        worker.SHUTDOWN.clear()
        self.store.close()
        self.tmp.cleanup()

    def job(self, publish=True):
        job, _ = self.store.create('Qwen/test', 'Qwen/test', self.quants, publish=publish,
                                   revision='a' * 40, execution_backend='oracle')
        self.jid = job['id']
        self.analysis = {'repo_id': 'Qwen/test', 'revision': 'a' * 40,
            'native_gguf_repo': {'repo_id': self.source['repo_id'], 'revision': self.source['revision']},
            'license_decision': {'redistributable': True}, 'trust_remote_code_executed': False}
        self.plan = {'route': 'native_gguf_stream', 'execution_backend': 'oracle',
                     'allow_new_patch': False, 'publish_allowed': False,
                     'release_repo_id': self.manifest['target_repo_id']}
        self.store.transition(self.jid, 'INSPECTING')
        self.store.update_fields(self.jid, analysis_json=self.analysis)
        self.store.transition(self.jid, 'PLANNING')
        self.store.update_fields(self.jid, plan_json=self.plan)
        self.store.transition(self.jid, 'WAITING_FOR_WORKER')
        self.ticket = self.root / (self.jid + '.json')
        self.ticket.write_text(json.dumps({'job_id': self.jid, 'execution_backend': 'oracle', 'manifest': self.manifest}))
        return job

    def fake_stream(self, remote, source, quant, native, **kwargs):
        return SimpleNamespace(source=source, quant=quant, native=native,
                               should_stop=kwargs['should_stop'])

    def fake_upload(self, api, target, revision, stem, stream):
        if stream.should_stop():
            raise InterruptedError('paused before stream')
        self.runs.append(stream.quant)
        if self.fail_quant == stream.quant:
            raise RuntimeError('Xet failed; hf_abcdefghijk_SECRET should be redacted')
        digest = str(self.quants.index(stream.quant) + 4) * 64
        receipt = {'source': copy.deepcopy(stream.source), 'native': copy.deepcopy(stream.native.identity),
            'quant': stream.quant, 'source_sha256_verified': True, 'stream_complete': True,
            'source_bytes_hashed': stream.source['size'], 'output_materialized': False,
            'source_materialized': False, 'full_inference_test_performed': False,
            'shard_count': 1, 'shards': [{'index': 0, 'size': 100, 'sha256': digest}]}
        stage = {'files': [{'path': shard_filename(stem, stream.quant, 0, 1), 'size': 100,
                           'sha256': digest, 'xet_hash': 'e' * 64}],
                 'receipt': receipt, 'native_sha256_verified': True, 'hub_commit_created': False, 'published': False}
        if self.after_quant:
            callback, self.after_quant = self.after_quant, None
            callback()
        return stage

    def run_worker(self):
        hub = SimpleNamespace(HfApi=lambda **kwargs: self.api)
        with patch('scripts.run_oracle_stream_ticket.NativeQuantizer', return_value=self.native), \
             patch('scripts.run_oracle_stream_ticket._inspect', return_value=object()) as inspect, \
             patch('scripts.run_oracle_stream_ticket.ShardedQuantizedStream', side_effect=self.fake_stream), \
             patch('scripts.run_oracle_stream_ticket.upload_xet_sharded_stream', side_effect=self.fake_upload), \
             patch('scripts.run_oracle_stream_ticket._read_token', return_value='hf_unit_test_only'), \
             patch.dict('sys.modules', {'huggingface_hub': hub}):
            result = worker.process_ticket(self.ticket, self.store, 'unused-bridge', 'unused-ggml', self.root / 'receipts')
        self.inspect_calls = inspect.call_count
        return result

    def retry(self):
        self.store.retry(self.jid)
        self.store.update_fields(self.jid, plan_json=self.plan)
        self.store.transition(self.jid, 'WAITING_FOR_WORKER')

    def test_all_quants_stage_without_completion_or_a_hub_commit(self):
        self.job()
        result = self.run_worker()
        self.assertEqual(self.runs, self.quants)
        self.assertEqual(result['state'], 'BLOCKED')
        self.assertEqual(result['result']['stage_state'], 'AWAITING_RUNTIME_VALIDATION')
        self.assertFalse(result['result']['published'])
        self.assertFalse(result['result']['inference_tested'])
        self.assertFalse(result['result']['source_materialized'])
        self.assertFalse(result['result']['output_materialized'])
        self.assertEqual(len(self.store.shards(self.jid)), 4)
        self.assertTrue(self.ticket.exists())
        self.api.create_commit.assert_not_called()
        self.api.create_repo.assert_called_once_with('webmp3/test-GGUF', repo_type='model', exist_ok=True, private=False)
        files = list((self.root / 'receipts' / self.jid).iterdir())
        self.assertEqual(len(files), 6)
        self.assertTrue(all(path.suffix == '.json' for path in files))

    def test_completed_quants_reused_without_http_or_quantization_on_retry(self):
        self.job()
        self.run_worker()
        self.retry()
        self.api.create_repo.reset_mock()
        result = self.run_worker()
        self.assertEqual(result['state'], 'BLOCKED')
        self.assertEqual(self.runs, self.quants)
        self.assertEqual(self.inspect_calls, 0)
        self.api.create_repo.assert_not_called()
        self.api.create_commit.assert_not_called()

    def test_pause_after_success_keeps_receipt_before_state_check_and_resumes(self):
        self.job()
        self.after_quant = lambda: self.store.transition(self.jid, 'PAUSED')
        paused = self.run_worker()
        self.assertEqual(paused['state'], 'PAUSED')
        self.assertEqual(self.runs, ['Q2_K'])
        self.assertTrue((self.root / 'receipts' / self.jid / 'Q2_K.json').exists())
        self.assertEqual(self.store.shards(self.jid), [])
        self.store.transition(self.jid, paused['paused_from'])
        result = self.run_worker()
        self.assertEqual(self.runs, self.quants)
        self.assertEqual(result['result']['staged_quants'], self.quants)
        self.assertEqual(len(self.store.shards(self.jid)), 4)

    def test_completed_stage_after_cancel_is_retained_without_more_work(self):
        self.job()
        self.after_quant = lambda: self.store.transition(self.jid, 'CANCELLED')
        result = self.run_worker()
        self.assertEqual(result['state'], 'CANCELLED')
        self.assertEqual(self.runs, ['Q2_K'])
        self.assertTrue((self.root / 'receipts' / self.jid / 'Q2_K.json').exists())
        self.api.create_commit.assert_not_called()

    def test_mid_quant_failure_keeps_previous_quant_and_redacts_secret(self):
        self.job()
        self.fail_quant = 'Q3_K_M'
        result = self.run_worker()
        self.assertEqual(result['state'], 'BLOCKED')
        self.assertNotIn('hf_abcdefghijk_SECRET', result['error'])
        self.assertIn('[REDACTED]', result['error'])
        self.assertTrue((self.root / 'receipts' / self.jid / 'Q2_K.json').exists())
        self.assertFalse((self.root / 'receipts' / self.jid / 'Q3_K_M.json').exists())
        self.fail_quant = None
        self.retry()
        self.run_worker()
        self.assertEqual(self.runs, ['Q2_K', 'Q3_K_M', 'Q3_K_M', 'Q4_K_M', 'Q5_K_M'])

    def test_changed_native_identity_blocks_checkpoint_reuse_without_recompute(self):
        self.job()
        self.run_worker()
        self.retry()
        self.native.identity['llama_sha256'] = 'f' * 64
        result = self.run_worker()
        self.assertEqual(result['state'], 'BLOCKED')
        self.assertIn('native identity', result['error'])
        self.assertEqual(self.runs, self.quants)
        self.assertEqual(self.inspect_calls, 0)

    def test_changed_manifest_blocks_before_native_or_network_work(self):
        self.job()
        self.run_worker()
        self.retry()
        ticket = json.loads(self.ticket.read_text())
        ticket['manifest']['source']['sha256'] = 'f' * 64
        self.ticket.write_text(json.dumps(ticket))
        result = self.run_worker()
        self.assertIn('manifest changed', result['error'])
        self.assertEqual(self.runs, self.quants)
        self.assertEqual(self.inspect_calls, 0)

    def test_staging_requires_publish_request_and_both_license_decisions(self):
        self.job(publish=False)
        result = self.run_worker()
        self.assertIn('requires publication requested', result['error'])
        self.assertEqual(self.runs, [])
        self.api.create_repo.assert_not_called()

    def test_inspected_license_denial_blocks_before_external_write(self):
        self.job()
        self.analysis['license_decision']['redistributable'] = False
        self.store.update_fields(self.jid, analysis_json=self.analysis)
        result = self.run_worker()
        self.assertEqual(result['state'], 'BLOCKED')
        self.assertIn('permit redistribution', result['error'])
        self.assertEqual(self.runs, [])
        self.api.create_repo.assert_not_called()

    def test_manifest_license_denial_blocks_before_external_write(self):
        self.manifest['source_license']['redistributable'] = False
        self.job()
        result = self.run_worker()
        self.assertEqual(result['state'], 'BLOCKED')
        self.assertIn('permit redistribution', result['error'])
        self.assertEqual(self.runs, [])
        self.api.create_repo.assert_not_called()

    def test_backend_and_upstream_source_must_match_inspected_plan(self):
        self.job()
        self.plan['execution_backend'] = 'github'
        self.store.update_fields(self.jid, plan_json=self.plan)
        result = self.run_worker()
        self.assertIn('explicitly select', result['error'])
        self.assertEqual(self.runs, [])
        self.retry()
        self.plan['execution_backend'] = 'oracle'
        self.analysis['native_gguf_repo']['revision'] = 'd' * 40
        self.store.update_fields(self.jid, plan_json=self.plan, analysis_json=self.analysis)
        result = self.run_worker()
        self.assertIn('inspected immutable GGUF', result['error'])
        self.api.create_repo.assert_not_called()

    def test_another_worker_lease_prevents_duplicate_work(self):
        self.job()
        self.assertTrue(self.store.claim(self.jid, 'other', ttl_seconds=60))
        with self.assertRaises(worker.WorkerBusy):
            self.run_worker()
        self.store.release_claim(self.jid, 'other')
        self.assertEqual(self.runs, [])

    def test_heartbeat_renews_even_when_quantizer_is_blocked(self):
        self.job()
        self.assertTrue(self.store.claim(self.jid, 'owner', ttl_seconds=180))
        guard = worker.LeaseGuard(self.store, self.jid, 'owner')
        with patch.object(guard.stop, 'wait', side_effect=[False, True]), patch.object(self.store, 'renew_claim', wraps=self.store.renew_claim) as renew:
            guard._heartbeat()
        renew.assert_called_once_with(self.jid, 'owner', ttl_seconds=180)
        self.assertFalse(guard.lost.is_set())
        self.store.release_claim(self.jid, 'owner')
        with patch.object(guard.stop, 'wait', return_value=False):
            guard._heartbeat()
        self.assertTrue(guard.lost.is_set())
        with self.assertRaises(worker.WorkerStopped):
            guard.check()

    def test_flat_ticket_schema_has_the_same_durable_manifest(self):
        self.job()
        self.ticket.write_text(json.dumps({'job_id': self.jid, 'execution_backend': 'oracle', **self.manifest}))
        result = self.run_worker()
        self.assertEqual(result['state'], 'BLOCKED')
        self.assertEqual(result['result']['stage_state'], 'AWAITING_RUNTIME_VALIDATION')
        self.assertEqual(self.runs, self.quants)

    def test_wrong_hf_identity_cannot_create_repository(self):
        self.job()
        self.api.whoami.return_value = {'name': 'different-publisher'}
        result = self.run_worker()
        self.assertIn('qualified publisher', result['error'])
        self.assertEqual(self.runs, [])
        self.api.create_repo.assert_not_called()


if __name__ == '__main__':
    unittest.main()
