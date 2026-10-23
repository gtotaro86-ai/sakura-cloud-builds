import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hfqa.gguf import GGUFError, validate_gguf, write_minimal_gguf
from hfqa.jobs import JobStore
from hfqa.publish import publication_gate, render_readme, sha256_file, upload_verified_release
from hfqa.quantize import quantize_file
from scripts import run_worker_ticket as worker


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source-q8_0.gguf'
        write_minimal_gguf(self.source, 'qwen2')
        self.quantizer = self.root / 'llama-quantize'
        self.runtime = self.root / 'llama-cli'
        self.quantizer.write_bytes(b'stock quantizer test fixture')
        self.runtime.write_bytes(b'stock runtime test fixture')
        self.store = JobStore(self.root / 'jobs.sqlite')
        self.analysis = {'repo_id': 'Qwen/test', 'revision': 'a' * 40, 'license': 'apache-2.0', 'license_decision': {'redistributable': True}, 'trust_remote_code_executed': False, 'native_gguf_repo': {'repo_id': 'Qwen/test-GGUF', 'revision': 'b'*40}}
        self.quants = ['Q2_K', 'Q3_K_M', 'Q4_K_M', 'Q5_K_M']
        self.quant_runs, self.infer_runs = [], []
        self.after_quant = None
        self.inference_failure = False

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def job(self, publish=False):
        job, _ = self.store.create('Qwen/test', 'Qwen/test', self.quants, publish=publish, revision='a' * 40)
        self.store.transition(job['id'], 'INSPECTING')
        self.store.update_fields(job['id'], analysis_json=self.analysis)
        self.store.transition(job['id'], 'PLANNING')
        self.store.update_fields(job['id'], plan_json={'route': 'native_gguf_requant', 'allow_new_patch': False, 'publish_allowed': True, 'quants': self.quants})
        self.store.transition(job['id'], 'WAITING_FOR_WORKER')
        self.ticket = self.root / (job['id'] + '.json')
        self.ticket.write_text(json.dumps({'job_id': job['id'], 'source': str(self.source), 'quant': 'Q2_K', 'download': {'repo_id': 'Qwen/test-GGUF', 'revision': 'b'*40, 'filename': self.source.name, 'size': self.source.stat().st_size, 'sha256': sha256_file(self.source)}}))
        self.jid = job['id']
        return job

    def fake_run(self, cmd, **kwargs):
        if '--allow-requantize' in cmd:
            self.quant_runs.append(cmd[4])
            write_minimal_gguf(cmd[3], 'qwen2')
            kwargs['stdout'].write(b'quantization complete\n')
            kwargs['stderr'].write(b'WARNING: 1 of 1 tensor(s) required fallback quantization\n')
            if self.after_quant:
                self.after_quant()
                self.after_quant = None
            return subprocess.CompletedProcess(cmd, 0, b'', b'')
        self.infer_runs.append(cmd[cmd.index('-m') + 1])
        kwargs['stdout'].write(b' Paris.\n')
        kwargs['stderr'].write(b'build: fixture-build-for-unit-test\nllama_perf_context_print: eval time = 100.0 ms / 8 runs\n')
        return subprocess.CompletedProcess(cmd, 1 if self.inference_failure else 0)

    def run_worker(self):
        with patch('scripts.run_worker_ticket.prepare_source', return_value=self.source), patch('hfqa.quantize.subprocess.run', side_effect=self.fake_run), patch.dict(os.environ, {'HFQA_SPARK_ENABLED': '0'}):
            return worker.process_ticket(self.ticket, self.store, self.quantizer, self.runtime, self.root / 'outputs')

    def test_all_quants_and_real_process_evidence_per_output(self):
        self.job()
        result = self.run_worker()
        self.assertEqual(result['state'], 'COMPLETED')
        self.assertEqual(self.quant_runs, self.quants)
        self.assertEqual(len(self.infer_runs), 4)
        self.assertEqual(len(result['result']['artifacts']), 4)
        self.assertFalse(result['result']['published'])
        self.assertFalse(self.ticket.exists())
        self.assertTrue(all(s['state'] == 'validated' for s in self.store.shards(self.jid)))
        artifact = result['result']['artifacts'][0]
        self.assertEqual(artifact['quantization']['tensor_type_histogram'], {'F32': 1})
        self.assertEqual(artifact['quantization']['fallback_tensor_count'], 1)
        self.assertTrue(Path(artifact['quantization']['quantize_stderr_log']).is_file())
        self.assertEqual(artifact['inference']['runtime_build'], 'fixture-build-for-unit-test')

    def test_quantizer_cannot_replace_source_tensor_names(self):
        output = self.root / 'wrong-tensors.gguf'
        def wrong_tensors(cmd, **kwargs):
            write_minimal_gguf(cmd[3], 'qwen2', tensor_name='wrong.weight')
            return subprocess.CompletedProcess(cmd, 0)
        with patch('hfqa.quantize.subprocess.run', side_effect=wrong_tensors):
            with self.assertRaisesRegex(ValueError, 'tensor count/name set'):
                quantize_file(self.quantizer, self.source, output, 'Q2_K')
        self.assertFalse(output.exists())
        write_minimal_gguf(output, 'qwen2', tensor_name='wrong.weight')
        with self.assertRaisesRegex(ValueError, 'tensor count/name set'):
            quantize_file(self.quantizer, self.source, output, 'Q2_K', expected_sha256=sha256_file(output))

    def test_card_discloses_requantization_runtime_and_fallback_types(self):
        self.job()
        result = self.run_worker()['result']
        card = render_readme(self.analysis, self.store.get(self.jid)['plan'], result, result['artifacts'])
        self.assertIn('existing-GGUF requantization', card)
        self.assertIn('did not convert the upstream BF16/safetensors', card)
        self.assertIn('No importance-matrix calibration was run', card)
        self.assertIn('F32: 1', card)
        self.assertIn('fixture-build-for-unit-test', card)
        self.assertIn(sha256_file(self.runtime), card)
        self.assertIn(validate_gguf(self.source)['tensor_name_set_sha256'], card)

    def test_pause_after_quant_resumes_without_recomputing(self):
        self.job()
        self.after_quant = lambda: self.store.transition(self.jid, 'PAUSED')
        paused = self.run_worker()
        self.assertEqual(paused['state'], 'PAUSED')
        self.assertEqual(len(self.quant_runs), 1)
        self.assertEqual(len(self.infer_runs), 0)
        self.assertTrue(self.ticket.exists())
        self.store.transition(self.jid, paused['paused_from'])
        final = self.run_worker()
        self.assertEqual(final['state'], 'COMPLETED')
        self.assertEqual(self.quant_runs, self.quants)
        self.assertEqual(len(self.infer_runs), 4)

    def test_cancel_after_quant_never_infers_or_completes(self):
        self.job()
        self.after_quant = lambda: self.store.transition(self.jid, 'CANCELLED')
        cancelled = self.run_worker()
        self.assertEqual(cancelled['state'], 'CANCELLED')
        self.assertEqual(len(self.infer_runs), 0)
        self.assertEqual(len(self.store.shards(self.jid)), 1)

    def test_runtime_failure_blocks_completion_and_preserves_ticket(self):
        self.job()
        self.inference_failure = True
        result = self.run_worker()
        self.assertEqual(result['state'], 'BLOCKED')
        self.assertTrue(self.ticket.exists())
        self.assertEqual(len(self.quant_runs), 1)
        self.assertFalse((result.get('result') or {}).get('inference_tested', False))
        self.assertEqual(self.store.shards(self.jid)[0]['state'], 'quantized')

    def test_missing_publication_token_never_marks_completed(self):
        self.job(publish=True)
        with patch('scripts.run_worker_ticket._read_token', side_effect=RuntimeError('HF token missing')):
            result = self.run_worker()
        self.assertEqual(result['state'], 'BLOCKED')
        self.assertTrue(result['result']['inference_tested'])
        self.assertFalse(result['result']['published'])
        self.assertEqual(len(self.quant_runs), 4)
        self.assertTrue(self.ticket.exists())

    def test_another_live_worker_prevents_duplicate_work(self):
        self.job()
        self.assertTrue(self.store.claim(self.jid, 'other', ttl_seconds=60))
        with self.assertRaises(worker.WorkerBusy):
            self.run_worker()
        self.assertEqual(self.quant_runs, [])
        self.store.release_claim(self.jid, 'other')

    def test_legacy_local_source_cannot_publish_without_upstream_provenance(self):
        self.job(publish=True)
        ticket = json.loads(self.ticket.read_text())
        ticket.pop('download')
        self.ticket.write_text(json.dumps(ticket))
        result = self.run_worker()
        self.assertEqual(result['state'], 'BLOCKED')
        self.assertIn('provenance', result['error'])
        self.assertEqual(self.quant_runs, [])

    def test_paused_upload_rejects_changed_runtime_instead_of_stalling(self):
        self.job(publish=True)
        with patch('scripts.run_worker_ticket._read_token', return_value='hf_test'), patch('scripts.run_worker_ticket.upload_verified_release', side_effect=worker.WorkerStopped('simulate crash at upload')):
            result = self.run_worker()
        self.assertEqual(result['state'], 'UPLOADING')
        self.runtime.write_bytes(b'changed runtime')
        blocked = self.run_worker()
        self.assertEqual(blocked['state'], 'BLOCKED')
        self.assertIn('runtime changed', blocked['error'])
        self.assertEqual(len(self.quant_runs), 4)

    def test_orphan_atomic_output_uses_manifest_without_recompute(self):
        output = self.root / 'out.gguf'
        with patch('hfqa.quantize.subprocess.run', side_effect=self.fake_run):
            first = quantize_file(self.quantizer, self.source, output, 'Q2_K')
            second = quantize_file(self.quantizer, self.source, output, 'Q2_K')
        self.assertEqual(len(self.quant_runs), 1)
        self.assertTrue(second['skipped'])
        self.assertEqual(first['sha256'], second['sha256'])
        output.write_bytes(b'corrupt')
        with self.assertRaises(FileExistsError):
            quantize_file(self.quantizer, self.source, output, 'Q2_K')

    def test_truncated_tensor_data_fails_structural_gate(self):
        self.source.write_bytes(self.source.read_bytes()[:-1])
        with self.assertRaises(GGUFError):
            validate_gguf(self.source)

    def test_publication_requires_requested_license_and_all_quant_artifacts(self):
        with self.assertRaisesRegex(RuntimeError, 'not requested'):
            publication_gate(False, self.analysis, {}, {}, [])
        with self.assertRaisesRegex(RuntimeError, 'license'):
            publication_gate(True, {}, {}, {}, [])
        self.job()
        result = self.run_worker()['result']
        files = [Path(a['path']) for a in result['artifacts']]
        plan = {'publish_allowed': True, 'quants': self.quants}
        publication_gate(True, self.analysis, plan, result, files)
        with self.assertRaisesRegex(RuntimeError, 'set is incomplete'):
            publication_gate(True, self.analysis, plan, result, files[:-1])
        files[0].write_bytes(b'corrupt')
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            publication_gate(True, self.analysis, plan, result, files)

    def test_publisher_resumes_by_hash_and_verifies_model_card(self):
        import sys
        self.job()
        validation = self.run_worker()['result']
        files = [Path(a['path']) for a in validation['artifacts']]
        readme = '# Model card\n'
        blobs = {p.name: p.read_bytes() for p in files}
        blobs['README.md'] = readme.encode()
        uploads = []
        def info(*args, **kwargs):
            siblings = [SimpleNamespace(rfilename=name, size=len(data), lfs=None, blob_id=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()) for name, data in blobs.items()]
            return SimpleNamespace(sha='b'*40, siblings=siblings, card_data={'base_model': 'Qwen/test', 'hfqa_source_revision': 'a'*40})
        api = SimpleNamespace(whoami=lambda: {'name': 'webmp3'}, create_repo=lambda *a, **k: None, model_info=info, create_commit=lambda **kw: uploads.append(kw))
        with patch.dict(sys.modules, {'huggingface_hub': SimpleNamespace(HfApi=lambda **kwargs: api, CommitOperationAdd=lambda **kw: kw)}):
            release = upload_verified_release('webmp3/test-GGUF', readme, files, 'hf_test', requested=True, analysis=self.analysis, plan={'publish_allowed': True, 'quants': self.quants}, validation=validation)
            self.assertTrue(release['verified'])
            self.assertTrue(release['readme_verified'])
            self.assertEqual(uploads, [])
            blobs[files[0].name] = b'x' * len(blobs[files[0].name])
            with self.assertRaisesRegex(RuntimeError, 'hash verification'):
                upload_verified_release('webmp3/test-GGUF', readme, files, 'hf_test', requested=True, analysis=self.analysis, plan={'publish_allowed': True, 'quants': self.quants}, validation=validation)
            # A new target receives every quant plus README in exactly one commit.
            blobs.clear()
            uploads.clear()
            def commit(**kwargs):
                uploads.append(kwargs)
                for op in kwargs['operations']:
                    payload = op['path_or_fileobj']
                    blobs[op['path_in_repo']] = payload if isinstance(payload, bytes) else Path(payload).read_bytes()
            api.create_commit = commit
            release = upload_verified_release('webmp3/test-GGUF', readme, files, 'hf_test', requested=True, analysis=self.analysis, plan={'publish_allowed': True, 'quants': self.quants}, validation=validation)
            self.assertTrue(release['verified'])
            self.assertEqual(len(uploads), 1)
            self.assertEqual(len(uploads[0]['operations']), 5)
            self.assertEqual(set(blobs), {p.name for p in files} | {'README.md'})

if __name__ == '__main__':
    unittest.main()
