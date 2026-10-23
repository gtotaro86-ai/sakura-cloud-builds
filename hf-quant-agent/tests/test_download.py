import json
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from hfqa.worker import download_decision, source_descriptor, source_destination, prepare_source
from hfqa.agent import advise
from hfqa.jobs import JobStore

class DownloadTests(unittest.TestCase):
    def test_source_pinned_and_limited(self):
        analysis = {"repo_id": "a/b", "revision": "a"*40, "model_type": "qwen2"}
        info = {"siblings": [{"rfilename": "b-q8_0.gguf", "size": 100, "lfs": {"sha256": "b"*64}}]}
        with patch('hfqa.inspect._json', return_value=info) as read:
            desc = source_descriptor(analysis)
        self.assertEqual(desc['sha256'], 'b'*64)
        self.assertIn('/revision/'+'a'*40, read.call_args.args[0])
        info['siblings'][0]['size'] = 10**12
        with patch('hfqa.inspect._json', return_value=info):
            with self.assertRaises(ValueError): source_descriptor(analysis)

    def test_traversal_rejected(self):
        for name in ['../outside.gguf', '/etc/a.gguf', 'a\\b.gguf', 'C:/x.gguf']:
            with self.assertRaises(ValueError):
                source_destination({'repo_id':'a/b','revision':'a'*40,'filename':name})

    def test_revision_required(self):
        with self.assertRaises(ValueError): source_descriptor({'repo_id':'a/b','revision':'main'})

    def test_spark_tool_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            with JobStore(Path(tmp)/'jobs.sqlite') as store:
                job,_=store.create('a/b','a/b',['Q4_K_M'])
                replies=iter([
                    {'choices':[{'message':{'role':'assistant','tool_calls':[{'id':'1','type':'function','function':{'name':'read_pinned_analysis','arguments':'{}'}}]}}]},
                    {'choices':[{'message':{'content':'Validate before publishing.'}}]}
                ])
                result=advise(store,job['id'],transport=lambda payload:next(replies))
                self.assertTrue(result['ok'])
                self.assertTrue(advise(store,job['id'])['reused'])

class FakeResponse:
    def __init__(self, body, *, status=200, headers=None, max_chunk=None, fail_on_read=None):
        self.body = io.BytesIO(body)
        self.status = status
        self.headers = headers or {}
        self.max_chunk = max_chunk
        self.fail_on_read = fail_on_read
        self.read_count = 0
        self.requested_sizes = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size):
        self.read_count += 1
        self.requested_sizes.append(size)
        if self.read_count == self.fail_on_read:
            raise ConnectionResetError("simulated interrupted HTTP stream")
        return self.body.read(min(size, self.max_chunk) if self.max_chunk else size)


class BoundedDownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.body = b"GGUF" + bytes(range(64))
        self.descriptor = {"repo_id": "owner/model-GGUF", "revision": "a" * 40,
                           "filename": "model-q8_0.gguf", "size": len(self.body),
                           "sha256": hashlib.sha256(self.body).hexdigest()}
        self.ticket = {"download": dict(self.descriptor)}
        self.job = {"analysis": {"repo_id": "owner/model", "revision": "b" * 40},
                    "quants": ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M"]}
        for patcher in (patch.dict("os.environ", {"HFQA_SOURCE_DIR": str(self.root)}),
                        patch("hfqa.worker.source_descriptor", return_value=self.descriptor),
                        patch("shutil.disk_usage", return_value=SimpleNamespace(free=100 * 1024**3))):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.destination = source_destination(self.descriptor)
        self.partial = self.destination.with_suffix(".gguf.partial")

    def test_interrupted_http_stream_resumes_with_range_and_pinned_digest(self):
        interrupted = FakeResponse(self.body, max_chunk=11, fail_on_read=2)
        resumed = FakeResponse(self.body[11:], status=206,
                               headers={"Content-Range": f"bytes 11-{len(self.body)-1}/{len(self.body)}"})
        with patch("urllib.request.urlopen", side_effect=[interrupted, resumed]) as network:
            with self.assertRaises(ConnectionResetError):
                prepare_source(self.ticket, self.job)
            self.assertEqual(self.partial.read_bytes(), self.body[:11])
            self.assertFalse(self.destination.exists())
            output = prepare_source(self.ticket, self.job)
        self.assertEqual(output.read_bytes(), self.body)
        self.assertFalse(self.partial.exists())
        first, second = [call.args[0] for call in network.call_args_list]
        self.assertIsNone(first.get_header("Range"))
        self.assertEqual(second.get_header("Range"), "bytes=11-")
        self.assertIn("/resolve/" + "a" * 40 + "/", second.full_url)
        provenance = json.loads(output.with_suffix(".source.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["sha256"], hashlib.sha256(self.body).hexdigest())

    def test_oversized_response_is_rejected_without_committing_source(self):
        response = FakeResponse(self.body + b"excess bytes", max_chunk=13)
        with patch("urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "exceeded pinned download size"):
                prepare_source(self.ticket, self.job)
        self.assertFalse(self.destination.exists())
        self.assertLessEqual(self.partial.stat().st_size, len(self.body))
        self.assertEqual(response.body.tell(), len(self.body) + 1)

    def test_hash_mismatch_discards_completed_corrupt_partial(self):
        corrupt = bytes([self.body[0] ^ 1]) + self.body[1:]
        with patch("urllib.request.urlopen", return_value=FakeResponse(corrupt)):
            with self.assertRaisesRegex(ValueError, "SHA-256 verification failed"):
                prepare_source(self.ticket, self.job)
        self.assertFalse(self.partial.exists())
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.destination.with_suffix(".source.json").exists())

    def test_insufficient_disk_rejected_before_network_or_partial_creation(self):
        required = len(self.body) * (len(self.job["quants"]) + 2) + 2 * 1024**3
        with patch("shutil.disk_usage", return_value=SimpleNamespace(free=required - 1)), \
             patch("urllib.request.urlopen") as network:
            with self.assertRaisesRegex(RuntimeError, "insufficient free disk"):
                prepare_source(self.ticket, self.job)
        network.assert_not_called()
        self.assertFalse(self.partial.exists())
        self.assertFalse(self.destination.exists())

    def test_ignored_range_restarts_full_response_without_appending(self):
        self.destination.parent.mkdir(parents=True)
        self.partial.write_bytes(self.body[:11])
        with patch("urllib.request.urlopen", return_value=FakeResponse(self.body)) as network:
            self.assertEqual(prepare_source(self.ticket, self.job).read_bytes(), self.body)
        self.assertEqual(network.call_args.args[0].get_header("Range"), "bytes=11-")

    def test_invalid_partial_range_cannot_corrupt_existing_partial(self):
        self.destination.parent.mkdir(parents=True)
        self.partial.write_bytes(self.body[:11])
        response = FakeResponse(self.body[11:], status=206, headers={"Content-Range": "bytes 12-67/68"})
        with patch("urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "invalid partial response"):
                prepare_source(self.ticket, self.job)
        self.assertEqual(self.partial.read_bytes(), self.body[:11])
        self.assertFalse(self.destination.exists())

    def test_changed_ticket_descriptor_rejected_before_network(self):
        self.ticket["download"]["sha256"] = "c" * 64
        with patch("urllib.request.urlopen") as network:
            with self.assertRaisesRegex(ValueError, "does not match pinned Hub source"):
                prepare_source(self.ticket, self.job)
        network.assert_not_called()

    def test_verified_cache_is_reused_without_network(self):
        self.destination.parent.mkdir(parents=True)
        self.destination.write_bytes(self.body)
        with patch("urllib.request.urlopen") as network:
            self.assertEqual(prepare_source(self.ticket, self.job), self.destination)
        network.assert_not_called()

    def test_short_eof_keeps_partial_and_next_attempt_resumes(self):
        short = FakeResponse(self.body[:11])
        resumed = FakeResponse(self.body[11:], status=206,
                               headers={"Content-Range": f"bytes 11-{len(self.body)-1}/{len(self.body)}"})
        with patch("urllib.request.urlopen", side_effect=[short, resumed]) as network:
            with self.assertRaisesRegex(OSError, "download incomplete"):
                prepare_source(self.ticket, self.job)
            self.assertEqual(self.partial.read_bytes(), self.body[:11])
            self.assertEqual(prepare_source(self.ticket, self.job).read_bytes(), self.body)
        self.assertEqual(network.call_args_list[1].args[0].get_header("Range"), "bytes=11-")

    def test_download_deadline_retains_resumable_partial(self):
        response = FakeResponse(self.body, max_chunk=11)
        with patch("urllib.request.urlopen", return_value=response), \
             patch("time.monotonic", side_effect=[100, 100, 1001]):
            with self.assertRaisesRegex(TimeoutError, "deadline reached"):
                prepare_source(self.ticket, self.job)
        self.assertEqual(self.partial.read_bytes(), self.body[:11])
        self.assertEqual(response.read_count, 1)
        self.assertFalse(self.destination.exists())

    def test_cached_source_still_requires_disk_reserve(self):
        self.destination.parent.mkdir(parents=True)
        self.destination.write_bytes(self.body)
        with patch("shutil.disk_usage", return_value=SimpleNamespace(free=1)), \
             patch("urllib.request.urlopen") as network:
            with self.assertRaisesRegex(RuntimeError, "insufficient free disk"):
                prepare_source(self.ticket, self.job)
        self.assertEqual(self.destination.read_bytes(), self.body)
        network.assert_not_called()

    def test_interrupted_legacy_adoption_never_commits_partial_copy(self):
        legacy = self.root / self.descriptor["filename"]
        legacy.write_bytes(self.body)
        def interrupted_copy(src, dst):
            Path(dst).write_bytes(Path(src).read_bytes()[:11])
            raise OSError("simulated interrupted filesystem copy")
        with patch("os.link", side_effect=OSError("hardlinks unavailable")), \
             patch("shutil.copyfile", side_effect=interrupted_copy), \
             patch("urllib.request.urlopen") as network:
            with self.assertRaisesRegex(OSError, "interrupted filesystem copy"):
                prepare_source(self.ticket, self.job)
        self.assertFalse(self.destination.exists())
        self.assertEqual(legacy.read_bytes(), self.body)
        network.assert_not_called()


if __name__ == '__main__': unittest.main()
