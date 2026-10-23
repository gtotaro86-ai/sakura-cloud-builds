"""Offline routing regressions: large models remain metadata on the controller."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hfqa.jobs import JobStore
from hfqa.orchestrator import Orchestrator
from hfqa.stream_controller import RUNTIME_REVISION, WORKFLOW, monitor_github, queue_stream


class StreamControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / "jobs.sqlite"
        self.store = JobStore(self.db)
        self.addCleanup(self.store.close)
        self.queue = self.root / "stream-queue"
        self.job_index = 0
        self.quants = ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M"]
        self.analysis = {"repo_id": "owner/model", "revision": "a" * 40,
                         "license_decision": {"redistributable": True},
                         "native_gguf_repo": {"repo_id": "quantizer/model-GGUF", "revision": "b" * 40}}
        self.selection = {"repo_id": "quantizer/model-GGUF", "revision": "b" * 40,
                          "total_source_bytes": 8_000_000_000,
                          "license_decision": {"redistributable": True},
                          "sources": [{"repo_id": "quantizer/model-GGUF", "revision": "b" * 40,
                                       "filename": f"model-Q8_0-{n:05}-of-00002.gguf",
                                       "size": 4_000_000_000, "sha256": str(n) * 64,
                                       "split_index": n, "split_count": 2} for n in (1, 2)]}
        for name, patcher in (
            ("env", patch.dict(os.environ, {"HFQA_STREAM_QUEUE_DIR": str(self.queue),
                                            "HFQA_QUEUE_DIR": str(self.root / "legacy-queue"),
                                            "HFQA_GITHUB_WORKER_SHA": ""})),
            ("select", patch("hfqa.stream_controller.select_stream_sources", side_effect=lambda _: copy.deepcopy(self.selection))),
            ("token", patch("hfqa.stream_controller.github_token", return_value="")),
            ("github", patch("hfqa.stream_controller.github_request", side_effect=AssertionError("unexpected GitHub request"))),
            ("network", patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden in routing test"))),
            ("download", patch("hfqa.worker.prepare_source", side_effect=AssertionError("full source materialization forbidden"))),
        ):
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)

    def job(self, backend="oracle", publish=True, variants=None):
        revision = "a" * 40 if self.job_index == 0 else f"{self.job_index:040x}"
        self.job_index += 1
        self.analysis["revision"] = revision
        job, _ = self.store.create("owner/model", "owner/model", self.quants,
                                   publish=publish, revision=revision,
                                   execution_backend=backend, variants=variants)
        self.store.transition(job["id"], "INSPECTING")
        self.store.update_fields(job["id"], analysis_json=self.analysis)
        self.store.transition(job["id"], "PLANNING")
        self.store.update_fields(job["id"], plan_json={"route": "native_gguf_requant", "quants": self.quants})
        self.store.transition(job["id"], "WAITING_FOR_WORKER")
        return self.store.get(job["id"])

    def test_oracle_ticket_contains_complete_pinned_manifest_without_changing_wait_state(self):
        job = self.job()
        result = queue_stream(self.store, job["id"], self.root / "work")
        self.assertEqual(result["state"], "WAITING_FOR_WORKER")
        ticket_path = self.queue / (job["id"] + ".json")
        ticket = json.loads(ticket_path.read_text(encoding="utf-8"))
        self.assertEqual(ticket["job_id"], job["id"])
        self.assertEqual(ticket["execution_backend"], "oracle")
        self.assertEqual(ticket["manifest"], {
            "sources": self.selection["sources"], "quants": self.quants,
            "runtime_revision": RUNTIME_REVISION,
            "source_license": {"redistributable": True},
            "original_source": {"repo_id": "owner/model", "revision": "a" * 40},
            "target_repo_id": "webmp3/model-GGUF", "target_revision": "main"})
        self.assertFalse(ticket_path.with_suffix(".tmp").exists())
        self.assertFalse(result["plan"]["publish_allowed"])
        self.assertTrue(result["plan"]["stream_stage_only"])
        self.assertFalse(result["plan"]["full_source_materialization"])
        self.assertEqual(result["plan"]["total_source_bytes"], 8_000_000_000)
        self.assertEqual(result["plan"]["source_shard_count"], 2)
        self.github.assert_not_called()
        self.network.assert_not_called()
        self.download.assert_not_called()

    def test_explicit_stream_backends_skip_legacy_source_resolution(self):
        for backend in ("oracle", "github"):
            with self.subTest(backend=backend):
                job = self.job(backend)
                with patch("hfqa.orchestrator.source_descriptor") as legacy, patch("hfqa.orchestrator.source_destination") as destination:
                    Orchestrator(self.store, self.root / "work")._queue_local_source(job["id"])
                legacy.assert_not_called()
                destination.assert_not_called()
                self.assertEqual(self.store.get(job["id"])["state"], "WAITING_FOR_WORKER")
        self.network.assert_not_called()
        self.download.assert_not_called()

    def test_auto_falls_back_to_oracle_stream_when_legacy_download_cap_rejects(self):
        job = self.job("auto")
        with patch("hfqa.orchestrator.source_descriptor", side_effect=ValueError("source exceeds download cap")) as legacy, patch("hfqa.orchestrator.source_destination") as destination:
            Orchestrator(self.store, self.root / "work")._queue_local_source(job["id"])
        legacy.assert_called_once()
        destination.assert_not_called()
        self.assertTrue((self.queue / (job["id"] + ".json")).is_file())
        self.assertFalse((self.root / "legacy-queue").exists())
        self.download.assert_not_called()

    def test_explicit_github_waits_without_token_or_pinned_worker_and_never_posts(self):
        for token, revision in (("", "c" * 40), ("fixture-token", ""),
                                 ("fixture-token", "main")):
            with self.subTest(token_present=bool(token), revision=revision):
                job = self.job("github")
                self.token.return_value = token
                with patch.dict(os.environ, {"HFQA_GITHUB_WORKER_SHA": revision}):
                    result = queue_stream(self.store, job["id"], self.root)
                self.assertEqual(result["state"], "WAITING_FOR_WORKER")
                self.assertFalse((result.get("result") or {}).get("github_stream"))
                self.assertIn("waiting", self.store.events_for(job["id"])[-1]["message"])
                self.github.assert_not_called()
                self.assertFalse(self.queue.exists())

    def test_publication_false_blocks_staging_before_any_queue_or_dispatch(self):
        job = self.job(publish=False)
        result = queue_stream(self.store, job["id"], self.root)
        self.assertEqual(result["state"], "BLOCKED")
        self.assertIn("publication was not requested", self.store.events_for(job["id"])[-1]["message"])
        self.assertFalse(self.queue.exists())
        self.github.assert_not_called()

    def test_both_original_and_native_licenses_must_explicitly_allow_staging(self):
        for original, native in ((False, True), (True, False), (None, True), (True, None), ("true", True)):
            with self.subTest(original=original, native=native):
                # Fresh pinned revisions keep blocked test cases independent.
                self.analysis["license_decision"] = {"redistributable": original}
                self.selection["license_decision"] = {"redistributable": native}
                job = self.job("github" if original is False else "oracle")
                result = queue_stream(self.store, job["id"], self.root)
                self.assertEqual(result["state"], "BLOCKED")
                self.assertFalse(self.queue.exists())
                self.github.assert_not_called()

    def test_backend_choice_persists_and_deduplicates_only_matching_backend(self):
        ids = {}
        for backend in ("auto", "oracle", "github"):
            job, created = self.store.create("owner/model", "owner/model", self.quants,
                                              revision="a" * 40, execution_backend=backend)
            self.assertTrue(created)
            ids[backend] = job["id"]
        self.assertEqual(len(set(ids.values())), 3)
        with JobStore(self.db) as reopened:
            for backend, job_id in ids.items():
                same, created = reopened.create("owner/model", "owner/model", list(reversed(self.quants)),
                                                 revision="a" * 40, execution_backend=backend)
                self.assertFalse(created)
                self.assertEqual(same["id"], job_id)
                self.assertEqual(same["execution_backend"], backend)
        with self.assertRaises(ValueError):
            self.store.create("owner/model", "owner/model", self.quants, execution_backend="remote-url")

    def test_github_dispatch_is_pinned_stage_only_and_repeated_queue_monitors_without_redispatch(self):
        job = self.job("github")
        self.token.return_value = "fixture-token"
        self.github.side_effect = [{"accepted": True}, {"workflow_runs": []}]
        with patch.dict(os.environ, {"HFQA_GITHUB_WORKER_SHA": "c" * 40}):
            first = queue_stream(self.store, job["id"], self.root)
            second = queue_stream(self.store, job["id"], self.root)
        self.assertEqual(first["state"], "WAITING_FOR_WORKER")
        self.assertEqual(second["state"], "WAITING_FOR_WORKER")
        self.assertEqual(self.github.call_count, 2)
        dispatch, monitor = self.github.call_args_list
        self.assertTrue(dispatch.args[0].endswith("/dispatches"))
        body = dispatch.kwargs["body"]
        self.assertEqual(body["ref"], "c" * 40)
        self.assertEqual(body["inputs"]["mode"], "stage")
        self.assertEqual(body["inputs"]["job_id"], job["id"])
        self.assertEqual(json.loads(body["inputs"]["manifest_json"])["quants"], self.quants)
        self.assertTrue(monitor.args[0].startswith("/actions/workflows/" + WORKFLOW + "/runs?"))
        self.assertNotIn("body", monitor.kwargs)
        self.assertTrue(second["result"]["github_stream"]["dispatch_intent_recorded"])
        self.assertFalse(second["result"]["published"])
        self.assertFalse(self.queue.exists())

    def test_ambiguous_dispatch_records_intent_and_cannot_post_again(self):
        job = self.job("github")
        self.token.return_value = "fixture-token"
        self.github.side_effect = [TimeoutError("secret fixture-token"), {"workflow_runs": []}]
        with patch.dict(os.environ, {"HFQA_GITHUB_WORKER_SHA": "c" * 40}):
            first = queue_stream(self.store, job["id"], self.root)
            queue_stream(self.store, job["id"], self.root)
        self.assertEqual(first["result"]["github_stream"]["state"], "DISPATCH_OUTCOME_REQUIRES_CHECK")
        self.assertNotIn("fixture-token", json.dumps(first))
        self.assertEqual(sum("body" in call.kwargs for call in self.github.call_args_list), 1)

    def test_monitor_matches_both_pinned_worker_and_job_identity_and_never_publishes(self):
        job = self.job("github")
        self.store.update_fields(job["id"], result_json={"published": False, "github_stream": {
            "dispatch_intent_recorded": True, "worker_revision": "c" * 40}})
        matching = {"id": 2, "head_sha": "c" * 40, "display_title": "HFQA stream " + job["id"],
                    "status": "completed", "conclusion": "success", "html_url": "https://github.com/example/actions/runs/2"}
        unrelated = [{**matching, "id": 99, "head_sha": "d" * 40},
                     {**matching, "id": 100, "display_title": "HFQA stream other-job"}]
        self.github.side_effect = [{"workflow_runs": unrelated}, {"workflow_runs": unrelated + [matching]}]
        waiting = monitor_github(self.store, job["id"])
        self.assertEqual(waiting["state"], "WAITING_FOR_WORKER")
        completed = monitor_github(self.store, job["id"])
        self.assertEqual(completed["state"], "BLOCKED")
        self.assertEqual(completed["result"]["github_stream"]["run_id"], 2)
        self.assertFalse(completed["result"]["published"])

    def test_nonwaiting_jobs_cannot_queue_or_dispatch(self):
        job = self.job()
        self.store.transition(job["id"], "PAUSED")
        result = queue_stream(self.store, job["id"], self.root)
        self.assertEqual(result["state"], "PAUSED")
        self.select.assert_not_called()
        self.github.assert_not_called()
        self.assertFalse(self.queue.exists())

    def test_heretic_variants_block_before_source_selection_or_queue(self):
        for variants in (["heretic"], ["normal", "heretic"]):
            with self.subTest(variants=variants):
                job = self.job(variants=variants)
                result = queue_stream(self.store, job["id"], self.root)
                self.assertEqual(result["state"], "BLOCKED")
                self.assertFalse(self.queue.exists())
        self.select.assert_not_called()
        self.github.assert_not_called()

    def test_oracle_existing_ticket_is_reused_and_changed_manifest_is_rejected(self):
        job = self.job()
        queue_stream(self.store, job["id"], self.root)
        ticket = self.queue / (job["id"] + ".json")
        initial = ticket.read_bytes()
        initial_time = ticket.stat().st_mtime_ns
        event_count = len(self.store.events_for(job["id"]))
        queue_stream(self.store, job["id"], self.root)
        self.assertEqual(ticket.stat().st_mtime_ns, initial_time)
        self.assertEqual(len(self.store.events_for(job["id"])), event_count)
        self.selection["sources"][0]["sha256"] = "f" * 64
        with self.assertRaisesRegex(RuntimeError, "differs from pinned job"):
            queue_stream(self.store, job["id"], self.root)
        self.assertEqual(ticket.read_bytes(), initial)

    def test_lost_github_credential_after_dispatch_intent_keeps_waiting_without_network(self):
        job = self.job("github")
        previous = {"dispatch_intent_recorded": True, "worker_revision": "c" * 40,
                    "state": "DISPATCH_OUTCOME_REQUIRES_CHECK"}
        self.store.update_fields(job["id"], result_json={"published": False, "github_stream": previous})
        result = queue_stream(self.store, job["id"], self.root)
        self.assertEqual(result["state"], "WAITING_FOR_WORKER")
        self.assertEqual(result["result"]["github_stream"], previous)
        self.github.assert_not_called()

    def test_dispatch_intent_is_durable_before_external_post(self):
        job = self.job("github")
        self.token.return_value = "fixture-token"
        def post_after_commit(path, *, body):
            with JobStore(self.db) as observer:
                checkpoint = observer.get(job["id"])["result"]["github_stream"]
                self.assertTrue(checkpoint["dispatch_intent_recorded"])
                self.assertEqual(checkpoint["state"], "DISPATCHING")
                self.assertEqual(checkpoint["worker_revision"], body["ref"])
            return {"accepted": True}
        self.github.side_effect = post_after_commit
        with patch.dict(os.environ, {"HFQA_GITHUB_WORKER_SHA": "c" * 40}):
            result = queue_stream(self.store, job["id"], self.root)
        self.assertEqual(result["result"]["github_stream"]["state"], "DISPATCHED")


if __name__ == "__main__":
    unittest.main()
