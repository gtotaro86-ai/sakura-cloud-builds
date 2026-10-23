import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from hfqa.heretic import assess, normalize_variants
from hfqa.jobs import JobStore, TransitionError
from hfqa.native import classify
from hfqa.planner import plan


REV = "a" * 40


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "jobs.sqlite"
        self.store = JobStore(self.db)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def create(self, **kwargs):
        return self.store.create("https://huggingface.co/owner/model", "owner/model", kwargs.pop("quants", ["Q4_K_M"]), **kwargs)

    def test_resolved_revision_dedupe_includes_variants_publish_and_quant_set(self):
        first, _ = self.create(revision=REV, quants=["Q4_K_M", "Q2_K"], variants=["both"])
        same, created = self.create(revision=REV, quants=["Q2_K", "Q4_K_M"], variants=["heretic", "normal"])
        self.assertFalse(created)
        self.assertEqual(first["id"], same["id"])
        for changes in ({"revision": "b" * 40}, {"variants": ["normal"]}, {"publish": True}):
            args = dict(revision=REV, quants=["Q4_K_M", "Q2_K"], variants=["both"])
            args.update(changes)
            self.assertTrue(self.create(**args)[1])

    def test_mutable_revision_only_dedupes_after_pinning(self):
        first, _ = self.create(revision="main")
        second, created = self.create(revision="main")
        self.assertTrue(created)
        self.assertTrue(self.store.pin_revision(first["id"], REV)[1])
        canonical, pinned = self.store.pin_revision(second["id"], REV)
        self.assertFalse(pinned)
        self.assertEqual(canonical["id"], first["id"])
        self.assertEqual(self.store.get(second["id"])["state"], "CANCELLED")
        self.assertEqual(self.store.events_for(second["id"])[-1]["data"]["canonical_job_id"], first["id"])

    def test_atomic_dedupe_across_connections(self):
        def submit(_):
            with JobStore(self.db) as store:
                return store.create("owner/model", "owner/model", ["Q4_K_M"], revision=REV)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(submit, range(8)))
        self.assertEqual(sum(created for _, created in results), 1)
        self.assertEqual(len({job["id"] for job, _ in results}), 1)

    def test_cancel_and_pause_cannot_be_overwritten_by_stale_worker(self):
        for target in ("PAUSED", "CANCELLED"):
            job, _ = self.create()
            self.store.transition(job["id"], "INSPECTING")
            self.store.transition(job["id"], target)
            with self.assertRaises(TransitionError):
                self.store.transition(job["id"], "PLANNING", expected_state="INSPECTING")
            with self.assertRaises(TransitionError):
                self.store.update_fields(job["id"], expected_state="INSPECTING", error="late worker")
            self.assertEqual(self.store.get(job["id"])["state"], target)
            self.assertEqual(self.store.events_for(job["id"])[-1]["to_state"], target)

    def test_state_and_event_rollback_together(self):
        job, _ = self.create()
        with patch.object(self.store, "_event", side_effect=RuntimeError("disk full")):
            with self.assertRaises(RuntimeError):
                self.store.transition(job["id"], "INSPECTING")
        self.assertEqual(self.store.get(job["id"])["state"], "CREATED")
        self.assertEqual(len(self.store.events_for(job["id"])), 1)

    def test_retry_preserves_artifact_and_upload_checkpoints_across_restart(self):
        job, _ = self.create(revision=REV)
        jid = job["id"]
        self.store.transition(jid, "INSPECTING")
        self.store.update_fields(jid, analysis_json={"revision": REV}, plan_json={"old": True}, result_json={"uploaded": ["q4.gguf"]}, error="network")
        self.store.upsert_shard(jid, "Q4_K_M", 0, "VALIDATED", checksum="digest", nbytes=123, path="q4.gguf")
        self.store.transition(jid, "FAILED")
        self.store.close()
        self.store = JobStore(self.db)
        recovered = self.store.retry(jid)
        self.assertEqual(recovered["state"], "PLANNING")
        self.assertIsNone(recovered["error"])
        self.assertIsNone(recovered["plan"])
        self.assertEqual(recovered["analysis"], {"revision": REV})
        self.assertEqual(recovered["result"]["uploaded"], ["q4.gguf"])
        self.assertEqual(self.store.shards(jid)[0]["checksum"], "digest")

    def test_leases_only_expire_or_release_by_owner(self):
        job, _ = self.create()
        jid = job["id"]
        with patch("hfqa.jobs.time.time", return_value=100):
            self.assertTrue(self.store.claim(jid, "worker-a", 10))
            self.assertFalse(self.store.claim(jid, "worker-b", 10))
            self.store.release_claim(jid, "worker-b")
            self.assertFalse(self.store.claim(jid, "worker-b", 10))
        with patch("hfqa.jobs.time.time", return_value=111):
            self.assertTrue(self.store.claim(jid, "worker-b", 10))
            self.assertFalse(self.store.renew_claim(jid, "worker-a"))
            self.assertTrue(self.store.renew_claim(jid, "worker-b"))
        self.store.release_claim(jid, "worker-b")
        self.assertTrue(self.store.claim(jid, "worker-c"))

    def test_retry_refuses_an_active_worker(self):
        job, _ = self.create()
        jid = job["id"]
        self.store.claim(jid, "worker")
        self.store.transition(jid, "INSPECTING")
        self.store.transition(jid, "FAILED")
        with self.assertRaises(TransitionError):
            self.store.retry(jid)
        self.store.release_claim(jid, "worker")
        self.assertEqual(self.store.retry(jid)["state"], "PLANNING")

    def test_heretic_selection_is_blocked_until_real_worker_exists(self):
        self.assertFalse(assess({"model_type": "qwen2"}, ["heretic"], host="gpu")["runnable_here"])
        with self.assertRaises(ValueError):
            normalize_variants(["both", "typo"])
        analysis = {"repo_id": "owner/model", "model_type": "qwen2", "formats": ["gguf"], "license_decision": {"redistributable": True}}
        built = plan(analysis, ["Q4_K_M"], variants=["both"])
        self.assertFalse(built["publish_allowed"])
        self.assertTrue(any("not configured" in item for item in built["blockers"]))

    def test_architecture_aliases_do_not_claim_incompatible_runtime(self):
        for model_type in ("gemma3", "phi", "phi3", "stablelm"):
            self.assertFalse(classify({"model_type": model_type})["in_stock_binary"])
        built = plan({"model_type": "phi3", "formats": ["gguf"], "parameters": 1000, "license_decision": {"redistributable": True}}, ["Q4_K_M"])
        self.assertEqual(built["route"], "blocked")

    def test_sibling_license_blocks_publication_and_unknown_estimates_allow_worker_inspection(self):
        analysis = {"repo_id": "owner/model", "model_type": "qwen2", "formats": ["safetensors"], "license_decision": {"redistributable": True}, "native_gguf_repo": {"repo_id": "owner/model-GGUF"}}
        self.assertFalse(plan(analysis, ["Q4_K_M"])["publish_allowed"])
        analysis["native_gguf_repo"]["license_decision"] = {"redistributable": True}
        self.assertTrue(plan(analysis, ["Q4_K_M"])["publish_allowed"])
        analysis["formats"] = ["gguf"]
        self.assertEqual(plan(analysis, ["Q4_K_M"])["route"], "oracle_local_requant")


if __name__ == "__main__":
    unittest.main()
