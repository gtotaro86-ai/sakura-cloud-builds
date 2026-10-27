"""Immutable multi-file source discovery; all Hub responses are local fixtures."""
import unittest
from unittest.mock import patch

from hfqa.stream_sources import select_stream_sources


def shard(number, count=2, *, stem="model-Q8_0", size=4_000_000_000, **changes):
    result = {"rfilename": f"{stem}-{number:05}-of-{count:05}.gguf", "size": size,
              "lfs": {"sha256": f"{number:064x}"}}
    result.update(changes)
    return result


class StreamSourceSelectionTests(unittest.TestCase):
    def select(self, siblings, analysis=None):
        analysis = analysis or {"repo_id": "owner/source-GGUF", "revision": "a" * 40}
        with patch("hfqa.stream_sources._json", return_value={"siblings": siblings}) as network:
            result = select_stream_sources(analysis)
        self.assertEqual(network.call_count, 1)
        self.assertIn("/revision/" + analysis.get("native_gguf_repo", analysis)["revision"],
                      network.call_args.args[0])
        self.assertTrue(network.call_args.args[0].endswith("?blobs=true"))
        return result

    def test_complete_shards_are_sorted_pinned_and_not_subject_to_download_cap(self):
        result = self.select([shard(2), shard(1)])
        self.assertEqual(result["total_source_bytes"], 8_000_000_000)
        self.assertFalse(result["weights_downloaded_on_oracle"])
        self.assertEqual([item["split_index"] for item in result["sources"]], [1, 2])
        for number, item in enumerate(result["sources"], 1):
            self.assertEqual(item["sha256"], f"{number:064x}")
            self.assertEqual(item["repo_id"], "owner/source-GGUF")
            self.assertEqual(item["revision"], "a" * 40)
            self.assertEqual(item["split_count"], 2)

    def test_twenty_one_large_shards_are_selected_by_metadata_only(self):
        result = self.select([shard(n, 21, size=3_700_000_000) for n in range(21, 0, -1)])
        self.assertEqual(len(result["sources"]), 21)
        self.assertEqual(result["total_source_bytes"], 77_700_000_000)

    def test_missing_duplicate_zero_and_inconsistent_shard_indices_fail(self):
        cases = ([shard(1)], [shard(1), shard(1)], [shard(0), shard(1)],
                 [shard(1), shard(3)], [shard(1), shard(2, 3)])
        for items in cases:
            with self.subTest(items=[i["rfilename"] for i in items]), self.assertRaisesRegex(ValueError, "complete"):
                self.select(items)

    def test_missing_invalid_digest_or_size_cannot_make_complete_source_set(self):
        for bad in (shard(2, lfs={}), shard(2, lfs={"sha256": "b" * 63}),
                    shard(2, size=True), shard(2, size=0), shard(2, size=23),
                    shard(2, size="4000000000")):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "complete"):
                self.select([shard(1), bad])

    def test_lfs_oid_and_lfs_size_are_accepted_and_normalized(self):
        result = self.select([{"rfilename": "model-BF16.gguf",
                               "lfs": {"oid": "sha256:" + "B" * 64, "size": 10**11}}])
        item = result["sources"][0]
        self.assertEqual(item["size"], 10**11)
        self.assertEqual(item["sha256"], "b" * 64)
        self.assertEqual((item["split_index"], item["split_count"]), (1, 1))

    def test_highest_precision_wins_then_the_smallest_complete_set(self):
        items = [shard(1, 1, stem="model-F16", size=100),
                 shard(1, 1, stem="large-Q8_0", size=1000),
                 shard(1, 1, stem="small-Q8_0", size=500)]
        self.assertEqual(self.select(items)["sources"][0]["filename"], "model-F16-00001-of-00001.gguf")
        items.append(shard(1, 1, stem="model-BF16", size=900))
        result = self.select(items)
        self.assertEqual((result["sources"][0]["filename"], result["source_precision"]), ("model-BF16-00001-of-00001.gguf", "bf16"))
        only_q8 = self.select(items[1:3])
        self.assertEqual((only_q8["sources"][0]["filename"], only_q8["source_precision"]), ("small-Q8_0-00001-of-00001.gguf", "q8_0"))

    def test_the_vision_projector_is_never_taken_for_the_model(self):
        """Regression (Ornith-1.5-122B): mmproj-...-BF16.gguf is 0.9 GB next to a 249 GB model and used to win 'smallest complete set'."""
        items = [shard(1, 1, stem="Ornith-BF16", size=249_000_000_000), shard(1, 1, stem="mmproj-Ornith-BF16", size=910_000_000), shard(1, 1, stem="imatrix-BF16", size=5)]
        self.assertEqual(self.select(items)["sources"][0]["filename"], "Ornith-BF16-00001-of-00001.gguf")
        with self.assertRaises(ValueError):
            self.select([shard(1, 1, stem="mmproj-model-F16")])

    def test_incomplete_q8_does_not_hide_complete_f16(self):
        result = self.select([shard(1), shard(1, 1, stem="model-F16")])
        self.assertIn("F16", result["sources"][0]["filename"])

    def test_low_bit_files_and_non_gguf_names_are_not_source_candidates(self):
        for stem in ("model-Q4_K_M", "model-Q6_K", "model", "model-q8_01"):
            with self.subTest(stem=stem), self.assertRaises(ValueError):
                self.select([shard(1, 1, stem=stem)])
        with self.assertRaises(ValueError):
            self.select([shard(1, 1, rfilename="model-Q8_0.safetensors")])

    def test_native_repository_revision_and_license_are_used(self):
        analysis = {"repo_id": "upstream/model", "revision": "b" * 40,
                    "license_decision": {"allowed": False},
                    "native_gguf_repo": {"repo_id": "quantizer/model-GGUF",
                                         "revision": "C" * 40,
                                         "license_decision": {"allowed": True}}}
        result = self.select([shard(1, 1)], analysis)
        self.assertEqual(result["repo_id"], "quantizer/model-GGUF")
        self.assertEqual(result["revision"], "c" * 40)
        self.assertEqual(result["license_decision"], {"allowed": True})

    def test_mutable_revision_is_rejected_without_metadata_request(self):
        for revision in (None, "main", "a" * 39, "a" * 41):
            with self.subTest(revision=revision), patch("hfqa.stream_sources._json") as network:
                with self.assertRaises(ValueError):
                    select_stream_sources({"repo_id": "owner/model", "revision": revision})
                network.assert_not_called()

    def test_oversized_shard_count_fails_closed(self):
        with self.assertRaises(ValueError):
            self.select([shard(1, 1025)])


if __name__ == "__main__":
    unittest.main()
