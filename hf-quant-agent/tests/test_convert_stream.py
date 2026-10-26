"""Runner side of the BF16 conversion: manifest checks, shard hand-over and the model card. The converter itself needs torch and is exercised on the runner."""
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts import convert_stream as cs

MANIFEST = {'job_id': 'j1', 'model_repo_id': 'owner/model', 'model_revision': 'a' * 40, 'outtype': 'bf16', 'split_max_size': '45G', 'target_repo_id': 'webmp3/model-BF16-GGUF',
            'llama_revision': 'b' * 40, 'output_stem': 'model', 'source_license': {'redistributable': True}}


class ManifestTests(unittest.TestCase):
    def test_a_complete_manifest_passes(self):
        self.assertIs(cs.validate_manifest(dict(MANIFEST))['outtype'], 'bf16')

    def test_every_unsafe_variation_is_refused(self):
        for name, change in (('license', {'source_license': {'redistributable': False}}), ('license missing', {'source_license': {}}), ('namespace', {'target_repo_id': 'someone/model-BF16-GGUF'}),
                             ('private', {'private': True}), ('same repo', {'target_repo_id': 'owner/model', 'model_repo_id': 'owner/model'}), ('outtype', {'outtype': 'q4_0'}),
                             ('revision', {'model_revision': 'main'}), ('llama revision', {'llama_revision': 'b11669'}), ('split', {'split_max_size': '45G; rm -rf /'}),
                             ('stem', {'output_stem': '../x'}), ('repo', {'model_repo_id': 'a b/c'})):
            with self.subTest(name), self.assertRaises(cs.ConvertError):
                cs.validate_manifest({**MANIFEST, **change})
        with self.assertRaises(cs.ConvertError):
            cs.validate_manifest({k: v for k, v in MANIFEST.items() if k != 'target_repo_id'})


class ShardTests(unittest.TestCase):
    def test_a_finished_shard_is_hashed_then_uploaded_then_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model-BF16-00001-of-00002.gguf'
            path.write_bytes(b'gguf' * 1000)
            seen = []
            shards = cs.Shards(lambda p, item: seen.append((p.name, p.exists(), item['size'])))
            shards.finished(path)
            self.assertEqual(seen, [(path.name, True, 4000)])  # still on disk while it is uploaded
            self.assertFalse(path.exists())
            self.assertEqual(shards.items, [{'name': path.name, 'size': 4000, 'sha256': hashlib.sha256(b'gguf' * 1000).hexdigest()}])


class CardTests(unittest.TestCase):
    def test_the_card_states_origin_hashes_and_what_was_not_tested(self):
        shards = [{'name': 'model-BF16.gguf', 'size': 3_000_000_000, 'sha256': 'e' * 64}]
        english, chinese = cs.card(MANIFEST, 'mit', shards)
        for text in (english, chinese):
            self.assertIn('e' * 64, text)
            self.assertIn('a' * 40, text)
            self.assertIn('b' * 40, text)
        self.assertIn('base_model: owner/model', english)
        self.assertIn('hfqa_source_revision: ' + 'a' * 40, english)
        self.assertIn('No inference test and no quality measurement', english)
        self.assertIn('未对此文件做推理测试', chinese)
        self.assertIn('a single file', english)
        self.assertNotIn('Oracle', english + chinese)
        two = cs.card(MANIFEST, 'mit', [{**shards[0], 'name': 'model-BF16-00001-of-00002.gguf'}, {**shards[0], 'name': 'model-BF16-00002-of-00002.gguf'}])[0]
        self.assertIn('2 numbered files of at most 45 GB each', two)

    def test_the_card_carries_the_markers_the_oracle_looks_for_and_the_license_link(self):
        english, _ = cs.card(MANIFEST, 'other', [{'name': 'model-BF16.gguf', 'size': 1, 'sha256': 'e' * 64}], {'license_name': 'vendor', 'license_link': 'https://vendor.example/l'})
        self.assertIn('hfqa_source_revision: ' + 'a' * 40, english)
        self.assertIn('hfqa_converter_revision: ' + 'b' * 40, english)
        self.assertIn('license_name: vendor', english)
        self.assertIn('license_link: https://vendor.example/l', english)

    def test_the_parity_claim_names_the_revision_it_was_made_with(self):
        shards = [{'name': 'model-BF16.gguf', 'size': 1, 'sha256': 'e' * 64}]
        same = cs.card({**MANIFEST, 'llama_revision': cs.PARITY_REVISION}, 'mit', shards)[0]
        self.assertIn('290 of 290 tensors', same)
        other = cs.card(MANIFEST, 'mit', shards)[0]
        self.assertIn('has not been repeated with this revision', other)
        self.assertNotIn('The same procedure produced tensors byte-identical', other)

    def test_the_repository_never_goes_public_without_a_card(self):
        text = cs.placeholder_card(MANIFEST, 'mit', {})
        self.assertIn('being filled', text)
        self.assertIn('Do not use it', text)
        self.assertIn('hfqa_source_revision: ' + 'a' * 40, text)


class NamingTests(unittest.TestCase):
    def test_only_parts_of_this_conversion_are_recognised(self):
        pattern = cs.part_pattern(MANIFEST)
        for name in ('model-BF16.gguf', 'model-BF16-00001-of-00004.gguf'):
            self.assertTrue(pattern.fullmatch(name), name)
        for name in ('model-Q4_K_M.gguf', 'other-BF16.gguf', 'model-BF16-1-of-4.gguf', 'sub/model-BF16.gguf', 'model-BF16.gguf.bak'):
            self.assertFalse(pattern.fullmatch(name), name)

    def test_the_split_size_is_counted_in_decimal_gigabytes_like_llama_cpp(self):
        self.assertEqual(cs.split_bytes({**MANIFEST, 'split_max_size': '20G'}), 20_000_000_000)


if __name__ == '__main__':
    unittest.main()
