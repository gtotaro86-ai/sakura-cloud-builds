"""Offline protocol and malformed-input regression tests for the range reader."""
import hashlib
import io
import struct
import unittest
from unittest.mock import Mock

from hfqa.gguf import GGUFError
from hfqa.remote_gguf import (MAX_RANGE_BYTES, ConcatenatedRangeReader, RangeReader,
                              descriptor_url, inspect_remote_gguf,
                              inspect_remote_source_set)


def string(value):
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def metadata(key, kind, raw):
    return string(key) + struct.pack("<I", kind) + raw


def tensor(name="weight", shape=(8,), kind=0, offset=0):
    return (string(name) + struct.pack("<I", len(shape))
            + b"".join(struct.pack("<Q", dim) for dim in shape)
            + struct.pack("<IQ", kind, offset))


def gguf(*, entries=None, tensors=None, payload=bytes(32), version=3):
    if entries is None:
        entries = [metadata("general.architecture", 8, string("llama"))]
    if tensors is None:
        tensors = [tensor()]
    header = (b"GGUF" + struct.pack("<IQQ", version, len(tensors), len(entries))
              + b"".join(entries) + b"".join(tensors))
    return header + bytes(-len(header) % 32) + payload


def descriptor(body=b"x" * 128, **changes):
    result = {"repo_id": "owner/model-GGUF", "revision": "a" * 40,
              "filename": "weights/model q8_0.gguf", "size": len(body),
              "sha256": hashlib.sha256(body).hexdigest()}
    result.update(changes)
    return result


class MemoryReader:
    def __init__(self, body, advertised_size=None):
        self.body = body
        self.size = len(body) if advertised_size is None else advertised_size
        self.ranges = []

    def read_range(self, offset, count):
        self.ranges.append((offset, count))
        return self.body[offset:offset + count]


class Response:
    def __init__(self, body, *, status=206, content_range="bytes 8-15/128",
                 encoding="identity"):
        self.status = status
        self.headers = {"Content-Range": content_range, "Content-Encoding": encoding}
        self.body = io.BytesIO(body)
        self.read_sizes = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size):
        self.read_sizes.append(size)
        return self.body.read(size)


class RangeReaderTests(unittest.TestCase):
    def test_url_is_pinned_and_filename_is_escaped(self):
        url = descriptor_url(descriptor())
        self.assertEqual(url, "https://huggingface.co/owner/model-GGUF/resolve/"
                         + "a" * 40 + "/weights/model%20q8_0.gguf")

    def test_mutable_revision_or_foreign_repo_is_rejected(self):
        for changes in ({"revision": "main"}, {"revision": "a" * 39},
                        {"revision": "a" * 40 + "/other"},
                        {"repo_id": "https://evil.example/model"},
                        {"repo_id": "owner/model/extra"},
                        {"repo_id": "owner@evil.example/model"},
                        {"repo_id": "../model"}, {"repo_id": "owner/.."}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                descriptor_url(descriptor(**changes))

    def test_path_traversal_and_absolute_paths_are_rejected(self):
        for name in ("../model.gguf", "nested/../model.gguf", "/model.gguf",
                     "C:/model.gguf", "nested\\model.gguf", "model.bin",
                     "./model.gguf", "nested//model.gguf", "nested/./model.gguf",
                     "model\x00.gguf", "model\n.gguf", "model\x7f.gguf"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                descriptor_url(descriptor(filename=name))

    def test_exact_range_uses_identity_encoding_and_bounded_body_read(self):
        response = Response(b"12345678")
        opener = Mock(return_value=response)
        reader = RangeReader(descriptor(), opener=opener)
        self.assertEqual(reader.read_range(8, 8), b"12345678")
        request = opener.call_args.args[0]
        self.assertEqual(request.get_header("Range"), "bytes=8-15")
        self.assertEqual(request.get_header("Accept-encoding"), "identity")
        self.assertEqual(opener.call_args.kwargs["timeout"], 45)
        self.assertEqual(response.read_sizes, [9])
        self.assertEqual((reader.bytes_read, reader.request_count), (8, 1))

    def test_ignored_range_never_reads_full_body_or_falls_back(self):
        response = Response(b"x" * 128, status=200)
        opener = Mock(return_value=response)
        reader = RangeReader(descriptor(), opener=opener)
        with self.assertRaisesRegex(OSError, "full download refused"):
            reader.read_range(8, 8)
        self.assertEqual(response.read_sizes, [])
        self.assertEqual(opener.call_count, 1)
        self.assertEqual((reader.bytes_read, reader.request_count), (0, 0))

    def test_content_range_requires_exact_start_end_and_pinned_total(self):
        for value in (None, "bytes 9-16/128", "bytes 8-16/128", "bytes 8-15/129",
                      "bytes 8-15/*", "bytes 0-127/128", "bytes=8-15/128"):
            response = Response(b"12345678", content_range=value)
            with self.subTest(value=value), self.assertRaises(OSError):
                RangeReader(descriptor(), opener=Mock(return_value=response)).read_range(8, 8)
            self.assertEqual(response.read_sizes, [])

    def test_compressed_partial_is_rejected_before_read(self):
        response = Response(b"12345678", encoding="gzip")
        with self.assertRaisesRegex(OSError, "compressed"):
            RangeReader(descriptor(), opener=Mock(return_value=response)).read_range(8, 8)
        self.assertEqual(response.read_sizes, [])

    def test_truncated_or_oversized_body_is_rejected_with_one_byte_probe(self):
        for body in (b"short", b"12345678x", b"x" * 100000):
            response = Response(body)
            reader = RangeReader(descriptor(), opener=Mock(return_value=response))
            with self.subTest(length=len(body)), self.assertRaises(OSError):
                reader.read_range(8, 8)
            self.assertEqual(response.read_sizes, [9])
            self.assertLessEqual(response.body.tell(), 9)
            self.assertEqual((reader.bytes_read, reader.request_count), (0, 0))

    def test_invalid_range_never_opens_connection(self):
        opener = Mock()
        reader = RangeReader(descriptor(), opener=opener)
        for offset, count in ((-1, 4), (0, -1), (128, 1), (120, 9),
                              (0, MAX_RANGE_BYTES + 1), (True, 1), (0, False),
                              (0.5, 1), (0, 1.5)):
            with self.subTest(offset=offset, count=count), self.assertRaises(ValueError):
                reader.read_range(offset, count)
        self.assertEqual(reader.read_range(128, 0), b"")
        opener.assert_not_called()

    def test_chunk_iterator_preserves_exact_source_coverage(self):
        body = bytes(range(128))
        ranges = []

        def open_range(request, **kwargs):
            start, end = map(int, request.get_header("Range")[6:].split("-"))
            ranges.append((start, end))
            return Response(body[start:end + 1], content_range=f"bytes {start}-{end}/128")

        reader = RangeReader(descriptor(body), opener=open_range)
        self.assertEqual(b"".join(reader.iter_range(4, 17, chunk_size=6)), body[4:21])
        self.assertEqual(ranges, [(4, 9), (10, 15), (16, 20)])
        self.assertEqual(reader.bytes_read, 17)

    def test_chunk_iterator_validates_empty_intervals_and_chunk_limits(self):
        opener = Mock()
        reader = RangeReader(descriptor(), opener=opener)
        for offset, count, chunk in ((129, 0, 8), (-1, 0, 8), (0, -1, 8),
                                      (0, 0, 0), (0, 0, MAX_RANGE_BYTES + 1),
                                      (False, 0, 8), (0, True, 8), (0, 0, 1.5)):
            with self.subTest(interval=(offset, count, chunk)), self.assertRaises(ValueError):
                list(reader.iter_range(offset, count, chunk_size=chunk))
        self.assertEqual(list(reader.iter_range(128, 0)), [])
        opener.assert_not_called()


class RemoteGGUFTests(unittest.TestCase):
    def inspect(self, body, **kwargs):
        reader = MemoryReader(body)
        return inspect_remote_gguf(descriptor(body), reader=reader, **kwargs)

    def assert_bad(self, body, pattern=None, **kwargs):
        with self.assertRaisesRegex(GGUFError, pattern or "."):
            self.inspect(body, **kwargs)

    def test_valid_v2_and_v3_catalogue_includes_full_tensor_layout(self):
        for version in (2, 3):
            parsed = self.inspect(gguf(version=version))
            self.assertEqual(parsed.version, version)
            self.assertEqual(parsed.metadata["general.architecture"], "llama")
            self.assertEqual(parsed.alignment, 32)
            self.assertEqual(parsed.data_offset % 32, 0)
            item = parsed.tensors[0]
            self.assertEqual((item.name, item.shape, item.type, item.offset, item.nbytes),
                             ("weight", (8,), 0, 0, 32))

    def test_array_metadata_preserves_exact_encoded_bytes(self):
        tokens = struct.pack("<IQ", 8, 3) + string("a") + string("ä") + string("")
        integers = struct.pack("<IQIII", 4, 3, 256, 256, 256)
        parsed = self.inspect(gguf(entries=[
            metadata("general.architecture", 8, string("llama")),
            metadata("tokenizer.ggml.tokens", 9, tokens),
            metadata("llama.feed_forward_length", 9, integers)]))
        raw = {key: (kind, value) for key, kind, value in parsed.metadata_entries}
        self.assertEqual(raw["tokenizer.ggml.tokens"], (9, tokens))
        self.assertEqual(raw["llama.feed_forward_length"], (9, integers))
        self.assertEqual(list(parsed.metadata["llama.feed_forward_length"]), [256] * 3)

    def test_header_budget_stops_fetching_before_large_model_payload(self):
        prefix = b"GGUF" + struct.pack("<IQQ", 3, 1, 1) + struct.pack("<Q", 4096)
        reader = MemoryReader(prefix + bytes(256), advertised_size=10**12)
        with self.assertRaisesRegex(GGUFError, "budget"):
            inspect_remote_gguf({}, reader=reader, max_header_bytes=64)
        self.assertTrue(reader.ranges)
        self.assertLessEqual(sum(count for _, count in reader.ranges), 64)
        self.assertTrue(all(offset + count <= 64 for offset, count in reader.ranges))

    def test_bad_magic_version_and_counts(self):
        self.assert_bad(b"FAIL" + gguf()[4:], "magic")
        for version in (0, 1, 4, 0xffffffff):
            self.assert_bad(gguf(version=version), "version")
        for tensors, entries in ((0, 0), (1000001, 0), (1, 100001)):
            self.assert_bad(b"GGUF" + struct.pack("<IQQ", 3, tensors, entries), "count")

    def test_duplicate_metadata_and_tensor_names_are_rejected(self):
        item = metadata("general.architecture", 8, string("llama"))
        self.assert_bad(gguf(entries=[item, item]), "duplicate")
        self.assert_bad(gguf(tensors=[tensor(), tensor(offset=32)], payload=bytes(64)), "duplicate")

    def test_empty_tensor_name_is_rejected(self):
        self.assert_bad(gguf(tensors=[tensor(name="")]), "name")
        self.assert_bad(gguf(tensors=[tensor(name="weight\x00hidden")]), "name")

    def test_invalid_dimensions_shape_type_and_alignment(self):
        for shape in ((), (1,) * 5, (0,), (2**40 + 1,), (31,)):
            kind = 8 if shape == (31,) else 0
            with self.subTest(shape=shape):
                self.assert_bad(gguf(tensors=[tensor(shape=shape, kind=kind)]))
        self.assert_bad(gguf(tensors=[tensor(kind=999)]))
        self.assert_bad(gguf(tensors=[tensor(offset=1)]))

    def test_quantized_tensor_row_alignment_and_byte_size(self):
        parsed = self.inspect(gguf(tensors=[tensor(shape=(32, 2), kind=8)], payload=bytes(68)))
        self.assertEqual(parsed.tensors[0].nbytes, 68)

    def test_overlapping_or_truncated_payload_is_rejected(self):
        self.assert_bad(gguf(tensors=[tensor(name="a", shape=(16,)),
                                     tensor(name="b", offset=32)], payload=bytes(96)), "overlapping")
        self.assert_bad(gguf(payload=bytes(31)), "truncated")

    def test_unsorted_nonoverlapping_payload_with_gaps_is_valid(self):
        parsed = self.inspect(gguf(tensors=[tensor(name="second", offset=64),
                                           tensor(name="first", offset=0)], payload=bytes(96)))
        self.assertEqual([(item.name, item.offset) for item in parsed.tensors],
                         [("second", 64), ("first", 0)])

    def test_architecture_must_be_nonempty_string(self):
        for kind, raw in ((8, string("")), (4, struct.pack("<I", 1)), (7, b"\1")):
            with self.subTest(kind=kind):
                self.assert_bad(gguf(entries=[metadata("general.architecture", kind, raw)]), "architecture")

    def test_invalid_alignment_values_are_rejected(self):
        for alignment in (0, 3, 8192):
            entries = [metadata("general.architecture", 8, string("llama")),
                       metadata("general.alignment", 4, struct.pack("<I", alignment))]
            self.assert_bad(gguf(entries=entries), "alignment")
        self.assert_bad(gguf(entries=[metadata("general.architecture", 8, string("llama")),
                                     metadata("general.alignment", 7, b"\1")]), "alignment")

    def test_split_source_is_not_silently_treated_as_single_file(self):
        self.assert_bad(gguf(entries=[metadata("general.architecture", 8, string("llama")),
                                     metadata("split.count", 2, struct.pack("<H", 2))]), "split")

    def test_unknown_metadata_types_and_excessive_arrays_are_rejected(self):
        architecture = metadata("general.architecture", 8, string("llama"))
        for raw in (struct.pack("<IQ", 9, 1), struct.pack("<IQ", 4, 2000001),
                    struct.pack("<IQ", 99, 0)):
            with self.subTest(raw=raw):
                self.assert_bad(gguf(entries=[architecture, metadata("bad", 9, raw)]))
        self.assert_bad(gguf(entries=[architecture, metadata("bad", 99, b"")]))


def split_file(index, *, name=None, architecture=None, count=2, total=2, omit=(), overrides=None):
    entries = []
    if index == 0 or architecture is not None:
        entries.append(metadata("general.architecture", 8, string(architecture or "llama")))
    split = {"split.no": (2, struct.pack("<H", index)),
             "split.count": (2, struct.pack("<H", count)),
             "split.tensors.count": (5, struct.pack("<i", total))}
    split.update(overrides or {})
    entries.extend(metadata(key, kind, raw) for key, (kind, raw) in split.items() if key not in omit)
    return gguf(entries=entries, tensors=[tensor(name=name or f"weight.{index}")],
                payload=bytes([index + 1]) * 32)


def source_set(bodies):
    return [descriptor(body, filename=f"model-Q8_0-{i:05}-of-{len(bodies):05}.gguf",
                       split_index=i, split_count=len(bodies))
            for i, body in enumerate(bodies, 1)]


class RemoteSourceSetTests(unittest.TestCase):
    def setUp(self):
        self.bodies = [split_file(0), split_file(1)]
        self.descriptors = source_set(self.bodies)
        self.readers = [MemoryReader(body) for body in self.bodies]

    def inspect(self, descriptors=None, readers=None):
        return inspect_remote_source_set(self.descriptors if descriptors is None else descriptors,
                                         readers=self.readers if readers is None else readers)

    def test_split_catalogues_merge_offsets_and_keep_first_metadata(self):
        result = self.inspect()
        self.assertEqual(result.data_offset, 0)
        self.assertEqual(result.metadata["general.architecture"], "llama")
        self.assertFalse(any(key.startswith("split.") for key in result.metadata))
        self.assertFalse(any(key.startswith("split.") for key, _, _ in result.metadata_entries))
        self.assertEqual([item.name for item in result.tensors], ["weight.0", "weight.1"])
        first = inspect_remote_gguf(self.descriptors[0], reader=MemoryReader(self.bodies[0]), allow_split=True)
        second = inspect_remote_gguf(self.descriptors[1], reader=MemoryReader(self.bodies[1]), allow_split=True)
        self.assertEqual([item.offset for item in result.tensors],
                         [first.data_offset, len(self.bodies[0]) + second.data_offset])
        self.assertEqual(result.reader.size, sum(map(len, self.bodies)))

    def test_single_unsplit_source_retains_native_offsets(self):
        body = gguf()
        result = inspect_remote_source_set([descriptor(body)], readers=[MemoryReader(body)])
        self.assertGreater(result.data_offset, 0)
        self.assertEqual(result.tensors[0].offset, 0)

    def test_descriptor_identity_and_completeness_are_required(self):
        cases = ([self.descriptors[0]],
                 [self.descriptors[0], {**self.descriptors[1], "split_index": 1}],
                 [self.descriptors[0], {**self.descriptors[1], "split_count": 3}],
                 [self.descriptors[0], {**self.descriptors[1], "revision": "b" * 40}],
                 [self.descriptors[0], {**self.descriptors[1], "repo_id": "other/model"}])
        for descriptors in cases:
            with self.subTest(descriptors=descriptors), self.assertRaises(GGUFError):
                self.inspect(descriptors)
        with self.assertRaises(GGUFError):
            self.inspect([])

    def test_missing_secondary_reader_fails_closed(self):
        with self.assertRaises(GGUFError):
            self.inspect(readers=self.readers[:1])

    def test_unsorted_descriptor_reader_pairs_are_reordered_together(self):
        result = self.inspect(list(reversed(self.descriptors)), list(reversed(self.readers)))
        self.assertEqual([item.name for item in result.tensors], ["weight.0", "weight.1"])
        result.reader.begin_verified_pass()
        self.assertEqual(result.reader.read_range(0, result.reader.size), b"".join(self.bodies))
        self.assertTrue(result.reader.finish_verified_pass()["all_source_shas_verified"])

    def test_duplicate_source_filename_is_rejected(self):
        bad = [self.descriptors[0], {**self.descriptors[1], "filename": self.descriptors[0]["filename"]}]
        with self.assertRaises(GGUFError):
            self.inspect(bad)

    def test_source_split_indices_and_counts_are_exact_integers(self):
        for change in ({"split_index": True}, {"split_index": 1.0}, {"split_count": 2.0}):
            bad = [{**self.descriptors[0], **change}, self.descriptors[1]]
            with self.subTest(change=change), self.assertRaises(GGUFError):
                self.inspect(bad)

    def test_split_header_indices_counts_and_totals_must_match(self):
        for second in (split_file(0), split_file(1, count=3),
                       split_file(1, omit=("split.no",)),
                       split_file(1, omit=("split.count",))):
            bodies = [self.bodies[0], second]
            with self.subTest(second=second), self.assertRaises(GGUFError):
                self.inspect(source_set(bodies), [MemoryReader(body) for body in bodies])
        bodies = [split_file(0, total=3), self.bodies[1]]
        with self.assertRaisesRegex(GGUFError, "completeness"):
            self.inspect(source_set(bodies), [MemoryReader(body) for body in bodies])

    def test_each_shard_requires_matching_tensor_total(self):
        for first, second in ((split_file(0, omit=("split.tensors.count",)), self.bodies[1]),
                               (self.bodies[0], split_file(1, omit=("split.tensors.count",))),
                               (self.bodies[0], split_file(1, total=3))):
            bodies = [first, second]
            with self.subTest(first=first, second=second), self.assertRaisesRegex(GGUFError, "completeness"):
                self.inspect(source_set(bodies), [MemoryReader(body) for body in bodies])

    def test_split_header_control_values_are_exact_integers(self):
        for key, kind, value in (("split.no", 7, b"\0"),
                                  ("split.no", 6, struct.pack("<f", 0)),
                                  ("split.count", 7, b"\1"),
                                  ("split.count", 6, struct.pack("<f", 1)),
                                  ("split.tensors.count", 7, b"\1")):
            body = split_file(0, count=1, total=1, overrides={key: (kind, value)})
            with self.subTest(key=key, kind=kind), self.assertRaises(GGUFError):
                self.inspect(source_set([body]), [MemoryReader(body)])

    def test_duplicate_tensor_across_shards_and_architecture_mismatch_fail(self):
        for second in (split_file(1, name="weight.0"), split_file(1, architecture="qwen2")):
            bodies = [self.bodies[0], second]
            with self.subTest(second=second), self.assertRaises(GGUFError):
                self.inspect(source_set(bodies), [MemoryReader(body) for body in bodies])

    def test_hash_verification_covers_original_headers_padding_and_payload(self):
        reader = self.inspect().reader
        reader.begin_verified_pass()
        joined = b"".join(self.bodies)
        self.assertEqual(reader.read_range(0, len(joined)), joined)
        proof = reader.finish_verified_pass()
        self.assertTrue(proof["all_source_shas_verified"])
        self.assertEqual(proof["total_source_bytes"], len(joined))
        self.assertEqual(len(proof["sources_verified"]), 2)
        self.assertTrue(all(item["sha256_verified"] for item in proof["sources_verified"]))

    def test_cross_shard_reads_are_split_at_exact_physical_boundary(self):
        reader = self.inspect().reader
        self.readers[0].ranges.clear()
        self.readers[1].ranges.clear()
        boundary = len(self.bodies[0])
        self.assertEqual(reader.read_range(boundary - 3, 8), self.bodies[0][-3:] + self.bodies[1][:5])
        self.assertEqual(self.readers[0].ranges, [(boundary - 3, 3)])
        self.assertEqual(self.readers[1].ranges, [(0, 5)])

    def test_verified_pass_requires_contiguous_and_complete_coverage(self):
        reader = self.inspect().reader
        with self.assertRaisesRegex(ValueError, "incomplete"):
            reader.finish_verified_pass()
        reader.begin_verified_pass()
        with self.assertRaisesRegex(ValueError, "contiguous"):
            reader.read_range(1, 1)
        reader.read_range(0, 16)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            reader.finish_verified_pass()
        with self.assertRaisesRegex(ValueError, "contiguous"):
            reader.read_range(0, 1)

    def test_changed_original_shard_bytes_cannot_produce_verification_proof(self):
        # Mutate after metadata inspection: the final proof must cover original
        # file headers and padding as well as the tensors used in the output.
        for index, position in ((0, 0), (0, -33), (1, -1)):
            readers = [MemoryReader(body) for body in self.bodies]
            reader = self.inspect(readers=readers).reader
            changed = bytearray(readers[index].body)
            changed[position] ^= 1
            readers[index].body = bytes(changed)
            reader.begin_verified_pass()
            reader.read_range(0, reader.size)
            with self.subTest(index=index, position=position), self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                reader.finish_verified_pass()

    def test_virtual_reader_rejects_bad_identity_and_range_bounds(self):
        for descriptors in ([{**self.descriptors[0], "size": 1}, self.descriptors[1]],
                            [{**self.descriptors[0], "sha256": "a"}, self.descriptors[1]]):
            with self.subTest(descriptors=descriptors), self.assertRaises(ValueError):
                ConcatenatedRangeReader(descriptors, self.readers)
        with self.assertRaises(ValueError):
            ConcatenatedRangeReader(self.descriptors, self.readers[:1])
        reader = self.inspect().reader
        for offset, count in ((-1, 1), (0, -1), (reader.size, 1), (0, MAX_RANGE_BYTES + 1)):
            with self.subTest(offset=offset, count=count), self.assertRaises(ValueError):
                reader.read_range(offset, count)


if __name__ == "__main__":
    unittest.main()
