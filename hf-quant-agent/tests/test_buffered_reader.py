"""Bounded-window contract tests; underlying HTTP is an explicit byte double."""
import hashlib
import unittest
from hfqa.remote_gguf import BufferedRangeReader, ConcatenatedRangeReader


class BytesReader:
    def __init__(self, blob):
        self.blob, self.size = blob, len(blob)
        self.bytes_read, self.request_count, self.calls = 0, 0, []

    def read_range(self, offset, count):
        self.bytes_read += count
        self.request_count += 1
        self.calls.append((offset, count))
        return self.blob[offset:offset + count]


class BufferedReaderTests(unittest.TestCase):
    def test_single_window_bound_reuses_sequential_subranges(self):
        blob = bytes(range(256)) * 4
        underlying = BytesReader(blob)
        reader = BufferedRangeReader(underlying, window_bytes=64)
        result = bytearray()
        for offset in range(0, len(blob), 7):
            result.extend(reader.read_range(offset, min(7, len(blob) - offset)))
            self.assertLessEqual(len(reader._cache), 64)
        self.assertEqual(result, blob)
        self.assertEqual(underlying.request_count, 16)
        self.assertEqual(underlying.bytes_read, len(blob))
        self.assertTrue(all(count <= 64 for _, count in underlying.calls))
        self.assertEqual(reader.read_range(0, 100), blob[:100])
        self.assertLessEqual(len(reader._cache), 64)
        with self.assertRaises(ValueError):
            reader.read_range(len(blob), 1)

    def test_buffered_multifile_source_pass_keeps_every_full_sha_proof(self):
        blobs = [bytes(range(97)), bytes(range(113))]
        descriptors = [{'repo_id': 'tests/model', 'revision': 'a' * 40,
                        'filename': f'part-{index}.gguf', 'size': len(blob),
                        'sha256': hashlib.sha256(blob).hexdigest()}
                       for index, blob in enumerate(blobs)]
        readers = [BufferedRangeReader(BytesReader(blob), window_bytes=7) for blob in blobs]
        virtual = ConcatenatedRangeReader(descriptors, readers)
        virtual.begin_verified_pass()
        result = bytearray()
        for offset in range(0, virtual.size, 3):
            result.extend(virtual.read_range(offset, min(3, virtual.size - offset)))
            self.assertTrue(all(len(reader._cache) <= 7 for reader in readers))
        report = virtual.finish_verified_pass()
        self.assertEqual(result, b''.join(blobs))
        self.assertTrue(report['all_source_shas_verified'])
        self.assertEqual(report['total_source_bytes'], sum(map(len, blobs)))
        self.assertEqual([item['sha256'] for item in report['sources_verified']],
                         [item['sha256'] for item in descriptors])
        self.assertTrue(all(item['sha256_verified'] for item in report['sources_verified']))


if __name__ == '__main__':
    unittest.main()
