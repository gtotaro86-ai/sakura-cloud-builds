"""Stream contract unit tests with explicit in-memory reader/native/Xet doubles.

These tests do not claim native kernel, network, Hub upload or inference proof.
Real stock/native comparison runs separately on the pinned Linux runtime.
"""
import base64
import ctypes
import hashlib
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hfqa.gguf import validate_gguf
from hfqa.remote_gguf import inspect_remote_gguf, ConcatenatedRangeReader, RemoteGGUF, RemoteTensor
from hfqa.streaming import (NativeQuantizer, QuantizedStream, ShardedQuantizedStream,
    model_hparams, stream_quantize, upload_xet_stream, upload_xet_sharded_stream)
from scripts import run_stream_worker


def string(value):
    raw = value.encode()
    return struct.pack('<Q', len(raw)) + raw


def fixture():
    metadata = [('general.architecture', 8, string('qwen2')),
                ('general.file_type', 4, struct.pack('<I', 0)),
                ('general.quantization_version', 4, struct.pack('<I', 1)),
                ('qwen2.embedding_length', 4, struct.pack('<I', 256)),
                ('qwen2.feed_forward_length', 4, struct.pack('<I', 512)),
                ('qwen2.block_count', 4, struct.pack('<I', 1)),
                ('qwen2.attention.head_count', 4, struct.pack('<I', 4)),
                ('tokenizer.ggml.tokens', 9, struct.pack('<IQ', 8, 2) + string('a') + string('b'))]
    # Catalogue order deliberately differs from physical byte order. The gap
    # and trailer must also be included in the complete source SHA-256 proof.
    tensors = [('output.weight', (256, 3), 0, 64), ('output_norm.weight', (8,), 0, 0)]
    header = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(metadata))
    for name, kind, raw in metadata:
        header += string(name) + struct.pack('<I', kind) + raw
    for name, shape, kind, offset in tensors:
        header += string(name) + struct.pack('<I', len(shape))
        header += b''.join(struct.pack('<Q', d) for d in shape)
        header += struct.pack('<IQ', kind, offset)
    header += b'\0' * (-len(header) % 32)
    payload = struct.pack('<8f', *range(8)) + b'gap-not-tensor'.ljust(32, b'!')
    payload += struct.pack('<768f', *(x / 1000 for x in range(768)))
    return header + payload + b'full-source-checksum-trailer'


class MemoryReader:
    def __init__(self, blob):
        self.blob, self.size, self.calls = blob, len(blob), []

    def read_range(self, offset, count):
        self.calls.append((offset, count))
        return self.blob[offset:offset + count]


class PassthroughQuantizer:
    """Contract double: returns original F32 data, never invokes native code."""
    identity = {'runtime_revision': 'b' * 40, 'test_double': True}

    def __init__(self):
        self.plans, self.rows = [], []

    def plan(self, remote, quant):
        self.plans.append([t.name for t in remote.tensors])
        return {tensor.name: tensor.type for tensor in remote.tensors}

    def quantize_rows(self, raw, input_type, output_type, cols):
        self.rows.append((len(raw), cols))
        return raw


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.blob = fixture()
        self.reader = MemoryReader(self.blob)
        self.descriptor = {'repo_id': 'tests/source-GGUF', 'revision': 'a' * 40,
                           'filename': 'test-f32.gguf', 'size': len(self.blob),
                           'sha256': hashlib.sha256(self.blob).hexdigest()}
        self.remote = inspect_remote_gguf(self.descriptor, reader=self.reader)
        self.reader.calls.clear()
        self.native = PassthroughQuantizer()

    def stream(self, descriptor=None, native=None, should_stop=None):
        return QuantizedStream(self.remote, descriptor or self.descriptor, 'Q3_K_M', native or self.native, should_stop)

    def test_complete_source_hash_includes_header_gaps_and_trailer(self):
        stream = self.stream()
        self.assertIsNone(stream.receipt)
        output = b''.join(stream)
        receipt = stream.receipt
        self.assertTrue(receipt['source_sha256_verified'])
        self.assertEqual(receipt['source_bytes_hashed'], len(self.blob))
        self.assertEqual(sum(count for _, count in self.reader.calls), len(self.blob))
        self.assertEqual(receipt['sha256'], hashlib.sha256(output).hexdigest())
        self.assertEqual(receipt['size'], len(output))
        self.assertFalse(receipt['full_inference_test_performed'])
        self.assertEqual(self.native.plans, [['output.weight', 'output_norm.weight']])
        out_descriptor = dict(self.descriptor, size=len(output), sha256=receipt['sha256'])
        out = inspect_remote_gguf(out_descriptor, reader=MemoryReader(output))
        self.assertEqual(out.metadata['general.file_type'], 12)
        self.assertEqual(out.metadata['general.quantization_version'], 2)
        self.assertEqual({t.name for t in out.tensors}, {t.name for t in self.remote.tensors})
        self.assertEqual(next(raw for key, _, raw in out.metadata_entries if key == 'tokenizer.ggml.tokens'),
                         next(raw for key, _, raw in self.remote.metadata_entries if key == 'tokenizer.ggml.tokens'))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.gguf'
            path.write_bytes(output)
            self.assertTrue(validate_gguf(path)['structure_valid'])

    def test_chunking_is_row_bounded_and_source_ordered(self):
        with patch('hfqa.streaming.ROW_WORK_BYTES', 1024):
            stream = self.stream()
            chunks = list(stream)
        self.assertTrue(stream.receipt)
        self.assertEqual([size for size, cols in self.native.rows if cols == 256], [1024, 1024, 1024])
        self.assertTrue(all(size <= 1024 for size, _ in self.native.rows))
        self.assertEqual([offset for offset, _ in self.reader.calls], sorted(offset for offset, _ in self.reader.calls))
        self.assertEqual(stream.size, sum(map(len, chunks)))

    def test_receipt_requires_exhaustion_and_stream_is_single_use(self):
        stream = self.stream()
        iterator = iter(stream)
        next(iterator)
        self.assertIsNone(stream.receipt)
        iterator.close()
        self.assertIsNone(stream.receipt)
        with self.assertRaisesRegex(RuntimeError, 'reused'):
            list(stream)

    def test_complete_source_checksum_mismatch_has_no_receipt(self):
        descriptor = dict(self.descriptor, sha256='0' * 64)
        stream = self.stream(descriptor)
        with self.assertRaisesRegex(ValueError, 'source SHA-256 mismatch'):
            list(stream)
        self.assertIsNone(stream.receipt)

    def test_descriptor_is_copied_so_caller_cannot_replace_expected_hash(self):
        descriptor = dict(self.descriptor)
        stream = self.stream(descriptor)
        descriptor['sha256'] = '0' * 64
        list(stream)
        self.assertEqual(stream.receipt['source']['sha256'], self.descriptor['sha256'])

    def test_pause_cancels_without_a_success_receipt(self):
        checks = []
        def stop():
            checks.append(True)
            return len(checks) > 2
        stream = self.stream(should_stop=stop)
        with self.assertRaises(InterruptedError):
            list(stream)
        self.assertIsNone(stream.receipt)

    def test_short_source_and_native_output_are_rejected(self):
        stream = self.stream()
        self.reader.read_range = lambda offset, count: b'\0' * (count - 1)
        with self.assertRaisesRegex(ValueError, 'short/excess'):
            list(stream)
        self.assertIsNone(stream.receipt)
        self.reader.read_range = lambda offset, count: self.blob[offset:offset + count]
        self.native.quantize_rows = lambda raw, *args: raw[:-1]
        stream = self.stream()
        with self.assertRaisesRegex(RuntimeError, 'output chunk byte count'):
            list(stream)
        self.assertIsNone(stream.receipt)

    def test_sink_short_write_aborts(self):
        sink = SimpleNamespace(write=lambda blob: len(blob) - 1)
        with self.assertRaisesRegex(RuntimeError, 'short write'):
            stream_quantize(self.remote, self.descriptor, 'Q2_K', self.native, sink)

    def test_source_pins_split_and_unknown_types_fail_before_payload(self):
        for changed in ({'revision': 'main'}, {'sha256': 'unknown'}, {'size': 1}, {'split_count': 2}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.stream(dict(self.descriptor, **changed))
        unknown = SimpleNamespace(plan=lambda remote, quant: {t.name: 999 for t in remote.tensors})
        with self.assertRaisesRegex(ValueError, 'unqualified output'):
            self.stream(native=unknown)
        missing = SimpleNamespace(plan=lambda remote, quant: {'output.weight': 0})
        with self.assertRaisesRegex(ValueError, 'complete catalogue'):
            self.stream(native=missing)
        self.assertEqual(self.reader.calls, [])

    def test_missing_or_heterogeneous_native_dimensions_rejected(self):
        metadata = dict(self.remote.metadata)
        arch, values = model_hparams(metadata)
        self.assertEqual(arch, 'qwen2')
        self.assertEqual(values, [256, 512, 1, 4, 4, 0, 64, 64])
        metadata['qwen2.attention.head_count'] = 3
        with self.assertRaisesRegex(ValueError, 'attention dimensions'):
            model_hparams(metadata)
        metadata['qwen2.attention.head_count'] = [4, 8]
        with self.assertRaisesRegex(ValueError, 'uniform metadata'):
            model_hparams(metadata)

    def test_native_rows_preflight_before_any_ctypes_call(self):
        native = NativeQuantizer.__new__(NativeQuantizer)
        for raw, cols in ((b'', 32), (b'x', 0), (b'x', -1), (b'x', 8 * 1024**2)):
            with self.subTest(cols=cols), self.assertRaises(ValueError):
                native.quantize_rows(raw, 8, 10, cols)
        with patch('hfqa.streaming.ROW_WORK_BYTES', 1024):
            with self.assertRaisesRegex(ValueError, 'dequantized chunk'):
                native.quantize_rows(b'\0' * 34 * 16, 8, 8, 32)

    def test_native_passthrough_still_validates_rows(self):
        native = NativeQuantizer.__new__(NativeQuantizer)
        calls = []
        native.lib = SimpleNamespace(ggml_validate_row_data=lambda *args: calls.append(args) or True)
        raw = struct.pack('<4f', 1, 2, 3, 4)
        self.assertEqual(native.quantize_rows(raw, 0, 0, 4), raw)
        self.assertEqual(len(calls), 1)
        native.lib.ggml_validate_row_data = lambda *args: False
        with self.assertRaisesRegex(RuntimeError, 'row validation'):
            native.quantize_rows(raw, 0, 0, 4)

    def fake_xet(self, *, wrong_hash=False, short=False):
        events, chunks = [], []
        class FakeStream:
            def write(inner, blob):
                events.append('write')
                chunks.append(blob)
                return len(blob) - 1 if short else None
            def finish(inner):
                events.append('finish')
                data = b''.join(chunks)
                return SimpleNamespace(xet_info=SimpleNamespace(hash='c' * 64, file_size=len(data),
                    sha256='0' * 64 if wrong_hash else hashlib.sha256(data).hexdigest()))
            def abort(inner): events.append('abort')
        class FakeCommit:
            def __enter__(inner): return inner
            def __exit__(inner, typ, value, tb): events.append('context_abort' if typ else 'cas_finalize')
            def start_upload_stream(inner, **kwargs):
                events.append('start')
                return FakeStream()
        class Config:
            def with_config(inner, config): return inner
        xet = SimpleNamespace(XetConfig=Config, COMPUTE_SHA256=object(),
                              XetSession=lambda **kwargs: SimpleNamespace(new_upload_commit=lambda **kw: FakeCommit()))
        helpers = SimpleNamespace(XetTokenType=SimpleNamespace(WRITE='write'),
            xet_connection_info_refresh_url=lambda **kw: 'https://example.invalid/no-network',
            xet_headers_without_auth=lambda headers: {})
        api = SimpleNamespace(whoami=lambda: {'name': 'webmp3'}, endpoint='https://huggingface.co',
                              _build_hf_headers=lambda: {'authorization': 'Bearer unit-test'})
        return {'hf_xet': xet, 'huggingface_hub.utils._xet': helpers}, api, events

    def test_xet_finalizes_only_after_independent_source_and_output_hash_proof(self):
        modules, api, events = self.fake_xet()
        with patch.dict(sys.modules, modules):
            result = upload_xet_stream(api, 'webmp3/test', 'main', 'test.gguf', self.stream())
        self.assertEqual(events[-2:], ['finish', 'cas_finalize'])
        self.assertTrue(result['native_sha256_verified'])
        self.assertFalse(result['hub_commit_created'])
        self.assertFalse(result['published'])
        self.assertEqual(base64.b64decode(result['sample_base64'])[:4], b'GGUF')
        json.dumps(result)  # Durable receipt may not contain raw Python bytes.

    def test_xet_source_mismatch_native_mismatch_and_short_write_abort(self):
        for source_bad, native_bad, short in ((True, False, False), (False, True, False), (False, False, True)):
            modules, api, events = self.fake_xet(wrong_hash=native_bad, short=short)
            source = dict(self.descriptor, sha256='0' * 64) if source_bad else self.descriptor
            with self.subTest(source_bad=source_bad, native_bad=native_bad, short=short):
                with patch.dict(sys.modules, modules), self.assertRaises((ValueError, RuntimeError)):
                    upload_xet_stream(api, 'webmp3/test', 'main', 'test.gguf', self.stream(source))
                self.assertIn('abort', events)
                self.assertEqual(events[-1], 'context_abort')
                self.assertNotIn('cas_finalize', events)
                if source_bad:
                    self.assertNotIn('finish', events)

    def test_manifest_gates_and_bounded_local_worker_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / 'manifest.json'
            data = {'source': self.descriptor, 'runtime_revision': 'b' * 40, 'quants': ['Q2_K', 'Q3_K_M']}
            manifest.write_text(json.dumps(data))
            args = SimpleNamespace(manifest=manifest, bridge='unused', ggml='unused',
                                   receipt_dir=root / 'receipts', output_dir=root / 'out', xet_repo=None)
            with patch.object(run_stream_worker, 'NativeQuantizer', return_value=self.native), patch.object(run_stream_worker, 'inspect_remote_gguf', return_value=self.remote):
                result = run_stream_worker.execute(args)
            self.assertFalse(result['published'])
            self.assertFalse(result['source_materialized'])
            self.assertFalse(result['full_inference_test_performed'])
            self.assertEqual(len(result['outputs']), 2)
            self.assertTrue((root / 'receipts/stream-result.json').is_file())
            self.assertTrue(all(item['receipt']['output_materialized'] for item in result['outputs']))
            data['source']['size'] = 3 * 1024**3
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'restricted to small proof'):
                run_stream_worker.execute(args)
            data['sources'] = [self.descriptor, self.descriptor]
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'every file'):
                run_stream_worker.read_manifest(manifest)

    def test_multisource_checks_each_complete_original_including_headers(self):
        blob2 = self.blob.replace(b'output', b'hidden')
        descriptor2 = dict(self.descriptor, filename='part-2.gguf', sha256=hashlib.sha256(blob2).hexdigest())
        reader2 = MemoryReader(blob2)
        part2 = inspect_remote_gguf(descriptor2, reader=reader2)
        descriptors = [self.descriptor, descriptor2]
        virtual = ConcatenatedRangeReader(descriptors, [self.reader, reader2])
        tensors = [RemoteTensor(t.name, t.shape, t.type, part.data_offset + t.offset + base, t.nbytes)
                   for part, base in ((self.remote, 0), (part2, len(self.blob))) for t in part.tensors]
        remote = RemoteGGUF(3, 32, self.remote.metadata, self.remote.metadata_entries, tensors, 0, virtual)
        aggregate = {'sources': descriptors, 'repo_id': self.descriptor['repo_id'],
                     'revision': self.descriptor['revision'], 'size': len(self.blob) + len(blob2)}
        stream = QuantizedStream(remote, aggregate, 'Q2_K', self.native)
        list(stream)
        proofs = stream.receipt['source_verification']['sources_verified']
        self.assertEqual([item['sha256'] for item in proofs], [item['sha256'] for item in descriptors])
        self.assertTrue(all(item['sha256_verified'] for item in proofs))
        self.assertEqual(stream.receipt['source_bytes_hashed'], aggregate['size'])
        # Corrupt a trailer byte beyond all tensor data: no tensor-only checksum
        # can detect this, but the per-file complete proof must refuse it.
        reader2.blob = blob2[:-1] + b'?'
        stream = QuantizedStream(remote, aggregate, 'Q2_K', self.native)
        with self.assertRaisesRegex(ValueError, 'source shard SHA-256 mismatch'):
            list(stream)
        self.assertIsNone(stream.receipt)

    def test_tensor_boundary_shards_keep_global_plan_and_full_source_proof(self):
        stream = ShardedQuantizedStream(self.remote, self.descriptor, 'Q3_K_M', self.native, target_shard_bytes=3650)
        self.assertEqual(len(stream.shards), 2)
        blobs = [bytearray() for _ in stream.shards]
        events = iter(stream)
        index, raw = next(events)
        blobs[index].extend(raw)
        self.assertIsNone(stream.receipt)
        for index, raw in events:
            blobs[index].extend(raw)
        self.assertEqual(len(self.native.plans), 1)
        self.assertEqual(stream.receipt['shard_count'], 2)
        self.assertTrue(stream.receipt['source_sha256_verified'])
        names = []
        for index, blob in enumerate(blobs):
            self.assertLessEqual(len(blob), 3650)
            descriptor = dict(self.descriptor, size=len(blob), sha256=hashlib.sha256(blob).hexdigest())
            part = inspect_remote_gguf(descriptor, reader=MemoryReader(blob), allow_split=True)
            self.assertEqual(part.metadata['split.no'], index)
            self.assertEqual(part.metadata['split.count'], 2)
            self.assertEqual(part.metadata['split.tensors.count'], 2)
            self.assertEqual(stream.receipt['shards'][index]['sha256'], hashlib.sha256(blob).hexdigest())
            names.extend(t.name for t in part.tensors)
        self.assertEqual(set(names), {tensor.name for tensor in self.remote.tensors})
        self.assertEqual(len(names), 2)
        self.assertEqual(sum(len(blob) for blob in blobs), stream.size)

    def test_output_shard_budget_rejects_single_tensor_before_reading_payload(self):
        with self.assertRaisesRegex(ValueError, 'single complete tensor'):
            ShardedQuantizedStream(self.remote, self.descriptor, 'Q2_K', self.native, target_shard_bytes=1024)
        self.assertEqual(self.reader.calls, [])

    def test_sharded_checksum_failure_exposes_no_success_receipt(self):
        stream = ShardedQuantizedStream(self.remote, dict(self.descriptor, sha256='0' * 64),
                                       'Q2_K', self.native, target_shard_bytes=3650)
        with self.assertRaisesRegex(ValueError, 'source SHA-256 mismatch'):
            list(stream)
        self.assertIsNone(stream.receipt)

    def test_sharded_xet_producer_hashes_and_no_hub_commit(self):
        modules, api, events = self.fake_xet()
        # fake_xet is deliberately single-file; replace only the stream factory
        # with independent file buffers while retaining its fake CAS context.
        xet = modules['hf_xet']
        finished = []
        class Upload:
            def __init__(inner): inner.chunks = []
            def write(inner, raw): inner.chunks.append(raw)
            def finish(inner):
                data = b''.join(inner.chunks)
                finished.append(data)
                return SimpleNamespace(xet_info=SimpleNamespace(hash='c' * 64, file_size=len(data), sha256=hashlib.sha256(data).hexdigest()))
            def abort(inner): events.append('abort')
        session = xet.XetSession()
        commit = session.new_upload_commit()
        commit.start_upload_stream = lambda **kwargs: Upload()
        xet.XetSession = lambda **kwargs: SimpleNamespace(new_upload_commit=lambda **kw: commit)
        stream = ShardedQuantizedStream(self.remote, self.descriptor, 'Q3_K_M', self.native, target_shard_bytes=3650)
        with patch.dict(sys.modules, modules):
            result = upload_xet_sharded_stream(api, 'webmp3/test', 'main', 'test', stream)
        self.assertEqual(len(result['files']), 2)
        self.assertEqual(len(finished), 2)
        self.assertTrue(result['native_sha256_verified'])
        self.assertFalse(result['hub_commit_created'])
        self.assertEqual(events[-1], 'cas_finalize')
        json.dumps(result)


if __name__ == '__main__':
    unittest.main()
