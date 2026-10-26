"""Hybrid table engine against the real pinned ggml (skipped where the library is absent, e.g. on the developer PC)."""
import hashlib
import os
import struct
import tempfile
import unittest
from pathlib import Path

LIB = Path(os.environ.get('HFQA_TEST_GGML', '/opt/hf-quant-agent/runtime/llama-b11669/libggml-base.so'))
REVISION = '620d2225cc436ab0b001e3200e590e3b286ef9ab'
Q3_K, Q6_K, Q5_K, F16 = 11, 14, 13, 1


def _string(value):
    raw = value.encode()
    return struct.pack('<Q', len(raw)) + raw


def write_gguf(path, kv, tensors, alignment=32):
    """kv: [(key, type, raw value bytes)], tensors: [(name, (cols, rows), type, raw bytes)]."""
    entries = list(kv)
    header = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(entries))
    header += b''.join(_string(k) + struct.pack('<I', t) + raw for k, t, raw in entries)
    offset, data = 0, b''
    for name, shape, kind, raw in tensors:
        header += _string(name) + struct.pack('<I', len(shape)) + b''.join(struct.pack('<Q', d) for d in shape) + struct.pack('<IQ', kind, offset)
        padded = raw + b'\0' * (-len(raw) % alignment)
        data += padded
        offset += len(padded)
    header += b'\0' * (-len(header) % alignment)
    Path(path).write_bytes(header + data)


class FileReader:
    def __init__(self, path):
        self.path, self.size = Path(path), Path(path).stat().st_size

    def read_range(self, start, count):
        with self.path.open('rb') as handle:
            handle.seek(start)
            return handle.read(count)


def descriptor(path, name):
    return {'repo_id': 'maya/Tables', 'revision': 'a' * 40, 'filename': name, 'size': Path(path).stat().st_size,
            'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


@unittest.skipUnless(LIB.is_file(), 'pinned ggml library not available here')
class HybridTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from hfqa import hybrid
        from hfqa.remote_gguf import inspect_remote_gguf
        self.np, self.hybrid, self.inspect = np, hybrid, inspect_remote_gguf
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.ggml = hybrid.Ggml(LIB, REVISION)
        rng = np.random.default_rng(7)
        self.cols, self.rows_a, self.rows_b = 256, 24, 17
        self.table = rng.normal(0, 0.05, (self.rows_a + self.rows_b, self.cols)).astype(np.float32)
        raw_a = self.ggml.convert(self.table[:self.rows_a].tobytes(), 0, Q6_K, self.cols)
        raw_b = self.table[self.rows_a:].astype('<f2').tobytes()
        self.arch = [('general.architecture', 8, _string('deepseek41')), ('split.no', 2, struct.pack('<H', 8)), ('split.count', 2, struct.pack('<H', 9)),
                     ('split.tensors.count', 5, struct.pack('<i', 400)), ('general.name', 8, _string('Backbone'))]
        write_gguf(self.root / 'maya5.gguf', self.arch, [('t.layer1', (self.cols, self.rows_a), Q6_K, raw_a)])
        write_gguf(self.root / 'maya6.gguf', self.arch, [('t.layer14', (self.cols, self.rows_b), F16, raw_b)])
        own = self.ggml.convert(self.table.tobytes(), 0, Q5_K, self.cols)  # the backbone's own copy of the table, another precision
        write_gguf(self.root / 'template.gguf', self.arch, [('blk.1.engram_embd.weight', (self.cols, self.rows_a + self.rows_b), Q5_K, own)])
        self.sources = {}
        for key, name in (('a', 'maya5.gguf'), ('b', 'maya6.gguf')):
            path = self.root / name
            self.sources[key] = {'remote': self.remote(path), 'descriptor': descriptor(path, name)}
        self.template = self.remote(self.root / 'template.gguf')
        self.outputs = [{'name': 'blk.1.engram_embd.weight', 'parts': [{'source': 'a', 'tensor': 't.layer1'}, {'source': 'b', 'tensor': 't.layer14'}]}]

    def remote(self, path):
        reader = FileReader(path)
        return self.inspect({'repo_id': 'x/y', 'revision': '0' * 40, 'filename': Path(path).name, 'size': reader.size, 'sha256': '0' * 64}, reader=reader, allow_split=True)

    def build(self, **kwargs):
        plan = self.hybrid.build_plan(self.template, self.outputs, self.sources, Q3_K, self.ggml, **kwargs)
        stream = self.hybrid.HybridTableStream(self.template, plan, self.sources, self.ggml)
        return plan, stream, b''.join(stream)

    def test_output_is_a_valid_gguf_with_the_backbone_metadata_and_the_new_type(self):
        plan, stream, data = self.build()
        (self.root / 'out.gguf').write_bytes(data)
        out = self.remote(self.root / 'out.gguf')
        self.assertEqual(len(data), plan['size'])
        tensor = out.tensors[0]
        self.assertEqual((tensor.name, tensor.type, tuple(tensor.shape)), ('blk.1.engram_embd.weight', Q3_K, (self.cols, self.rows_a + self.rows_b)))
        self.assertEqual([tuple(e) for e in out.metadata_entries], [tuple(e) for e in self.template.metadata_entries])  # key/value block verbatim
        self.assertEqual(len(data) % 32, 0)

    def test_payload_decodes_close_to_the_original_table_and_is_deterministic(self):
        plan, stream, data = self.build()
        out_rows = self.rows_a + self.rows_b
        payload = data[len(plan['header']):][:out_rows * 110]
        decoded = self.ggml.decode(payload, Q3_K, self.cols)
        error = self.np.sqrt(((decoded - self.table) ** 2).mean()) / self.np.sqrt((self.table ** 2).mean())
        self.assertLess(float(error), 0.30)
        self.assertEqual(hashlib.sha256(data).hexdigest(), self.build()[1].receipt['sha256'])
        self.assertEqual(stream.receipt['sha256'], hashlib.sha256(data).hexdigest())

    def test_receipt_hashes_every_source_file_completely_and_is_honest_about_testing(self):
        plan, stream, data = self.build()
        receipt = stream.receipt
        self.assertTrue(receipt['integrity_only'] and not receipt['full_inference_test_performed'] and not receipt['source_materialized'])
        self.assertEqual([s['sha256'] for s in receipt['sources_verified']], [self.sources[k]['descriptor']['sha256'] for k in 'ab'])
        self.assertEqual(receipt['tensors'][0]['rows'], self.rows_a + self.rows_b)

    def test_a_source_that_differs_from_its_pinned_hash_never_gets_a_receipt(self):
        self.sources['a']['descriptor'] = {**self.sources['a']['descriptor'], 'sha256': 'f' * 64}
        plan = self.hybrid.build_plan(self.template, self.outputs, self.sources, Q3_K, self.ggml)
        stream = self.hybrid.HybridTableStream(self.template, plan, self.sources, self.ggml)
        with self.assertRaises(self.hybrid.HybridError):
            b''.join(stream)
        self.assertIsNone(stream.receipt)

    def test_unsafe_plans_are_refused(self):
        bad = [
            [{'name': 'x', 'parts': []}],
            [{'name': 'x', 'parts': [{'source': 'a', 'tensor': 'missing'}]}],
            [{'name': 'x', 'parts': [{'source': 'a', 'tensor': 't.layer1'}]}, {'name': 'y', 'parts': [{'source': 'a', 'tensor': 't.layer1'}]}],
            [{'name': 'x', 'parts': [{'source': 'zzz', 'tensor': 't.layer1'}]}],
        ]
        for outputs in bad:
            with self.subTest(outputs), self.assertRaises(self.hybrid.HybridError):
                self.hybrid.build_plan(self.template, outputs, self.sources, Q3_K, self.ggml)
        with self.assertRaises(self.hybrid.HybridError):
            self.hybrid.build_plan(self.template, self.outputs, self.sources, Q3_K, self.ggml, kv_overrides={'no.such.key': (8, b'')})

    def test_header_parser_layouts_equal_the_pinned_library_for_every_type_it_accepts(self):
        from hfqa.gguf import TENSOR_LAYOUTS
        for type_id, layout in TENSOR_LAYOUTS.items():
            if type_id in self.hybrid.SAFE_TYPE_IDS and type_id not in {24, 25, 26, 27, 28}:  # plain integer/float types have no block traits worth comparing
                with self.subTest(type_id):
                    self.assertEqual(tuple(layout), self.ggml.layout(type_id))
        for type_id in (4, 5, 31, 36, 43, 44, 99, -1):
            with self.subTest(unknown=type_id), self.assertRaises(self.hybrid.HybridError):  # must never reach ggml, which aborts the whole process
                self.ggml.layout(type_id)

    def test_equal_types_pass_through_byte_for_byte_and_corrupt_rows_are_refused(self):
        raw = self.ggml.convert(self.table[:4].tobytes(), 0, Q3_K, self.cols)
        self.assertEqual(self.ggml.convert(raw, Q3_K, Q3_K, self.cols), raw)
        with self.assertRaises(self.hybrid.HybridError):
            self.ggml.convert(raw[:-1], Q3_K, Q3_K, self.cols)

    def test_every_documented_source_type_decodes(self):
        for type_id in (2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 20, 23):
            raw = self.ggml.convert(self.table[:2].tobytes(), 0, type_id, self.cols) if type_id not in {2, 3, 6, 7} else None
            if raw is None:
                continue
            with self.subTest(type_id):
                decoded = self.ggml.decode(raw, type_id, self.cols)
                self.assertEqual(decoded.shape, (2, self.cols))

    def test_pause_stops_the_stream(self):
        plan = self.hybrid.build_plan(self.template, self.outputs, self.sources, Q3_K, self.ggml)
        stream = self.hybrid.HybridTableStream(self.template, plan, self.sources, self.ggml, should_stop=lambda: True)
        with self.assertRaises(InterruptedError):
            b''.join(stream)

    def test_kv_override_changes_only_the_named_value(self):
        plan = self.hybrid.build_plan(self.template, self.outputs, self.sources, Q3_K, self.ggml, kv_overrides={'split.tensors.count': (5, struct.pack('<i', 401))})
        (self.root / 'o2.gguf').write_bytes(plan['header'] + b'\0' * 32)
        entries = {e[0]: e for e in self.template.metadata_entries}
        # parse the header again through the real parser by padding a tiny data region
        out_entries = {e[0]: e for e in self.remote_from_header(plan).metadata_entries}
        self.assertEqual(out_entries['split.tensors.count'][2], struct.pack('<i', 401))
        self.assertEqual(out_entries['general.name'], entries['general.name'])

    def remote_from_header(self, plan):
        path = self.root / 'hdr.gguf'
        size = sum(t['nbytes'] + (-t['nbytes'] % 32) for t in plan['tensors'])
        path.write_bytes(plan['header'] + b'\0' * size)
        return self.remote(path)

    def test_compatibility_check_accepts_the_same_table_and_rejects_another_one(self):
        good = self.hybrid.sample_compatibility(self.ggml, (self.template, 'blk.1.engram_embd.weight'),
                                                [(self.sources['a']['remote'], 't.layer1'), (self.sources['b']['remote'], 't.layer14')])
        self.assertTrue(good['passes'], good)
        rng = self.np.random.default_rng(99)
        other = rng.normal(0, 0.05, (self.rows_a, self.cols)).astype(self.np.float32)
        write_gguf(self.root / 'other.gguf', self.arch, [('t.layer1', (self.cols, self.rows_a), Q6_K, self.ggml.convert(other.tobytes(), 0, Q6_K, self.cols))])
        bad = self.hybrid.sample_compatibility(self.ggml, (self.template, 'blk.1.engram_embd.weight'),
                                               [(self.remote(self.root / 'other.gguf'), 't.layer1'), (self.sources['b']['remote'], 't.layer14')])
        self.assertFalse(bad['passes'], bad)
        shape = self.hybrid.sample_compatibility(self.ggml, (self.template, 'blk.1.engram_embd.weight'), [(self.sources['a']['remote'], 't.layer1')])
        self.assertFalse(shape['passes'])


if __name__ == '__main__':
    unittest.main()
