"""Measured deviation from the source: parsing of the real llama-perplexity output, the pinned evaluation texts, and the skip-never-fail rule. Offline."""
import hashlib
import tempfile
import unittest
from pathlib import Path

from hfqa import runner_quality

# The statistics block exactly as llama-perplexity (b11669) printed it for Spark-X2.6-1.7B Q2_K against Q5_K_M (6 chunks, 128 tokens).
REAL_OUTPUT = """
====== Perplexity statistics ======
Mean PPL(Q)                   :  53.738647 ±   9.448440
Mean PPL(base)                :  19.849005 ±   3.183624
Cor(ln(PPL(Q)), ln(PPL(base))):  85.95%
Mean ln(PPL(Q)/PPL(base))     :   0.995979 ±   0.090360
Mean PPL(Q)/PPL(base)         :   2.707372 ±   0.244638
Mean PPL(Q)-PPL(base)         :  33.889642 ±   6.906745

====== KL divergence statistics ======
Mean    KLD:   1.162820 ±   0.049944
Maximum KLD:   5.366421
99.9%   KLD:   5.200749
99.0%   KLD:   4.007181
95.0%   KLD:   3.041921
Median  KLD:   0.926483
Minimum KLD:   0.000023

====== Token probability statistics ======
Mean    Δp: -12.785 ± 1.404 %
RMS Δp    : 30.112 ± 1.632 %
Same top p: 47.090 ± 2.571 %
"""


def book(prefix_chars=60000, marker='*** START OF THE PROJECT GUTENBERG EBOOK X ***', shift=0):
    return ('Header text of the book.\r\n' + marker + '\r\n' + ''.join(chr(97 + (i + shift) % 26) for i in range(prefix_chars))).encode('utf-8')


def spec_for(raw, start=100, characters=500):
    text = raw.decode('utf-8').replace('\r\n', '\n')
    import re
    piece = text[re.search(r'\*\*\* START OF[^\n]*\n', text).end():][start:start + characters]
    return {'name': 'english', 'url': 'https://example.org/book.txt', 'start': start, 'characters': characters, 'bytes': len(piece.encode('utf-8')),
            'sha256': hashlib.sha256(piece.encode('utf-8')).hexdigest()}


class ParseTests(unittest.TestCase):
    def test_the_real_output_is_read_completely(self):
        stats = runner_quality.parse_statistics(REAL_OUTPUT)
        self.assertEqual(stats['ppl_quant'], {'value': 53.738647, 'error': 9.44844})
        self.assertEqual(stats['ppl_source']['value'], 19.849005)
        self.assertEqual(stats['ppl_ratio'], {'value': 2.707372, 'error': 0.244638})  # not confused with the "Mean PPL(Q)" or the difference line
        self.assertEqual(stats['mean_kld'], {'value': 1.16282, 'error': 0.049944})
        self.assertEqual((stats['median_kld'], stats['p99_kld'], stats['max_kld']), (0.926483, 4.007181, 5.366421))
        self.assertEqual(stats['same_top_token_percent'], {'value': 47.09, 'error': 2.571})
        self.assertEqual(stats['rms_delta_p_percent']['value'], 30.112)

    def test_an_output_without_the_statistics_gives_no_numbers(self):
        stats = runner_quality.parse_statistics('llama_model_load: error loading model')
        self.assertIsNone(stats['mean_kld'])
        self.assertIsNone(stats['ppl_ratio'])


class TextTests(unittest.TestCase):
    def test_the_pinned_slice_is_cut_after_the_start_marker_and_verified(self):
        raw = book()
        spec = spec_for(raw)
        self.assertEqual(len(runner_quality.fetch_text(spec, lambda url: raw)), 500)

    def test_anything_that_is_not_exactly_the_pinned_text_is_unavailable_not_a_failure(self):
        raw = book()
        spec = spec_for(raw)
        for name, download in (('changed text', lambda url: book(marker='*** START OF OTHER ***', shift=1)), ('no marker', lambda url: b'plain text only'),
                               ('network error', lambda url: (_ for _ in ()).throw(OSError('offline'))), ('not utf-8', lambda url: b'\xff\xfe\x00')):
            with self.subTest(name), self.assertRaises(runner_quality.Unavailable):
                runner_quality.fetch_text(spec, download)

    def test_the_real_pins_name_two_public_domain_books_and_differ_from_the_calibration_text(self):
        from hfqa import imatrix_data
        self.assertEqual([t['name'] for t in runner_quality.EVAL_TEXTS], ['english', 'chinese'])
        for item in runner_quality.EVAL_TEXTS:
            self.assertRegex(item['sha256'], r'^[a-f0-9]{64}$')
            self.assertTrue(item['url'].startswith('https://www.gutenberg.org/'))
        self.assertNotEqual(runner_quality.EVAL_TEXTS[0]['sha256'], imatrix_data.DATASET_SHA256)


class MeasureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        raw = book()
        self.raw = raw
        self.specs = [spec_for(raw)]
        self.spec = {'method': 'stock_perplexity_kld', 'context': 512, 'texts': self.specs}
        self.tool = self.work / 'llama-perplexity'
        self.tool.write_bytes(b'tool')
        self.source = self.work / 'source.gguf'
        self.source.write_bytes(b's')
        self.quants = {'Q2_K': self.work / 'q2.gguf', 'Q4_K_M': self.work / 'q4.gguf'}
        for path in self.quants.values():
            path.write_bytes(b'q')
        self.calls = []
        # the module reads its own SPEC; point it at the fake text for these tests
        from unittest.mock import patch
        patcher = patch.object(runner_quality, 'SPEC', self.spec)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_fake(self, command, logs, name, env):
        self.calls.append((name, command))
        return REAL_OUTPUT if '--kl-divergence' in command else 'Final estimate: PPL = 19.8'

    def measure(self, **kw):
        return runner_quality.measure(self.source, self.quants, self.tool, self.work, 4, 3_420_000_000, run=kw.pop('run', self.run_fake),
                                      download=kw.pop('download', lambda url: self.raw), log=lambda *a, **k: None)

    def test_the_source_is_scored_once_and_every_quant_against_it(self):
        result = self.measure()
        self.assertEqual(sorted(result['per_quant']), ['Q2_K', 'Q4_K_M'])
        self.assertEqual([name for name, _ in self.calls], ['source', 'Q2_K', 'Q4_K_M'])
        base = self.calls[0][1]
        self.assertIn('--kl-divergence-base', base)
        self.assertNotIn('--kl-divergence', [part for part in base if part == '--kl-divergence'])
        self.assertIn('--kl-divergence', self.calls[1][1])
        self.assertEqual(result['evidence']['chunks'], 64)
        self.assertEqual(result['evidence']['context'], 512)
        self.assertEqual(result['evidence']['texts'][0]['sha256'], self.specs[0]['sha256'])

    def test_a_larger_source_is_scored_on_fewer_windows(self):
        self.assertEqual(runner_quality.chunks_for(3_400_000_000), 64)
        self.assertEqual(runner_quality.chunks_for(7_000_000_000), 12)

    def test_nothing_that_goes_wrong_here_can_fail_a_release(self):
        def crash(command, logs, name, env):
            raise RuntimeError('llama-perplexity failed (exit 1)')
        self.assertIn('did not finish', self.measure(run=crash)['skipped'])
        self.assertIn('could not be read', self.measure(run=lambda *a: 'nothing useful')['skipped'])
        self.assertIn('differs from its pinned hash', self.measure(download=lambda url: book(marker='*** START OF X ***', shift=1))['skipped'])
        self.assertIn('could not be fetched', self.measure(download=lambda url: (_ for _ in ()).throw(OSError('offline')))['skipped'])


if __name__ == '__main__':
    unittest.main()
