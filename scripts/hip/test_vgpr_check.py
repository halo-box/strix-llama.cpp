import copy
import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location('vgpr_check', Path(__file__).with_name('gcn-cdna-vgpr-check.py'))
assert spec is not None and spec.loader is not None
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

ASSEMBLY = '''
    .amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx1151"
    .amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .name: _Z6kernelv
    .private_segment_fixed_size: 3392
    .vgpr_count: 256
    .vgpr_spill_count: 1036
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
    .end_amdgpu_metadata
'''


class TestVgprCheck(unittest.TestCase):
    def setUp(self):
        self.kernels = checker.parse_assembly(ASSEMBLY, 'gfx1151')
        self.units = {'sample': 1}
        self.budgets = {'_Z6kernelv': dict(self.kernels['_Z6kernelv'], reason='compiler_change')}
        self.baseline = {
            'schema_version': 1,
            'architecture': 'gfx1151',
            'compiler_version': 'fixture compiler',
            'translation_units': {'sample': 1},
            'reasons': {'compiler_change': 'Same source, compiler change; runtime cost is unqualified.'},
            'kernels': self.budgets,
        }

    def check(self):
        return checker.find_regressions(self.kernels, self.units, self.baseline, 'fixture compiler')

    def test_assembly_units_and_unchanged_baseline(self):
        self.assertEqual(self.kernels['_Z6kernelv'], {'vgprs': 256, 'spill_stores': 1036, 'scratch_bytes': 3392})
        self.assertEqual(self.check(), [])

    def test_each_increased_budget_fails(self):
        for field in checker.FIELDS:
            with self.subTest(field=field):
                self.kernels['_Z6kernelv'][field] += 1
                self.assertTrue(any(field in e or 'architectural limit' in e for e in self.check()))
                self.kernels['_Z6kernelv'][field] -= 1

    def test_improvements_pass(self):
        self.kernels['_Z6kernelv'] = {'vgprs': 192, 'spill_stores': 0, 'scratch_bytes': 0}
        self.assertEqual(self.check(), [])

    def test_metrics_cannot_offset_each_other(self):
        self.kernels['_Z6kernelv']['vgprs'] = 192
        self.kernels['_Z6kernelv']['spill_stores'] += 1
        self.assertTrue(any('spill_stores' in e for e in self.check()))

    def test_new_small_spill_fails(self):
        self.kernels['_Z3newv'] = {'vgprs': 32, 'spill_stores': 1, 'scratch_bytes': 4}
        self.assertTrue(any('new spill' in e for e in self.check()))

    def test_new_nonspilling_kernel_passes(self):
        self.kernels['_Z3newv'] = {'vgprs': 256, 'spill_stores': 0, 'scratch_bytes': 0}
        self.assertEqual(self.check(), [])

    def test_empty_or_reduced_coverage_fails(self):
        self.units = {}
        self.assertTrue(any('coverage' in e for e in self.check()))
        self.kernels = {}
        self.assertTrue(any('missing baseline kernel' in e for e in self.check()))

    def test_wrong_compiler_fails(self):
        with self.assertRaisesRegex(ValueError, 'compiler identity'):
            checker.find_regressions(self.kernels, self.units, self.baseline, 'another compiler')

    def test_reason_is_required(self):
        self.budgets['_Z6kernelv']['reason'] = ''
        with self.assertRaisesRegex(ValueError, 'baseline reason'):
            self.check()

    def test_incomplete_or_wrong_architecture_metadata_fails(self):
        for text in [ASSEMBLY.replace('gfx1151', 'gfx908'), ASSEMBLY.replace('    .vgpr_spill_count: 1036\n', ''), ASSEMBLY.replace('.end_amdgpu_metadata', ''), ASSEMBLY.replace('.wavefront_size: 32', '.wavefront_size: 64')]:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    checker.parse_assembly(text, 'gfx1151')

    def test_no_unbounded_legacy_ignore(self):
        usage = self.kernels.pop('_Z6kernelv')
        self.kernels['_ZL12rwkv_wkv_f32ILi128EEviiiiPKfS1_S1_S1_S1_S1_Pf'] = usage
        self.assertTrue(any('new spill' in e for e in self.check()))

    def test_empty_translation_unit_metadata(self):
        text = ASSEMBLY.split('amdhsa.kernels:', 1)[0] + 'amdhsa.kernels: []\n.end_amdgpu_metadata\n'
        self.assertEqual(checker.parse_assembly(text, 'gfx1151'), {})
        with self.assertRaises(ValueError):
            checker.parse_assembly(text.replace('amdhsa.kernels: []', 'amdhsa.kernels: unknown'), 'gfx1151')

    def test_invalid_baseline_fails(self):
        for change in [{'schema_version': 2}, {'architecture': 'gfx908'}, {'translation_units': {}}, {'kernels': {}}]:
            baseline = copy.deepcopy(self.baseline)
            baseline.update(change)
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    checker.find_regressions(self.kernels, self.units, baseline, 'fixture compiler')


if __name__ == '__main__':
    unittest.main()
