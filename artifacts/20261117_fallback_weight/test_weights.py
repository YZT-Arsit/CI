"""Check the intended fixed-distribution weighting before training."""
import json
import unittest
from pathlib import Path

R = Path(__file__).resolve().parent


class WeightTests(unittest.TestCase):
    def test_minority_weight_and_global_normalization(self):
        from weighting import weights
        cases = [json.loads(x) for x in (R.parent/'20261117_frozen_crossover/train_oracle.jsonl').read_text().splitlines()]
        baseline = weights(cases, 1.)
        changed = weights(cases, 4.)
        zero = [c['observation']['role']=='Worker' and not any(x['safe_completion'] for x in c['candidates']) for c in cases]
        self.assertEqual(sum(zero), 18)
        self.assertEqual(baseline, [1.]*320)
        self.assertAlmostEqual(sum(changed), 320.)
        self.assertAlmostEqual(changed[zero.index(True)]/changed[zero.index(False)], 4.)


if __name__ == '__main__':
    unittest.main()
