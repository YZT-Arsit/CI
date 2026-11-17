"""Regression: repeated semantic batches must add names without changing other factors."""
import copy
import sys
import unittest
from pathlib import Path
R = Path(__file__).resolve().parent
sys.path.insert(0, str(R.parent / '20261117_small_fit'))
from fit_data import batches, schedule
sys.path.insert(0, str(R))
from rename_views import views_for


class ViewTests(unittest.TestCase):
    def test_sixteen_distinct_name_seeds_per_batch(self):
        bs = batches()
        for index, batch in enumerate(bs):
            values = [v['names'] for step, bi in enumerate(schedule()) if bi == index for v in views_for(batch, step)]
            self.assertEqual(len(values), 16)
            self.assertEqual(len(set(values)), 16, index)

    def test_first_epoch_and_all_non_name_fields_unchanged(self):
        bs = batches(); original = copy.deepcopy(bs)
        for step, index in enumerate(schedule()):
            batch = bs[index]; views = views_for(batch, step)
            if step < 18: self.assertEqual(views, batch['views'])
            for before, after in zip(batch['views'], views):
                self.assertEqual({k: v for k, v in before.items() if k != 'names'}, {k: v for k, v in after.items() if k != 'names'})
        self.assertEqual(bs, original)


if __name__ == '__main__':
    unittest.main()
