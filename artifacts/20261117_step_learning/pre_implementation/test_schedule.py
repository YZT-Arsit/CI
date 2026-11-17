"""Every local update must have balanced labels and include control coverage."""
import collections
import unittest
from learning_data import training_schedule


class ScheduleTests(unittest.TestCase):
    def test_balanced_batches_and_all_strata(self):
        for seed in [17, 29, 41]:
            records = training_schedule(seed)
            self.assertEqual(len(records), 144)
            self.assertEqual([r['step'] for r in records], list(range(144)))
            for row in records:
                self.assertEqual(len(row['ids']), 4)
                self.assertEqual(collections.Counter(row['labels']), {0: 2, 1: 2})
                self.assertEqual(len(row['names']), 4)
                if row['kind'] == 'switch':
                    self.assertEqual(row['names'][0], row['names'][1])
                    self.assertEqual(row['names'][2], row['names'][3])
            counts = collections.Counter(r['kind'] for r in records)
            self.assertEqual(counts, {'switch': 108, 'control': 36})
            strata = collections.Counter(tuple(r['stratum']) for r in records if r['kind'] == 'switch')
            self.assertEqual(len(strata), 6)
            self.assertEqual(set(strata.values()), {18})

    def test_reproducible_distinct_schedules(self):
        self.assertEqual(training_schedule(17), training_schedule(17))
        self.assertNotEqual(training_schedule(17), training_schedule(29))
        self.assertNotEqual(training_schedule(29), training_schedule(41))


if __name__ == '__main__': unittest.main()
